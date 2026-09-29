// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reconcile loop: ModelExpressServer -> Deployment + Service + PVC.

use crate::crd::{CacheStorage, ModelExpressServer, ModelExpressServerStatus};
use crate::deployment::{DesiredState, render};
use crate::labels;
use crate::rbac::{
    ServerRbac, auth_delegator_binding_name, render_auth_delegator_binding, render_rbac, role_name,
    service_account_name,
};
use crate::tls::{self, TlsDefaults, TlsDefaultsError, TlsSettings};
use futures::StreamExt;
use k8s_openapi::api::apps::v1::Deployment;
use k8s_openapi::api::core::v1::{PersistentVolumeClaim, Service, ServiceAccount};
use k8s_openapi::api::networking::v1::NetworkPolicy;
use k8s_openapi::api::rbac::v1::{ClusterRoleBinding, Role, RoleBinding};
use k8s_openapi::apimachinery::pkg::apis::meta::v1::Condition;
use kube::api::{Api, ObjectMeta, PartialObjectMeta, Patch, PatchParams};
use kube::runtime::WatchStreamExt;
use kube::runtime::controller::{Action, Controller};
use kube::runtime::watcher;
use kube::runtime::watcher::metadata_watcher;
use kube::{Client, Resource, ResourceExt};
use std::sync::Arc;
use std::time::Duration;

pub const FIELD_MANAGER: &str = "modelexpress-operator";

/// Carried only by CRs that own a ClusterRoleBinding. Cluster-scoped objects
/// cannot be garbage collected through a namespaced owner, so enforce mode
/// needs the operator to delete the binding itself.
pub const AUTH_DELEGATOR_FINALIZER: &str = "modelexpress.opendatahub.io/auth-delegator";

const WATCH_TIMEOUT_SECS: u32 = 290;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("kube api: {0}")]
    Kube(#[from] kube::Error),
    #[error("TLS defaults: {0}")]
    TlsDefaults(TlsDefaultsError),
    #[error("spec.image is unset and the operator has no default server image")]
    ServerImageUnset,
    #[error("CR has no namespace")]
    MissingNamespace,
    #[error("CR has no uid")]
    MissingUid,
    #[error("existingClaim {claim} not found")]
    MissingClaim { claim: String },
    #[error(
        "existingClaim {claim} is single-node ({modes:?}) but replicas is {replicas}; \
         scheduling would deadlock"
    )]
    SingleNodeClaim {
        claim: String,
        modes: Vec<String>,
        replicas: i32,
    },
}

pub struct Ctx {
    pub client: Client,
    pub tls_defaults: Arc<dyn TlsDefaults>,
    /// The server image for CRs that leave spec.image unset.
    pub default_server_image: Option<String>,
}

pub async fn run(
    client: Client,
    tls_defaults: Arc<dyn TlsDefaults>,
    default_server_image: Option<String>,
) -> Result<(), kube::Error> {
    let servers = Api::<ModelExpressServer>::all(client.clone());
    let deployments = Api::<Deployment>::all(client.clone());
    let services = Api::<Service>::all(client.clone());
    let pvcs = Api::<PersistentVolumeClaim>::all(client.clone());

    let netpols = Api::<NetworkPolicy>::all(client.clone());
    let sas = Api::<ServiceAccount>::all(client.clone());
    let roles = Api::<Role>::all(client.clone());
    let bindings = Api::<RoleBinding>::all(client.clone());

    // Unfiltered these cache every object of their kind in the cluster, and
    // ServiceAccounts and RoleBindings run to thousands in a real one.
    let owned = watcher::Config::default()
        .labels(labels::MANAGED_BY_SELECTOR)
        .timeout(WATCH_TIMEOUT_SECS);

    let controller = Controller::new(
        servers,
        watcher::Config::default().timeout(WATCH_TIMEOUT_SECS),
    )
    .owns_stream(owned_meta(deployments, owned.clone()))
    .owns_stream(owned_meta(services, owned.clone()))
    .owns_stream(owned_meta(pvcs, owned.clone()))
    .owns_stream(owned_meta(netpols, owned.clone()))
    .owns_stream(owned_meta(sas, owned.clone()))
    .owns_stream(owned_meta(roles, owned.clone()))
    .owns_stream(owned_meta(bindings, owned));

    // A change to the TLS defaults re-renders every server that relies on
    // them. They are global, so a full requeue is the cheapest correct mapping.
    let defaults_changed = tls_defaults.updates().await.map(|_| ());
    let controller = controller.reconcile_all_on(defaults_changed);

    controller
        .shutdown_on_signal()
        .run(
            reconcile,
            error_policy,
            Arc::new(Ctx {
                client,
                tls_defaults,
                default_server_image,
            }),
        )
        .for_each(|result| async move {
            match result {
                Ok((obj, _)) => tracing::debug!(name = %obj.name, "reconciled"),
                Err(err) => tracing::warn!(%err, "reconcile failed"),
            }
        })
        .await;
    Ok(())
}

/// Owned objects are only read for their ownerReferences, so the spec and
/// status a full watch would cache are dead weight.
fn owned_meta<K>(
    api: Api<K>,
    config: watcher::Config,
) -> impl futures::Stream<Item = Result<PartialObjectMeta<K>, watcher::Error>> + Send
where
    K: Resource<DynamicType = ()>
        + Clone
        + serde::de::DeserializeOwned
        + std::fmt::Debug
        + Send
        + 'static,
{
    metadata_watcher(api, config).touched_objects()
}

async fn reconcile(cr: Arc<ModelExpressServer>, ctx: Arc<Ctx>) -> Result<Action, Error> {
    let start = std::time::Instant::now();
    let result = reconcile_inner(cr, ctx).await;
    metrics::histogram!("mxop_reconcile_duration_seconds").record(start.elapsed().as_secs_f64());
    match &result {
        Ok(_) => metrics::counter!("mxop_reconcile_total", "outcome" => "ok").increment(1),
        Err(err) => {
            metrics::counter!("mxop_reconcile_total", "outcome" => "error").increment(1);
            metrics::counter!("mxop_reconcile_errors_total", "reason" => reason(err)).increment(1);
        }
    }
    result
}

#[tracing::instrument(skip_all, fields(name = %cr.name_any(), namespace = cr.namespace()))]
async fn reconcile_inner(cr: Arc<ModelExpressServer>, ctx: Arc<Ctx>) -> Result<Action, Error> {
    let ns = cr.namespace().ok_or(Error::MissingNamespace)?;
    let name = cr.name_any();

    if cr.metadata.deletion_timestamp.is_some() {
        release_auth_delegator(&cr, &ns, &name, &ctx).await?;
        return Ok(Action::await_change());
    }

    let result = apply(&cr, &ns, &name, &ctx).await;
    let (condition, endpoint, observed_generation) = match &result {
        Ok(()) => (
            ready_condition(&cr, "True", "Applied", "resources applied"),
            Some(endpoint(&name, &ns, cr.spec.port, cr.spec.tls.is_some())),
            cr.metadata.generation,
        ),
        Err(err) => (
            ready_condition(&cr, "False", reason(err), &err.to_string()),
            None,
            // clients compare this against metadata.generation to decide
            // whether the current spec has been acted on
            cr.status.as_ref().and_then(|s| s.observed_generation),
        ),
    };
    write_status(&ns, &name, &ctx, condition, endpoint, observed_generation).await?;

    result.map(|()| Action::requeue(Duration::from_secs(300)))
}

/// What clients set MODEL_EXPRESS_ENDPOINT to. The Service is always named
/// after the CR. The scheme is what the clients parse: the Rust client
/// negotiates TLS for `https` only, and the Python client strips `http://`
/// and `https://` and nothing else.
pub fn endpoint(name: &str, ns: &str, port: i32, tls: bool) -> String {
    let scheme = if tls { "https" } else { "http" };
    format!("{scheme}://{name}.{ns}.svc.cluster.local:{port}")
}

/// The CR's image, else the operator's default.
fn server_image<'a>(
    spec: &'a crate::crd::ModelExpressServerSpec,
    default: Option<&'a str>,
) -> Result<&'a str, Error> {
    spec.image
        .as_deref()
        .or(default)
        .ok_or(Error::ServerImageUnset)
}

#[tracing::instrument(skip_all)]
async fn apply(cr: &ModelExpressServer, ns: &str, name: &str, ctx: &Ctx) -> Result<(), Error> {
    check_existing_claim(cr, ns, ctx).await?;
    let uid = cr.metadata.uid.clone().ok_or(Error::MissingUid)?;

    let tls_defaults = match &cr.spec.tls {
        Some(config) if tls::needs_defaults(config) => ctx
            .tls_defaults
            .current()
            .await
            .map_err(Error::TlsDefaults)?,
        _ => TlsSettings::default(),
    };
    let DesiredState {
        mut deployment,
        mut service,
        pvc,
        network_policy,
    } = render(
        name,
        &cr.spec,
        server_image(&cr.spec, ctx.default_server_image.as_deref())?,
        &tls_defaults,
    );

    let owner = cr.controller_owner_ref(&());
    stamp(&mut deployment.metadata, ns, owner.clone());
    stamp(&mut service.metadata, ns, owner.clone());

    let params = PatchParams::apply(FIELD_MANAGER).force();

    // RBAC before the Deployment: pods referencing a not-yet-existing SA
    // fail admission at the ReplicaSet level.
    apply_rbac(cr, ns, ctx, &params, &uid).await?;

    if let Some(mut pvc) = pvc {
        stamp(&mut pvc.metadata, ns, owner);
        let api = Api::<PersistentVolumeClaim>::namespaced(ctx.client.clone(), ns);
        let pvc_name = pvc.metadata.name.clone().unwrap_or_default();
        api.patch(&pvc_name, &params, &Patch::Apply(&pvc)).await?;
    }

    let api = Api::<Deployment>::namespaced(ctx.client.clone(), ns);
    api.patch(name, &params, &Patch::Apply(&deployment)).await?;

    let api = Api::<Service>::namespaced(ctx.client.clone(), ns);
    api.patch(name, &params, &Patch::Apply(&service)).await?;

    let api = Api::<NetworkPolicy>::namespaced(ctx.client.clone(), ns);
    match network_policy {
        Some(mut netpol) => {
            stamp(&mut netpol.metadata, ns, cr.controller_owner_ref(&()));
            api.patch(name, &params, &Patch::Apply(&netpol)).await?;
        }
        // config removed: clean up the stale policy rather than leave a
        // stray ingress restriction behind
        None => delete_if_owned(&api, name, &uid).await?,
    }

    tracing::debug!("desired state applied");
    Ok(())
}

#[tracing::instrument(skip_all)]
async fn apply_rbac(
    cr: &ModelExpressServer,
    ns: &str,
    ctx: &Ctx,
    params: &PatchParams,
    uid: &str,
) -> Result<(), Error> {
    let name = cr.name_any();
    let ServerRbac {
        service_account,
        role,
        role_binding,
    } = render_rbac(&name, &cr.spec);
    let owner = cr.controller_owner_ref(&());

    let sa_api = Api::<ServiceAccount>::namespaced(ctx.client.clone(), ns);
    let generated_sa = service_account_name(&name, &cr.spec);
    match service_account {
        Some(mut sa) => {
            stamp(&mut sa.metadata, ns, owner.clone());
            sa_api
                .patch(&generated_sa, params, &Patch::Apply(&sa))
                .await?;
        }
        // user brought their own SA: remove the generated one if it lingers
        None => delete_if_owned(&sa_api, &format!("{name}-server"), uid).await?,
    }

    let role_api = Api::<Role>::namespaced(ctx.client.clone(), ns);
    let binding_api = Api::<RoleBinding>::namespaced(ctx.client.clone(), ns);
    let rname = role_name(&name);
    match (role, role_binding) {
        (Some(mut role), Some(mut binding)) => {
            stamp(&mut role.metadata, ns, owner.clone());
            stamp(&mut binding.metadata, ns, owner);
            role_api.patch(&rname, params, &Patch::Apply(&role)).await?;
            binding_api
                .patch(&rname, params, &Patch::Apply(&binding))
                .await?;
        }
        // backend no longer needs grants (redis, or user-managed SA)
        _ => {
            delete_if_owned(&binding_api, &rname, uid).await?;
            delete_if_owned(&role_api, &rname, uid).await?;
        }
    }

    apply_auth_delegator(cr, ns, &name, ctx, params).await
}

/// The binding outlives the CR unless the operator deletes it, so the
/// finalizer goes on before the binding is created.
#[tracing::instrument(skip_all)]
async fn apply_auth_delegator(
    cr: &ModelExpressServer,
    ns: &str,
    name: &str,
    ctx: &Ctx,
    params: &PatchParams,
) -> Result<(), Error> {
    let api = Api::<ClusterRoleBinding>::all(ctx.client.clone());
    let binding_name = auth_delegator_binding_name(name, ns);
    match render_auth_delegator_binding(name, ns, &cr.spec) {
        Some(binding) => {
            set_finalizer(cr, ns, name, ctx, true).await?;
            api.patch(&binding_name, params, &Patch::Apply(&binding))
                .await?;
        }
        None => {
            delete_if_managed(&api, &binding_name, name).await?;
            set_finalizer(cr, ns, name, ctx, false).await?;
        }
    }
    Ok(())
}

/// Runs on a CR that is going away: drop the binding, then the finalizer that
/// held the CR open for it.
#[tracing::instrument(skip_all)]
async fn release_auth_delegator(
    cr: &ModelExpressServer,
    ns: &str,
    name: &str,
    ctx: &Ctx,
) -> Result<(), Error> {
    let api = Api::<ClusterRoleBinding>::all(ctx.client.clone());
    delete_if_managed(&api, &auth_delegator_binding_name(name, ns), name).await?;
    set_finalizer(cr, ns, name, ctx, false).await
}

async fn set_finalizer(
    cr: &ModelExpressServer,
    ns: &str,
    name: &str,
    ctx: &Ctx,
    present: bool,
) -> Result<(), Error> {
    let current = cr.finalizers();
    let held = current.iter().any(|f| f == AUTH_DELEGATOR_FINALIZER);
    if held == present {
        return Ok(());
    }
    // Server-side apply: metadata.finalizers is a set, so applying only this
    // one leaves finalizers other controllers own alone, and applying none
    // removes just this one. A merge patch would write back the whole array
    // from a snapshot that may already be stale.
    let held: &[&str] = if present {
        &[AUTH_DELEGATOR_FINALIZER]
    } else {
        &[]
    };
    let api = Api::<ModelExpressServer>::namespaced(ctx.client.clone(), ns);
    let patch = serde_json::json!({
        "apiVersion": ModelExpressServer::api_version(&()),
        "kind": ModelExpressServer::kind(&()),
        "metadata": { "name": name, "finalizers": held },
    });
    match api
        .patch(
            name,
            &PatchParams::apply(FIELD_MANAGER).force(),
            &Patch::Apply(&patch),
        )
        .await
    {
        Ok(_) => Ok(()),
        // the CR is already gone; nothing left to hold open
        Err(kube::Error::Api(e)) if e.code == 404 => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Cluster-scoped objects carry no ownerReference back to a namespaced CR, so
/// the managed labels are the only proof the operator created this one.
async fn delete_if_managed(
    api: &Api<ClusterRoleBinding>,
    name: &str,
    cr_name: &str,
) -> Result<(), Error> {
    let Some(existing) = api.get_opt(name).await? else {
        return Ok(());
    };
    let labels = existing.labels();
    let managed = labels.get(labels::MANAGED_BY_LABEL).map(String::as_str)
        == Some(labels::MANAGED_BY)
        && labels.get(labels::INSTANCE_LABEL).map(String::as_str) == Some(cr_name);
    if !managed {
        tracing::debug!(name, "not operator-owned, leaving in place");
        return Ok(());
    }
    match api.delete(name, &Default::default()).await {
        Ok(_) => Ok(()),
        Err(kube::Error::Api(e)) if e.code == 404 => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Cleanup targets names derived from the CR, not names the operator can prove
/// it created. `spec.networkPolicy` is unset by default, so without this check
/// the netpol branch deletes a user's unrelated same-named policy on every
/// reconcile.
async fn delete_if_owned<K>(api: &Api<K>, name: &str, owner_uid: &str) -> Result<(), Error>
where
    K: kube::Resource + Clone + serde::de::DeserializeOwned + std::fmt::Debug,
{
    let Some(existing) = api.get_opt(name).await? else {
        return Ok(());
    };
    if !is_owned_by(existing.meta(), owner_uid) {
        tracing::debug!(name, "not operator-owned, leaving in place");
        return Ok(());
    }
    match api.delete(name, &Default::default()).await {
        Ok(_) => Ok(()),
        // lost a race with the garbage collector
        Err(kube::Error::Api(e)) if e.code == 404 => Ok(()),
        Err(e) => Err(e.into()),
    }
}

fn is_owned_by(meta: &ObjectMeta, owner_uid: &str) -> bool {
    meta.owner_references
        .iter()
        .flatten()
        .any(|owner| owner.uid == owner_uid)
}

/// The admission rules can't see an existing claim's access modes; enforce the
/// RWO-multi-replica exclusion here instead.
async fn check_existing_claim(cr: &ModelExpressServer, ns: &str, ctx: &Ctx) -> Result<(), Error> {
    let Some(CacheStorage::ExistingClaim(existing)) = cr
        .spec
        .cache
        .as_ref()
        .and_then(|cache| cache.storage.as_ref())
    else {
        return Ok(());
    };
    if cr.spec.replicas <= 1 {
        return Ok(());
    }

    let api = Api::<PersistentVolumeClaim>::namespaced(ctx.client.clone(), ns);
    let claim = api
        .get_opt(&existing.claim_name)
        .await?
        .ok_or_else(|| Error::MissingClaim {
            claim: existing.claim_name.clone(),
        })?;
    let modes = claim
        .spec
        .and_then(|spec| spec.access_modes)
        .unwrap_or_default();
    if !modes
        .iter()
        .any(|m| m == "ReadWriteMany" || m == "ReadOnlyMany")
    {
        return Err(Error::SingleNodeClaim {
            claim: existing.claim_name.clone(),
            modes,
            replicas: cr.spec.replicas,
        });
    }
    Ok(())
}

fn stamp(
    meta: &mut ObjectMeta,
    ns: &str,
    owner: Option<k8s_openapi::apimachinery::pkg::apis::meta::v1::OwnerReference>,
) {
    meta.namespace = Some(ns.to_string());
    meta.owner_references = owner.map(|o| vec![o]);
}

fn reason(err: &Error) -> &'static str {
    match err {
        Error::Kube(_) => "ApplyFailed",
        Error::TlsDefaults(_) => "TlsDefaultsUnavailable",
        Error::ServerImageUnset => "ServerImageUnset",
        Error::MissingNamespace => "MissingNamespace",
        Error::MissingUid => "MissingUid",
        Error::MissingClaim { .. } => "CacheClaimMissing",
        Error::SingleNodeClaim { .. } => "CacheClaimSingleNode",
    }
}

pub const READY_CONDITION: &str = "Ready";

/// Carries lastTransitionTime forward while the condition holds, per the API
/// conventions. Restamping it makes every status write differ from the stored
/// object, and since the controller watches its own CR that write schedules
/// the next reconcile: an unbounded loop the requeue never gates.
fn ready_condition(
    cr: &ModelExpressServer,
    status: &str,
    reason: &str,
    message: &str,
) -> Condition {
    let previous = cr
        .status
        .as_ref()
        .and_then(|s| s.conditions.iter().find(|c| c.type_ == READY_CONDITION));
    let last_transition_time = match previous {
        Some(prev) if prev.status == status && prev.reason == reason && prev.message == message => {
            prev.last_transition_time.clone()
        }
        _ => k8s_openapi::apimachinery::pkg::apis::meta::v1::Time(chrono::Utc::now()),
    };

    Condition {
        type_: READY_CONDITION.to_string(),
        status: status.to_string(),
        reason: reason.to_string(),
        message: message.to_string(),
        observed_generation: cr.metadata.generation,
        last_transition_time,
    }
}

#[tracing::instrument(skip_all)]
async fn write_status(
    ns: &str,
    name: &str,
    ctx: &Ctx,
    condition: Condition,
    endpoint: Option<String>,
    observed_generation: Option<i64>,
) -> Result<(), Error> {
    let status = ModelExpressServerStatus {
        observed_generation,
        conditions: vec![condition],
        endpoint,
    };
    let api = Api::<ModelExpressServer>::namespaced(ctx.client.clone(), ns);
    // from the Resource impl so the group is spelled once, in the derive
    let patch = serde_json::json!({
        "apiVersion": ModelExpressServer::api_version(&()),
        "kind": ModelExpressServer::kind(&()),
        "status": status,
    });
    api.patch_status(
        name,
        &PatchParams::apply(FIELD_MANAGER).force(),
        &Patch::Apply(&patch),
    )
    .await?;
    Ok(())
}

fn error_policy(_cr: Arc<ModelExpressServer>, err: &Error, _ctx: Arc<Ctx>) -> Action {
    match err {
        // user-fixable config problems: no point hammering the apiserver
        Error::MissingClaim { .. } | Error::SingleNodeClaim { .. } | Error::ServerImageUnset => {
            Action::requeue(Duration::from_secs(120))
        }
        _ => Action::requeue(Duration::from_secs(15)),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use crate::controller::*;
    use crate::crd::{MetadataBackend, ModelExpressServerSpec, RedisBackend};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::{OwnerReference, Time};

    fn server(status: Option<ModelExpressServerStatus>) -> ModelExpressServer {
        let mut cr = ModelExpressServer::new(
            "mx",
            ModelExpressServerSpec {
                image: Some("img".into()),
                replicas: 1,
                metadata_backend: MetadataBackend::Redis(RedisBackend {
                    url: Some("redis://mx-redis:6379".into()),
                    url_secret: None,
                }),
                port: 8001,
                log: None,
                cache: None,
                security: None,
                tls: None,
                reaper: None,
                credentials: None,
                pod_metadata: None,
                service_metadata: None,
                resources: None,
                node_selector: None,
                tolerations: None,
                affinity: None,
                network_policy: None,
                service_account_name: None,
                image_pull_secrets: None,
            },
        );
        cr.metadata.generation = Some(1);
        cr.status = status;
        cr
    }

    fn stamped(at: chrono::DateTime<chrono::Utc>, status: &str, reason: &str) -> Condition {
        Condition {
            type_: READY_CONDITION.to_string(),
            status: status.to_string(),
            reason: reason.to_string(),
            message: "resources applied".to_string(),
            observed_generation: Some(1),
            last_transition_time: Time(at),
        }
    }

    fn with_ready(condition: Condition) -> Option<ModelExpressServerStatus> {
        Some(ModelExpressServerStatus {
            observed_generation: Some(1),
            conditions: vec![condition],
            endpoint: None,
        })
    }

    #[test]
    fn server_image_prefers_the_cr() {
        let mut spec = server(None).spec;
        spec.image = Some("cr-image".into());
        assert_eq!(
            server_image(&spec, Some("default-image")).ok(),
            Some("cr-image")
        );
    }

    #[test]
    fn server_image_falls_back_to_the_default() {
        let mut spec = server(None).spec;
        spec.image = None;
        assert_eq!(
            server_image(&spec, Some("default-image")).ok(),
            Some("default-image")
        );
    }

    #[test]
    fn server_image_without_either_is_a_user_fixable_error() {
        let mut spec = server(None).spec;
        spec.image = None;
        let err = server_image(&spec, None).expect_err("no image anywhere");
        assert!(matches!(err, Error::ServerImageUnset));
        assert_eq!(reason(&err), "ServerImageUnset");
    }

    #[test]
    fn ownership_gates_the_cleanup_deletes() {
        const UID: &str = "11111111-2222-3333-4444-555555555555";
        const OTHER: &str = "99999999-8888-7777-6666-555555555555";

        fn owned_by(uids: &[&str]) -> ObjectMeta {
            ObjectMeta {
                owner_references: Some(
                    uids.iter()
                        .map(|uid| OwnerReference {
                            uid: (*uid).to_string(),
                            ..OwnerReference::default()
                        })
                        .collect(),
                ),
                ..ObjectMeta::default()
            }
        }

        assert!(is_owned_by(&owned_by(&[UID]), UID));
        assert!(
            is_owned_by(&owned_by(&[OTHER, UID]), UID),
            "one of several owners still counts"
        );
        assert!(
            !is_owned_by(&owned_by(&[OTHER]), UID),
            "owned by a different CR"
        );
        // a user's own NetworkPolicy that merely shares the CR's name
        assert!(
            !is_owned_by(&ObjectMeta::default(), UID),
            "an unowned object must never be deleted"
        );
    }

    #[test]
    fn unchanged_condition_keeps_its_transition_time() {
        let earlier = chrono::Utc::now() - chrono::Duration::hours(3);
        let cr = server(with_ready(stamped(earlier, "True", "Applied")));
        let next = ready_condition(&cr, "True", "Applied", "resources applied");
        assert_eq!(
            next.last_transition_time,
            Time(earlier),
            "restamping an unchanged condition re-triggers our own watch"
        );
    }

    #[test]
    fn flipping_status_restamps_transition_time() {
        let earlier = chrono::Utc::now() - chrono::Duration::hours(3);
        let cr = server(with_ready(stamped(earlier, "True", "Applied")));
        let next = ready_condition(&cr, "False", "ApplyFailed", "kube api: boom");
        assert!(next.last_transition_time.0 > earlier);
    }

    #[test]
    fn same_status_but_new_reason_restamps() {
        let earlier = chrono::Utc::now() - chrono::Duration::hours(3);
        let cr = server(with_ready(stamped(earlier, "False", "CacheClaimMissing")));
        let next = ready_condition(&cr, "False", "ApplyFailed", "resources applied");
        assert!(next.last_transition_time.0 > earlier);
    }

    #[test]
    fn same_status_and_reason_but_new_message_restamps() {
        let earlier = chrono::Utc::now() - chrono::Duration::hours(3);
        let cr = server(with_ready(stamped(earlier, "False", "ApplyFailed")));
        let next = ready_condition(&cr, "False", "ApplyFailed", "kube api: different");
        assert!(next.last_transition_time.0 > earlier);
    }

    #[test]
    fn first_reconcile_stamps_a_fresh_time() {
        let before = chrono::Utc::now();
        let cr = server(None);
        let next = ready_condition(&cr, "True", "Applied", "resources applied");
        assert!(next.last_transition_time.0 >= before);
        assert_eq!(next.type_, READY_CONDITION);
        assert_eq!(next.observed_generation, Some(1));
    }

    #[test]
    fn a_foreign_condition_type_does_not_supply_the_timestamp() {
        let earlier = chrono::Utc::now() - chrono::Duration::hours(3);
        let mut other = stamped(earlier, "True", "Applied");
        other.type_ = "Degraded".to_string();
        let before = chrono::Utc::now();
        let cr = server(with_ready(other));
        let next = ready_condition(&cr, "True", "Applied", "resources applied");
        assert!(next.last_transition_time.0 >= before);
    }
}
