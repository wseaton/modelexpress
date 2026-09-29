// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use k8s_openapi::apiextensions_apiserver::pkg::apis::apiextensions::v1::{
    JSONSchemaProps, JSONSchemaPropsOrArray, JSONSchemaPropsOrBool,
};
use kube::{CELSchema, CustomResource};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The group this operator owns. The backend CRs the server reads and writes
/// stay under modelexpress.nvidia.com (see rbac::UPSTREAM_API_GROUP); only
/// ModelExpressServer, which is a midstream addition, lives here.
///
/// The kube derive needs a literal, so `group_matches_the_derive` guards the
/// two against drifting.
pub const API_GROUP: &str = "modelexpress.opendatahub.io";

/// Desired state of a ModelExpress metadata coordinator deployment.
///
/// Validation lives in CEL rules on the schema (enforced at admission by the
/// apiserver) instead of a webhook, so the operator stays OLMv1-clean.
#[derive(CustomResource, CELSchema, Clone, Debug, Deserialize, Serialize)]
#[kube(
    group = "modelexpress.opendatahub.io",
    version = "v1alpha1",
    kind = "ModelExpressServer",
    namespaced,
    status = "ModelExpressServerStatus",
    shortname = "mxs",
    printcolumn = r#"{"name":"Endpoint","type":"string","jsonPath":".status.endpoint"}"#,
    // the controller only writes the Ready condition, so index 0 is stable
    // (printcolumn JSONPath does not support [?(@...)] filters)
    printcolumn = r#"{"name":"Ready","type":"string","jsonPath":".status.conditions[0].status"}"#,
    printcolumn = r#"{"name":"Reason","type":"string","jsonPath":".status.conditions[0].reason"}"#
)]
#[serde(rename_all = "camelCase")]
// A ReadWriteOnce cache PVC pins every replica to one node: the second pod
// wedges Pending forever (the upstream chart's values-production.yaml ships
// exactly this bug). ExistingClaim access modes aren't knowable at admission;
// the controller checks those at reconcile.
#[cel_validate(rule = Rule::new(
    "self.replicas <= 1 || !has(self.cache) || !has(self.cache.storage) || !has(self.cache.storage.pvc) || (has(self.cache.storage.pvc.spec.accessModes) && self.cache.storage.pvc.spec.accessModes.exists(m, m == 'ReadWriteMany' || m == 'ReadOnlyMany'))"
).message("replicas > 1 with a ReadWriteOnce cache PVC deadlocks scheduling; use ReadWriteMany or emptyDir"))]
pub struct ModelExpressServerSpec {
    /// Server image ref. The tag pins the mx version; mx_source_id embeds it,
    /// so a rolling image change cold-starts P2P discovery between old and new
    /// workers. Unset follows the default server image the operator was
    /// started with, and a change to that default rolls the server.
    #[cel_validate(rule = Rule::new("self != ''").message("image must not be empty"))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,

    /// Pull secrets for the server image, for a private or mirrored registry.
    /// Set on the pod spec, so they apply whether the pods run as the
    /// generated ServiceAccount or one named by `serviceAccountName`.
    #[cel_validate(
        rule = Rule::new("self.all(secret, secret.name != '')")
            .message("image pull secret name must not be empty")
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_pull_secrets: Option<Vec<k8s_openapi::api::core::v1::LocalObjectReference>>,

    /// Replicas are stateless and share no locks; safe to scale as long as the
    /// cache volume is RWX or per-replica.
    #[serde(default = "default_replicas")]
    #[cel_validate(rule = Rule::new("self >= 0").message("replicas must not be negative"))]
    pub replicas: i32,

    /// Metadata backend. Upstream has no default (MX_METADATA_BACKEND is
    /// mandatory), so it is required here too.
    pub metadata_backend: MetadataBackend,

    /// gRPC port, MODEL_EXPRESS_SERVER_PORT.
    #[serde(default = "default_port")]
    #[cel_validate(rule = Rule::new("self > 0 && self < 65536").message("port must be 1-65535"))]
    pub port: i32,

    /// Logging, MODEL_EXPRESS_LOG_LEVEL / MODEL_EXPRESS_LOG_FORMAT.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log: Option<LogConfig>,

    /// Model cache, MODEL_EXPRESS_CACHE_DIRECTORY / _CACHE_EVICTION_ENABLED.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheConfig>,

    /// ServiceAccount token auth, MODEL_EXPRESS_SECURITY_*.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub security: Option<SecurityConfig>,

    /// TLS termination on the gRPC listener, MX_TLS_*.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tls: Option<TlsConfig>,

    /// Stale-worker reaper timings, MX_REAPER_* / MX_HEARTBEAT_* / MX_GC_*.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reaper: Option<ReaperConfig>,

    /// Download credentials, injected as secretKeyRef env vars.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credentials: Option<CredentialsConfig>,

    /// Extra labels/annotations for the server pods. The escape hatch for
    /// mesh sidecar injection, ambient enrollment, scrape annotations, etc.
    /// Operator-owned selector labels win on conflict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pod_metadata: Option<MetadataOverrides>,

    /// Extra labels/annotations for the Service, e.g. a certificate issuer's
    /// annotation for tls.secretName. Operator-owned labels win on conflict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_metadata: Option<MetadataOverrides>,

    /// Compute resources for the server container. Defaults to modest
    /// requests and no limits; unset entirely would put the pod in BestEffort,
    /// first to be evicted under node pressure, which is a poor default for
    /// something holding a model cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<k8s_openapi::api::core::v1::ResourceRequirements>,

    /// Node labels the server pods must match. Required on clusters whose
    /// nodes are not interchangeable, e.g. a mixed-architecture cluster where
    /// `spec.image` only runs on one of them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_selector: Option<std::collections::BTreeMap<String, String>>,

    /// Taints the server pods tolerate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerations: Option<Vec<k8s_openapi::api::core::v1::Toleration>>,

    /// Node and pod affinity rules for the server pods, for placement that
    /// `nodeSelector` cannot express.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub affinity: Option<k8s_openapi::api::core::v1::Affinity>,

    /// Opt-in NetworkPolicy for the gRPC port. Governs the metadata plane
    /// only; worker-to-worker RDMA never crosses the CNI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network_policy: Option<NetworkPolicyConfig>,

    /// Run the server as an existing ServiceAccount instead of the generated
    /// `<cr-name>-server`. Binding the metadata Role to it becomes your job.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(max = 253))]
    pub service_account_name: Option<String>,
}

/// Restricts ingress to the server's gRPC port to the listed peers. Peers are
/// verbatim networking/v1 NetworkPolicyPeer, so namespaceSelector/podSelector/
/// ipBlock all work. Enforcement is up to the cluster CNI.
#[derive(CELSchema, Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkPolicyConfig {
    /// Who may connect. Deliberately required and non-empty: an empty `from`
    /// in a NetworkPolicy rule means allow-everywhere, which is what *not*
    /// setting networkPolicy already gives you.
    #[schemars(length(min = 1, max = 32))]
    pub allow_from: Vec<k8s_openapi::api::networking::v1::NetworkPolicyPeer>,
}

fn default_replicas() -> i32 {
    1
}

fn default_port() -> i32 {
    8001
}

/// Where source/cache metadata lives. Mirrors MX_METADATA_BACKEND.
/// CELSchema can't derive on enums, so per-variant validation lives on the
/// inner structs.
#[derive(JsonSchema, Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum MetadataBackend {
    /// External Redis via REDIS_URL.
    Redis(RedisBackend),
    /// CRD-backed storage in the server's own namespace. Requires the
    /// ModelMetadata/ModelCacheEntry CRDs and namespace RBAC. The namespace
    /// comes from the downward API (POD_NAMESPACE).
    Kubernetes {},
}

#[derive(CELSchema, Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
// Redis URLs routinely embed credentials (rediss://user:pass@host:6380). A
// literal url lands in the CR and in the pod template, readable by anyone with
// get on either, so urlSecret exists for that case.
#[cel_validate(rule = Rule::new("has(self.url) != has(self.urlSecret)")
    .message("set exactly one of url or urlSecret"))]
pub struct RedisBackend {
    /// Full connection URL, e.g. redis://host:6379. Use urlSecret instead when
    /// the URL embeds credentials.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cel_validate(rule = Rule::new("self.startsWith('redis://') || self.startsWith('rediss://')")
        .message("url must be a redis:// or rediss:// URL"))]
    pub url: Option<String>,

    /// Secret holding the connection URL, rendered as a secretKeyRef so the
    /// credentials never enter the CR.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url_secret: Option<SecretKeyRef>,
}

#[derive(JsonSchema, Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LogConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<LogLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<LogFormat>,
}

/// Mirrors upstream LogLevel; env value is the lowercase name.
#[derive(JsonSchema, Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn as_env_value(self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

/// Mirrors upstream LogFormat; env value is the lowercase name.
#[derive(JsonSchema, Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    Json,
    Pretty,
    Compact,
}

impl LogFormat {
    pub fn as_env_value(self) -> &'static str {
        match self {
            LogFormat::Json => "json",
            LogFormat::Pretty => "pretty",
            LogFormat::Compact => "compact",
        }
    }
}

#[derive(JsonSchema, Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheConfig {
    /// MODEL_EXPRESS_CACHE_DIRECTORY, also the volume mount path. Defaults to
    /// /var/cache/modelexpress; the operator always sets the env var so the
    /// server never falls back to a HOME it cannot write.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directory: Option<String>,
    /// MODEL_EXPRESS_CACHE_EVICTION_ENABLED.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eviction_enabled: Option<bool>,
    /// What backs the cache mount. Defaults to an emptyDir (models re-download
    /// after a pod restart, but any replica count works).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<CacheStorage>,
}

/// CELSchema can't derive on enums, so per-variant validation lives on the
/// inner structs.
#[derive(JsonSchema, Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CacheStorage {
    /// Node-local scratch, lost on pod restart.
    EmptyDir(EmptyDirStorage),
    /// A PVC the user manages; the operator only mounts it.
    ExistingClaim(ExistingClaimStorage),
    /// A PVC the operator creates and owns, named `<cr-name>-model-cache`.
    /// Boxed: the embedded claim spec dwarfs the other variants.
    Pvc(Box<ManagedPvcStorage>),
}

#[derive(CELSchema, Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmptyDirStorage {
    /// Optional cap, e.g. "200Gi". Unbounded when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(max = 24))]
    #[cel_validate(rule = Rule::new("self.matches('^[0-9]+(\\\\.[0-9]+)?(Ki|Mi|Gi|Ti|Pi|Ei|k|M|G|T|P|E|m)?$')")
        .message("sizeLimit must be a Kubernetes quantity"))]
    pub size_limit: Option<String>,
}

#[derive(CELSchema, Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExistingClaimStorage {
    #[cel_validate(rule = Rule::new("self != ''").message("claimName must not be empty"))]
    pub claim_name: String,
}

/// StatefulSet volumeClaimTemplates-style: a full claim spec plus the
/// metadata the user may set. The claim name is operator-assigned
/// (`<cr-name>-model-cache`); a name in metadata is not accepted.
#[derive(CELSchema, Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
#[cel_validate(rule = Rule::new(
    "has(self.spec.resources) && has(self.spec.resources.requests) && 'storage' in self.spec.resources.requests"
).message("spec.resources.requests.storage is required"))]
pub struct ManagedPvcStorage {
    /// Labels and annotations applied to the claim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MetadataOverrides>,
    /// Verbatim PVC spec. accessModes defaults to ReadWriteOnce, which the
    /// spec-level rule rejects for replicas > 1. Quantity strings are only
    /// validated by the apiserver at claim creation; a bad size shows up as a
    /// status condition, not an admission error.
    pub spec: k8s_openapi::api::core::v1::PersistentVolumeClaimSpec,
}

#[derive(JsonSchema, Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MetadataOverrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub labels: Option<std::collections::BTreeMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<std::collections::BTreeMap<String, String>>,
}

/// TokenReview-based caller auth. Enforce needs both audiences and an
/// allowlist; the upstream chart fails the render on that combination, we
/// reject it at admission instead.
#[derive(CELSchema, Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
#[cel_validate(rule = Rule::new(
    "!(has(self.mode) && self.mode == 'enforce') || (has(self.tokenAudiences) && size(self.tokenAudiences) > 0 && has(self.allowedServiceAccounts) && size(self.allowedServiceAccounts) > 0)"
).message("enforce mode requires tokenAudiences and allowedServiceAccounts"))]
pub struct SecurityConfig {
    /// MODEL_EXPRESS_SECURITY_MODE.
    #[serde(default)]
    pub mode: AuthMode,
    /// MODEL_EXPRESS_SECURITY_TOKEN_AUDIENCES (rendered comma-separated).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(max = 32))]
    pub token_audiences: Vec<String>,
    /// MODEL_EXPRESS_SECURITY_ALLOWED_SERVICE_ACCOUNTS (rendered `ns:sa,...`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(max = 256))]
    pub allowed_service_accounts: Vec<ServiceAccountRef>,
    /// MODEL_EXPRESS_SECURITY_CACHE_TTL_SECS (upstream default 60).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_ttl_secs: Option<u32>,
}

/// Serve TLS from a kubernetes.io/tls Secret. Version, ciphers and groups left
/// unset take the platform's defaults, or the server's own when there are
/// none. Kubernetes gRPC probes cannot speak TLS, so the probes switch to TCP
/// when this is set.
#[derive(CELSchema, Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TlsConfig {
    /// Secret holding tls.crt and tls.key, mounted read-only.
    #[cel_validate(rule = Rule::new("self != ''").message("secretName must not be empty"))]
    #[schemars(length(max = 253))]
    pub secret_name: String,
    /// MX_TLS_MIN_VERSION, `VersionTLS12` or `TLS1.2` style.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(max = 16))]
    #[cel_validate(rule = Rule::new("self.matches('^(VersionTLS1[0-3]|TLS1[.][0-3])$')")
        .message("minVersion must be VersionTLS10..13 or TLS1.0..1.3"))]
    pub min_version: Option<String>,
    /// MX_TLS_CIPHER_SUITES, OpenSSL names (rendered comma-separated).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(max = 64))]
    pub cipher_suites: Vec<String>,
    /// MX_TLS_GROUPS, key exchange groups in preference order
    /// (rendered comma-separated).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(max = 16))]
    pub groups: Vec<String>,
}

/// Mirrors upstream AuthMode, but the CRD value is `disabled` rather than
/// upstream's `off`: YAML 1.1 parsers (kubectl, the apiserver) read an
/// unquoted `off` as a boolean, which corrupts the schema's enum/default.
#[derive(JsonSchema, Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AuthMode {
    #[default]
    Disabled,
    Enforce,
}

impl AuthMode {
    pub fn as_env_value(self) -> &'static str {
        match self {
            AuthMode::Disabled => "off",
            AuthMode::Enforce => "enforce",
        }
    }
}

/// An allowed caller, rendered as `<namespace>:<serviceAccount>`.
/// maxLength bounds keep the regex rules inside the apiserver's CEL cost
/// budget; unbounded strings get costed at worst case and rejected.
#[derive(CELSchema, Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ServiceAccountRef {
    #[schemars(length(max = 63))]
    #[cel_validate(rule = Rule::new("self.matches('^[a-z0-9]([-a-z0-9]*[a-z0-9])?$')")
        .message("namespace must be a DNS-1123 label"))]
    pub namespace: String,
    #[schemars(length(max = 253))]
    #[cel_validate(rule = Rule::new("self.matches('^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$')")
        .message("serviceAccount must be a DNS-1123 subdomain"))]
    pub service_account: String,
}

/// Reaper timings interact: a heartbeat can only be declared stale after at
/// least one scan interval, and GC must not fire before staleness.
#[derive(CELSchema, Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
#[cel_validate(rule = Rule::new(
    "!has(self.scanIntervalSecs) || !has(self.heartbeatTimeoutSecs) || self.heartbeatTimeoutSecs > self.scanIntervalSecs"
).message("heartbeatTimeoutSecs must be greater than scanIntervalSecs"))]
#[cel_validate(rule = Rule::new(
    "!has(self.heartbeatTimeoutSecs) || !has(self.gcTimeoutSecs) || self.gcTimeoutSecs >= self.heartbeatTimeoutSecs"
).message("gcTimeoutSecs must be at least heartbeatTimeoutSecs"))]
pub struct ReaperConfig {
    /// MX_REAPER_SCAN_INTERVAL_SECS (upstream default 30).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cel_validate(rule = Rule::new("self > 0").message("scanIntervalSecs must be positive"))]
    pub scan_interval_secs: Option<u32>,
    /// MX_HEARTBEAT_TIMEOUT_SECS (upstream default 90).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cel_validate(rule = Rule::new("self > 0").message("heartbeatTimeoutSecs must be positive"))]
    pub heartbeat_timeout_secs: Option<u32>,
    /// MX_GC_TIMEOUT_SECS (upstream default 3600).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cel_validate(rule = Rule::new("self > 0").message("gcTimeoutSecs must be positive"))]
    pub gc_timeout_secs: Option<u32>,
}

#[derive(JsonSchema, Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialsConfig {
    /// Secret holding the HuggingFace token, injected as HF_TOKEN.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_token_secret: Option<SecretKeyRef>,
    /// Secret holding the NGC API key, injected as NGC_API_KEY.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ngc_api_key_secret: Option<SecretKeyRef>,
}

/// Same shape as core/v1 SecretKeySelector (which lacks JsonSchema).
#[derive(CELSchema, Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SecretKeyRef {
    #[cel_validate(rule = Rule::new("self != ''").message("secret name must not be empty"))]
    pub name: String,
    #[cel_validate(rule = Rule::new("self != ''").message("secret key must not be empty"))]
    pub key: String,
}

#[derive(JsonSchema, Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelExpressServerStatus {
    /// Generation last acted on by the controller.
    pub observed_generation: Option<i64>,
    /// Standard conditions; Ready is the rollup.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<k8s_openapi::apimachinery::pkg::apis::meta::v1::Condition>,
    /// gRPC endpoint clients should set MODEL_EXPRESS_ENDPOINT to, as
    /// `http://host:port`, or `https://host:port` when spec.tls is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
}

/// The CRD to install: `ModelExpressServer::crd()` plus bounds the derive
/// can't express. The deadlock rule iterates the embedded (upstream-typed)
/// pvc.spec.accessModes; without maxItems/maxLength there, the apiserver
/// costs the rule at worst case and rejects the whole CRD.
pub fn generate_crd()
-> k8s_openapi::apiextensions_apiserver::pkg::apis::apiextensions::v1::CustomResourceDefinition {
    use kube::CustomResourceExt;

    let mut crd = ModelExpressServer::crd();
    let versions = &mut crd.spec.versions;
    for version in versions.iter_mut() {
        let Some(schema) = version
            .schema
            .as_mut()
            .and_then(|s| s.open_api_v3_schema.as_mut())
        else {
            continue;
        };
        restore_int_or_string(schema);

        let access_modes = ["spec", "cache", "storage", "pvc", "spec"]
            .iter()
            .try_fold(schema, |props, key| {
                props.properties.as_mut()?.get_mut(*key)
            })
            .and_then(|pvc_spec| pvc_spec.properties.as_mut()?.get_mut("accessModes"));
        if let Some(modes) = access_modes {
            modes.max_items = Some(8);
            if let Some(k8s_openapi::apiextensions_apiserver::pkg::apis::apiextensions::v1::JSONSchemaPropsOrArray::Schema(items)) =
                modes.items.as_mut()
            {
                items.max_length = Some(32);
            }
        }
    }
    crd
}

/// k8s-openapi 0.24 renders Quantity as a plain string, where Kubernetes' own
/// PVC schema marks it int-or-string. Left alone, `storage: 100` is rejected
/// where `storage: 100Gi` is accepted, which is not how a Quantity behaves
/// anywhere else in the API.
///
/// Matching on the description is ugly, but the Quantity schema carries no
/// other marker by the time schemars is done with it.
fn restore_int_or_string(schema: &mut JSONSchemaProps) {
    const QUANTITY_DOC: &str = "Quantity is a fixed-point representation of a number.";

    let is_quantity = schema.type_.as_deref() == Some("string")
        && schema
            .description
            .as_deref()
            .is_some_and(|d| d.starts_with(QUANTITY_DOC));
    if is_quantity {
        // the apiserver rejects x-kubernetes-int-or-string alongside a type
        schema.type_ = None;
        schema.x_kubernetes_int_or_string = Some(true);
        return;
    }

    if let Some(properties) = schema.properties.as_mut() {
        for child in properties.values_mut() {
            restore_int_or_string(child);
        }
    }
    if let Some(JSONSchemaPropsOrBool::Schema(child)) = schema.additional_properties.as_mut() {
        restore_int_or_string(child);
    }
    match schema.items.as_mut() {
        Some(JSONSchemaPropsOrArray::Schema(child)) => restore_int_or_string(child),
        Some(JSONSchemaPropsOrArray::Schemas(children)) => {
            for child in children {
                restore_int_or_string(child);
            }
        }
        None => {}
    }
    for group in [
        schema.all_of.as_mut(),
        schema.any_of.as_mut(),
        schema.one_of.as_mut(),
    ]
    .into_iter()
    .flatten()
    {
        for child in group {
            restore_int_or_string(child);
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    // JSON, not YAML: YAML style-quoting mangles rule strings (''enforce'').
    fn crd_json() -> String {
        serde_json::to_string(&generate_crd()).expect("CRD serializes")
    }

    // Until kube's client-side CEL evaluator is usable (kube-cel 0.8.0 pins a
    // yanked cel version), assert the rules are present in the schema; the
    // apiserver executes them.
    #[test]
    fn cel_rules_present_in_schema() {
        let json = crd_json();
        for rule in [
            "!(has(self.mode) && self.mode == 'enforce')",
            "self.heartbeatTimeoutSecs > self.scanIntervalSecs",
            "self.gcTimeoutSecs >= self.heartbeatTimeoutSecs",
            "self.startsWith('redis://')",
            "self >= 0",
            "self > 0 && self < 65536",
            "self.all(secret, secret.name != '')",
            "self.cache.storage.pvc.spec.accessModes.exists(m, m == 'ReadWriteMany' || m == 'ReadOnlyMany')",
            "'storage' in self.spec.resources.requests",
            "must be a Kubernetes quantity",
        ] {
            assert!(json.contains(rule), "missing CEL rule: {rule}");
        }
        assert!(json.contains("x-kubernetes-validations"));
    }

    #[test]
    fn image_pull_secret_names_are_validated() {
        let crd = serde_json::to_value(generate_crd()).expect("CRD serializes");
        let image_pull_secrets = &crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"]
            ["spec"]["properties"]["imagePullSecrets"];
        let validations = image_pull_secrets["x-kubernetes-validations"]
            .as_array()
            .expect("imagePullSecrets validations");

        assert!(validations.iter().any(|validation| {
            validation["rule"] == "self.all(secret, secret.name != '')"
                && validation["message"] == "image pull secret name must not be empty"
        }));
    }

    #[test]
    fn printer_columns_present_in_crd() {
        let crd = generate_crd();
        let columns = crd.spec.versions[0]
            .additional_printer_columns
            .as_ref()
            .expect("printer columns");
        let names: Vec<_> = columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["Endpoint", "Ready", "Reason"]);
        assert_eq!(columns[0].json_path, ".status.endpoint");
    }

    #[test]
    fn endpoint_format_is_service_dns() {
        assert_eq!(
            crate::controller::endpoint("mx", "weaton-dev", 8001, false),
            "http://mx.weaton-dev.svc.cluster.local:8001"
        );
    }

    #[test]
    fn tls_endpoint_uses_https() {
        assert_eq!(
            crate::controller::endpoint("mx", "weaton-dev", 8001, true),
            "https://mx.weaton-dev.svc.cluster.local:8001"
        );
    }

    #[test]
    fn minimal_cr_deserializes_with_defaults() {
        let spec: ModelExpressServerSpec = serde_json::from_value(serde_json::json!({
            "image": "nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.5.0",
            "metadataBackend": {"kubernetes": {}},
        }))
        .expect("minimal spec deserializes");
        assert_eq!(spec.replicas, 1);
        assert_eq!(spec.port, 8001);
        assert!(spec.security.is_none());
    }

    #[test]
    fn backend_is_required() {
        let result = serde_json::from_value::<ModelExpressServerSpec>(serde_json::json!({
            "image": "img",
        }));
        assert!(result.is_err(), "spec without metadataBackend must fail");
    }

    #[test]
    fn required_fields_marked_in_schema() {
        let json = crd_json();
        assert!(json.contains("metadataBackend"));
        // spec-level required list must include the backend; image may be
        // left to the operator's default
        let crd = generate_crd();
        let schema = serde_json::to_value(&crd).expect("crd to json");
        let required = schema["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"]
            ["spec"]["required"]
            .as_array()
            .expect("spec.required is an array")
            .iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>();
        assert!(!required.contains(&"image"));
        assert!(required.contains(&"metadataBackend"));
    }

    #[test]
    fn quantities_are_int_or_string() {
        fn walk(node: &serde_json::Value, quantities: &mut Vec<serde_json::Value>) {
            if let Some(obj) = node.as_object() {
                let is_quantity = obj
                    .get("description")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|d| d.starts_with("Quantity is a fixed-point"));
                if is_quantity {
                    quantities.push(node.clone());
                }
                for child in obj.values() {
                    walk(child, quantities);
                }
            } else if let Some(items) = node.as_array() {
                for child in items {
                    walk(child, quantities);
                }
            }
        }

        let crd = serde_json::to_value(generate_crd()).expect("CRD serializes");
        let mut quantities = Vec::new();
        walk(&crd, &mut quantities);

        // pvc.spec.resources and spec.resources, requests and limits each
        assert_eq!(quantities.len(), 4, "Quantity schema count changed");
        for q in &quantities {
            assert_eq!(
                q.get("x-kubernetes-int-or-string"),
                Some(&serde_json::Value::Bool(true)),
                "k8s-openapi renders Quantity as a plain string; `storage: 100` \
                 would be rejected where `storage: 100Gi` is accepted"
            );
            // the apiserver rejects int-or-string alongside a type
            assert_eq!(q.get("type"), None);
        }
    }

    #[test]
    fn group_matches_the_derive() {
        let crd = generate_crd();
        assert_eq!(crd.spec.group, API_GROUP);
        assert_eq!(
            crd.metadata.name.as_deref(),
            Some(format!("modelexpressservers.{API_GROUP}").as_str())
        );
        assert!(
            <ModelExpressServer as kube::Resource>::api_version(&()).starts_with(API_GROUP),
            "the status apply patch would target the wrong group"
        );
    }
}
