// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TLS defaults from the cluster TLS profile in
//! `apiservers.config.openshift.io/cluster`.
//!
//! The profile applies only when the APIServer's `tlsAdherence` asks every
//! component to follow it, as `ShouldHonorClusterTLSProfile` in library-go
//! decides. Otherwise, and on clusters without the OpenShift config API, the
//! defaults are the Intermediate profile.
//!
//! The named profiles mirror `TLSProfiles` in openshift/api
//! (config/v1/types_tlssecurityprofile.go). The server takes the values as
//! written there, so nothing is translated here.

use futures::StreamExt;
use kube::Client;
use kube::api::{Api, ApiResource, DynamicObject, GroupVersionKind};
use kube::runtime::WatchStreamExt;
use kube::runtime::watcher;
use kube::runtime::watcher::metadata_watcher;
use modelexpress_operator::tls::{TlsDefaults, TlsDefaultsError, TlsSettings, TlsUpdates};
use serde::Deserialize;
use std::time::Duration;
use tokio_stream::wrappers::ReceiverStream;

pub const API_GROUP: &str = "config.openshift.io";
pub const API_VERSION: &str = "v1";
pub const KIND: &str = "APIServer";
pub const PLURAL: &str = "apiservers";
pub const CLUSTER_OBJECT: &str = "cluster";

const WATCH_TIMEOUT_SECS: u32 = 290;

/// The groups every named profile carries in openshift/api. The server drops
/// names its TLS library cannot negotiate, so post-quantum entries are safe
/// to pass on.
const PROFILE_GROUPS: [&str; 4] = ["X25519MLKEM768", "X25519", "secp256r1", "secp384r1"];

const TLS13_SUITES: [&str; 3] = [
    "TLS_AES_128_GCM_SHA256",
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
];

const INTERMEDIATE_TLS12_CIPHERS: [&str; 6] = [
    "ECDHE-ECDSA-AES128-GCM-SHA256",
    "ECDHE-RSA-AES128-GCM-SHA256",
    "ECDHE-ECDSA-AES256-GCM-SHA384",
    "ECDHE-RSA-AES256-GCM-SHA384",
    "ECDHE-ECDSA-CHACHA20-POLY1305",
    "ECDHE-RSA-CHACHA20-POLY1305",
];

const OLD_EXTRA_CIPHERS: [&str; 15] = [
    "ECDHE-ECDSA-AES128-SHA256",
    "ECDHE-RSA-AES128-SHA256",
    "ECDHE-ECDSA-AES128-SHA",
    "ECDHE-RSA-AES128-SHA",
    "ECDHE-ECDSA-AES256-SHA384",
    "ECDHE-RSA-AES256-SHA384",
    "ECDHE-ECDSA-AES256-SHA",
    "ECDHE-RSA-AES256-SHA",
    "AES128-GCM-SHA256",
    "AES256-GCM-SHA384",
    "AES128-SHA256",
    "AES256-SHA256",
    "AES128-SHA",
    "AES256-SHA",
    "DES-CBC3-SHA",
];

fn strings(parts: &[&[&str]]) -> Vec<String> {
    parts
        .iter()
        .flat_map(|part| part.iter().map(|s| (*s).to_string()))
        .collect()
}

#[must_use]
pub fn old() -> TlsSettings {
    TlsSettings {
        min_version: Some("VersionTLS10".to_string()),
        ciphers: strings(&[
            &TLS13_SUITES,
            &INTERMEDIATE_TLS12_CIPHERS,
            &OLD_EXTRA_CIPHERS,
        ]),
        groups: strings(&[&PROFILE_GROUPS]),
    }
}

/// What OpenShift applies when the profile is unset, and what this operator
/// uses whenever the cluster profile is not honored.
#[must_use]
pub fn intermediate() -> TlsSettings {
    TlsSettings {
        min_version: Some("VersionTLS12".to_string()),
        ciphers: strings(&[&TLS13_SUITES, &INTERMEDIATE_TLS12_CIPHERS]),
        groups: strings(&[&PROFILE_GROUPS]),
    }
}

#[must_use]
pub fn modern() -> TlsSettings {
    TlsSettings {
        min_version: Some("VersionTLS13".to_string()),
        ciphers: strings(&[&TLS13_SUITES]),
        groups: strings(&[&PROFILE_GROUPS]),
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SecurityProfile {
    #[serde(rename = "type")]
    kind: Option<String>,
    custom: Option<CustomProfile>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CustomProfile {
    ciphers: Option<Vec<String>>,
    groups: Option<Vec<String>>,
    #[serde(rename = "minTLSVersion")]
    min_tls_version: Option<String>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ProfileError {
    #[error("tlsSecurityProfile type is Custom but has no custom body")]
    CustomMissing,
}

/// Interpret an APIServer's `spec.tlsSecurityProfile`, matching
/// `GetTLSProfileSpec` in openshift/controller-runtime-common: null, an
/// empty type and an unknown type are Intermediate; Custom is taken as
/// written and is an error without its body.
pub fn from_security_profile(value: &serde_json::Value) -> Result<TlsSettings, ProfileError> {
    let profile: SecurityProfile = serde_json::from_value(value.clone()).unwrap_or_default();
    Ok(match profile.kind.as_deref() {
        Some("Old") => old(),
        Some("Modern") => modern(),
        Some("Custom") => {
            let custom = profile.custom.ok_or(ProfileError::CustomMissing)?;
            TlsSettings {
                min_version: custom.min_tls_version.filter(|v| !v.is_empty()),
                ciphers: custom.ciphers.unwrap_or_default(),
                groups: custom.groups.unwrap_or_default(),
            }
        }
        _ => intermediate(),
    })
}

/// `spec.tlsAdherence` of the cluster APIServer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TlsAdherence {
    /// Unset or empty.
    NoOpinion,
    LegacyAdheringComponentsOnly,
    StrictAllComponents,
    /// A value newer than this operator.
    Unknown(String),
}

impl TlsAdherence {
    #[must_use]
    pub fn from_spec(value: Option<&str>) -> Self {
        match value {
            None | Some("") => Self::NoOpinion,
            Some("LegacyAdheringComponentsOnly") => Self::LegacyAdheringComponentsOnly,
            Some("StrictAllComponents") => Self::StrictAllComponents,
            Some(other) => Self::Unknown(other.to_string()),
        }
    }

    /// Whether components outside the legacy set follow the cluster profile.
    /// Unknown values do, so a newer, stricter policy fails secure.
    #[must_use]
    pub fn honors_cluster_profile(&self) -> bool {
        match self {
            Self::NoOpinion | Self::LegacyAdheringComponentsOnly => false,
            Self::StrictAllComponents | Self::Unknown(_) => true,
        }
    }
}

/// The settings to follow given an APIServer `spec`: the cluster profile
/// when `tlsAdherence` honors it, Intermediate otherwise. The profile is not
/// parsed when it is not honored, so a broken Custom profile the cluster does
/// not enforce is not an error here either.
pub fn from_apiserver_spec(spec: &serde_json::Value) -> Result<TlsSettings, ProfileError> {
    let adherence = TlsAdherence::from_spec(spec["tlsAdherence"].as_str());
    if !adherence.honors_cluster_profile() {
        return Ok(intermediate());
    }
    from_security_profile(&spec["tlsSecurityProfile"])
}

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error(transparent)]
    Kube(#[from] kube::Error),
    #[error("apiservers.config.openshift.io/cluster: {0}")]
    Profile(#[from] ProfileError),
}

fn gvk() -> GroupVersionKind {
    GroupVersionKind::gvk(API_GROUP, API_VERSION, KIND)
}

fn api_resource() -> ApiResource {
    let mut resource = ApiResource::from_gvk(&gvk());
    resource.plural = PLURAL.to_string();
    resource
}

/// Fetch the settings to follow. A cluster that does not serve the API, which
/// is any non-OpenShift cluster, is Intermediate, and so is a missing object.
///
/// Discovery answers the first question, not the read: the apiserver
/// authorizes a request before it looks the resource up, so without RBAC for
/// a group that does not exist the read is a 403, not a 404. Where the API is
/// served a 403 is returned as-is: the operator is missing RBAC it ships, and
/// that must not silently degrade to a default profile.
pub async fn fetch(client: &Client) -> Result<TlsSettings, FetchError> {
    if matches!(served(client).await, Served::No) {
        return Ok(intermediate());
    }
    read(client).await
}

/// Read the object from an API already known to be served.
async fn read(client: &Client) -> Result<TlsSettings, FetchError> {
    let api = Api::<DynamicObject>::all_with(client.clone(), &api_resource());
    match api.get_opt(CLUSTER_OBJECT).await? {
        Some(object) => Ok(from_apiserver_spec(&object.data["spec"])?),
        None => Ok(intermediate()),
    }
}

/// How long to wait before asking again when discovery could not answer.
const DISCOVERY_RETRY: Duration = Duration::from_secs(30);

/// Whether the cluster serves the OpenShift config API.
enum Served {
    Yes,
    No,
    Unknown(kube::Error),
}

async fn served(client: &Client) -> Served {
    match kube::discovery::oneshot::pinned_kind(client, &gvk()).await {
        Ok(_) => Served::Yes,
        Err(e) if absent(&e) => Served::No,
        Err(e) => Served::Unknown(e),
    }
}

/// A definite "this cluster has no such API", as opposed to an apiserver that
/// could not be asked.
fn absent(error: &kube::Error) -> bool {
    match error {
        kube::Error::Discovery(_) => true,
        kube::Error::Api(response) => response.code == 404,
        _ => false,
    }
}

/// [`TlsDefaults`] backed by `apiservers.config.openshift.io/cluster`.
#[derive(Clone)]
pub struct ApiServerTlsDefaults {
    client: Client,
}

impl ApiServerTlsDefaults {
    #[must_use]
    pub fn new(client: Client) -> Self {
        Self { client }
    }
}

#[async_trait::async_trait]
impl TlsDefaults for ApiServerTlsDefaults {
    async fn current(&self) -> Result<TlsSettings, TlsDefaultsError> {
        Ok(fetch(&self.client).await?)
    }

    /// Re-reads the object on every change and yields only when the settings
    /// to follow differ from the last ones. The first read is always yielded,
    /// so a change between a caller's `current()` and this one is not lost.
    ///
    /// Discovery says whether the API is there: a definite no ends the stream
    /// (any non-OpenShift cluster), while an apiserver that is merely
    /// unreachable is retried, since giving up would pin the caller to the
    /// settings it started with until the process restarts.
    async fn updates(&self) -> TlsUpdates {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let client = self.client.clone();
        tokio::spawn(async move {
            let mut last: Option<TlsSettings> = None;
            loop {
                match served(&client).await {
                    Served::No => {
                        tracing::info!(
                            "{API_GROUP} not served; TLS follows the Intermediate profile"
                        );
                        return;
                    }
                    Served::Unknown(e) => {
                        tracing::warn!(
                            "checking whether {API_GROUP} is served: {e}; retrying in {}s",
                            DISCOVERY_RETRY.as_secs()
                        );
                        tokio::time::sleep(DISCOVERY_RETRY).await;
                        continue;
                    }
                    Served::Yes => {}
                }
                tracing::info!(
                    "watching apiservers.{API_GROUP}/{CLUSTER_OBJECT} for TLS profile changes"
                );
                let api = Api::<DynamicObject>::all_with(client.clone(), &api_resource());
                let config = watcher::Config::default().timeout(WATCH_TIMEOUT_SECS);
                let mut events = metadata_watcher(api, config).touched_objects().boxed();
                // The first pass through has no event to wait for: read now so
                // the caller starts from what the cluster says.
                let mut read_now = true;
                loop {
                    if !read_now {
                        match events.next().await {
                            None => break,
                            Some(Err(e)) => {
                                tracing::warn!("APIServer watch error: {e}");
                                continue;
                            }
                            Some(Ok(_)) => {}
                        }
                    }
                    read_now = false;
                    match read(&client).await {
                        Ok(settings) if last.as_ref() == Some(&settings) => {}
                        Ok(settings) => {
                            last = Some(settings.clone());
                            if tx.send(settings).await.is_err() {
                                return;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("re-reading cluster TLS profile after a change: {e}");
                        }
                    }
                }
            }
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use crate::apiserver::{
        ProfileError, TlsAdherence, absent, from_apiserver_spec, from_security_profile,
        intermediate, modern, old,
    };
    use modelexpress_operator::tls::TlsSettings;
    use serde_json::json;

    fn profile(value: serde_json::Value) -> TlsSettings {
        from_security_profile(&value).expect("valid profile")
    }

    fn api_error(code: u16) -> kube::Error {
        kube::Error::Api(kube::core::ErrorResponse {
            status: "Failure".to_string(),
            message: "boom".to_string(),
            reason: "Boom".to_string(),
            code,
        })
    }

    #[test]
    fn a_missing_api_is_absent() {
        assert!(absent(&kube::Error::Discovery(
            kube::error::DiscoveryError::MissingApiGroup("config.openshift.io".to_string())
        )));
        assert!(absent(&api_error(404)));
    }

    #[test]
    fn an_unreachable_apiserver_is_not_absent() {
        assert!(!absent(&api_error(503)));
        assert!(!absent(&api_error(500)));
        assert!(!absent(&api_error(403)));
        assert!(!absent(&kube::Error::LinesCodecMaxLineLengthExceeded));
    }

    #[test]
    fn named_profiles_match_openshift_api() {
        let intermediate = intermediate();
        assert_eq!(intermediate.min_version.as_deref(), Some("VersionTLS12"));
        assert_eq!(intermediate.ciphers.len(), 9);
        assert_eq!(intermediate.ciphers[0], "TLS_AES_128_GCM_SHA256");
        assert_eq!(intermediate.ciphers[8], "ECDHE-RSA-CHACHA20-POLY1305");

        let modern = modern();
        assert_eq!(modern.min_version.as_deref(), Some("VersionTLS13"));
        assert_eq!(modern.ciphers.len(), 3);

        let old = old();
        assert_eq!(old.min_version.as_deref(), Some("VersionTLS10"));
        assert_eq!(old.ciphers.len(), 24);
        assert_eq!(old.ciphers[23], "DES-CBC3-SHA");
        assert!(old.ciphers.starts_with(&intermediate.ciphers));

        for profile in [&old, &intermediate, &modern] {
            assert_eq!(
                profile.groups,
                ["X25519MLKEM768", "X25519", "secp256r1", "secp384r1"]
            );
        }
    }

    #[test]
    fn null_empty_and_unknown_types_are_intermediate() {
        assert_eq!(profile(serde_json::Value::Null), intermediate());
        assert_eq!(profile(json!({})), intermediate());
        assert_eq!(profile(json!({"type": ""})), intermediate());
        assert_eq!(profile(json!({"type": "Quantum"})), intermediate());
    }

    #[test]
    fn named_types_resolve() {
        assert_eq!(profile(json!({"type": "Old", "old": {}})), old());
        assert_eq!(
            profile(json!({"type": "Intermediate", "intermediate": {}})),
            intermediate()
        );
        assert_eq!(profile(json!({"type": "Modern", "modern": {}})), modern());
    }

    #[test]
    fn custom_profile_is_taken_verbatim() {
        let custom = profile(json!({
            "type": "Custom",
            "custom": {
                "minTLSVersion": "VersionTLS13",
                "ciphers": ["TLS_AES_256_GCM_SHA384"],
                "groups": ["X25519"]
            }
        }));
        assert_eq!(custom.min_version.as_deref(), Some("VersionTLS13"));
        assert_eq!(custom.ciphers, ["TLS_AES_256_GCM_SHA384"]);
        assert_eq!(custom.groups, ["X25519"]);
    }

    #[test]
    fn custom_profile_without_body_is_an_error() {
        assert_eq!(
            from_security_profile(&json!({"type": "Custom"})),
            Err(ProfileError::CustomMissing)
        );
    }

    #[test]
    fn custom_profile_is_not_padded_from_intermediate() {
        let custom = profile(json!({
            "type": "Custom",
            "custom": {"minTLSVersion": "VersionTLS11"}
        }));
        assert_eq!(custom.min_version.as_deref(), Some("VersionTLS11"));
        assert!(custom.ciphers.is_empty());
        assert!(custom.groups.is_empty());
    }

    #[test]
    fn custom_profile_with_empty_version_leaves_it_unset() {
        let custom = profile(json!({"type": "Custom", "custom": {"minTLSVersion": ""}}));
        assert_eq!(custom.min_version, None);
    }

    #[test]
    fn adherence_values_parse_like_the_api() {
        assert_eq!(TlsAdherence::from_spec(None), TlsAdherence::NoOpinion);
        assert_eq!(TlsAdherence::from_spec(Some("")), TlsAdherence::NoOpinion);
        assert_eq!(
            TlsAdherence::from_spec(Some("LegacyAdheringComponentsOnly")),
            TlsAdherence::LegacyAdheringComponentsOnly
        );
        assert_eq!(
            TlsAdherence::from_spec(Some("StrictAllComponents")),
            TlsAdherence::StrictAllComponents
        );
        assert_eq!(
            TlsAdherence::from_spec(Some("strictallcomponents")),
            TlsAdherence::Unknown("strictallcomponents".to_string())
        );
    }

    #[test]
    fn only_strict_and_unknown_adherence_honor_the_profile() {
        assert!(!TlsAdherence::NoOpinion.honors_cluster_profile());
        assert!(!TlsAdherence::LegacyAdheringComponentsOnly.honors_cluster_profile());
        assert!(TlsAdherence::StrictAllComponents.honors_cluster_profile());
        assert!(TlsAdherence::Unknown("FutureStrict".to_string()).honors_cluster_profile());
    }

    #[test]
    fn unset_adherence_keeps_intermediate_under_a_modern_profile() {
        let spec = json!({"tlsSecurityProfile": {"type": "Modern", "modern": {}}});
        assert_eq!(from_apiserver_spec(&spec), Ok(intermediate()));
        let spec = json!({"tlsAdherence": "", "tlsSecurityProfile": {"type": "Modern"}});
        assert_eq!(from_apiserver_spec(&spec), Ok(intermediate()));
    }

    #[test]
    fn legacy_adherence_keeps_intermediate_under_an_old_profile() {
        let spec = json!({
            "tlsAdherence": "LegacyAdheringComponentsOnly",
            "tlsSecurityProfile": {"type": "Old", "old": {}}
        });
        assert_eq!(from_apiserver_spec(&spec), Ok(intermediate()));
    }

    #[test]
    fn strict_adherence_applies_the_profile() {
        let spec = json!({
            "tlsAdherence": "StrictAllComponents",
            "tlsSecurityProfile": {"type": "Modern", "modern": {}}
        });
        assert_eq!(from_apiserver_spec(&spec), Ok(modern()));
    }

    #[test]
    fn unknown_adherence_applies_the_profile() {
        let spec = json!({
            "tlsAdherence": "SomePolicyFromTheFuture",
            "tlsSecurityProfile": {"type": "Old", "old": {}}
        });
        assert_eq!(from_apiserver_spec(&spec), Ok(old()));
    }

    #[test]
    fn strict_adherence_with_no_profile_is_intermediate() {
        let spec = json!({"tlsAdherence": "StrictAllComponents"});
        assert_eq!(from_apiserver_spec(&spec), Ok(intermediate()));
    }

    #[test]
    fn broken_custom_profile_matters_only_when_honored() {
        let broken = json!({"type": "Custom"});
        assert_eq!(
            from_apiserver_spec(&json!({"tlsSecurityProfile": broken})),
            Ok(intermediate())
        );
        assert_eq!(
            from_apiserver_spec(&json!({
                "tlsAdherence": "StrictAllComponents",
                "tlsSecurityProfile": broken
            })),
            Err(ProfileError::CustomMissing)
        );
    }

    #[test]
    fn empty_spec_is_intermediate() {
        assert_eq!(
            from_apiserver_spec(&serde_json::Value::Null),
            Ok(intermediate())
        );
        assert_eq!(from_apiserver_spec(&json!({})), Ok(intermediate()));
    }
}
