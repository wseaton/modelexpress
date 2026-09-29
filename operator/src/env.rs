// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Renders a ModelExpressServerSpec into the server container's env vars.

use crate::crd::{MetadataBackend, ModelExpressServerSpec, SecretKeyRef};
use crate::tls::{MOUNT_PATH, ResolvedTls};
use k8s_openapi::api::core::v1::{EnvVar, EnvVarSource, ObjectFieldSelector, SecretKeySelector};

// The subset of the server's env vars the operator sets. Declared here rather
// than taken from modelexpress-common so the operator binary doesn't link the
// gRPC and download stack for seventeen strings; `envs_match_the_server` fails
// the build if they ever drift.
pub const HF_TOKEN: &str = "HF_TOKEN";
pub const MODEL_EXPRESS_CACHE_DIRECTORY: &str = "MODEL_EXPRESS_CACHE_DIRECTORY";
pub const MODEL_EXPRESS_CACHE_EVICTION_ENABLED: &str = "MODEL_EXPRESS_CACHE_EVICTION_ENABLED";
pub const MODEL_EXPRESS_LOG_FORMAT: &str = "MODEL_EXPRESS_LOG_FORMAT";
pub const MODEL_EXPRESS_LOG_LEVEL: &str = "MODEL_EXPRESS_LOG_LEVEL";
pub const MODEL_EXPRESS_SECURITY_ALLOWED_SERVICE_ACCOUNTS: &str =
    "MODEL_EXPRESS_SECURITY_ALLOWED_SERVICE_ACCOUNTS";
pub const MODEL_EXPRESS_SECURITY_CACHE_TTL_SECS: &str = "MODEL_EXPRESS_SECURITY_CACHE_TTL_SECS";
pub const MODEL_EXPRESS_SECURITY_MODE: &str = "MODEL_EXPRESS_SECURITY_MODE";
pub const MODEL_EXPRESS_SECURITY_TOKEN_AUDIENCES: &str = "MODEL_EXPRESS_SECURITY_TOKEN_AUDIENCES";
pub const MODEL_EXPRESS_SERVER_PORT: &str = "MODEL_EXPRESS_SERVER_PORT";
pub const MX_TLS_CERT_FILE: &str = "MX_TLS_CERT_FILE";
pub const MX_TLS_CIPHER_SUITES: &str = "MX_TLS_CIPHER_SUITES";
pub const MX_TLS_GROUPS: &str = "MX_TLS_GROUPS";
pub const MX_TLS_KEY_FILE: &str = "MX_TLS_KEY_FILE";
pub const MX_TLS_MIN_VERSION: &str = "MX_TLS_MIN_VERSION";
pub const MX_GC_TIMEOUT_SECS: &str = "MX_GC_TIMEOUT_SECS";
pub const MX_HEARTBEAT_TIMEOUT_SECS: &str = "MX_HEARTBEAT_TIMEOUT_SECS";
pub const MX_METADATA_BACKEND: &str = "MX_METADATA_BACKEND";
pub const MX_REAPER_SCAN_INTERVAL_SECS: &str = "MX_REAPER_SCAN_INTERVAL_SECS";
pub const NGC_API_KEY: &str = "NGC_API_KEY";
pub const POD_NAMESPACE: &str = "POD_NAMESPACE";
pub const REDIS_URL: &str = "REDIS_URL";

fn literal(name: &str, value: impl Into<String>) -> EnvVar {
    EnvVar {
        name: name.to_string(),
        value: Some(value.into()),
        value_from: None,
    }
}

fn from_secret(name: &str, secret: &SecretKeyRef) -> EnvVar {
    EnvVar {
        name: name.to_string(),
        value: None,
        value_from: Some(EnvVarSource {
            secret_key_ref: Some(SecretKeySelector {
                name: secret.name.clone(),
                key: secret.key.clone(),
                optional: Some(false),
            }),
            ..EnvVarSource::default()
        }),
    }
}

fn downward_namespace(name: &str) -> EnvVar {
    EnvVar {
        name: name.to_string(),
        value: None,
        value_from: Some(EnvVarSource {
            field_ref: Some(ObjectFieldSelector {
                field_path: "metadata.namespace".to_string(),
                api_version: None,
            }),
            ..EnvVarSource::default()
        }),
    }
}

/// The server binds 0.0.0.0 by default, so no host var is set.
pub fn render_env(spec: &ModelExpressServerSpec, tls: Option<&ResolvedTls>) -> Vec<EnvVar> {
    let mut env = Vec::new();

    match &spec.metadata_backend {
        MetadataBackend::Redis(redis) => {
            env.push(literal(MX_METADATA_BACKEND, "redis"));
            // admission enforces exactly one of the two
            if let Some(secret) = &redis.url_secret {
                env.push(from_secret(REDIS_URL, secret));
            } else if let Some(url) = &redis.url {
                env.push(literal(REDIS_URL, url));
            }
        }
        MetadataBackend::Kubernetes {} => {
            env.push(literal(MX_METADATA_BACKEND, "kubernetes"));
            env.push(downward_namespace(POD_NAMESPACE));
        }
    }

    env.push(literal(MODEL_EXPRESS_SERVER_PORT, spec.port.to_string()));

    if let Some(log) = &spec.log {
        if let Some(level) = log.level {
            env.push(literal(MODEL_EXPRESS_LOG_LEVEL, level.as_env_value()));
        }
        if let Some(format) = log.format {
            env.push(literal(MODEL_EXPRESS_LOG_FORMAT, format.as_env_value()));
        }
    }

    // unset makes the server fall back to HOME and write outside the mount
    env.push(literal(
        MODEL_EXPRESS_CACHE_DIRECTORY,
        crate::volume::mount_path(spec),
    ));

    if let Some(eviction) = spec.cache.as_ref().and_then(|c| c.eviction_enabled) {
        env.push(literal(
            MODEL_EXPRESS_CACHE_EVICTION_ENABLED,
            eviction.to_string(),
        ));
    }

    if let Some(security) = &spec.security {
        env.push(literal(
            MODEL_EXPRESS_SECURITY_MODE,
            security.mode.as_env_value(),
        ));
        if !security.token_audiences.is_empty() {
            env.push(literal(
                MODEL_EXPRESS_SECURITY_TOKEN_AUDIENCES,
                security.token_audiences.join(","),
            ));
        }
        if !security.allowed_service_accounts.is_empty() {
            let allowlist = security
                .allowed_service_accounts
                .iter()
                .map(|sa| format!("{}:{}", sa.namespace, sa.service_account))
                .collect::<Vec<_>>()
                .join(",");
            env.push(literal(
                MODEL_EXPRESS_SECURITY_ALLOWED_SERVICE_ACCOUNTS,
                allowlist,
            ));
        }
        if let Some(ttl) = security.cache_ttl_secs {
            env.push(literal(
                MODEL_EXPRESS_SECURITY_CACHE_TTL_SECS,
                ttl.to_string(),
            ));
        }
    }

    if let Some(tls) = tls {
        env.push(literal(MX_TLS_CERT_FILE, format!("{MOUNT_PATH}/tls.crt")));
        env.push(literal(MX_TLS_KEY_FILE, format!("{MOUNT_PATH}/tls.key")));
        let settings = &tls.settings;
        if let Some(min_version) = &settings.min_version {
            env.push(literal(MX_TLS_MIN_VERSION, min_version.clone()));
        }
        if !settings.ciphers.is_empty() {
            env.push(literal(MX_TLS_CIPHER_SUITES, settings.ciphers.join(",")));
        }
        if !settings.groups.is_empty() {
            env.push(literal(MX_TLS_GROUPS, settings.groups.join(",")));
        }
    }

    if let Some(reaper) = &spec.reaper {
        if let Some(secs) = reaper.scan_interval_secs {
            env.push(literal(MX_REAPER_SCAN_INTERVAL_SECS, secs.to_string()));
        }
        if let Some(secs) = reaper.heartbeat_timeout_secs {
            env.push(literal(MX_HEARTBEAT_TIMEOUT_SECS, secs.to_string()));
        }
        if let Some(secs) = reaper.gc_timeout_secs {
            env.push(literal(MX_GC_TIMEOUT_SECS, secs.to_string()));
        }
    }

    if let Some(creds) = &spec.credentials {
        if let Some(secret) = &creds.hf_token_secret {
            env.push(from_secret(HF_TOKEN, secret));
        }
        if let Some(secret) = &creds.ngc_api_key_secret {
            env.push(from_secret(NGC_API_KEY, secret));
        }
    }

    env
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::crd::{
        AuthMode, CacheConfig, CredentialsConfig, LogConfig, LogFormat, LogLevel, ReaperConfig,
        RedisBackend, SecurityConfig, ServiceAccountRef,
    };

    /// The operator writes these names, the server reads them. Renaming one on
    /// either side without the other silently drops config, so pin them to the
    /// server's own definitions.
    #[test]
    fn envs_match_the_server() {
        use modelexpress_common::envs as server;
        for (ours, theirs) in [
            (HF_TOKEN, server::HF_TOKEN),
            (
                MODEL_EXPRESS_CACHE_DIRECTORY,
                server::MODEL_EXPRESS_CACHE_DIRECTORY,
            ),
            (
                MODEL_EXPRESS_CACHE_EVICTION_ENABLED,
                server::MODEL_EXPRESS_CACHE_EVICTION_ENABLED,
            ),
            (MODEL_EXPRESS_LOG_FORMAT, server::MODEL_EXPRESS_LOG_FORMAT),
            (MODEL_EXPRESS_LOG_LEVEL, server::MODEL_EXPRESS_LOG_LEVEL),
            (
                MODEL_EXPRESS_SECURITY_ALLOWED_SERVICE_ACCOUNTS,
                server::MODEL_EXPRESS_SECURITY_ALLOWED_SERVICE_ACCOUNTS,
            ),
            (
                MODEL_EXPRESS_SECURITY_CACHE_TTL_SECS,
                server::MODEL_EXPRESS_SECURITY_CACHE_TTL_SECS,
            ),
            (
                MODEL_EXPRESS_SECURITY_MODE,
                server::MODEL_EXPRESS_SECURITY_MODE,
            ),
            (
                MODEL_EXPRESS_SECURITY_TOKEN_AUDIENCES,
                server::MODEL_EXPRESS_SECURITY_TOKEN_AUDIENCES,
            ),
            (MODEL_EXPRESS_SERVER_PORT, server::MODEL_EXPRESS_SERVER_PORT),
            (MX_TLS_CERT_FILE, server::MX_TLS_CERT_FILE),
            (MX_TLS_CIPHER_SUITES, server::MX_TLS_CIPHER_SUITES),
            (MX_TLS_GROUPS, server::MX_TLS_GROUPS),
            (MX_TLS_KEY_FILE, server::MX_TLS_KEY_FILE),
            (MX_TLS_MIN_VERSION, server::MX_TLS_MIN_VERSION),
            (MX_GC_TIMEOUT_SECS, server::MX_GC_TIMEOUT_SECS),
            (MX_HEARTBEAT_TIMEOUT_SECS, server::MX_HEARTBEAT_TIMEOUT_SECS),
            (MX_METADATA_BACKEND, server::MX_METADATA_BACKEND),
            (
                MX_REAPER_SCAN_INTERVAL_SECS,
                server::MX_REAPER_SCAN_INTERVAL_SECS,
            ),
            (NGC_API_KEY, server::NGC_API_KEY),
            (POD_NAMESPACE, server::POD_NAMESPACE),
            (REDIS_URL, server::REDIS_URL),
        ] {
            assert_eq!(ours, theirs);
        }
    }

    fn render_env_plain(spec: &ModelExpressServerSpec) -> Vec<EnvVar> {
        render_env(spec, None)
    }

    fn base_spec(backend: MetadataBackend) -> ModelExpressServerSpec {
        ModelExpressServerSpec {
            image: Some("nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.5.0".into()),
            replicas: 1,
            metadata_backend: backend,
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
        }
    }

    fn value_of<'a>(env: &'a [EnvVar], name: &str) -> Option<&'a str> {
        env.iter()
            .find(|e| e.name == name)
            .and_then(|e| e.value.as_deref())
    }

    #[test]
    fn redis_backend_sets_backend_and_url() {
        let env = render_env_plain(&base_spec(MetadataBackend::Redis(RedisBackend {
            url: Some("redis://mx-redis:6379".into()),
            url_secret: None,
        })));
        assert_eq!(value_of(&env, MX_METADATA_BACKEND), Some("redis"));
        assert_eq!(value_of(&env, REDIS_URL), Some("redis://mx-redis:6379"));
        assert!(!env.iter().any(|e| e.name == POD_NAMESPACE));
    }

    #[test]
    fn kubernetes_backend_uses_downward_namespace() {
        let env = render_env_plain(&base_spec(MetadataBackend::Kubernetes {}));
        assert_eq!(value_of(&env, MX_METADATA_BACKEND), Some("kubernetes"));
        let ns = env
            .iter()
            .find(|e| e.name == POD_NAMESPACE)
            .expect("POD_NAMESPACE present");
        let field_path = ns
            .value_from
            .as_ref()
            .and_then(|v| v.field_ref.as_ref())
            .map(|f| f.field_path.as_str());
        assert_eq!(field_path, Some("metadata.namespace"));
    }

    #[test]
    fn minimal_spec_renders_no_optional_vars() {
        let env = render_env_plain(&base_spec(MetadataBackend::Kubernetes {}));
        assert_eq!(value_of(&env, MODEL_EXPRESS_SERVER_PORT), Some("8001"));
        for name in [
            MODEL_EXPRESS_LOG_LEVEL,
            MODEL_EXPRESS_SECURITY_MODE,
            MX_REAPER_SCAN_INTERVAL_SECS,
            HF_TOKEN,
        ] {
            assert!(!env.iter().any(|e| e.name == name), "{name} unexpected");
        }
    }

    #[test]
    fn redis_url_renders_from_a_secret_when_set() {
        let env = render_env_plain(&base_spec(MetadataBackend::Redis(RedisBackend {
            url: None,
            url_secret: Some(crate::crd::SecretKeyRef {
                name: "mx-redis".into(),
                key: "url".into(),
            }),
        })));
        let var = env
            .iter()
            .find(|e| e.name == REDIS_URL)
            .expect("REDIS_URL rendered");
        assert!(
            var.value.is_none(),
            "credentials must not land in the pod template"
        );
        let secret = var
            .value_from
            .as_ref()
            .and_then(|v| v.secret_key_ref.as_ref())
            .expect("secretKeyRef");
        assert_eq!(secret.name, "mx-redis");
        assert_eq!(secret.key, "url");
    }

    #[test]
    fn redis_url_secret_wins_over_a_literal() {
        // admission rejects both being set; this pins the render if it slips
        let env = render_env_plain(&base_spec(MetadataBackend::Redis(RedisBackend {
            url: Some("redis://plain:6379".into()),
            url_secret: Some(crate::crd::SecretKeyRef {
                name: "mx-redis".into(),
                key: "url".into(),
            }),
        })));
        let var = env.iter().find(|e| e.name == REDIS_URL).expect("REDIS_URL");
        assert_eq!(var.value, None);
        assert!(var.value_from.is_some());
    }

    #[test]
    fn cache_directory_always_matches_the_mount_path() {
        let spec = base_spec(MetadataBackend::Kubernetes {});
        assert_eq!(
            value_of(&render_env(&spec, None), MODEL_EXPRESS_CACHE_DIRECTORY),
            Some(crate::volume::DEFAULT_MOUNT_PATH),
            "unset means the server falls back to HOME and misses the volume"
        );

        let mut spec = base_spec(MetadataBackend::Kubernetes {});
        spec.cache = Some(crate::crd::CacheConfig {
            directory: Some("/cache".into()),
            ..crate::crd::CacheConfig::default()
        });
        assert_eq!(
            value_of(&render_env(&spec, None), MODEL_EXPRESS_CACHE_DIRECTORY),
            Some("/cache")
        );
        assert_eq!(
            crate::volume::render_cache_volume("mx", &spec)
                .mount
                .mount_path,
            "/cache"
        );
    }

    #[test]
    fn full_spec_renders_every_var() {
        let mut spec = base_spec(MetadataBackend::Redis(RedisBackend {
            url: Some("rediss://mx-redis:6380".into()),
            url_secret: None,
        }));
        spec.port = 9000;
        spec.log = Some(LogConfig {
            level: Some(LogLevel::Debug),
            format: Some(LogFormat::Json),
        });
        spec.cache = Some(CacheConfig {
            directory: Some("/cache".into()),
            eviction_enabled: Some(true),
            ..CacheConfig::default()
        });
        spec.security = Some(SecurityConfig {
            mode: AuthMode::Enforce,
            token_audiences: vec!["mx".into(), "mx-alt".into()],
            allowed_service_accounts: vec![
                ServiceAccountRef {
                    namespace: "llm-d".into(),
                    service_account: "decode".into(),
                },
                ServiceAccountRef {
                    namespace: "llm-d".into(),
                    service_account: "prefill".into(),
                },
            ],
            cache_ttl_secs: Some(120),
        });
        spec.reaper = Some(ReaperConfig {
            scan_interval_secs: Some(10),
            heartbeat_timeout_secs: Some(30),
            gc_timeout_secs: Some(600),
        });
        spec.credentials = Some(CredentialsConfig {
            hf_token_secret: Some(SecretKeyRef {
                name: "hf-secret".into(),
                key: "HF_TOKEN".into(),
            }),
            ngc_api_key_secret: None,
        });

        let env = render_env(&spec, None);
        assert_eq!(value_of(&env, MODEL_EXPRESS_SERVER_PORT), Some("9000"));
        assert_eq!(value_of(&env, MODEL_EXPRESS_LOG_LEVEL), Some("debug"));
        assert_eq!(value_of(&env, MODEL_EXPRESS_LOG_FORMAT), Some("json"));
        assert_eq!(
            value_of(&env, MODEL_EXPRESS_CACHE_DIRECTORY),
            Some("/cache")
        );
        assert_eq!(
            value_of(&env, MODEL_EXPRESS_CACHE_EVICTION_ENABLED),
            Some("true")
        );
        assert_eq!(value_of(&env, MODEL_EXPRESS_SECURITY_MODE), Some("enforce"));
        assert_eq!(
            value_of(&env, MODEL_EXPRESS_SECURITY_TOKEN_AUDIENCES),
            Some("mx,mx-alt")
        );
        assert_eq!(
            value_of(&env, MODEL_EXPRESS_SECURITY_ALLOWED_SERVICE_ACCOUNTS),
            Some("llm-d:decode,llm-d:prefill")
        );
        assert_eq!(
            value_of(&env, MODEL_EXPRESS_SECURITY_CACHE_TTL_SECS),
            Some("120")
        );
        assert_eq!(value_of(&env, MX_REAPER_SCAN_INTERVAL_SECS), Some("10"));
        assert_eq!(value_of(&env, MX_HEARTBEAT_TIMEOUT_SECS), Some("30"));
        assert_eq!(value_of(&env, MX_GC_TIMEOUT_SECS), Some("600"));

        let hf = env.iter().find(|e| e.name == HF_TOKEN).expect("HF_TOKEN");
        let secret_name = hf
            .value_from
            .as_ref()
            .and_then(|v| v.secret_key_ref.as_ref())
            .map(|s| s.name.as_str());
        assert_eq!(secret_name, Some("hf-secret"));
        assert!(!env.iter().any(|e| e.name == NGC_API_KEY));
    }

    #[test]
    fn env_names_are_unique() {
        let mut spec = base_spec(MetadataBackend::Kubernetes {});
        spec.log = Some(LogConfig {
            level: Some(LogLevel::Info),
            format: Some(LogFormat::Compact),
        });
        let env = render_env(&spec, None);
        let mut names: Vec<_> = env.iter().map(|e| e.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), env.len());
    }
}
