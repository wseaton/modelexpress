// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fault-injecting decorator over a real `RegistryBackend`.
//!
//! Wraps any backend (in practice the in-memory one) and injects failures, latency,
//! hangs, or panics at the trait boundary, so tests can drive the server through backend
//! failure modes that are painful to reproduce against a live Redis. The success path
//! delegates to the inner backend, so state stays real: this is fault injection, not a
//! mock. The faults it injects (transient errors, slow responses, dropped connections)
//! all happen in production; we just make them happen on demand.
//!
//! A test supplies a policy closure `(op, nth_call) -> Fault`. Faults are injected
//! *before* the inner call, so a `Fault::Fail` leaves the store unmodified (it models a
//! request that never reached the backend).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use modelexpress_common::models::{ModelProvider, ModelStatus};

use crate::registry::backend::{ClaimOutcome, ModelRecord, RegistryBackend, RegistryResult};

/// The operation a policy is being consulted about. Carries the bits of the arguments a
/// policy is likely to branch on (model name, target status), not the whole call.
#[derive(Debug, Clone)]
pub enum RegistryOp {
    Connect,
    GetStatus {
        model_name: String,
    },
    GetModelRecord {
        model_name: String,
    },
    SetStatus {
        model_name: String,
        status: ModelStatus,
    },
    TouchModel {
        model_name: String,
    },
    DeleteModel {
        model_name: String,
    },
    GetModelsByLastUsed,
    GetStatusCounts,
    TryClaimForDownload {
        model_name: String,
    },
    TryResetErrorForRetry {
        model_name: String,
    },
    RefreshDownloadClaim {
        model_name: String,
    },
    FinishDownloadClaim {
        model_name: String,
        status: ModelStatus,
    },
}

impl RegistryOp {
    /// Stable key for per-op call counting. One arm per trait method.
    fn kind(&self) -> &'static str {
        match self {
            Self::Connect => "connect",
            Self::GetStatus { .. } => "get_status",
            Self::GetModelRecord { .. } => "get_model_record",
            Self::SetStatus { .. } => "set_status",
            Self::TouchModel { .. } => "touch_model",
            Self::DeleteModel { .. } => "delete_model",
            Self::GetModelsByLastUsed => "get_models_by_last_used",
            Self::GetStatusCounts => "get_status_counts",
            Self::TryClaimForDownload { .. } => "try_claim_for_download",
            Self::TryResetErrorForRetry { .. } => "try_reset_error_for_retry",
            Self::RefreshDownloadClaim { .. } => "refresh_download_claim",
            Self::FinishDownloadClaim { .. } => "finish_download_claim",
        }
    }
}

/// What the decorator should do for a consulted op. Anything other than `Pass` is
/// injected before the inner backend is touched.
pub enum Fault {
    /// Delegate to the real backend.
    Pass,
    /// Return this error without calling the backend (state unchanged).
    Fail(String),
    /// Sleep, then delegate (latency injection).
    Delay(Duration),
    /// Sleep, then return the error (slow failure / timeout path).
    DelayThenFail(Duration, String),
    /// Never resolve. Tests server-side timeouts; race it with `tokio::time::timeout`.
    Hang,
    /// Panic. Tests request-level panic isolation.
    Panic(String),
}

/// Consulted on every call: `(op, nth_call_of_this_op_kind) -> Fault`. The count is
/// 1-based and per-op-kind, so `n == 2` means "the second `set_status`".
type Policy = dyn Fn(&RegistryOp, u32) -> Fault + Send + Sync;

/// A `RegistryBackend` that consults a policy before delegating to an inner backend.
pub struct FaultyRegistryBackend {
    inner: Arc<dyn RegistryBackend>,
    policy: Box<Policy>,
    counts: Mutex<HashMap<&'static str, u32>>,
}

impl FaultyRegistryBackend {
    pub fn new(
        inner: Arc<dyn RegistryBackend>,
        policy: impl Fn(&RegistryOp, u32) -> Fault + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner,
            policy: Box::new(policy),
            counts: Mutex::new(HashMap::new()),
        }
    }

    /// Consult the policy and apply any pre-call fault. `Ok(())` means "proceed to inner".
    async fn gate(&self, op: RegistryOp) -> RegistryResult<()> {
        let count = {
            let mut counts = self.counts.lock().unwrap_or_else(PoisonError::into_inner);
            let entry = counts.entry(op.kind()).or_insert(0);
            *entry = entry.saturating_add(1);
            *entry
        };
        match (self.policy)(&op, count) {
            Fault::Pass => Ok(()),
            Fault::Fail(msg) => Err(msg.into()),
            Fault::Delay(d) => {
                tokio::time::sleep(d).await;
                Ok(())
            }
            Fault::DelayThenFail(d, msg) => {
                tokio::time::sleep(d).await;
                Err(msg.into())
            }
            Fault::Hang => std::future::pending::<RegistryResult<()>>().await,
            Fault::Panic(msg) => panic!("fault-injected panic: {msg}"),
        }
    }
}

#[async_trait]
impl RegistryBackend for FaultyRegistryBackend {
    async fn connect(&self) -> RegistryResult<()> {
        self.gate(RegistryOp::Connect).await?;
        self.inner.connect().await
    }

    async fn get_status(&self, model_name: &str) -> RegistryResult<Option<ModelStatus>> {
        self.gate(RegistryOp::GetStatus {
            model_name: model_name.to_string(),
        })
        .await?;
        self.inner.get_status(model_name).await
    }

    async fn get_model_record(&self, model_name: &str) -> RegistryResult<Option<ModelRecord>> {
        self.gate(RegistryOp::GetModelRecord {
            model_name: model_name.to_string(),
        })
        .await?;
        self.inner.get_model_record(model_name).await
    }

    async fn set_status(
        &self,
        model_name: &str,
        provider: ModelProvider,
        status: ModelStatus,
        message: Option<String>,
    ) -> RegistryResult<()> {
        self.gate(RegistryOp::SetStatus {
            model_name: model_name.to_string(),
            status,
        })
        .await?;
        self.inner
            .set_status(model_name, provider, status, message)
            .await
    }

    async fn touch_model(&self, model_name: &str) -> RegistryResult<()> {
        self.gate(RegistryOp::TouchModel {
            model_name: model_name.to_string(),
        })
        .await?;
        self.inner.touch_model(model_name).await
    }

    async fn delete_model(&self, model_name: &str) -> RegistryResult<()> {
        self.gate(RegistryOp::DeleteModel {
            model_name: model_name.to_string(),
        })
        .await?;
        self.inner.delete_model(model_name).await
    }

    async fn get_models_by_last_used(
        &self,
        limit: Option<u32>,
    ) -> RegistryResult<Vec<ModelRecord>> {
        self.gate(RegistryOp::GetModelsByLastUsed).await?;
        self.inner.get_models_by_last_used(limit).await
    }

    async fn get_status_counts(&self) -> RegistryResult<(u32, u32, u32)> {
        self.gate(RegistryOp::GetStatusCounts).await?;
        self.inner.get_status_counts().await
    }

    async fn try_claim_for_download(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: Duration,
    ) -> RegistryResult<ClaimOutcome> {
        self.gate(RegistryOp::TryClaimForDownload {
            model_name: model_name.to_string(),
        })
        .await?;
        self.inner
            .try_claim_for_download(model_name, provider, claim_id, lease_duration)
            .await
    }

    async fn try_reset_error_for_retry(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: Duration,
    ) -> RegistryResult<bool> {
        self.gate(RegistryOp::TryResetErrorForRetry {
            model_name: model_name.to_string(),
        })
        .await?;
        self.inner
            .try_reset_error_for_retry(model_name, provider, claim_id, lease_duration)
            .await
    }

    async fn refresh_download_claim(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: Duration,
    ) -> RegistryResult<bool> {
        self.gate(RegistryOp::RefreshDownloadClaim {
            model_name: model_name.to_string(),
        })
        .await?;
        self.inner
            .refresh_download_claim(model_name, provider, claim_id, lease_duration)
            .await
    }

    async fn finish_download_claim(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        status: ModelStatus,
        message: Option<String>,
    ) -> RegistryResult<bool> {
        self.gate(RegistryOp::FinishDownloadClaim {
            model_name: model_name.to_string(),
            status,
        })
        .await?;
        self.inner
            .finish_download_claim(model_name, provider, claim_id, status, message)
            .await
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::registry::backend::memory::InMemoryRegistryBackend;

    fn mem() -> Arc<dyn RegistryBackend> {
        Arc::new(InMemoryRegistryBackend::new())
    }

    // A no-op policy leaves the wrapped backend fully functional: state is real.
    #[tokio::test]
    async fn pass_through_delegates_to_inner() {
        let backend = FaultyRegistryBackend::new(mem(), |_, _| Fault::Pass);
        backend
            .set_status(
                "m",
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
                None,
            )
            .await
            .expect("set");
        assert_eq!(
            backend.get_status("m").await.expect("get"),
            Some(ModelStatus::DOWNLOADED)
        );
    }

    // Counter-based: fail the 2nd set_status. The fault is pre-call, so the store
    // reflects only the first (successful) write.
    #[tokio::test]
    async fn fails_the_nth_call_without_mutating_state() {
        let backend = FaultyRegistryBackend::new(mem(), |op, n| match op {
            RegistryOp::SetStatus { .. } if n == 2 => Fault::Fail("redis blip".to_string()),
            _ => Fault::Pass,
        });
        backend
            .set_status(
                "m",
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADING,
                None,
            )
            .await
            .expect("first set ok");
        let err = backend
            .set_status(
                "m",
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
                None,
            )
            .await
            .expect_err("second set fails");
        assert!(err.to_string().contains("redis blip"));
        assert_eq!(
            backend.get_status("m").await.expect("get"),
            Some(ModelStatus::DOWNLOADING),
            "failed write never reached the store"
        );
    }

    // Argument-based: one poisoned model fails while healthy ones keep serving.
    #[tokio::test]
    async fn arg_based_fault_isolates_other_models() {
        let backend = FaultyRegistryBackend::new(mem(), |op, _| match op {
            RegistryOp::SetStatus { model_name, .. } if model_name == "boom" => {
                Fault::Fail("boom is cursed".to_string())
            }
            _ => Fault::Pass,
        });
        assert!(
            backend
                .set_status(
                    "boom",
                    ModelProvider::HuggingFace,
                    ModelStatus::DOWNLOADED,
                    None
                )
                .await
                .is_err()
        );
        backend
            .set_status(
                "fine",
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
                None,
            )
            .await
            .expect("healthy model still writes");
    }

    // Latency: Delay still delegates, just later.
    #[tokio::test]
    async fn delay_then_passes_still_writes() {
        let backend = FaultyRegistryBackend::new(mem(), |op, _| match op {
            RegistryOp::SetStatus { .. } => Fault::Delay(Duration::from_millis(5)),
            _ => Fault::Pass,
        });
        backend
            .set_status(
                "m",
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
                None,
            )
            .await
            .expect("delayed set still writes");
        assert_eq!(
            backend.get_status("m").await.expect("get"),
            Some(ModelStatus::DOWNLOADED)
        );
    }
}
