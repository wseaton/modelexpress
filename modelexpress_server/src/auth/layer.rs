// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Async tower middleware that authenticates `P2pService` gRPC callers.
//!
//! A `Layer`/`Service` rather than a tonic `Interceptor` because verification awaits
//! `TokenReview` and store lookups. Applied to the `P2pService` route only (by wrapping
//! the generated server), so routing and gating share one source of truth.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use http::{Request, Response};
use tonic::body::Body;
use tonic::server::NamedService;
use tower::{Layer, Service};
use tracing::{debug, warn};

use crate::auth::AuthState;
use crate::config::AuthMode;

/// Gates a service on SA-token + device possession.
#[derive(Clone)]
pub struct AuthLayer {
    state: Arc<AuthState>,
}

impl AuthLayer {
    #[must_use]
    pub fn new(state: Arc<AuthState>) -> Self {
        Self { state }
    }
}

impl<S> Layer<S> for AuthLayer {
    type Service = AuthService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuthService {
            inner,
            state: self.state.clone(),
        }
    }
}

#[derive(Clone)]
pub struct AuthService<S> {
    inner: S,
    state: Arc<AuthState>,
}

/// Forward the wrapped service's gRPC name so the tonic router still dispatches to it.
impl<S: NamedService> NamedService for AuthService<S> {
    const NAME: &'static str = S::NAME;
}

impl<S> Service<Request<Body>> for AuthService<S>
where
    S: Service<Request<Body>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response<Body>;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Response<Body>, S::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        // Clone-and-swap so we call the *ready* inner (poll_ready applied to self.inner).
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        let state = self.state.clone();

        Box::pin(async move {
            match state.verify(req.headers()).await {
                Ok(caller) => {
                    debug!(
                        namespace = %caller.namespace,
                        service_account = %caller.service_account,
                        pod = caller.pod_name.as_deref().unwrap_or("?"),
                        path = %req.uri().path(),
                        "device authorization ok"
                    );
                    inner.call(req).await
                }
                Err(denial) if state.mode == AuthMode::Enforce => {
                    warn!(reason = %denial, path = %req.uri().path(), "device authorization denied");
                    Ok(denial.into_status().into_http::<Body>())
                }
                Err(denial) => {
                    warn!(
                        reason = %denial,
                        path = %req.uri().path(),
                        "device authorization violation (permissive, allowing)"
                    );
                    inner.call(req).await
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dummy;
    impl NamedService for Dummy {
        const NAME: &'static str = "model_express.p2p.P2pService";
    }

    #[test]
    fn forwards_wrapped_service_name() {
        assert_eq!(
            <AuthService<Dummy> as NamedService>::NAME,
            <Dummy as NamedService>::NAME
        );
    }
}
