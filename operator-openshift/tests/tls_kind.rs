// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The cluster TLS profile, end to end on a real API server.
//!
//! kind stands in for OpenShift: the `apiservers.config.openshift.io` CRD is
//! installed (tests/kind_crds), so the `cluster` object can be edited freely
//! without rolling a real kube-apiserver. Against it the test drives the
//! OpenShift operator as shipped (OpenSSL) and two operands, one
//! per server TLS backend, and checks every policy with real handshakes
//! through pod port-forwards:
//!
//! ```text
//!   tlsAdherence   profile   operands            operator /metrics
//!   (unset)        Modern    Intermediate        Intermediate
//!   Strict         Modern    TLS1.3 only         TLS1.3 only, no restart
//!   Strict         Custom    ciphers + groups    ciphers + groups, no restart
//!   Legacy         Custom    Intermediate        Intermediate, no restart
//!   (metrics certificate rotated)                new cert served, no restart
//! ```
//!
//! The operator comes from one of two installs, picked by `test_tls_kind.sh
//! --overlay`: config/manifests/openshift in its own namespace (tests/tls_kind),
//! or config/manifests/overlays/odh the way a platform operator applies it, in
//! a namespace the overlay does not name and with both images set through
//! base/params.env (tests/odh_kind). The scenario is the same for both.
//!
//! A third install, config/manifests/overlays/odh-xks (tests/odh_xks_kind), is
//! the platform install on a cluster that is not OpenShift. It gets its own
//! scenario, `platform_install_without_openshift`: with no APIServer to edit,
//! what matters is that operands still come up on the Intermediate profile.
//!
//! The prometheus-operator API is installed too, as a schemaless CRD for the
//! same group and kind, so the ServiceMonitor the operator applies for its own
//! metrics can be checked without running prometheus-operator.
//!
//! The APIServer CRD is openshift/api's TechPreviewNoUpgrade variant, the one carrying
//! `spec.tlsAdherence`, stored as JSON: config/v1/zz_generated.crd-manifests/
//! 0000_10_config-operator_01_apiservers-TechPreviewNoUpgrade.crd.yaml at
//! openshift/api fba11a566839afbcdab1162978eb836ad8b86cad.
//!
//! `#[ignore]`: needs the cluster `test_tls_kind.sh` sets up, which also runs it.

#![allow(clippy::expect_used)]

use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use k8s_openapi::ByteString;
use k8s_openapi::api::apps::v1::Deployment;
use k8s_openapi::api::core::v1::{Namespace, Pod, Secret};
use kube::api::{
    Api, ApiResource, DeleteParams, DynamicObject, GroupVersionKind, ListParams, Patch,
    PatchParams, PostParams,
};
use kube::config::{KubeConfigOptions, Kubeconfig};
use kube::{Client, ResourceExt};
use modelexpress_client::Client as MxClient;
use modelexpress_common::client_config::ClientConfig;
use modelexpress_common::config::ConnectionConfig;
use rcgen::{
    BasicConstraints, CertificateParams, CertifiedIssuer, DnType, IsCa, KeyPair,
    PKCS_ECDSA_P256_SHA256,
};
use rustls::crypto::{CryptoProvider, ring};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::{
    CipherSuite, ClientConfig as TlsClientConfig, NamedGroup, ProtocolVersion, RootCertStore,
    SupportedProtocolVersion,
};
use serde_json::json;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio_rustls::TlsConnector;

/// Set by test_tls_kind.sh: this test is destructive, so it does not run
/// against whatever cluster happens to be current.
const OPT_IN_ENV: &str = "MX_TLS_KIND_E2E";
const CONTEXT_ENV: &str = "MX_TLS_KIND_CONTEXT";
const DEFAULT_CONTEXT: &str = "kind-mx-tls-e2e";

/// Set by test_tls_kind.sh to the namespace its overlay installs into.
const OPERATOR_NS_ENV: &str = "MX_TLS_KIND_OPERATOR_NS";
const DEFAULT_OPERATOR_NS: &str = "modelexpress-operator-system";
const OPERATOR_SELECTOR: &str = "app.kubernetes.io/name=modelexpress-operator";
const METRICS_SECRET: &str = "modelexpress-operator-metrics-tls";
const METRICS_SERVICE: &str = "modelexpress-operator-metrics";
const METRICS_PORT: u16 = 8443;

const SERVER: &str = "mx";
const SERVER_PORT: u16 = 8001;
const SERVER_SECRET: &str = "mx-tls";

/// How long a policy change may take to show up: an APIServer watch event,
/// a reconcile and a Deployment rollout, or an acceptor swap.
const CONVERGE: Duration = Duration::from_secs(180);
/// kubelet syncs a Secret volume within about a minute, and the operator
/// checks its certificate every minute.
const ROTATE: Duration = Duration::from_secs(300);
const POLL: Duration = Duration::from_secs(2);

static TLS12: &[&SupportedProtocolVersion] = &[&rustls::version::TLS12];
static TLS13: &[&SupportedProtocolVersion] = &[&rustls::version::TLS13];

/// One operand per server TLS backend, each in its own namespace.
#[derive(Clone, Copy, Debug)]
enum Backend {
    OpenSsl,
    Rustls,
}

impl Backend {
    const ALL: [Self; 2] = [Self::OpenSsl, Self::Rustls];

    fn namespace(self) -> &'static str {
        match self {
            Self::OpenSsl => "mx-tls-kind-openssl",
            Self::Rustls => "mx-tls-kind-rustls",
        }
    }

    /// The image the operand must run.
    fn image(self) -> &'static str {
        match self {
            Self::OpenSsl => "mx-e2e/server-openssl:kind",
            Self::Rustls => "mx-e2e/server-rustls:kind",
        }
    }

    /// spec.image on the CR. The OpenSSL operand leaves it unset, so its image
    /// comes from RELATED_IMAGE_ODH_MODELEXPRESS_IMAGE on the operator.
    fn cr_image(self) -> Option<&'static str> {
        match self {
            Self::OpenSsl => None,
            Self::Rustls => Some(self.image()),
        }
    }

    fn host(self) -> String {
        format!("{SERVER}.{}.svc", self.namespace())
    }
}

// ---------------------------------------------------------------------------
// Certificates
// ---------------------------------------------------------------------------

/// A CA every serving certificate in the test chains to, playing service-ca.
struct Pki {
    issuer: CertifiedIssuer<'static, KeyPair>,
    ca_file: tempfile::NamedTempFile,
}

impl Pki {
    fn new() -> Result<Self> {
        let mut params = CertificateParams::new(Vec::<String>::new())?;
        params
            .distinguished_name
            .push(DnType::CommonName, "mx tls kind e2e CA");
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let issuer =
            CertifiedIssuer::self_signed(params, KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256)?)?;
        let ca_file = tempfile::NamedTempFile::new()?;
        std::fs::write(ca_file.path(), issuer.pem())?;
        Ok(Self { issuer, ca_file })
    }

    /// An ECDSA P-256 leaf for `names`, as a kubernetes.io/tls Secret body.
    fn leaf(&self, names: &[&str]) -> Result<BTreeMap<String, ByteString>> {
        let names: Vec<String> = names.iter().map(|name| (*name).to_string()).collect();
        let mut params = CertificateParams::new(names)?;
        params
            .distinguished_name
            .push(DnType::CommonName, "mx tls kind e2e");
        let key = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256)?;
        let cert = params.signed_by(&key, &self.issuer)?;
        Ok(BTreeMap::from([
            ("tls.crt".to_string(), ByteString(cert.pem().into_bytes())),
            (
                "tls.key".to_string(),
                ByteString(key.serialize_pem().into_bytes()),
            ),
        ]))
    }

    fn roots(&self) -> Result<RootCertStore> {
        let mut roots = RootCertStore::empty();
        roots.add(CertificateDer::from_pem_slice(
            self.issuer.pem().as_bytes(),
        )?)?;
        Ok(roots)
    }

    fn ca_path(&self) -> PathBuf {
        self.ca_file.path().to_path_buf()
    }
}

// ---------------------------------------------------------------------------
// Handshakes through port-forwards
// ---------------------------------------------------------------------------

/// What the test client offers. Empty lists mean ring's defaults.
#[derive(Clone)]
struct Offer {
    versions: &'static [&'static SupportedProtocolVersion],
    suites: Vec<CipherSuite>,
    groups: Vec<NamedGroup>,
    alpn: &'static [u8],
}

impl Offer {
    fn tls12() -> Self {
        Self {
            versions: TLS12,
            suites: Vec::new(),
            groups: Vec::new(),
            alpn: b"h2",
        }
    }

    fn tls13() -> Self {
        Self {
            versions: TLS13,
            ..Self::tls12()
        }
    }

    fn suite(mut self, suite: CipherSuite) -> Self {
        self.suites = vec![suite];
        self
    }

    fn group(mut self, group: NamedGroup) -> Self {
        self.groups = vec![group];
        self
    }

    fn http1(mut self) -> Self {
        self.alpn = b"http/1.1";
        self
    }

    fn config(&self, roots: RootCertStore) -> Result<TlsClientConfig> {
        let base = ring::default_provider();
        let provider = CryptoProvider {
            cipher_suites: base
                .cipher_suites
                .iter()
                .copied()
                .filter(|suite| self.suites.is_empty() || self.suites.contains(&suite.suite()))
                .collect(),
            kx_groups: base
                .kx_groups
                .iter()
                .copied()
                .filter(|group| self.groups.is_empty() || self.groups.contains(&group.name()))
                .collect(),
            ..base
        };
        let mut config = TlsClientConfig::builder_with_provider(Arc::new(provider))
            .with_protocol_versions(self.versions)?
            .with_root_certificates(roots)
            .with_no_client_auth();
        config.alpn_protocols = vec![self.alpn.to_vec()];
        Ok(config)
    }
}

/// A TLS endpoint inside the cluster: a pod port and the name its
/// certificate is verified against.
struct Target {
    namespace: String,
    pod: String,
    port: u16,
    host: String,
}

impl Target {
    fn describe(&self) -> String {
        format!("{}/{}:{}", self.namespace, self.pod, self.port)
    }
}

type Stream = tokio_rustls::client::TlsStream<Box<dyn Duplex>>;

trait Duplex: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> Duplex for T {}

async fn port_stream(
    client: &Client,
    namespace: &str,
    pod: &str,
    port: u16,
) -> Result<Box<dyn Duplex>> {
    let pods: Api<Pod> = Api::namespaced(client.clone(), namespace);
    let mut forwarder = pods
        .portforward(pod, &[port])
        .await
        .with_context(|| format!("port-forward to {namespace}/{pod}:{port}"))?;
    let stream = forwarder.take_stream(port).context("port-forward stream")?;
    tokio::spawn(async move {
        let _ = forwarder.join().await;
    });
    Ok(Box::new(stream))
}

/// One handshake with `offer`. `Err` carries the TLS failure.
async fn connect(client: &Client, pki: &Pki, target: &Target, offer: &Offer) -> Result<Stream> {
    let raw = port_stream(client, &target.namespace, &target.pod, target.port).await?;
    let connector = TlsConnector::from(Arc::new(offer.config(pki.roots()?)?));
    let name = ServerName::try_from(target.host.clone())?;
    let stream = tokio::time::timeout(Duration::from_secs(20), connector.connect(name, raw))
        .await
        .with_context(|| format!("handshake with {} timed out", target.describe()))?
        .with_context(|| format!("handshake with {}", target.describe()))?;
    Ok(stream)
}

struct Negotiated {
    version: ProtocolVersion,
    suite: CipherSuite,
    group: Option<NamedGroup>,
}

async fn negotiate(
    client: &Client,
    pki: &Pki,
    target: &Target,
    offer: &Offer,
) -> Result<Negotiated> {
    let stream = connect(client, pki, target, offer).await?;
    let conn = stream.get_ref().1;
    Ok(Negotiated {
        version: conn.protocol_version().context("negotiated version")?,
        suite: conn
            .negotiated_cipher_suite()
            .context("negotiated suite")?
            .suite(),
        group: conn
            .negotiated_key_exchange_group()
            .map(|group| group.name()),
    })
}

async fn expect_accepted(
    client: &Client,
    pki: &Pki,
    target: &Target,
    offer: &Offer,
    what: &str,
) -> Result<Negotiated> {
    negotiate(client, pki, target, offer)
        .await
        .with_context(|| format!("{what} must be accepted by {}", target.describe()))
}

async fn expect_refused(
    client: &Client,
    pki: &Pki,
    target: &Target,
    offer: &Offer,
    what: &str,
) -> Result<()> {
    match negotiate(client, pki, target, offer).await {
        Ok(negotiated) => bail!(
            "{what} must be refused by {}, but negotiated {:?} {:?}",
            target.describe(),
            negotiated.version,
            negotiated.suite
        ),
        Err(_) => Ok(()),
    }
}

/// Poll `check` until it holds or `limit` passes.
async fn eventually<F, Fut>(what: &str, limit: Duration, mut check: F) -> Result<()>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<()>>,
{
    let waited = tokio::time::timeout(limit, async {
        loop {
            match check().await {
                Ok(()) => return Ok::<(), anyhow::Error>(()),
                Err(e) => tracing::debug!("waiting for {what}: {e:#}"),
            }
            tokio::time::sleep(POLL).await;
        }
    })
    .await;
    match waited {
        Ok(result) => result,
        Err(_) => match check().await {
            Ok(()) => Ok(()),
            Err(e) => Err(e.context(format!("timed out after {limit:?} waiting for {what}"))),
        },
    }
}

/// An HTTP/1.1 request on an open TLS connection. Reads the whole response,
/// so the connection is ready for the next request, and returns its status
/// line.
async fn http_status(stream: &mut Stream, host: &str, path: &str) -> Result<String> {
    stream
        .write_all(format!("GET {path} HTTP/1.1\r\nhost: {host}\r\n\r\n").as_bytes())
        .await?;
    let mut head = Vec::new();
    let mut byte = [0_u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).await? == 0 {
            bail!("connection closed mid-response");
        }
        head.extend_from_slice(&byte);
    }
    let head = String::from_utf8(head)?;
    let mut lines = head.lines();
    let status = lines.next().context("status line")?.to_string();
    let length = lines
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .map(|(_, value)| value.trim().parse::<usize>())
        .transpose()?
        .with_context(|| format!("response without content-length: {head}"))?;
    let mut body = vec![0_u8; length];
    stream.read_exact(&mut body).await?;
    Ok(status)
}

// ---------------------------------------------------------------------------
// Cluster objects
// ---------------------------------------------------------------------------

/// This test deletes apiservers/cluster, the operator's pods and its own
/// namespaces, so it refuses to run until it has been told to and the
/// kubeconfig points at the throwaway cluster. `#[ignore]` only keeps it out
/// of a plain `cargo test`.
async fn kube_client() -> Result<Client> {
    if std::env::var(OPT_IN_ENV).ok().as_deref() != Some("1") {
        bail!(
            "{OPT_IN_ENV}=1 is required: this test deletes apiservers/cluster, \
             the operator's pods and the namespaces it uses. ./test_tls_kind.sh sets it."
        );
    }
    let expected = std::env::var(CONTEXT_ENV).unwrap_or_else(|_| DEFAULT_CONTEXT.to_string());
    let kubeconfig = Kubeconfig::read().context("reading the kubeconfig")?;
    let current = kubeconfig.current_context.clone().unwrap_or_default();
    if current != expected {
        bail!(
            "kubeconfig context is {current:?}, expected {expected:?}; \
             refusing to touch another cluster (set {CONTEXT_ENV} to override)"
        );
    }
    let config = kube::Config::from_custom_kubeconfig(kubeconfig, &KubeConfigOptions::default())
        .await
        .context("building a client for {expected}")?;
    Ok(Client::try_from(config)?)
}

fn apiservers(client: &Client) -> Api<DynamicObject> {
    let gvk = GroupVersionKind::gvk("config.openshift.io", "v1", "APIServer");
    let mut resource = ApiResource::from_gvk(&gvk);
    resource.plural = "apiservers".to_string();
    Api::all_with(client.clone(), &resource)
}

fn model_express_servers(client: &Client, namespace: &str) -> Api<DynamicObject> {
    let gvk = GroupVersionKind::gvk(
        "modelexpress.opendatahub.io",
        "v1alpha1",
        "ModelExpressServer",
    );
    Api::namespaced_with(client.clone(), namespace, &ApiResource::from_gvk(&gvk))
}

/// Recreate `apiservers/cluster` with `spec`. tlsAdherence cannot be removed
/// once set, so a rerun has to start from a fresh object.
async fn reset_apiserver(client: &Client, spec: serde_json::Value) -> Result<()> {
    let api = apiservers(client);
    if api.get_opt("cluster").await?.is_some() {
        api.delete("cluster", &DeleteParams::default()).await?;
        eventually("apiservers/cluster to be deleted", CONVERGE, || {
            let api = api.clone();
            async move {
                match api.get_opt("cluster").await? {
                    None => Ok(()),
                    Some(_) => bail!("still present"),
                }
            }
        })
        .await?;
    }
    let object: DynamicObject = serde_json::from_value(json!({
        "apiVersion": "config.openshift.io/v1",
        "kind": "APIServer",
        "metadata": {"name": "cluster"},
        "spec": spec,
    }))?;
    api.create(&PostParams::default(), &object).await?;
    Ok(())
}

async fn patch_apiserver(client: &Client, spec: serde_json::Value) -> Result<()> {
    apiservers(client)
        .patch(
            "cluster",
            &PatchParams::default(),
            &Patch::Merge(json!({"spec": spec})),
        )
        .await
        .context("patch apiservers/cluster")?;
    Ok(())
}

async fn recreate_namespace(client: &Client, name: &str) -> Result<()> {
    let api: Api<Namespace> = Api::all(client.clone());
    if api.get_opt(name).await?.is_some() {
        api.delete(name, &DeleteParams::default()).await?;
        eventually(&format!("namespace {name} to be deleted"), CONVERGE, || {
            let api = api.clone();
            let name = name.to_string();
            async move {
                match api.get_opt(&name).await? {
                    None => Ok(()),
                    Some(_) => bail!("terminating"),
                }
            }
        })
        .await?;
    }
    let namespace: Namespace = serde_json::from_value(json!({"metadata": {"name": name}}))?;
    api.create(&PostParams::default(), &namespace).await?;
    Ok(())
}

async fn apply_tls_secret(
    client: &Client,
    namespace: &str,
    name: &str,
    data: BTreeMap<String, ByteString>,
) -> Result<()> {
    let api: Api<Secret> = Api::namespaced(client.clone(), namespace);
    let secret = Secret {
        metadata: kube::api::ObjectMeta {
            name: Some(name.to_string()),
            ..Default::default()
        },
        type_: Some("kubernetes.io/tls".to_string()),
        data: Some(data),
        ..Default::default()
    };
    api.patch(
        name,
        &PatchParams::apply("tls-kind-e2e").force(),
        &Patch::Apply(&secret),
    )
    .await
    .with_context(|| format!("apply secret {namespace}/{name}"))?;
    Ok(())
}

fn is_ready(pod: &Pod) -> bool {
    pod.metadata.deletion_timestamp.is_none()
        && pod
            .status
            .as_ref()
            .and_then(|status| status.conditions.as_ref())
            .is_some_and(|conditions| {
                conditions
                    .iter()
                    .any(|c| c.type_ == "Ready" && c.status == "True")
            })
}

/// The single ready pod matching `selector`, once there is exactly one.
async fn ready_pod(client: &Client, namespace: &str, selector: &str) -> Result<Pod> {
    let pods: Api<Pod> = Api::namespaced(client.clone(), namespace);
    let all = pods.list(&ListParams::default().labels(selector)).await?;
    let live: Vec<Pod> = all
        .items
        .into_iter()
        .filter(|pod| pod.metadata.deletion_timestamp.is_none())
        .collect();
    match live.as_slice() {
        [pod] if is_ready(pod) => Ok(pod.clone()),
        [pod] => bail!("{namespace}/{} not ready yet", pod.name_any()),
        pods => bail!("{} live pods for {selector} in {namespace}", pods.len()),
    }
}

fn restart_count(pod: &Pod) -> i32 {
    pod.status
        .as_ref()
        .and_then(|status| status.container_statuses.as_ref())
        .map(|statuses| statuses.iter().map(|s| s.restart_count).sum())
        .unwrap_or_default()
}

fn operator_namespace() -> String {
    std::env::var(OPERATOR_NS_ENV).unwrap_or_else(|_| DEFAULT_OPERATOR_NS.to_string())
}

fn metrics_host(namespace: &str) -> String {
    format!("{METRICS_SERVICE}.{namespace}.svc")
}

/// The operator pod, pinned by UID so any restart or replacement fails the test.
struct OperatorPod {
    namespace: String,
    name: String,
    uid: String,
    restarts: i32,
}

impl OperatorPod {
    async fn current(client: &Client, namespace: &str) -> Result<Self> {
        let pod = ready_pod(client, namespace, OPERATOR_SELECTOR).await?;
        Ok(Self {
            namespace: namespace.to_string(),
            name: pod.name_any(),
            uid: pod.uid().context("operator pod uid")?,
            restarts: restart_count(&pod),
        })
    }

    async fn assert_not_restarted(&self, client: &Client, during: &str) -> Result<()> {
        let now = Self::current(client, &self.namespace).await?;
        if now.uid != self.uid || now.restarts != self.restarts {
            bail!(
                "operator restarted during {during}: {} (restarts {}) became {} (restarts {})",
                self.name,
                self.restarts,
                now.name,
                now.restarts
            );
        }
        Ok(())
    }

    fn metrics(&self) -> Target {
        Target {
            namespace: self.namespace.clone(),
            pod: self.name.clone(),
            port: METRICS_PORT,
            host: metrics_host(&self.namespace),
        }
    }
}

/// The operand's container env, from its Deployment.
async fn operand_env(client: &Client, backend: Backend) -> Result<BTreeMap<String, String>> {
    let deployments: Api<Deployment> = Api::namespaced(client.clone(), backend.namespace());
    let deployment = deployments.get(SERVER).await?;
    let env = deployment
        .spec
        .and_then(|spec| spec.template.spec)
        .and_then(|pod| pod.containers.into_iter().find(|c| c.name == "server"))
        .and_then(|container| container.env)
        .unwrap_or_default();
    Ok(env
        .into_iter()
        .filter_map(|var| var.value.map(|value| (var.name, value)))
        .collect())
}

/// Wait until the operand's Deployment renders `min_version`, `ciphers` and
/// `groups`, its rollout finished, and a single ready pod runs that template.
async fn operand_converged(
    client: &Client,
    backend: Backend,
    min_version: &str,
    ciphers: &str,
    groups: &str,
) -> Result<Target> {
    let what = format!("{backend:?} operand to render {min_version} [{ciphers}] [{groups}]");
    eventually(&what, CONVERGE, || async move {
        let env = operand_env(client, backend).await?;
        for (name, want) in [
            ("MX_TLS_MIN_VERSION", min_version),
            ("MX_TLS_CIPHER_SUITES", ciphers),
            ("MX_TLS_GROUPS", groups),
        ] {
            let got = env.get(name).map(String::as_str);
            if got != Some(want) {
                bail!("{name} is {got:?}");
            }
        }
        let deployments: Api<Deployment> = Api::namespaced(client.clone(), backend.namespace());
        let deployment = deployments.get(SERVER).await?;
        let image = deployment
            .spec
            .as_ref()
            .and_then(|spec| spec.template.spec.as_ref())
            .and_then(|pod| pod.containers.iter().find(|c| c.name == "server"))
            .and_then(|container| container.image.clone());
        if image.as_deref() != Some(backend.image()) {
            bail!("server image is {image:?}, want {}", backend.image());
        }
        let generation = deployment.metadata.generation.unwrap_or_default();
        let status = deployment.status.unwrap_or_default();
        if status.observed_generation.unwrap_or_default() < generation
            || status.updated_replicas != Some(1)
            || status.ready_replicas != Some(1)
            || status.replicas != Some(1)
        {
            bail!(
                "rollout in progress: {:?} updated, {:?} ready, {:?} total",
                status.updated_replicas,
                status.ready_replicas,
                status.replicas
            );
        }
        ready_pod(
            client,
            backend.namespace(),
            &format!("app.kubernetes.io/instance={SERVER}"),
        )
        .await?;
        Ok(())
    })
    .await?;
    let pod = ready_pod(
        client,
        backend.namespace(),
        &format!("app.kubernetes.io/instance={SERVER}"),
    )
    .await?;
    Ok(Target {
        namespace: backend.namespace().to_string(),
        pod: pod.name_any(),
        port: SERVER_PORT,
        host: backend.host(),
    })
}

/// A health RPC from the real ModelExpress client over TLS, verified against
/// the test CA, through a local listener bridged to the pod.
async fn grpc_health(client: &Client, pki: &Pki, target: &Target) -> Result<()> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let local = listener.local_addr()?;
    let bridge_client = client.clone();
    let (namespace, pod, port) = (target.namespace.clone(), target.pod.clone(), target.port);
    let bridge = tokio::spawn(async move {
        while let Ok((mut tcp, _)) = listener.accept().await {
            let Ok(mut upstream) = port_stream(&bridge_client, &namespace, &pod, port).await else {
                continue;
            };
            tokio::spawn(async move {
                let _ = tokio::io::copy_bidirectional(&mut tcp, &mut upstream).await;
            });
        }
    });
    let config = ClientConfig {
        connection: ConnectionConfig {
            tls_ca_file: Some(pki.ca_path()),
            ..ConnectionConfig::new(format!("https://127.0.0.1:{}", local.port()))
        },
        ..Default::default()
    };
    let outcome = async {
        let mut mx = MxClient::new(config)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        mx.health_check()
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok::<(), anyhow::Error>(())
    }
    .await;
    bridge.abort();
    outcome.with_context(|| format!("gRPC health over TLS to {}", target.describe()))
}

async fn create_operand(client: &Client, pki: &Pki, backend: Backend) -> Result<()> {
    let namespace = backend.namespace();
    recreate_namespace(client, namespace).await?;
    let host = backend.host();
    let cluster_local = format!("{host}.cluster.local");
    apply_tls_secret(
        client,
        namespace,
        SERVER_SECRET,
        pki.leaf(&[&host, &cluster_local, "127.0.0.1"])?,
    )
    .await?;
    let mut spec = json!({
        "replicas": 1,
        "port": SERVER_PORT,
        "metadataBackend": {"kubernetes": {}},
        "tls": {"secretName": SERVER_SECRET},
    });
    if let Some(image) = backend.cr_image() {
        spec["image"] = json!(image);
    }
    let server: DynamicObject = serde_json::from_value(json!({
        "apiVersion": "modelexpress.opendatahub.io/v1alpha1",
        "kind": "ModelExpressServer",
        "metadata": {"name": SERVER, "namespace": namespace},
        "spec": spec,
    }))?;
    model_express_servers(client, namespace)
        .create(&PostParams::default(), &server)
        .await
        .with_context(|| format!("create ModelExpressServer in {namespace}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Profiles, as the operator renders them
// ---------------------------------------------------------------------------

const PROFILE_GROUPS: &str = "X25519MLKEM768,X25519,secp256r1,secp384r1";
const INTERMEDIATE_CIPHERS: &str = "TLS_AES_128_GCM_SHA256,TLS_AES_256_GCM_SHA384,TLS_CHACHA20_POLY1305_SHA256,ECDHE-ECDSA-AES128-GCM-SHA256,ECDHE-RSA-AES128-GCM-SHA256,ECDHE-ECDSA-AES256-GCM-SHA384,ECDHE-RSA-AES256-GCM-SHA384,ECDHE-ECDSA-CHACHA20-POLY1305,ECDHE-RSA-CHACHA20-POLY1305";
const MODERN_CIPHERS: &str =
    "TLS_AES_128_GCM_SHA256,TLS_AES_256_GCM_SHA384,TLS_CHACHA20_POLY1305_SHA256";
const CUSTOM_CIPHERS: &str = "ECDHE-ECDSA-AES256-GCM-SHA384,TLS_AES_256_GCM_SHA384";
const CUSTOM_GROUPS: &str = "secp384r1";

fn modern_profile() -> serde_json::Value {
    json!({"type": "Modern", "modern": {}})
}

fn custom_profile() -> serde_json::Value {
    json!({
        "type": "Custom",
        "modern": null,
        "custom": {
            "minTLSVersion": "VersionTLS12",
            "ciphers": CUSTOM_CIPHERS.split(',').collect::<Vec<_>>(),
            "groups": [CUSTOM_GROUPS],
        },
    })
}

// ---------------------------------------------------------------------------
// Policy checks, shared by operands and the operator
// ---------------------------------------------------------------------------

async fn assert_intermediate(client: &Client, pki: &Pki, target: &Target) -> Result<()> {
    expect_accepted(client, pki, target, &Offer::tls12(), "TLS1.2").await?;
    let tls13 = expect_accepted(client, pki, target, &Offer::tls13(), "TLS1.3").await?;
    if tls13.version != ProtocolVersion::TLSv1_3 {
        bail!("expected TLS1.3, got {:?}", tls13.version);
    }
    expect_accepted(
        client,
        pki,
        target,
        &Offer::tls12().suite(CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256),
        "Intermediate TLS1.2 cipher ECDHE-ECDSA-AES128-GCM-SHA256",
    )
    .await?;
    expect_accepted(
        client,
        pki,
        target,
        &Offer::tls13().group(NamedGroup::X25519),
        "group X25519",
    )
    .await?;
    Ok(())
}

async fn assert_modern(client: &Client, pki: &Pki, target: &Target) -> Result<()> {
    expect_refused(client, pki, target, &Offer::tls12(), "TLS1.2 under Modern").await?;
    expect_accepted(client, pki, target, &Offer::tls13(), "TLS1.3 under Modern").await?;
    Ok(())
}

async fn assert_custom(client: &Client, pki: &Pki, target: &Target) -> Result<()> {
    let listed = expect_accepted(
        client,
        pki,
        target,
        &Offer::tls12().suite(CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384),
        "listed TLS1.2 cipher",
    )
    .await?;
    if listed.group != Some(NamedGroup::secp384r1) {
        bail!("expected secp384r1, got {:?}", listed.group);
    }
    expect_refused(
        client,
        pki,
        target,
        &Offer::tls12().suite(CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256),
        "unlisted TLS1.2 cipher",
    )
    .await?;
    expect_accepted(
        client,
        pki,
        target,
        &Offer::tls13().suite(CipherSuite::TLS13_AES_256_GCM_SHA384),
        "listed TLS1.3 suite",
    )
    .await?;
    expect_refused(
        client,
        pki,
        target,
        &Offer::tls13().suite(CipherSuite::TLS13_AES_128_GCM_SHA256),
        "unlisted TLS1.3 suite",
    )
    .await?;
    expect_refused(
        client,
        pki,
        target,
        &Offer::tls13().group(NamedGroup::X25519),
        "unlisted group X25519",
    )
    .await?;
    Ok(())
}

async fn operands_converge_to(
    client: &Client,
    pki: &Pki,
    min_version: &str,
    ciphers: &str,
    groups: &str,
) -> Result<Vec<Target>> {
    let mut targets = Vec::new();
    for backend in Backend::ALL {
        let target = operand_converged(client, backend, min_version, ciphers, groups).await?;
        grpc_health(client, pki, &target).await?;
        targets.push(target);
    }
    Ok(targets)
}

async fn served_certificate(client: &Client, pki: &Pki, target: &Target) -> Result<Vec<u8>> {
    let stream = connect(client, pki, target, &Offer::tls13()).await?;
    let cert = stream
        .get_ref()
        .1
        .peer_certificates()
        .and_then(|chain| chain.first())
        .context("peer certificate")?;
    Ok(cert.as_ref().to_vec())
}

// ---------------------------------------------------------------------------
// The scenario
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "needs the kind cluster from test_tls_kind.sh"]
async fn cluster_tls_profile_end_to_end() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("tls_kind=debug")
        .with_test_writer()
        .try_init();
    let client = kube_client().await?;
    let pki = Pki::new()?;
    let operator_ns = operator_namespace();
    let metrics_host = metrics_host(&operator_ns);
    println!("operator namespace: {operator_ns}");

    println!("setup: apiservers/cluster with a Modern profile and no tlsAdherence");
    reset_apiserver(&client, json!({"tlsSecurityProfile": modern_profile()})).await?;

    println!("setup: operator metrics certificate, fresh operator pod");
    apply_tls_secret(
        &client,
        &operator_ns,
        METRICS_SECRET,
        pki.leaf(&[metrics_host.as_str(), "127.0.0.1"])?,
    )
    .await?;
    let pods: Api<Pod> = Api::namespaced(client.clone(), &operator_ns);
    pods.delete_collection(
        &DeleteParams::default(),
        &ListParams::default().labels(OPERATOR_SELECTOR),
    )
    .await?;
    eventually("a fresh operator pod", CONVERGE, || async {
        OperatorPod::current(&client, &operator_ns)
            .await
            .map(|_| ())
    })
    .await?;
    let operator = OperatorPod::current(&client, &operator_ns).await?;
    let metrics = operator.metrics();

    println!("setup: the operator applies its own ServiceMonitor");
    let monitors = Api::<DynamicObject>::namespaced_with(
        client.clone(),
        &operator_ns,
        &ApiResource::from_gvk(&GroupVersionKind::gvk(
            "monitoring.coreos.com",
            "v1",
            "ServiceMonitor",
        )),
    );
    eventually("the metrics ServiceMonitor", CONVERGE, || async {
        let monitor = monitors
            .get_opt("modelexpress-operator")
            .await?
            .context("not applied yet")?;
        let server_name = monitor.data["spec"]["endpoints"][0]["tlsConfig"]["serverName"].as_str();
        if server_name != Some(metrics_host.as_str()) {
            bail!("serverName is {server_name:?}, want {metrics_host}");
        }
        Ok(())
    })
    .await?;

    println!("setup: one ModelExpressServer per server TLS backend");
    for backend in Backend::ALL {
        create_operand(&client, &pki, backend).await?;
    }

    println!("phase 1: tlsAdherence unset, the Modern profile is not honored");
    for target in operands_converge_to(
        &client,
        &pki,
        "VersionTLS12",
        INTERMEDIATE_CIPHERS,
        PROFILE_GROUPS,
    )
    .await?
    {
        assert_intermediate(&client, &pki, &target).await?;
    }
    eventually("operator metrics on Intermediate", CONVERGE, || {
        assert_intermediate(&client, &pki, &metrics)
    })
    .await?;
    let mut open = connect(&client, &pki, &metrics, &Offer::tls12().http1()).await?;
    let status = http_status(&mut open, &metrics.host, "/metrics").await?;
    if !status.starts_with("HTTP/1.1 401") {
        bail!("unauthenticated scrape must get 401, got {status}");
    }

    println!("phase 2: StrictAllComponents, the Modern profile applies");
    patch_apiserver(&client, json!({"tlsAdherence": "StrictAllComponents"})).await?;
    for target in operands_converge_to(
        &client,
        &pki,
        "VersionTLS13",
        MODERN_CIPHERS,
        PROFILE_GROUPS,
    )
    .await?
    {
        assert_modern(&client, &pki, &target).await?;
    }
    eventually("operator metrics on Modern", CONVERGE, || {
        assert_modern(&client, &pki, &metrics)
    })
    .await?;
    operator
        .assert_not_restarted(&client, "the Modern swap")
        .await?;
    let status = http_status(&mut open, &metrics.host, "/metrics")
        .await
        .context("a TLS1.2 connection opened before the swap keeps serving")?;
    if !status.starts_with("HTTP/1.1 401") {
        bail!("open connection after the swap got {status}");
    }
    drop(open);

    println!("phase 3: StrictAllComponents, a Custom profile narrows ciphers and groups");
    patch_apiserver(&client, json!({"tlsSecurityProfile": custom_profile()})).await?;
    for target in
        operands_converge_to(&client, &pki, "VersionTLS12", CUSTOM_CIPHERS, CUSTOM_GROUPS).await?
    {
        assert_custom(&client, &pki, &target).await?;
    }
    eventually("operator metrics on the Custom profile", CONVERGE, || {
        assert_custom(&client, &pki, &metrics)
    })
    .await?;
    operator
        .assert_not_restarted(&client, "the Custom swap")
        .await?;

    println!("phase 4: LegacyAdheringComponentsOnly, back to Intermediate");
    patch_apiserver(
        &client,
        json!({"tlsAdherence": "LegacyAdheringComponentsOnly"}),
    )
    .await?;
    for target in operands_converge_to(
        &client,
        &pki,
        "VersionTLS12",
        INTERMEDIATE_CIPHERS,
        PROFILE_GROUPS,
    )
    .await?
    {
        assert_intermediate(&client, &pki, &target).await?;
    }
    eventually("operator metrics back on Intermediate", CONVERGE, || {
        assert_intermediate(&client, &pki, &metrics)
    })
    .await?;
    operator
        .assert_not_restarted(&client, "the Legacy swap")
        .await?;

    println!("phase 5: the metrics certificate rotates in place");
    let before = served_certificate(&client, &pki, &metrics).await?;
    apply_tls_secret(
        &client,
        &operator_ns,
        METRICS_SECRET,
        pki.leaf(&[metrics_host.as_str(), "127.0.0.1"])?,
    )
    .await?;
    eventually(
        "the rotated metrics certificate to be served",
        ROTATE,
        || async {
            let now = served_certificate(&client, &pki, &metrics).await?;
            if now == before {
                bail!("still serving the previous certificate");
            }
            Ok(())
        },
    )
    .await?;
    operator
        .assert_not_restarted(&client, "certificate rotation")
        .await?;

    println!("cleanup");
    for backend in Backend::ALL {
        let api: Api<Namespace> = Api::all(client.clone());
        api.delete(backend.namespace(), &DeleteParams::default())
            .await?;
    }
    Ok(())
}

/// The operator reads the cluster TLS profile with no RBAC for it here, and
/// the API group does not exist. The apiserver authorizes before it looks the
/// resource up, so that read is a 403, not a 404, and it must not stop a
/// ModelExpressServer from reconciling.
#[tokio::test]
#[ignore = "needs the kind cluster from test_tls_kind.sh --overlay odh-xks"]
async fn platform_install_without_openshift() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("tls_kind=debug")
        .with_test_writer()
        .try_init();
    let client = kube_client().await?;
    let pki = Pki::new()?;
    let operator_ns = operator_namespace();
    println!("operator namespace: {operator_ns}");

    let apiserver = GroupVersionKind::gvk("config.openshift.io", "v1", "APIServer");
    if kube::discovery::oneshot::pinned_kind(&client, &apiserver)
        .await
        .is_ok()
    {
        bail!("config.openshift.io is served; this scenario needs a cluster without it");
    }

    println!("setup: the operator is up with plaintext metrics");
    eventually("the operator pod", CONVERGE, || async {
        OperatorPod::current(&client, &operator_ns)
            .await
            .map(|_| ())
    })
    .await?;
    let operator = OperatorPod::current(&client, &operator_ns).await?;

    println!("setup: one ModelExpressServer per server TLS backend, TLS settings unpinned");
    for backend in Backend::ALL {
        create_operand(&client, &pki, backend).await?;
    }

    println!("operands come up on the Intermediate profile");
    for target in operands_converge_to(
        &client,
        &pki,
        "VersionTLS12",
        INTERMEDIATE_CIPHERS,
        PROFILE_GROUPS,
    )
    .await?
    {
        assert_intermediate(&client, &pki, &target).await?;
    }
    operator
        .assert_not_restarted(&client, "reconciling without an APIServer")
        .await?;

    println!("cleanup");
    for backend in Backend::ALL {
        let api: Api<Namespace> = Api::all(client.clone());
        api.delete(backend.namespace(), &DeleteParams::default())
            .await?;
    }
    Ok(())
}
