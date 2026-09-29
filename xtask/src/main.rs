// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod objects;
mod odh;
mod openshift;

use clap::{Parser, Subcommand};
use kube::CustomResourceExt;
use modelexpress_operator::crd::generate_crd;
use modelexpress_server::p2p::k8s_types::ModelMetadata;
use modelexpress_server::registry::k8s_types::ModelCacheEntry;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
enum XtaskError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("serializing {what}: {source}")]
    Yaml {
        what: &'static str,
        source: serde_norway::Error,
    },
    #[error("{path} is stale, run `cargo xtask {cmd}`")]
    Stale { path: PathBuf, cmd: &'static str },
    #[error("{path}: not a valid .yaml file name")]
    BadFileName { path: PathBuf },
}

type Result<T> = std::result::Result<T, XtaskError>;

fn io_err(path: &Path) -> impl FnOnce(std::io::Error) -> XtaskError {
    let path = path.to_path_buf();
    move |source| XtaskError::Io { path, source }
}

#[derive(Parser)]
#[command(about = "repo plumbing: cargo xtask <cmd>")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate CRD manifests into config/manifests/crd/
    Crdgen {
        /// Fail if the on-disk manifests are stale instead of writing (for CI)
        #[arg(long)]
        check: bool,
    },
    /// Generate the operator's deploy manifests into config/manifests/rbac/,
    /// config/manifests/manager/, config/manifests/components/,
    /// config/manifests/openshift/, config/manifests/overlays/ and
    /// config/manifests/base/params.env
    Manifests {
        /// Fail if the on-disk manifests are stale instead of writing (for CI)
        #[arg(long)]
        check: bool,
        /// Controller image to render, e.g. a release digest
        #[arg(long, default_value = objects::DEFAULT_IMAGE)]
        image: String,
    },
}

fn repo_root() -> PathBuf {
    // xtask/ lives directly under the repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Root of the packaged kustomize tree; generating elsewhere produces files
/// nothing deploys.
fn manifests_root() -> PathBuf {
    repo_root().join("config/manifests")
}

fn write(path: &Path, contents: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(io_err(dir))?;
    }
    std::fs::write(path, contents).map_err(io_err(path))?;
    println!("wrote {}", path.display());
    Ok(())
}

/// copyright-check.ps1 scopes `.yaml`, so regeneration has to re-emit this.
const SPDX_HEADER: &str = "\
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

";

fn to_yaml<T: serde::Serialize>(value: &T) -> Result<String> {
    let body = serde_norway::to_string(value).map_err(|source| XtaskError::Yaml {
        what: std::any::type_name::<T>(),
        source,
    })?;
    Ok(format!("{SPDX_HEADER}{body}"))
}

/// Write each (path, contents) pair, or with `check` fail on any drift.
fn write_or_check(files: Vec<(PathBuf, String)>, check: bool, cmd: &'static str) -> Result<()> {
    for (path, rendered) in files {
        if check {
            let on_disk = std::fs::read_to_string(&path).map_err(io_err(&path))?;
            if on_disk != rendered {
                return Err(XtaskError::Stale { path, cmd });
            }
            println!("{} is up to date", path.display());
        } else {
            write(&path, &rendered)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Crdgen { check } => write_or_check(all_crds()?, check, "crdgen"),
        Cmd::Manifests { check, image } => write_or_check(manifests(&image)?, check, "manifests"),
    }
}

/// All CRDs the operator installs: ours, plus the upstream backend CRDs
/// generated from the pinned modelexpress-types crate (guaranteed to match
/// the linked server version, unlike vendored YAML).
fn all_crds() -> Result<Vec<(PathBuf, String)>> {
    let dir = manifests_root().join("crd");
    let mut out = Vec::new();
    let ours = generate_crd();
    for (crd, sub) in [
        (ours, ""),
        (ModelMetadata::crd(), "upstream"),
        (ModelCacheEntry::crd(), "upstream"),
    ] {
        let name = crd
            .metadata
            .name
            .clone()
            .ok_or_else(|| XtaskError::BadFileName { path: dir.clone() })?;
        let path = dir.join(sub).join(format!("{name}.yaml"));
        out.push((path, to_yaml(&crd)?));
    }
    Ok(out)
}

fn yaml_files(
    dir: &Path,
    files: Vec<(&'static str, serde_json::Value)>,
) -> Result<Vec<(PathBuf, String)>> {
    files
        .into_iter()
        .map(|(file, value)| Ok((dir.join(file), to_yaml(&value)?)))
        .collect()
}

fn manifests(image: &str) -> Result<Vec<(PathBuf, String)>> {
    let config = manifests_root();
    let rbac = config.join("rbac");
    Ok(vec![
        (
            rbac.join("serviceaccount.yaml"),
            to_yaml(&objects::service_account())?,
        ),
        (
            rbac.join("clusterrole.yaml"),
            to_yaml(&objects::cluster_role())?,
        ),
        (
            rbac.join("clusterrolebinding.yaml"),
            to_yaml(&objects::cluster_role_binding())?,
        ),
        (
            config.join("manager/deployment.yaml"),
            to_yaml(&objects::deployment(image))?,
        ),
        (
            config.join("manager/service.yaml"),
            to_yaml(&objects::metrics_service())?,
        ),
        (config.join("base/params.env"), objects::params_env(image)),
    ]
    .into_iter()
    .chain(yaml_files(
        &config.join(openshift::COMPONENT_DIR),
        openshift::component(),
    )?)
    .chain(yaml_files(&config.join("openshift"), openshift::overlay())?)
    .chain(yaml_files(
        &config.join(openshift::RELATED_IMAGE_COMPONENT_DIR),
        openshift::related_image_component(),
    )?)
    .chain(yaml_files(&config.join(odh::OVERLAY_DIR), odh::overlay())?)
    .chain(yaml_files(
        &config.join(odh::XKS_OVERLAY_DIR),
        odh::xks_overlay(),
    )?)
    .collect())
}
