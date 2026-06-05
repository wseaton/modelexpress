// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve a model name to the on-disk snapshot the node holds, so the
//! [`crate::cached::transfer::stager::CacheServer`] can serve it.

use std::path::{Path, PathBuf};

use crate::cached::transfer::stager::ModelLocator;

/// Locates HuggingFace models in a ModelExpress cache root, matching the layout
/// the download path produces (`models--<org>--<name>/snapshots/<rev>/`).
pub struct HfLocator {
    cache_root: PathBuf,
}

impl HfLocator {
    pub fn new(cache_root: PathBuf) -> Self {
        Self { cache_root }
    }
}

impl ModelLocator for HfLocator {
    fn locate(&self, model: &str) -> Option<(PathBuf, String)> {
        locate_hf(&self.cache_root, model)
    }
}

/// Find a HuggingFace model's snapshot directory and revision under `cache_root`,
/// or `None` if it is not present. Returns the first snapshot found (a node
/// holds one revision per model in the v1 cache daemon).
pub fn locate_hf(cache_root: &Path, model: &str) -> Option<(PathBuf, String)> {
    let repo = cache_root.join(format!("models--{}", model.replace('/', "--")));
    let snapshots = repo.join("snapshots");
    for entry in std::fs::read_dir(&snapshots).ok()? {
        let entry = entry.ok()?;
        if entry.file_type().ok()?.is_dir() {
            let revision = entry.file_name().to_str()?.to_string();
            return Some((entry.path(), revision));
        }
    }
    None
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn locates_hf_snapshot_and_revision() {
        let cache = tempfile::tempdir().expect("tempdir");
        let snap = cache
            .path()
            .join("models--google-t5--t5-small")
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snap).expect("mkdir");
        std::fs::write(snap.join("config.json"), b"{}").expect("write");

        let (dir, rev) = locate_hf(cache.path(), "google-t5/t5-small").expect("present");
        assert_eq!(dir, snap);
        assert_eq!(rev, "abc123");
        // Absent model -> None.
        assert!(locate_hf(cache.path(), "nobody/here").is_none());
    }
}
