// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve a model name to the on-disk snapshot the node holds, so the
//! [`crate::cache::transfer::stager::CacheServer`] can serve it.

use std::path::{Path, PathBuf};

use crate::cache::transfer::stager::ModelLocator;

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

/// Every HuggingFace model present under `cache_root`, recovered from the
/// `models--<org>--<name>` repo directory names (the inverse of the
/// `/`-to-`--` encoding the download path applies). Backs auto-expand capture:
/// the daemon advertises models that appeared locally without going through its
/// own fetch path. Unreadable cache root yields an empty list, not an error.
pub fn list_cached_models(cache_root: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(cache_root) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|entry| entry.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|entry| {
            entry
                .file_name()
                .to_str()
                .and_then(|name| name.strip_prefix("models--"))
                .map(|rest| rest.replace("--", "/"))
        })
        .collect()
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

    #[test]
    fn lists_cached_models_decoding_repo_names() {
        let cache = tempfile::tempdir().expect("tempdir");
        for repo in ["models--google-t5--t5-small", "models--Qwen--Qwen3-0.6B"] {
            std::fs::create_dir_all(cache.path().join(repo).join("snapshots")).expect("mkdir");
        }
        // A non-model directory and a file are ignored.
        std::fs::create_dir_all(cache.path().join("version.txt.lock")).expect("mkdir");
        std::fs::write(cache.path().join("not-a-repo"), b"x").expect("write");

        let mut models = list_cached_models(cache.path());
        models.sort();
        // `--` decodes back to `/`; single dashes in org/name are preserved.
        assert_eq!(models, vec!["Qwen/Qwen3-0.6B", "google-t5/t5-small"]);

        // Empty / missing cache root is not an error.
        assert!(list_cached_models(&cache.path().join("nope")).is_empty());
    }
}
