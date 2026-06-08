// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The local use signal that drives the demand TTL. Under `--auto-expand` a node
//! keeps a model alive fleet-wide (advertises it) only while the model is pinned
//! or *recently used*; this is the "recently used" half. A model nobody has used
//! within the TTL decays out of the registry and is then garbage-collected.
//!
//! Hybrid by design (the chosen Phase B path): [`AtimeUsage`] reads filesystem
//! times today; a future `ReportedUsage` fed by an mx-client/vLLM hook can be
//! `max`-combined in without touching callers, since they depend on the trait.

use std::path::Path;
use std::time::SystemTime;

/// How recently a model's snapshot was used on this node. `None` means unknown,
/// which callers treat as "not recently used".
pub trait UsageSignal: Send + Sync {
    fn last_used(&self, snapshot_dir: &Path) -> Option<SystemTime>;
}

/// Filesystem-time use signal: the most recent access-or-modify time across a
/// model's snapshot files. `mtime` is folded in (and survives restarts) so a
/// freshly-downloaded model reads as recently used even where `relatime`
/// suppresses `atime` bumps. That makes the signal conservative: it can miss a
/// read, but it never invents staleness earlier than the download, so the TTL
/// errs toward keeping a model rather than evicting one in use.
pub struct AtimeUsage;

impl UsageSignal for AtimeUsage {
    fn last_used(&self, snapshot_dir: &Path) -> Option<SystemTime> {
        let mut latest: Option<SystemTime> = None;
        latest_touch(snapshot_dir, &mut latest);
        latest
    }
}

fn latest_touch(dir: &Path, latest: &mut Option<SystemTime>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // Follows symlinks (HF snapshot entries point into blobs/).
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        if meta.is_dir() {
            latest_touch(&path, latest);
            continue;
        }
        for touch in [meta.accessed().ok(), meta.modified().ok()]
            .into_iter()
            .flatten()
        {
            if latest.is_none_or(|current| touch > current) {
                *latest = Some(touch);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn last_used_is_the_newest_file_time() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.json"), b"{}").expect("write");
        std::fs::create_dir_all(dir.path().join("nested")).expect("mkdir");
        std::fs::write(dir.path().join("nested/model.bin"), b"weights").expect("write");

        let last = AtimeUsage.last_used(dir.path()).expect("some time");
        // The newest touch is within a moment of now (just wrote the files).
        let age = SystemTime::now()
            .duration_since(last)
            .unwrap_or(Duration::ZERO);
        assert!(
            age < Duration::from_secs(60),
            "freshly written reads as recent"
        );
    }

    #[test]
    fn missing_dir_is_unknown() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(AtimeUsage.last_used(&dir.path().join("nope")).is_none());
    }
}
