// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! On-disk layout for the cache transfer: turning a model snapshot directory
//! into a shard [`Manifest`], opening files for the transfer legs, SHA-256
//! verification, and atomically publishing a freshly pulled model.
//!
//! The whole snapshot is served (weights, config, tokenizer, ...) so a pulled
//! model is immediately usable. Source files are never mutated: the transfer is
//! byte-exact (no padding), unlike the throwaway-holder spike.
//!
//! A model is published in two steps so a crashed pull never leaves a
//! half-written file masquerading as real: each file lands under a temp name
//! (`<file>.mxtmp`) and is renamed into place only after its SHA matches, then
//! the model directory gets a [`COMPLETE_SENTINEL`] once every file is in. A
//! model directory without the sentinel is treated as absent and re-pulled.

use std::fs::File;
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use sha2::{Digest, Sha256};

use super::{Manifest, Shard};

/// Marker file dropped in a model directory once all files have landed and
/// verified. Its presence means the model is complete and safe to serve.
pub const COMPLETE_SENTINEL: &str = ".mxcomplete";

/// Suffix for a not-yet-verified file landing on the puller.
const TEMP_SUFFIX: &str = ".mxtmp";

// `O_DIRECT` is Linux-only. Gate it so this module still compiles for
// type-checking on non-Linux hosts, where it is a no-op. Phase 3 runs buffered
// (`direct = false`); Phase 4 re-enables `O_DIRECT` with partial-tail handling.
#[cfg(target_os = "linux")]
const DIRECT_FLAG: i32 = libc::O_DIRECT;
#[cfg(not(target_os = "linux"))]
const DIRECT_FLAG: i32 = 0;

/// Open a file for a transfer leg. With `direct`, `O_DIRECT` bypasses the page
/// cache. Tests and Phase 3 pass `direct = false` (tmpfs rejects `O_DIRECT`, and
/// byte-exact transfers need partial-tail handling before `O_DIRECT` is safe).
pub fn open_direct(path: &Path, write: bool, direct: bool) -> std::io::Result<File> {
    let mut opts = File::options();
    opts.read(true);
    if direct {
        opts.custom_flags(DIRECT_FLAG);
    }
    if write {
        opts.write(true).create(true);
    }
    opts.open(path)
}

/// SHA-256 of the first `n` bytes of `path` (pass the file's size to hash it all).
pub fn sha256_prefix(path: &Path, n: u64) -> std::io::Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut left = n;
    while left > 0 {
        let want = usize::try_from(left.min(buf.len() as u64)).unwrap_or(0);
        let r = f.read(&mut buf[..want])?;
        if r == 0 {
            break;
        }
        hasher.update(&buf[..r]);
        left = left.saturating_sub(r as u64);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Scan every file under `dir` (recursively, following the symlinks an HF
/// snapshot uses) into a manifest for `revision`. Records each file's path
/// relative to `dir`, its exact size, and its SHA-256. Our own scratch and
/// sentinel files are skipped; nothing is mutated.
pub fn scan_manifest(dir: &Path, revision: &str) -> anyhow::Result<Manifest> {
    let mut shards = Vec::new();
    collect_shards(dir, dir, &mut shards)?;
    shards.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    if shards.is_empty() {
        bail!("no files to serve in {}", dir.display());
    }
    Ok(Manifest {
        revision: revision.to_string(),
        shards,
    })
}

fn collect_shards(root: &Path, dir: &Path, out: &mut Vec<Shard>) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        // `metadata` follows symlinks (HF snapshot entries point into blobs/).
        let meta = std::fs::metadata(&path)?;
        if meta.is_dir() {
            collect_shards(root, &path, out)?;
            continue;
        }
        if !meta.is_file() {
            continue;
        }
        let rel = path.strip_prefix(root).unwrap_or(&path);
        let rel_path = rel
            .to_str()
            .context("non-utf8 path in snapshot")?
            .to_string();
        if is_mx_internal(&rel_path) {
            continue;
        }
        let true_size = meta.len();
        let sha256 = sha256_prefix(&path, true_size)?;
        out.push(Shard {
            rel_path,
            true_size,
            sha256,
        });
    }
    Ok(())
}

/// Whether a relative path is one of our own scratch/sentinel files (not part of
/// the model, must not be served).
fn is_mx_internal(rel_path: &str) -> bool {
    rel_path.ends_with(TEMP_SUFFIX) || rel_path == COMPLETE_SENTINEL
}

/// Temp path a file is written to before it is verified and renamed into place:
/// a sibling of the final path so the rename is atomic on the same filesystem.
pub fn temp_path(final_path: &Path) -> PathBuf {
    let mut s = final_path.as_os_str().to_os_string();
    s.push(TEMP_SUFFIX);
    PathBuf::from(s)
}

/// Create the parent directory of `path` if needed (snapshot files may sit in
/// subdirectories).
pub fn ensure_parent(path: &Path) -> std::io::Result<()> {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => std::fs::create_dir_all(parent),
        _ => Ok(()),
    }
}

/// Atomically publish a verified file: rename its temp file to the final name.
pub fn finalize_shard(tmp: &Path, final_path: &Path) -> std::io::Result<()> {
    std::fs::rename(tmp, final_path)
}

/// Mark a model directory complete once every file has landed and verified.
pub fn mark_complete(dir: &Path) -> std::io::Result<()> {
    File::create(dir.join(COMPLETE_SENTINEL)).map(drop)
}

/// Whether a model directory carries its completion sentinel.
pub fn is_complete(dir: &Path) -> bool {
    dir.join(COMPLETE_SENTINEL).exists()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(path: &Path, bytes: &[u8]) {
        ensure_parent(path).expect("mkdir");
        let mut f = File::create(path).expect("create");
        f.write_all(bytes).expect("write");
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(bytes);
        format!("{:x}", h.finalize())
    }

    #[test]
    fn scan_collects_all_files_with_rel_paths_unmodified() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_file(&dir.path().join("model.safetensors"), b"weights!");
        write_file(&dir.path().join("config.json"), b"{}");
        write_file(&dir.path().join("nested/tokenizer.json"), b"tok");

        let manifest = scan_manifest(dir.path(), "rev-1").expect("scan");
        assert_eq!(manifest.revision, "rev-1");
        let paths: Vec<&str> = manifest
            .shards
            .iter()
            .map(|s| s.rel_path.as_str())
            .collect();
        // Sorted, all files, config included, subdir path preserved.
        assert_eq!(
            paths,
            vec!["config.json", "model.safetensors", "nested/tokenizer.json"]
        );
        // Sizes are exact and content is hashed (not padded).
        let cfg = manifest
            .shards
            .iter()
            .find(|s| s.rel_path == "config.json")
            .unwrap();
        assert_eq!(cfg.true_size, 2);
        assert_eq!(cfg.sha256, sha256_hex(b"{}"));
        // Source file is untouched (no padding to a CHUNK multiple).
        assert_eq!(
            std::fs::metadata(dir.path().join("config.json"))
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn scan_skips_internal_files_and_errors_when_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_file(&dir.path().join("model.safetensors.mxtmp"), b"partial");
        mark_complete(dir.path()).expect("sentinel");
        // Only internal files present -> nothing to serve.
        assert!(scan_manifest(dir.path(), "r").is_err());
    }

    #[test]
    fn sentinel_round_trips() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(!is_complete(dir.path()));
        mark_complete(dir.path()).expect("mark");
        assert!(is_complete(dir.path()));
    }

    #[test]
    fn temp_is_sibling_and_finalize_renames_through_subdirs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let final_path = dir.path().join("sub/model.safetensors");
        let tmp = temp_path(&final_path);
        assert_eq!(tmp, dir.path().join("sub/model.safetensors.mxtmp"));
        write_file(&tmp, b"bytes");
        finalize_shard(&tmp, &final_path).expect("finalize");
        assert!(!tmp.exists());
        assert!(final_path.exists());
    }
}
