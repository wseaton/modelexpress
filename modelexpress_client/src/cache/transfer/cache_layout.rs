// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! On-disk layout for the cache transfer: turning a model snapshot directory
//! into a shard [`Manifest`], opening files for the transfer legs, BLAKE3
//! verification, and atomically publishing a freshly pulled model.
//!
//! The whole snapshot is served (weights, config, tokenizer, ...) so a pulled
//! model is immediately usable. Source files are never mutated: the transfer is
//! byte-exact (no padding), unlike the throwaway-holder spike.
//!
//! A model is published in two steps so a crashed pull never leaves a
//! half-written file masquerading as real: each file lands under a temp name
//! (`<file>.mxtmp`) and is renamed into place only after its hash matches, then
//! the model directory gets a [`COMPLETE_SENTINEL`] once every file is in. A
//! model directory without the sentinel is treated as absent and re-pulled.

use std::fs::File;
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};

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

/// Round `n` up to the next multiple of `align`, which must be a power of two
/// (the logical block size for `O_DIRECT`). On add overflow `n` is returned
/// unchanged. The puller writes this rounded length so an `O_DIRECT` write's
/// final partial chunk is block-aligned, then truncates the file back to the
/// exact size.
pub fn align_up(n: u64, align: u64) -> u64 {
    if align == 0 {
        return n;
    }
    let mask = align.wrapping_sub(1);
    match n.checked_add(mask) {
        Some(sum) => sum & !mask,
        None => n,
    }
}

/// BLAKE3 of an in-memory byte slice, multithreaded across the rayon pool. The
/// puller hashes the freshly-received staging-buffer bytes directly instead of
/// reading the file back off disk (which would compete with the write-bound NVMe
/// path); BLAKE3 keeps verification from becoming the bottleneck. This verifies
/// the data that arrived over the wire (the FS/NVMe is trusted for the write).
pub fn hash_mem(bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update_rayon(bytes);
    hasher.finalize().to_hex().to_string()
}

/// BLAKE3 of the first `n` bytes of `path` (pass the file's size to hash it all).
/// Streamed, so a multi-GB shard is not read wholly into memory; the holder runs
/// this once per model when scanning the manifest. Produces the same hash as
/// [`hash_mem`] over the same bytes (BLAKE3 is chunking-independent).
pub fn hash_prefix(path: &Path, n: u64) -> std::io::Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
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
    Ok(hasher.finalize().to_hex().to_string())
}

/// Scan every file under `dir` (recursively, following the symlinks an HF
/// snapshot uses) into a manifest for `revision`. Records each file's path
/// relative to `dir`, its exact size, and its BLAKE3 hash. Our own scratch and
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
        let hash = hash_prefix(&path, true_size)?;
        out.push(Shard {
            rel_path,
            true_size,
            hash,
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

/// Whether `repo_dir` has an in-progress HuggingFace download: `huggingface_hub`
/// writes a blob to `blobs/<etag>.incomplete` and only renames it into place when
/// the download finishes, so any `*.incomplete` means a file is still landing and
/// the snapshot must not be advertised yet. A missing `blobs/` means none.
pub fn has_incomplete_blobs(repo_dir: &Path) -> bool {
    match std::fs::read_dir(repo_dir.join("blobs")) {
        Ok(entries) => entries.flatten().any(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|name| name.ends_with(".incomplete"))
        }),
        Err(_) => false,
    }
}

/// A cheap (file count, total bytes) fingerprint of a snapshot, following the
/// symlinks an HF snapshot uses and skipping our own scratch/sentinel files.
/// Used to detect that an externally-downloaded snapshot has stopped changing
/// between reconcile passes before advertising it; far cheaper than re-hashing
/// (that happens once, at serve time). A dangling symlink (a blob mid-rename) is
/// skipped, so an in-flight download reads as a different, smaller fingerprint.
pub fn snapshot_fingerprint(snapshot_dir: &Path) -> (u64, u64) {
    let mut count = 0u64;
    let mut bytes = 0u64;
    fingerprint_walk(snapshot_dir, snapshot_dir, &mut count, &mut bytes);
    (count, bytes)
}

fn fingerprint_walk(root: &Path, dir: &Path, count: &mut u64, bytes: &mut u64) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // Follows symlinks (HF snapshot entries point into blobs/); a dangling
        // link errors and is skipped.
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        if meta.is_dir() {
            fingerprint_walk(root, &path, count, bytes);
            continue;
        }
        if !meta.is_file() {
            continue;
        }
        let rel = path.strip_prefix(root).unwrap_or(&path);
        if rel.to_str().is_some_and(is_mx_internal) {
            continue;
        }
        *count = count.saturating_add(1);
        *bytes = bytes.saturating_add(meta.len());
    }
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

    fn blake3_hex(bytes: &[u8]) -> String {
        blake3::hash(bytes).to_hex().to_string()
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
        assert_eq!(cfg.hash, blake3_hex(b"{}"));
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
    fn align_up_rounds_to_block_multiple() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096, "already aligned is unchanged");
        assert_eq!(align_up(4097, 4096), 8192);
        // A 16 MiB chunk plus a 100-byte tail rounds the tail up by one block.
        assert_eq!(align_up((16 << 20) + 100, 4096), (16 << 20) + 4096);
        // Degenerate alignment is a no-op.
        assert_eq!(align_up(12345, 0), 12345);
    }

    #[test]
    fn incomplete_blobs_gate_settling() {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir.path();
        // No blobs/ dir at all -> nothing in progress.
        assert!(!has_incomplete_blobs(repo));
        let blobs = repo.join("blobs");
        std::fs::create_dir_all(&blobs).expect("mkdir");
        write_file(&blobs.join("abc123"), b"finished");
        assert!(
            !has_incomplete_blobs(repo),
            "a finished blob is not in progress"
        );
        write_file(&blobs.join("def456.incomplete"), b"partial");
        assert!(
            has_incomplete_blobs(repo),
            "an .incomplete blob is detected"
        );
    }

    #[test]
    fn fingerprint_counts_files_and_bytes_skipping_internal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snap = dir.path();
        write_file(&snap.join("config.json"), b"{}"); // 2 bytes
        write_file(&snap.join("nested/tok.json"), b"tok"); // 3 bytes
        assert_eq!(snapshot_fingerprint(snap), (2, 5));
        // Our scratch and sentinel files are not part of the model.
        write_file(&snap.join("model.safetensors.mxtmp"), b"partial");
        mark_complete(snap).expect("sentinel");
        assert_eq!(
            snapshot_fingerprint(snap),
            (2, 5),
            "internal files do not change the fingerprint"
        );
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
