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

/// File extensions that mark a real model weights file. A snapshot with only
/// config/tokenizer files is never a complete model.
const WEIGHT_EXTS: &[&str] = &["safetensors", "bin", "gguf", "pt", "pth"];

/// Whether a download into `repo_dir` is in progress right now: a `*.incomplete`
/// blob (the standard `hf_hub_download` staging file), or a currently-held
/// HuggingFace lock under `<cache>/.locks/<repo>/`. Either means a file is being
/// written, so the snapshot must not be advertised.
pub fn download_in_progress(repo_dir: &Path) -> bool {
    has_incomplete_blobs(repo_dir) || has_held_lock(repo_dir)
}

fn has_held_lock(repo_dir: &Path) -> bool {
    let (Some(parent), Some(name)) = (repo_dir.parent(), repo_dir.file_name()) else {
        return false;
    };
    match std::fs::read_dir(parent.join(".locks").join(name)) {
        Ok(entries) => entries.flatten().any(|e| {
            let path = e.path();
            path.extension().and_then(|x| x.to_str()) == Some("lock") && lock_is_held(&path)
        }),
        Err(_) => false,
    }
}

/// Non-blocking test of whether another process holds `path`'s `flock`. The
/// `filelock` library HuggingFace uses leaves the lock file on disk after release,
/// so existence is meaningless; only a held OS lock means a live download. We grab
/// the lock non-blocking and immediately release it: success means nobody held it.
fn lock_is_held(path: &Path) -> bool {
    use std::os::unix::io::AsRawFd;
    let Ok(file) = File::open(path) else {
        return false;
    };
    // SAFETY: `file` owns a valid fd for the duration of these calls; we only
    // test the lock and release it, never hold it past this function.
    let fd = file.as_raw_fd();
    if unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        unsafe { libc::flock(fd, libc::LOCK_UN) };
        return false;
    }
    // Only a would-block means held; any other errno is treated as not held so a
    // stray unreadable lock can't wedge capture forever.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EWOULDBLOCK)
}

/// Whether a snapshot's weights are fully present. If a shard index
/// (`*.index.json`) is present, every distinct file in its `weight_map` must
/// exist and resolve; otherwise a single weights file must be present. A
/// config/tokenizer-only snapshot (a download not yet at the weights), a sharded
/// model missing shards, and an unparseable (still-downloading) index all return
/// false. This is the deterministic completeness signal, not a time guess.
pub fn weights_complete(snapshot_dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(snapshot_dir) else {
        return false;
    };
    let mut index_file = None;
    let mut has_single_weight = false;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.ends_with(".index.json") {
            index_file = Some(path);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| WEIGHT_EXTS.contains(&ext))
        {
            has_single_weight = true;
        }
    }
    match index_file {
        Some(index) => index_shards_present(&index, snapshot_dir),
        None => has_single_weight,
    }
}

fn index_shards_present(index: &Path, snapshot_dir: &Path) -> bool {
    let Ok(bytes) = std::fs::read(index) else {
        return false;
    };
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    let Some(map) = json.get("weight_map").and_then(|m| m.as_object()) else {
        return false;
    };
    let shards: std::collections::HashSet<&str> = map.values().filter_map(|v| v.as_str()).collect();
    !shards.is_empty()
        && shards
            .iter()
            .all(|shard| std::fs::metadata(snapshot_dir.join(shard)).is_ok())
}

/// The newest mtime across a snapshot's files (following symlinks, skipping our
/// internals). A short quiescence backstop against this covers the residual gap
/// where a sharded model's index has not landed yet, so a lone shard would look
/// like a complete single-file model.
pub fn newest_mtime(snapshot_dir: &Path) -> Option<std::time::SystemTime> {
    let mut newest = None;
    newest_walk(snapshot_dir, &mut newest);
    newest
}

fn newest_walk(dir: &Path, newest: &mut Option<std::time::SystemTime>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        if meta.is_dir() {
            newest_walk(&path, newest);
            continue;
        }
        if !meta.is_file() {
            continue;
        }
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(is_mx_internal)
        {
            continue;
        }
        if let Ok(mtime) = meta.modified()
            && newest.is_none_or(|cur| mtime > cur)
        {
            *newest = Some(mtime);
        }
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
    fn weights_complete_single_file_needs_a_weights_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snap = dir.path();
        // Config/tokenizer only: not yet a complete model.
        write_file(&snap.join("config.json"), b"{}");
        write_file(&snap.join("tokenizer.json"), b"tok");
        assert!(!weights_complete(snap), "no weights file yet");
        // A weights file makes it complete.
        write_file(&snap.join("model.safetensors"), b"weights");
        assert!(weights_complete(snap), "single safetensors is complete");
    }

    #[test]
    fn weights_complete_sharded_needs_all_shards() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snap = dir.path();
        let index = br#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#;
        write_file(&snap.join("model.safetensors.index.json"), index);
        write_file(&snap.join("model-00001-of-00002.safetensors"), b"shard1");
        // Index present, one shard missing -> not complete (the bug we hit).
        assert!(!weights_complete(snap), "missing shard is not complete");
        write_file(&snap.join("model-00002-of-00002.safetensors"), b"shard2");
        assert!(weights_complete(snap), "all shards present is complete");
    }

    #[test]
    fn newest_mtime_and_incomplete_blobs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snap = dir.path().join("snapshots").join("r1");
        write_file(&snap.join("model.safetensors"), b"weights");
        assert!(newest_mtime(&snap).is_some());
        assert!(newest_mtime(&dir.path().join("nope")).is_none());

        // A repo with an .incomplete blob is in progress.
        let repo = dir.path();
        let blobs = repo.join("blobs");
        write_file(&blobs.join("abc.incomplete"), b"partial");
        assert!(download_in_progress(repo), "in-progress download detected");
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
