// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phase-0 spike: a two-node holder/puller port of the validated Python
//! benchmark, driving [`super::nixl::NixlAgent`] over real RDMA.
//!
//! This proves the Rust FFI end to end (dlopen libnixl, register, transfer, SHA
//! verify) before the clean stager/puller refactor and the registry
//! integration land. Metadata moves as hex blobs (`get_local_md`/`load_remote`)
//! out of band, which is the same shape the P2P registry will carry; the puller
//! hands its own blob to the holder in the first notification.

use std::fs::{File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use sha2::{Digest, Sha256};

use super::buffer::StagingBuffer;
use super::nixl::NixlAgent;
use super::{CHUNK, Manifest, Shard, notif};

const POLL: Duration = Duration::from_micros(200);
const GIB: u64 = 1 << 30;

/// Throughput in GB/s (decimal) for moving `bytes` in `dur`.
fn gbps(bytes: u64, dur: Duration) -> f64 {
    let secs = dur.as_secs_f64();
    if secs <= 0.0 {
        return 0.0;
    }
    (bytes as f64) / secs / 1e9
}

/// Lowercase hex of a byte blob (for passing NIXL metadata via env/stdout).
fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len().saturating_mul(2));
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn from_hex(s: &str) -> Option<Vec<u8>> {
    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    bytes
        .chunks_exact(2)
        .map(|c| {
            let hi = (c[0] as char).to_digit(16)?;
            let lo = (c[1] as char).to_digit(16)?;
            u8::try_from((hi << 4) | lo).ok()
        })
        .collect()
}

// `O_DIRECT` is Linux-only (the daemon's real target). Gate it so the FFI still
// compiles for local type-checking on non-Linux hosts, where it is a no-op.
#[cfg(target_os = "linux")]
const DIRECT_FLAG: i32 = libc::O_DIRECT;
#[cfg(not(target_os = "linux"))]
const DIRECT_FLAG: i32 = 0;

/// Open a file for the transfer with `O_DIRECT` (page-cache bypass, the path
/// the benchmark measured). The staging buffer is page-aligned and chunk sizes
/// are 4K multiples, so the alignment requirements hold.
fn open_direct(path: &Path, write: bool) -> std::io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.read(true).custom_flags(DIRECT_FLAG);
    if write {
        opts.write(true).create(true);
    }
    opts.open(path)
}

/// SHA-256 of the first `n` bytes of `path` (the true, unpadded content).
fn sha256_prefix(path: &Path, n: u64) -> std::io::Result<String> {
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

/// Build the manifest for every `*.safetensors` shard in `dir`, zero-extending
/// each file to a `CHUNK` multiple so every transfer is chunk-aligned.
fn scan_manifest(dir: &Path) -> anyhow::Result<Manifest> {
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_none_or(|e| e != "safetensors") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .context("non-utf8 shard name")?
            .to_string();
        let true_size = std::fs::metadata(&path)?.len();
        let sha256 = sha256_prefix(&path, true_size)?;
        let padded = true_size.div_ceil(CHUNK).saturating_mul(CHUNK);
        File::options().write(true).open(&path)?.set_len(padded)?;
        shards.push(Shard {
            name,
            true_size,
            sha256,
        });
    }
    shards.sort_by(|a, b| a.name.cmp(&b.name));
    if shards.is_empty() {
        bail!("no .safetensors shards in {}", dir.display());
    }
    Ok(Manifest { shards })
}

fn buf_chunks(buf_gib: u32, manifest: &Manifest) -> u64 {
    let per_gib = GIB / CHUNK;
    let requested = u64::from(buf_gib).saturating_mul(per_gib);
    let max_shard = manifest
        .shards
        .iter()
        .map(Shard::n_chunks)
        .max()
        .unwrap_or(0);
    requested.max(max_shard)
}

/// `PULL<idx><base_addr_le>` -> `(idx, base_addr)`.
fn parse_pull(msg: &[u8]) -> Option<(usize, u64)> {
    if !msg.starts_with(notif::PULL) {
        return None;
    }
    let idx: usize = std::str::from_utf8(msg.get(4..8)?).ok()?.parse().ok()?;
    let base: [u8; 8] = msg.get(8..16)?.try_into().ok()?;
    Some((idx, u64::from_le_bytes(base)))
}

/// Block draining notifications until one with `want` as a prefix arrives,
/// returning it. The spike protocol is strictly request/response, so nothing
/// else is in flight to drop. Times out so a dead peer fails fast instead of
/// hanging forever.
fn wait_notif(agent: &NixlAgent, want: &[u8]) -> anyhow::Result<Vec<u8>> {
    let start = std::time::Instant::now();
    loop {
        for (_, msg) in agent.drain_notifs()? {
            if msg.starts_with(want) {
                return Ok(msg);
            }
        }
        if start.elapsed() >= Duration::from_secs(300) {
            bail!(
                "timed out waiting for notif {:?}",
                String::from_utf8_lossy(want)
            );
        }
        std::thread::sleep(POLL);
    }
}

/// Holder: own the shards, stage them out on request. Prints its agent name and
/// metadata blob (hex) on stdout so the harness can hand them to the puller.
pub fn run_holder(dir: &Path, name: &str, port: u16, buf_gib: u32) -> anyhow::Result<()> {
    let manifest = scan_manifest(dir)?;
    let buf = StagingBuffer::new(buf_chunks(buf_gib, &manifest))?;
    let mut agent = NixlAgent::new(name, port)?;
    agent.register_dram(buf.base_addr(), buf.len())?;

    println!("HOLDER_NAME={name}");
    println!("HOLDER_MD={}", to_hex(&agent.local_md()?));
    println!(
        "[holder] {} shards staged, waiting for puller",
        manifest.shards.len()
    );

    // Keep opened shard files alive (their fds back the NIXL registrations) for
    // the session.
    let mut open_files: Vec<File> = Vec::new();
    let mut puller: Option<String> = None;
    loop {
        for (_, msg) in agent.drain_notifs()? {
            if msg.starts_with(notif::MANIFEST_REQUEST) {
                let blob = msg
                    .get(notif::MANIFEST_REQUEST.len()..)
                    .context("short M?")?;
                let pname = agent.load_remote(blob)?;
                let mut reply = notif::MANIFEST.to_vec();
                reply.extend_from_slice(&serde_json::to_vec(&manifest)?);
                agent.send_notif(&pname, &reply)?;
                puller = Some(pname);
            } else if msg.starts_with(notif::PULL) {
                let (idx, peer_base) = parse_pull(&msg).context("bad PULL notif")?;
                let pname = puller.as_deref().context("PULL before manifest request")?;
                let shard = manifest
                    .shards
                    .get(idx)
                    .context("PULL index out of range")?;
                let path = dir.join(&shard.name);
                let file = open_direct(&path, false)?;
                let fd = file.as_raw_fd();
                let n = shard.n_chunks();
                let padded = shard.padded_size();
                agent.register_file(fd, usize::try_from(padded)?)?;
                let t_read = Instant::now();
                agent.read_file_to_dram(buf.base_addr(), fd, n)?;
                let read_dur = t_read.elapsed();
                let t_net = Instant::now();
                agent.write_dram_to_peer(buf.base_addr(), usize::try_from(peer_base)?, pname, n)?;
                let net_dur = t_net.elapsed();
                agent.send_notif(pname, &notif::indexed(notif::DONE, idx))?;
                open_files.push(file);
                println!(
                    "[holder] shard {idx} {:.2}GB  NVMe-read {:.2} GB/s  RDMA-write {:.2} GB/s",
                    shard.true_size as f64 / 1e9,
                    gbps(padded, read_dur),
                    gbps(padded, net_dur),
                );
            } else if msg.starts_with(notif::BYE) {
                println!("[holder] session complete");
                return Ok(());
            }
        }
        std::thread::sleep(POLL);
    }
}

/// Puller: pull every shard from the holder into local NVMe and SHA-verify.
pub fn run_puller(
    dir: &Path,
    name: &str,
    holder_name: &str,
    holder_md_hex: &str,
    buf_gib: u32,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let holder_md = from_hex(holder_md_hex).context("invalid HOLDER_MD hex")?;

    // Fixed receive buffer, registered up front so the holder has our rkeys
    // before it ever writes (the register-before-metadata-exchange lesson). Each
    // shard transfers its own n_chunks into the first part of this buffer.
    let cap = u64::from(buf_gib).saturating_mul(GIB / CHUNK).max(1);
    let buf = StagingBuffer::new(cap)?;
    let mut agent = NixlAgent::new(name, 0)?;
    agent.register_dram(buf.base_addr(), buf.len())?;
    let loaded = agent.load_remote(&holder_md)?;
    if loaded != holder_name {
        bail!("holder md names {loaded:?}, expected {holder_name:?}");
    }

    // Request the manifest, handing the holder our metadata in-band so it can
    // reply and later RDMA-write into our registered buffer.
    let mut req = notif::MANIFEST_REQUEST.to_vec();
    req.extend_from_slice(&agent.local_md()?);
    agent.send_notif(holder_name, &req)?;
    let reply = wait_notif(&agent, notif::MANIFEST)?;
    let body = reply
        .get(notif::MANIFEST.len()..)
        .context("short MANIFEST notif")?;
    let manifest: Manifest = serde_json::from_slice(body)?;
    println!("[puller] manifest: {} shards", manifest.shards.len());

    let mut agg_recv = Duration::ZERO;
    let mut agg_write = Duration::ZERO;
    let mut agg_bytes: u64 = 0;
    for (idx, shard) in manifest.shards.iter().enumerate() {
        let n = shard.n_chunks();
        if n > cap {
            bail!("shard {idx} needs {n} chunks > buffer {cap}; raise --buf-gib");
        }
        let padded = shard.padded_size();
        let dest = dir.join(&shard.name);
        let file = open_direct(&dest, true)?;
        file.set_len(padded)?;
        let fd = file.as_raw_fd();
        agent.register_file(fd, usize::try_from(padded)?)?;

        // Ask for the shard (PULL + our buffer base); the holder writes it in.
        // The request->DONE window is the holder's read+network, observed here.
        let mut pull = notif::indexed(notif::PULL, idx);
        pull.extend_from_slice(&(buf.base_addr() as u64).to_le_bytes());
        let t_recv = Instant::now();
        agent.send_notif(holder_name, &pull)?;
        wait_notif(&agent, &notif::indexed(notif::DONE, idx))?;
        let recv_dur = t_recv.elapsed();

        // Land it on NVMe, trim the padding, and verify against the manifest.
        let t_write = Instant::now();
        agent.write_dram_to_file(buf.base_addr(), fd, n)?;
        file.sync_all()?;
        let write_dur = t_write.elapsed();
        file.set_len(shard.true_size)?;
        drop(file);
        let got = sha256_prefix(&dest, shard.true_size)?;
        if got != shard.sha256 {
            bail!("shard {idx} sha mismatch: {got} != {}", shard.sha256);
        }
        agg_recv = agg_recv.saturating_add(recv_dur);
        agg_write = agg_write.saturating_add(write_dur);
        agg_bytes = agg_bytes.saturating_add(padded);
        println!(
            "[puller] shard {idx} {:.2}GB  recv(read+net) {:.2} GB/s  NVMe-write {:.2} GB/s  sha OK",
            shard.true_size as f64 / 1e9,
            gbps(padded, recv_dur),
            gbps(padded, write_dur),
        );
    }

    agent.send_notif(holder_name, notif::BYE)?;
    println!(
        "[puller] SUMMARY {:.2} GB  avg recv(read+net) {:.2} GB/s  avg NVMe-write {:.2} GB/s  all {} shards verified",
        agg_bytes as f64 / 1e9,
        gbps(agg_bytes, agg_recv),
        gbps(agg_bytes, agg_write),
        manifest.shards.len(),
    );
    Ok(())
}
