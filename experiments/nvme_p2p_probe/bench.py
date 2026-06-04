#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-node GPU-less NVMe->NVMe model-weight transfer benchmark over NIXL.

Path measured (the cache-repop pattern, no GPU):

    holder NVMe --POSIX/O_DIRECT--> holder DRAM --UCX/RDMA--> puller DRAM
        --POSIX/O_DIRECT--> puller NVMe

Roles:
  holder  owns the real safetensors shards on local NVMe, acts as the active
          storage server (files can't be one-sided RDMA-read, see the probe).
  puller  initiates, receives each shard into a registered DRAM buffer, writes
          it to its own local NVMe, and SHA-256 verifies vs the holder original.

Each side keeps ONE fixed staging buffer (--buf-gib, default 4) registered once
up front. The puller must register before the metadata exchange so the holder
learns its rkeys; every shard transfers exactly buf-gib bytes (shards are
zero-extended to that size, dest truncated back before verify) so the local and
remote descriptor lists always line up. This fixed buffer is also the real
memory cap: model size never enters the DRAM footprint.

Per shard we report holder NVMe read, RDMA network, and puller NVMe write
bandwidth. Transport is NIXL UCX (auto-detects IB) + POSIX (O_DIRECT).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import mmap
import os
import time

from nixl._api import nixl_agent, nixl_agent_config

CHUNK = 16 * 1024 * 1024  # 16 MiB transfer/registration granularity (4K-aligned)


def _sha256_prefix(path: str, n: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        left = n
        while left:
            b = f.read(min(1 << 20, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def _alloc(size: int):
    """Page-aligned anonymous host buffer (O_DIRECT-safe, registerable, 64-bit).

    Returns (buf, addr). The temporary from_buffer exporter is released as soon
    as addressof returns, so buf.close() later won't hit 'exported pointers
    exist'. The address stays valid while buf is alive. NIXL's malloc_passthru
    is 32-bit and rejects >2GB, hence mmap.
    """
    buf = mmap.mmap(-1, size)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    return buf, addr


def _free(buf) -> None:
    try:
        buf.close()
    except BufferError:
        pass


def _make_agent(name: str, port: int) -> nixl_agent:
    agent = nixl_agent(name, nixl_agent_config(True, True, port, backends=[]))
    plugins = agent.get_plugin_list()
    for need in ("POSIX", "UCX"):
        if need not in plugins:
            raise RuntimeError(f"{need} backend missing; have {plugins}")
    agent.create_backend("POSIX")
    agent.create_backend("UCX")
    return agent


def _chunk_descs(base_addr: int, nchunks: int):
    return [(base_addr + i * CHUNK, CHUNK, 0, str(i)) for i in range(nchunks)]


def _file_descs(fd: int, nchunks: int):
    return [(i * CHUNK, CHUNK, fd, str(i)) for i in range(nchunks)]


def _run(agent, op, local_descs, remote_descs, remote_name, backends):
    h = agent.initialize_xfer(op, local_descs, remote_descs, remote_name, backends=backends)
    agent.transfer(h)
    while True:
        st = agent.check_xfer_state(h)
        if st == "DONE":
            break
        if st == "ERR":
            agent.release_xfer_handle(h)
            raise RuntimeError(f"{op} transfer errored")
    agent.release_xfer_handle(h)


def _recv_notif(agent, want_prefix: bytes, timeout=600.0):
    t0 = time.perf_counter()
    while True:
        for sender, msgs in agent.get_new_notifs().items():
            for m in msgs:
                if m.startswith(want_prefix):
                    return sender, m
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError(f"no notif {want_prefix!r} in {timeout}s")
        time.sleep(0.001)


# ---------------------------------------------------------------- holder ------
def run_holder(args):
    bufbytes = args.buf_gib * (1 << 30)
    nbuf = bufbytes // CHUNK
    agent = _make_agent(args.name, args.port)

    shards = sorted(
        os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith(".safetensors")
    )
    if not shards:
        raise RuntimeError(f"no .safetensors in {args.dir}")
    manifest = []
    for path in shards:
        true = os.path.getsize(path)
        if true > bufbytes:
            raise RuntimeError(f"{path} {true}B exceeds buffer {bufbytes}B; raise --buf-gib")
        manifest.append({"name": os.path.basename(path), "true": true,
                         "sha": _sha256_prefix(path, true)})
    total = sum(m["true"] for m in manifest)

    sbuf, saddr = _alloc(bufbytes)
    smem = agent.register_memory(_chunk_descs(saddr, nbuf), "DRAM")
    print(f"[holder] {len(manifest)} shards, {total/1e9:.2f} GB, buf {args.buf_gib}GiB, "
          f"waiting for puller", flush=True)

    _recv_notif(agent, b"M?")
    # The puller pushed its metadata (send_local_metadata) just before M?, but
    # loading it on our side is async. Wait until it's actually here, else the
    # reply notif (and later the RDMA write to the puller's rkeys) has no route.
    while not agent.check_remote_metadata(args.peer):
        time.sleep(0.01)
    agent.send_notif(args.peer, b"MANIFEST" + json.dumps(manifest).encode())

    for idx, m in enumerate(manifest):
        path = os.path.join(args.dir, m["name"])
        fd_rw = os.open(path, os.O_RDWR)
        os.ftruncate(fd_rw, bufbytes)  # zero-extend so every O_DIRECT read is buf-sized
        os.close(fd_rw)
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            fdescs = agent.register_memory(_file_descs(fd, nbuf), "FILE")
            _, msg = _recv_notif(agent, b"PULL%04d" % idx)
            puller_descs = agent.deserialize_descs(msg[8:])

            t0 = time.perf_counter()
            _run(agent, "READ", smem.trim(), fdescs.trim(), agent.name, ["POSIX"])
            t1 = time.perf_counter()
            _run(agent, "WRITE", smem.trim(), puller_descs, args.peer, ["UCX"])
            t2 = time.perf_counter()

            print(f"[holder] shard {idx} {m['true']/1e9:.2f}GB  "
                  f"NVMe-read {bufbytes/(t1-t0)/1e9:.2f} GB/s  "
                  f"RDMA-write {bufbytes/(t2-t1)/1e9:.2f} GB/s", flush=True)
            agent.send_notif(args.peer, b"DONE%04d" % idx)
            agent.deregister_memory(fdescs, backends=["POSIX"])
        finally:
            os.close(fd)

    _recv_notif(agent, b"BYE")
    agent.deregister_memory(smem)
    _free(sbuf)
    print("[holder] done", flush=True)


# ---------------------------------------------------------------- puller ------
def run_puller(args):
    bufbytes = args.buf_gib * (1 << 30)
    nbuf = bufbytes // CHUNK
    agent = _make_agent(args.name, args.port)

    # Register the receive buffer BEFORE the metadata exchange so the holder
    # learns its rkeys (registering after connect was the NIXL_ERR_NOT_FOUND bug).
    rbuf, raddr = _alloc(bufbytes)
    rmem = agent.register_memory(_chunk_descs(raddr, nbuf), "DRAM")
    rmem_descs = agent.get_serialized_descs(rmem.trim())

    agent.fetch_remote_metadata(args.peer, args.peer_ip, args.peer_port)
    while not agent.check_remote_metadata(args.peer):
        time.sleep(0.05)
    agent.send_local_metadata(args.peer_ip, args.peer_port)
    print(f"[puller] connected to {args.peer} at {args.peer_ip}:{args.peer_port}, "
          f"buf {args.buf_gib}GiB", flush=True)

    agent.send_notif(args.peer, b"M?")
    _, msg = _recv_notif(agent, b"MANIFEST")
    manifest = json.loads(msg[len(b"MANIFEST"):].decode())
    os.makedirs(args.dir, exist_ok=True)
    total = sum(m["true"] for m in manifest)
    print(f"[puller] manifest: {len(manifest)} shards, {total/1e9:.2f} GB", flush=True)

    agg_net = agg_wr = agg_bytes = 0.0
    for idx, m in enumerate(manifest):
        dest = os.path.join(args.dir, m["name"])
        fd = os.open(dest, os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o644)
        os.ftruncate(fd, bufbytes)
        try:
            fdescs = agent.register_memory(_file_descs(fd, nbuf), "FILE")

            t0 = time.perf_counter()
            agent.send_notif(args.peer, b"PULL%04d" % idx + rmem_descs)
            _recv_notif(agent, b"DONE%04d" % idx)
            t1 = time.perf_counter()  # holder read+net as observed by puller
            _run(agent, "WRITE", rmem.trim(), fdescs.trim(), agent.name, ["POSIX"])
            t2 = time.perf_counter()

            os.fsync(fd)
            os.ftruncate(fd, m["true"])
            got = _sha256_prefix(dest, m["true"])
            ok = got == m["sha"]
            agg_net += t1 - t0
            agg_wr += t2 - t1
            agg_bytes += bufbytes
            print(f"[puller] shard {idx} {m['true']/1e9:.2f}GB  "
                  f"recv(read+net) {bufbytes/(t1-t0)/1e9:.2f} GB/s  "
                  f"NVMe-write {bufbytes/(t2-t1)/1e9:.2f} GB/s  "
                  f"sha {'OK' if ok else 'MISMATCH'}", flush=True)
            if not ok:
                raise RuntimeError(f"shard {idx} sha mismatch: {got} != {m['sha']}")
            agent.deregister_memory(fdescs, backends=["POSIX"])
        finally:
            os.close(fd)

    agent.send_notif(args.peer, b"BYE")
    agent.deregister_memory(rmem)
    _free(rbuf)
    print(f"\n[puller] SUMMARY {agg_bytes/1e9:.2f} GB moved  "
          f"avg recv(read+net) {agg_bytes/agg_net/1e9:.2f} GB/s  "
          f"avg NVMe-write {agg_bytes/agg_wr/1e9:.2f} GB/s  ALL SHARDS VERIFIED", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["holder", "puller"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--peer", required=True)
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--peer-ip")
    ap.add_argument("--peer-port", type=int, default=7000)
    ap.add_argument("--buf-gib", type=int, default=4)
    args = ap.parse_args()
    (run_holder if args.role == "holder" else run_puller)(args)


if __name__ == "__main__":
    main()
