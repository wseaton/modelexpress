#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU-less NIXL probe for the NVMe-cache-repopulation idea.

One process, two NIXL agents, UCX over TCP, host (DRAM) and FILE memory only.
No CUDA, no RDMA hardware required. The single question we care about:

    Can a NIXL agent RDMA-READ a *remote* agent's FILE region directly,
    or must the source stage file bytes into registered host memory first?

The answer decides the architecture. Three sub-probes, each isolated so one
failure does not mask the others. Every probe prints PASS/FAIL plus the exact
exception, because the *way* it fails is the finding.

Run inside an image that ships the production nixl + UCX stack
(vllm/vllm-openai:*). UCX_TLS=tcp keeps it off RDMA hardware.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
import traceback

os.environ.setdefault("UCX_TLS", "tcp")

from nixl._api import nixl_agent, nixl_agent_config  # noqa: E402

PAYLOAD = b"".join(((i % 251).to_bytes(1, "little")) for i in range(1 << 20))  # 1 MiB, non-trivial pattern
N = len(PAYLOAD)
DRAM = "DRAM"
FILE = "FILE"


def _host_buf(init: bytes | None = None) -> tuple[ctypes.Array, int]:
    """Allocate a host buffer, return (buf, addr). Keep `buf` alive."""
    buf = ctypes.create_string_buffer(N)
    if init is not None:
        ctypes.memmove(ctypes.addressof(buf), init, N)
    return buf, ctypes.addressof(buf)


def _buf_bytes(buf: ctypes.Array) -> bytes:
    return ctypes.string_at(ctypes.addressof(buf), N)


def _make_agent(name: str, backends: list[str]) -> nixl_agent:
    cfg = nixl_agent_config(backends=backends)
    return nixl_agent(name, cfg)


def _run_read(dst_agent, src_agent, remote_name, remote_list, remote_mem,
              local_list, local_mem, backends, timeout=30.0):
    """Issue a READ pulling remote_list -> local_list, block until done."""
    src_prepped = dst_agent.prep_xfer_dlist(
        agent_name=remote_name, xfer_list=remote_list,
        mem_type=remote_mem, backends=backends,
    )
    dst_prepped = dst_agent.prep_xfer_dlist(
        agent_name="", xfer_list=local_list,
        mem_type=local_mem, backends=backends,
    )
    idx = list(range(len(remote_list)))
    handle = dst_agent.make_prepped_xfer(
        operation="READ", local_xfer_side=dst_prepped, local_indices=idx,
        remote_xfer_side=src_prepped, remote_indices=idx, backends=backends,
    )
    dst_agent.transfer(handle)
    t0 = time.perf_counter()
    while True:
        st = dst_agent.check_xfer_state(handle)
        if st in ("DONE", "SUCCESS"):
            dst_agent.release_xfer_handle(handle)
            return
        if st in ("ERR", "ERROR", "FAIL"):
            dst_agent.release_xfer_handle(handle)
            raise RuntimeError(f"transfer state {st}")
        if time.perf_counter() - t0 > timeout:
            dst_agent.release_xfer_handle(handle)
            raise TimeoutError("transfer timed out")
        time.sleep(0.001)


def _connect(dst_agent, src_agent):
    """Exchange metadata so dst can reach src; return src's remote name."""
    return dst_agent.add_remote_agent(src_agent.get_agent_metadata())


def probe_dram_to_dram() -> None:
    """Sanity: cross-agent host->host READ over UCX. Proves the harness."""
    src = _make_agent("src-dram", ["UCX"])
    dst = _make_agent("dst-dram", ["UCX"])
    src_buf, src_addr = _host_buf(PAYLOAD)
    dst_buf, dst_addr = _host_buf(b"\x00" * N)
    src.register_memory([(src_addr, N, 0, "")], DRAM)
    dst.register_memory([(dst_addr, N, 0, "")], DRAM)
    remote = _connect(dst, src)
    _run_read(dst, src, remote, [(src_addr, N, 0)], DRAM,
              [(dst_addr, N, 0)], DRAM, ["UCX"])
    got = _buf_bytes(dst_buf)
    assert got == PAYLOAD, f"payload mismatch ({got[:8]!r} != {PAYLOAD[:8]!r})"


def _local_storage_xfer(agent, mem_reg, file_reg, op: str) -> None:
    """Local storage<->memory step: agent talks to ITSELF (remote_name=name).

    This is the API the NIXL remote_storage_example uses for the storage leg:
    initialize_xfer(op, mem_descs, file_descs, agent.name, backends=[POSIX]).
    op="READ" pulls FILE->mem, op="WRITE" pushes mem->FILE.
    """
    handle = agent.initialize_xfer(
        op, mem_reg.trim(), file_reg.trim(), agent.name, backends=["POSIX"])
    agent.transfer(handle)
    t0 = time.perf_counter()
    try:
        while True:
            st = agent.check_xfer_state(handle)
            if st in ("DONE", "SUCCESS"):
                return
            if st in ("ERR", "ERROR", "FAIL"):
                raise RuntimeError(f"local {op} state {st}")
            if time.perf_counter() - t0 > 30:
                raise TimeoutError(f"local {op} timed out")
            time.sleep(0.001)
    finally:
        agent.release_xfer_handle(handle)


def probe_local_file_to_dram(path: str) -> None:
    """Storage leg in isolation: POSIX-backed FILE -> host DRAM, no GPU.

    Proves the shipped image can do storage<->memory on a CPU-only pod via the
    POSIX backend (GDS would need nvidia_fs + a GPU). This is the per-node
    building block both ends of a repop need.
    """
    with open(path, "wb") as f:
        f.write(PAYLOAD)
    fd = os.open(path, os.O_RDONLY)
    try:
        agent = _make_agent("file-local", ["POSIX"])
        mem_buf, mem_addr = _host_buf(b"\x00" * N)
        mem_reg = agent.register_memory([(mem_addr, N, 0, "")], DRAM)
        file_reg = agent.register_memory([(0, N, fd, "")], FILE)
        _local_storage_xfer(agent, mem_reg, file_reg, "READ")
        assert _buf_bytes(mem_buf) == PAYLOAD, "FILE->DRAM payload mismatch"
    finally:
        os.close(fd)


def probe_nvme_to_nvme_staged(src_path: str, dst_path: str) -> None:
    """The whole dream path, GPU-less, in one process across two agents:

        source NVMe --POSIX--> source DRAM --UCX--> dest DRAM --POSIX--> dest NVMe

    There is no one-sided RDMA read of a peer's file (NIXL has no rkey for
    file-backed memory; the earlier direct attempt returned NIXL_ERR_NOT_FOUND
    by design). The supported shape is: the data-holder stages FILE->DRAM
    locally, a memory<->memory network transfer moves it, and the receiver
    writes DRAM->FILE locally. This probe runs all four legs and verifies the
    bytes land on the destination file.
    """
    with open(src_path, "wb") as f:
        f.write(PAYLOAD)
    src_fd = os.open(src_path, os.O_RDONLY)
    dst_fd = os.open(dst_path, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(dst_fd, N)
    try:
        holder = _make_agent("holder", ["UCX", "POSIX"])  # owns the source NVMe file
        puller = _make_agent("puller", ["UCX", "POSIX"])  # rebuilding its cache

        # Leg 1: holder stages its NVMe file into its own DRAM (local, POSIX).
        hold_buf, hold_addr = _host_buf(b"\x00" * N)
        hold_mem = holder.register_memory([(hold_addr, N, 0, "")], DRAM)
        hold_file = holder.register_memory([(0, N, src_fd, "")], FILE)
        _local_storage_xfer(holder, hold_mem, hold_file, "READ")
        assert _buf_bytes(hold_buf) == PAYLOAD, "leg1 holder FILE->DRAM mismatch"

        # Leg 2: puller pulls holder's DRAM over UCX (memory<->memory network).
        pull_buf, pull_addr = _host_buf(b"\x00" * N)
        puller.register_memory([(pull_addr, N, 0, "")], DRAM)
        remote = _connect(puller, holder)
        _run_read(puller, holder, remote, [(hold_addr, N, 0)], DRAM,
                  [(pull_addr, N, 0)], DRAM, ["UCX"])
        assert _buf_bytes(pull_buf) == PAYLOAD, "leg2 network DRAM->DRAM mismatch"

        # Leg 3: puller writes its DRAM down to its own NVMe (local, POSIX).
        pull_file = puller.register_memory([(0, N, dst_fd, "")], FILE)
        pull_mem = puller.register_memory([(pull_addr, N, 0, "")], DRAM)
        _local_storage_xfer(puller, pull_mem, pull_file, "WRITE")

        # Verify the bytes actually hit the destination file on disk.
        os.fsync(dst_fd)
        with open(dst_path, "rb") as f:
            assert f.read() == PAYLOAD, "leg3 dest NVMe file content mismatch"
    finally:
        os.close(src_fd)
        os.close(dst_fd)


def main() -> int:
    tmp = os.environ.get("PROBE_TMPDIR", "/tmp")
    probes = [
        ("dram_to_dram", lambda: probe_dram_to_dram()),
        ("local_file_to_dram", lambda: probe_local_file_to_dram(f"{tmp}/probe_file.bin")),
        ("nvme_to_nvme_staged", lambda: probe_nvme_to_nvme_staged(
            f"{tmp}/probe_src.bin", f"{tmp}/probe_dst.bin")),
    ]
    results = {}
    for name, fn in probes:
        print(f"\n=== PROBE: {name} ===", flush=True)
        try:
            fn()
            results[name] = "PASS"
            print(f"RESULT {name}: PASS", flush=True)
        except Exception as e:  # noqa: BLE001 - probe wants every failure verbatim
            results[name] = f"FAIL: {type(e).__name__}: {e}"
            print(f"RESULT {name}: FAIL", flush=True)
            traceback.print_exc()

    print("\n=== SUMMARY ===", flush=True)
    for name, _ in probes:
        print(f"  {name}: {results[name]}", flush=True)
    # Exit nonzero only if the sanity probe failed; the FILE probes are
    # exploratory and a FAIL there is itself a valid, recorded outcome.
    return 0 if results["dram_to_dram"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
