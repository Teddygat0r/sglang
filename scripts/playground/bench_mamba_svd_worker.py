"""Microbenchmark: old (CPU svd_lowrank) vs new (GPU rsvd_eigh) mamba SVD worker.

Isolates the two things that determine inference impact of the async SVD
compression worker in MambaRadixCache:

  1. Scheduler-thread blocking time  -- how long the *main* thread is stalled
     inside `_enqueue_compression`. The old path calls `.cpu().float()`, a
     device-synchronizing D2H copy; the new path calls `.clone()` (async) +
     `event.record()`. This is the number that maps most directly to TTFT/ITL
     regressions, because it steals time from the scheduler loop. We measure it
     both on an idle GPU and with a backlog of inference-like GPU work queued,
     since the sync cost only bites when there is pending work to drain.

  2. Worker math latency + reconstruction quality -- wall-clock to produce one
     packed result, and the relative Frobenius error of the rank-k recon. The
     worker runs off-thread so this does not directly block inference, but it
     bounds compression throughput, and the error must not regress.

Shapes default to nvidia/Nemotron-H-8B-Base-8K mamba states:
  L=24 mamba layers, H=128 heads, D=64 head_dim, S=128 state_size, rank=16.

Usage:
    python scripts/playground/bench_mamba_svd_worker.py
    python scripts/playground/bench_mamba_svd_worker.py --L 24 --H 128 --D 64 --S 128 --rank 16 --iters 30
"""

import argparse
import time

import torch

from sglang.srt.mem_cache.rsvd_eigh import randomized_svd_eigh


def _make_state(L, H, D, S, device, dtype, seed=0):
    """Low-rank-ish state with a decaying spectrum, like real recurrent states."""
    g = torch.Generator(device=device).manual_seed(seed)
    # rank-~8 core plus small full-rank noise -> decaying singular values
    a = torch.randn(L, H, D, 8, device=device, dtype=dtype, generator=g)
    b = torch.randn(L, H, 8, S, device=device, dtype=dtype, generator=g)
    noise = 0.01 * torch.randn(L, H, D, S, device=device, dtype=dtype, generator=g)
    return a @ b + noise


def _rel_err(approx, ref):
    return (approx.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)


def _pack_new(state, rank):
    L, H, D, S = state.shape
    r = min(rank, D, S)
    u, s, vh = randomized_svd_eigh(state, rank=r, n_iter=2, oversample=8)
    u_flat = u.reshape(L, H, D * r)
    v_flat = vh.transpose(-2, -1).reshape(L, H, S * r)
    packed = torch.cat([u_flat, s, v_flat], dim=-1)
    # reconstruction for quality check
    approx = (u * s.unsqueeze(-2)) @ vh
    return packed, approx


def _pack_old(cpu_state, rank):
    L, H, D, S = cpu_state.shape
    q = min(rank + 4, min(D, S))
    u, s, v = torch.svd_lowrank(cpu_state, q=q, niter=1)
    r = rank
    u_t = u[..., :r]
    s_t = s[..., :r]
    v_t = v[..., :r]
    packed = torch.cat(
        [u_t.reshape(L, H, D * r), s_t, v_t.reshape(L, H, S * r)], dim=-1
    )
    approx = (u_t * s_t.unsqueeze(-2)) @ v_t.transpose(-2, -1)
    return packed, approx


def _busy_default_stream(device, n=40, size=4096):
    """Queue inference-like matmuls on the default stream WITHOUT syncing, to
    simulate a backlog the scheduler thread would have pending."""
    x = torch.randn(size, size, device=device)
    acc = x
    for _ in range(n):
        acc = acc @ x
    return acc  # not synced; kernels are in-flight on the default stream


def bench_enqueue_blocking(state, rank, device, iters, with_backlog):
    """Measure host-side time the scheduler thread spends in the enqueue snapshot."""
    is_cuda = device.type == "cuda"
    # New path: detach + clone (+ event). Old path: detach + cpu + float.
    old_times, new_times = [], []
    for i in range(iters):
        if with_backlog and is_cuda:
            _busy_default_stream(device)
        if is_cuda:
            torch.cuda.synchronize()  # clean baseline before the timed region
            if with_backlog:
                _busy_default_stream(device)  # re-queue so work is pending
        # --- OLD enqueue host cost: .cpu().float() (device sync) ---
        t0 = time.perf_counter()
        _ = state.squeeze().detach().cpu().float()
        old_times.append((time.perf_counter() - t0) * 1e3)

        if is_cuda:
            torch.cuda.synchronize()
            if with_backlog:
                _busy_default_stream(device)
        # --- NEW enqueue host cost: .clone() + event.record() (async) ---
        t0 = time.perf_counter()
        _ = state.squeeze().detach().clone()
        if is_cuda:
            ev = torch.cuda.Event()
            ev.record()
        new_times.append((time.perf_counter() - t0) * 1e3)
        if is_cuda:
            torch.cuda.synchronize()
    return old_times, new_times


def bench_worker_math(state, rank, device, iters):
    """Measure wall-clock per packed result + reconstruction error, both paths."""
    is_cuda = device.type == "cuda"
    cpu_state = state.detach().cpu().float()

    # warmup
    _pack_new(state, rank)
    _pack_old(cpu_state, rank)
    if is_cuda:
        torch.cuda.synchronize()

    new_times, old_times = [], []
    new_err = old_err = None
    for i in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        packed_n, approx_n = _pack_new(state, rank)
        if is_cuda:
            torch.cuda.synchronize()
        new_times.append((time.perf_counter() - t0) * 1e3)
        if new_err is None:
            new_err = _rel_err(approx_n, state).item()

    for i in range(iters):
        t0 = time.perf_counter()
        packed_o, approx_o = _pack_old(cpu_state, rank)
        old_times.append((time.perf_counter() - t0) * 1e3)
        if old_err is None:
            old_err = _rel_err(approx_o, cpu_state).item()
    return (old_times, old_err), (new_times, new_err)


def _stats(xs):
    xs = sorted(xs)
    n = len(xs)
    mean = sum(xs) / n
    p50 = xs[n // 2]
    p99 = xs[min(n - 1, int(n * 0.99))]
    return mean, p50, p99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=24)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    print(f"device={device}  dtype={args.dtype}  shape=[L={args.L}, H={args.H}, "
          f"D={args.D}, S={args.S}]  rank={args.rank}  iters={args.iters}")
    print(f"batched matrices per snapshot: L*H = {args.L * args.H}, each {args.D}x{args.S}\n")

    # one slot snapshot has a leading singleton (matches temporal[:, idx]); the
    # enqueue code does .squeeze(1). We keep [L,H,D,S] and squeeze() is a no-op.
    state = _make_state(args.L, args.H, args.D, args.S, device, dtype)

    print("=== (1) scheduler-thread BLOCKING time in _enqueue_compression (ms) ===")
    for backlog in (False, True):
        old_t, new_t = bench_enqueue_blocking(state, args.rank, device, args.iters, backlog)
        om, o50, o99 = _stats(old_t)
        nm, n50, n99 = _stats(new_t)
        tag = "GPU busy (inference backlog queued)" if backlog else "GPU idle"
        print(f"\n  [{tag}]")
        print(f"    OLD  .cpu().float() :  mean={om:7.3f}  p50={o50:7.3f}  p99={o99:7.3f}")
        print(f"    NEW  .clone()+event :  mean={nm:7.3f}  p50={n50:7.3f}  p99={n99:7.3f}")
        if nm > 0:
            print(f"    speedup (mean)      :  {om / nm:6.1f}x")

    print("\n=== (2) worker math latency per packed result (ms) + recon error ===")
    (old_t, old_e), (new_t, new_e) = bench_worker_math(state, args.rank, device, args.iters)
    om, o50, o99 = _stats(old_t)
    nm, n50, n99 = _stats(new_t)
    print(f"    OLD  svd_lowrank (CPU)  :  mean={om:8.3f}  p50={o50:8.3f}  p99={o99:8.3f}   rel_err={old_e:.4e}")
    print(f"    NEW  rsvd_eigh   (GPU)  :  mean={nm:8.3f}  p50={n50:8.3f}  p99={n99:8.3f}   rel_err={new_e:.4e}")
    if nm > 0:
        print(f"    speedup (mean)          :  {om / nm:6.1f}x")


if __name__ == "__main__":
    main()
