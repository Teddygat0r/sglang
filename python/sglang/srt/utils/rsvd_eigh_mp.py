"""
Fast batched randomized SVD via eigh-on-Gram + mixed-precision Cholesky-QR.

This is a minimal, single-path variant of `rsvd_eigh.py` that hard-codes the
`chol_mp` orthonormalization: a single Cholesky-QR sweep performed in fp64.
On ill-conditioned but full-rank inputs (decaying spectra, ML activations,
KV-cache states), this lands at fp32-machine-epsilon orthonormality and
the rank-k truncation optimum, without the systematic accuracy bias of the
jitter-ladder paths (`chol_v6`) or the slow Householder fallback path of
`cholqr2` on these inputs.

Algorithm (one call):
  Omega ~ N(0, 1) of shape [..., n, q] where q = rank + oversample.
  Y = A · Omega                       sketch the column space
  Q = chol_mp(Y)                      fp64 CholQR → fp32 Q
  repeat n_iter times:
      Y = A · (Aᵀ · Q)                power iteration
      Q = chol_mp(Y)
  Bproj = Qᵀ · A                      project A onto the subspace
  C = Bproj · Bprojᵀ                  small [q, q] SPD Gram
  λ, V = eigh(C)                      ascending eigvals/vecs
  S = sqrt(λ_top-k)
  U = Q · V_top-k
  Vh = V_top-kᵀ · Bproj / S

Why this path:
  - Replacing the inner SVD with eigh-on-Gram is the algorithm's signature
    optimization: cuSOLVER's tiny-SPD eigh is much faster than batched
    Jacobi SVD on the projected matrix. This is the same shape advantage
    that distinguishes rsvd_eigh from `torch.svd_lowrank`.
  - fp64 inside CholQR absorbs the κ² · ε amplification that breaks plain
    fp32 Cholesky on decaying spectra. On a typical Qwen3.5-4B recurrent-
    state batch ([768, 128, 128] → rank 16), single-pass fp32 CholQR fails
    on ~30% of states; fp32 with jitter ladder works but biases the recon
    by ~1pp; fp64 single-pass succeeds with no bias.
  - No fallback path: on the workloads this is designed for, the fp64
    Cholesky succeeds (the matrix is mathematically full-rank, just badly
    conditioned). If your inputs can be exactly rank-deficient at q (e.g.
    σ_q = 0 in fp64), use `_chol_mp_with_fallback` in `rsvd_eigh.py` or
    `cholqr2` instead.

Cost summary on Qwen3.5-4B recurrent states, [768, 128, 128] → rank 16:
  - chol_mp single-pass (this file)        ~15 ms
  - cholqr2 / house  (fallback path fires) ~110 ms
  - chol_v6 + proactive jitter             ~10 ms (but +1.2pp recon bias)
  - torch.svd_lowrank, GPU                 ~2150 ms
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _chol_mp(Y: Tensor) -> Tensor:
    """Single-pass Cholesky-QR in fp64.

    Computes the Gram, Cholesky factor, and triangular solve all in fp64,
    then casts the result back to Y's original dtype. Buys ~8 extra
    decimal digits over plain fp32 CholQR, which is enough to handle
    decaying-spectrum sketches where fp32 CholQR fails (κ² · ε ≈ 1)
    without introducing the systematic bias of jitter-based paths.

    Cost: dominated by the fp64 Gram matmul `Yᵀ Y` on the tall-skinny
    sketch. On GPU at the standard [B, 128, q] shape this is ~10–25 ms;
    on CPU it's faster than Householder QR.

    Raises `torch.linalg.LinAlgError` if the Cholesky fails — which only
    happens when Y is exactly rank-deficient (σ_q = 0 to fp64 precision).
    On natural data this effectively never occurs; if you need belt-and-
    suspenders coverage for that case, wrap this in a try/except and fall
    back to `torch.linalg.qr`.
    """
    Y64 = Y.double()
    G = Y64.transpose(-2, -1) @ Y64
    R = torch.linalg.cholesky(G, upper=True)
    Q = torch.linalg.solve_triangular(R, Y64, upper=True, left=False)
    return Q.to(Y.dtype)


@torch.no_grad()
def randomized_svd_eigh(
    A: Tensor,
    *,
    rank: int,
    n_iter: int = 2,
    oversample: int = 8,
    power_dtype: torch.dtype | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Randomized truncated SVD with eigh-on-Gram and fp64 Cholesky-QR.

    Parameters
    ----------
    A:
        Input tensor of shape [..., m, n]. Computation is batched over all
        leading dimensions. The dominant cost is matmuls of A and Aᵀ
        against tall-skinny [..., m, q] sketches, so this is fastest when
        m, n are small (e.g. 128) and the leading batch is large (e.g.
        1024).
    rank:
        Target rank k. Must satisfy `rank <= min(m, n)`.
    n_iter:
        Number of subspace (power) iterations. Each iteration does
        `Y = A · (Aᵀ · Q)` then re-orthonormalizes. `n_iter=2` is the
        accuracy point that matches `torch.svd_lowrank(niter=4)`.
    oversample:
        Extra sketch columns beyond `rank`. The sketch dimension is
        `q = rank + oversample`. q=8 is a good default; larger q tightens
        the approximation at moderate cost.
    power_dtype:
        Optional dtype for the power-iteration matmuls. Defaults to
        `A.dtype`. Pass `torch.bfloat16` (or `torch.float16`) on GPU to
        run matmuls in lower precision; the orthonormalization runs in
        fp64 and the projection + eigh run in fp32 regardless, for
        numerical safety.

    Returns
    -------
    U  : [..., m, rank]
    S  : [..., rank]   (descending)
    Vh : [..., rank, n]

    such that A ≈ U · diag(S) · Vh in the truncated rank-k Frobenius
    sense, and UᵀU = I, Vh·Vhᵀ = I at fp32 machine epsilon (≈ 1e-7).
    """
    m, n = A.shape[-2], A.shape[-1]
    if rank <= 0 or rank > min(m, n):
        raise ValueError(f"rank must be in [1, min(m, n) = {min(m, n)}], got {rank}")

    q = min(rank + oversample, m, n)
    pdtype = power_dtype if power_dtype is not None else A.dtype
    Ap = A.to(pdtype) if pdtype != A.dtype else A

    # Initial Gaussian sketch.
    Omega = torch.randn(n, q, dtype=Ap.dtype, device=A.device)
    Y = Ap @ Omega
    if pdtype != torch.float32:
        Y = Y.float()
    Q = _chol_mp(Y)

    # Subspace iteration. Each step: Y = A · Aᵀ · Q (one fused power step,
    # no intermediate orthonormalize), then re-orthonormalize via fp64
    # Cholesky-QR.
    for _ in range(n_iter):
        Q_p = Q.to(Ap.dtype) if pdtype != torch.float32 else Q
        Z = Ap.transpose(-2, -1) @ Q_p           # [..., n, q]
        Y = Ap @ Z                                # [..., m, q]
        if pdtype != torch.float32:
            Y = Y.float()
        Q = _chol_mp(Y)

    # Final projection and small SVD via eigh on the small Gram.
    Bproj = Q.transpose(-2, -1) @ A.float()       # [..., q, n] in fp32
    C = Bproj @ Bproj.transpose(-2, -1)           # [..., q, q] SPD
    C = 0.5 * (C + C.transpose(-2, -1))           # cheap symmetrize for safety
    evals, evecs = torch.linalg.eigh(C)           # ascending eigvals

    # Take top-k (last k columns), flip to descending.
    Sk = torch.sqrt(evals[..., -rank:].flip(-1).clamp_min(0))
    Ub = evecs[..., :, -rank:].flip(-1)            # [..., q, rank]
    U = Q @ Ub                                     # [..., m, rank]
    Vh = (Ub.transpose(-2, -1) @ Bproj) / Sk.unsqueeze(-1).clamp_min(1e-30)
    return U, Sk, Vh


@torch.no_grad()
def low_rank_svd(
    tensor: Tensor,
    n: int = 16,
    oversample: int = 8,
    niter: int = 2,
) -> Tensor:
    """Rank-n reconstruction of `tensor` via `randomized_svd_eigh`.

    Returns `U · diag(S) · Vh` cast back to `tensor.dtype`, on the input
    device (no CPU round-trip). Batched over all leading dimensions.
    """
    if tensor.dim() < 2:
        raise ValueError(f"SVD expects tensor rank >= 2, got shape {tuple(tensor.shape)}")
    orig_dtype = tensor.dtype
    rank = min(n, min(tensor.shape[-2:]))
    u, s, vh = randomized_svd_eigh(
        tensor.detach(), rank=rank, n_iter=niter, oversample=oversample
    )
    approx = (u * s.unsqueeze(-2)) @ vh
    return approx.to(orig_dtype)


__all__ = ["randomized_svd_eigh", "low_rank_svd"]
