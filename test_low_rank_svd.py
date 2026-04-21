"""Exercise low_rank_svd on realistic SSM-state shapes (mirroring the mamba
pool's temporal-state layout for Qwen3.5 GDN):
  - CPU bfloat16 input
  - GPU bfloat16 input
  - Shape: [num_selected, num_v_heads, head_v_dim, head_k_dim]
Check shape/dtype/device are preserved and output has rank <= n.
"""
import torch

from sglang.srt.models.qwen3_5 import low_rank_svd


def effective_rank(mat: torch.Tensor, tol: float = 1e-2) -> int:
    """True numerical rank via full SVD, using a relative tolerance loose
    enough to ignore bfloat16 roundoff noise below the truncated singular
    values."""
    s = torch.linalg.svdvals(mat.to(torch.float32))
    if s.numel() == 0:
        return 0
    cutoff = s.max() * tol
    return int((s > cutoff).sum().item())


def run_one(device: str, dtype: torch.dtype):
    if device == "cuda" and not torch.cuda.is_available():
        print(f"  [skip {device} {dtype}: no cuda]")
        return
    num_selected, H, V, K = 3, 4, 128, 128
    n = 16
    torch.manual_seed(0)
    state = torch.randn(num_selected, H, V, K, device=device, dtype=dtype)
    out = low_rank_svd(state, n=n)

    assert out.shape == state.shape, (out.shape, state.shape)
    assert out.dtype == state.dtype, (out.dtype, state.dtype)
    assert out.device.type == state.device.type, (out.device, state.device)
    assert torch.isfinite(out.to(torch.float32)).all().item()

    # Every VxK matrix in the batch should be rank <= n.
    flat = out.reshape(-1, V, K)
    ranks = [effective_rank(flat[i]) for i in range(flat.shape[0])]
    assert max(ranks) <= n, f"ranks={ranks} n={n}"

    # And the approximation should be different from the original (SVD actually ran).
    err = (out.to(torch.float32) - state.to(torch.float32)).abs().mean().item()
    assert err > 0, err

    print(
        f"  device={device} dtype={dtype} shape={tuple(state.shape)} "
        f"max_rank={max(ranks)} mean_abs_err={err:.4g} ... ok"
    )


print("low_rank_svd realistic-shape check")
run_one("cpu", torch.bfloat16)
run_one("cpu", torch.float32)
run_one("cuda", torch.bfloat16)
run_one("cuda", torch.float32)
print("all low_rank_svd tests passed")
