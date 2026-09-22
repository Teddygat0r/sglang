"""Memory accounting for dense, compressed, and staged Mamba cache states."""

from math import prod


def validate_explicit_allocation(args):
    if (
        getattr(args, "mamba_svd_compression", False)
        and getattr(args, "tp_size", 1) > 1
    ):
        # The collective drain is wired into the standard autoregressive
        # scheduler, using its CPU TP cache group on every scheduling step.
        unsupported = (
            getattr(args, "dp_size", 1) != 1
            or getattr(args, "pp_size", 1) != 1
            or getattr(args, "speculative_algorithm", None)
            or getattr(args, "dllm_algorithm", None)
            or getattr(args, "enable_pdmux", False)
            or getattr(args, "disaggregation_mode", "null") != "null"
            or getattr(args, "enable_hierarchical_cache", False)
        )
        if unsupported:
            raise ValueError(
                "Tensor-parallel Mamba compression requires DP=1, PP=1 and standard "
                "autoregressive scheduling without speculation, disaggregation, "
                "PDMux or hierarchical caching"
            )
    if getattr(args, "mamba_svd_max_pending", 8) < 1:
        raise ValueError("mamba_svd_max_pending must be positive")
    slots = getattr(args, "mamba_svd_cache_size", None)
    reserve = getattr(args, "mamba_svd_staging_reserve_bytes", 0)
    if reserve < 0:
        raise ValueError("Mamba staging reserve cannot be negative")
    if slots is None:
        if reserve:
            raise ValueError("Staging reserve requires --mamba-svd-cache-size")
        return
    if not args.mamba_svd_compression:
        raise ValueError("Explicit compressed sizing requires --mamba-svd-compression")
    if slots <= 0 or not args.max_mamba_cache_size or args.max_mamba_cache_size <= 0:
        raise ValueError("Explicit sizing requires positive full and compressed slots")
    if args.mamba_svd_rank <= 0:
        raise ValueError("Mamba SVD rank must be positive")
    if getattr(args, "dp_size", 1) != 1 or getattr(args, "pp_size", 1) != 1:
        raise ValueError("Explicit Mamba sizing currently supports DP=1 and PP=1 only")
    if getattr(args, "speculative_algorithm", None) is not None:
        raise ValueError(
            "Explicit Mamba sizing does not yet support speculative decoding"
        )
    if getattr(args, "disable_radix_cache", False):
        raise ValueError("Explicit compressed sizing requires radix caching")


def compressed_state_bytes(params, rank):
    heads, dim, state = params.shape.temporal
    packed = rank * (dim + 1 + state)
    if rank <= 0 or packed > dim * state:
        raise ValueError("SVD rank must be positive and fit within the dense state")
    conv = sum(prod(shape) for shape in params.shape.conv) * params.dtype.conv.itemsize
    return int(
        len(params.layers) * (conv + heads * packed * params.dtype.temporal.itemsize)
    )


def compression_staging_bytes(params, rank, max_pending=8, worker_batch=2):
    """Storage retained by bounded jobs, excluding temporary SVD workspace.

    Completed tensors are views into an entire worker batch. Conservatively
    charge that full batch for each retained result until its final view dies.
    Additional linalg workspace uses the dynamic memory budget or an explicit
    staging reserve larger than this minimum.
    """
    compressed_state_bytes(params, rank)  # Validate the geometry and rank.
    if max_pending < 1 or worker_batch < 1:
        raise ValueError("Compression staging and worker batch must be positive")
    heads, dim, state = params.shape.temporal
    matrices = len(params.layers) * heads
    snapshot = matrices * dim * state * params.dtype.temporal.itemsize
    packed = matrices * rank * (dim + 1 + state) * 4  # SVD returns float32.
    return max_pending * (snapshot + min(worker_batch, max_pending) * packed)


def fit_mamba_cache_size(params, pool_budget_bytes, rank):
    """Largest dense pool whose sentinel and default compressed pool fit."""
    dense = params.mamba_cache_per_req
    compressed = compressed_state_bytes(params, rank)
    low, high = 0, max(0, int(pool_budget_bytes) // dense)
    while low < high:
        mid = (low + high + 1) // 2
        required = (mid + 1) * dense + max(1, mid // 2) * compressed
        if required <= pool_budget_bytes:
            low = mid
        else:
            high = mid - 1
    return low


def explicit_mamba_bytes(
    params,
    full_slots,
    compressed_slots,
    rank,
    staging_bytes,
    *,
    max_pending=8,
    worker_batch=2,
):
    if full_slots <= 0 or compressed_slots <= 0 or staging_bytes < 0:
        raise ValueError("Invalid explicit Mamba pool allocation")
    return int(
        (full_slots + 1) * params.mamba_cache_per_req
        + compressed_slots * compressed_state_bytes(params, rank)
        + max(
            staging_bytes,
            compression_staging_bytes(params, rank, max_pending, worker_batch),
        )
    )
