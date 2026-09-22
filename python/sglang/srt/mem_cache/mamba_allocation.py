"""Opt-in static Mamba pool accounting; not runtime queue-budget enforcement."""

from math import prod


def validate_explicit_allocation(args):
    # Compression commits change cache capacity and may evict prefix nodes.
    # Until completions are coordinated across ranks, each worker must own
    # the entire model so asynchronous completion cannot diverge TP batches.
    if getattr(args, "mamba_svd_compression", False) and getattr(args, "tp_size", 1) != 1:
        raise ValueError("Mamba SVD compression currently requires --tp-size 1")
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
        raise ValueError("Explicit Mamba sizing does not yet support speculative decoding")
    if getattr(args, "disable_radix_cache", False):
        raise ValueError("Explicit compressed sizing requires radix caching")


def compressed_state_bytes(params, rank):
    heads, dim, state = params.shape.temporal
    packed = rank * (dim + 1 + state)
    if rank <= 0 or packed > dim * state:
        raise ValueError("SVD rank must be positive and fit within the dense state")
    conv = sum(prod(shape) for shape in params.shape.conv) * params.dtype.conv.itemsize
    return int(len(params.layers) * (conv + heads * packed * params.dtype.temporal.itemsize))


def explicit_mamba_bytes(params, full_slots, compressed_slots, rank, staging_bytes):
    if full_slots <= 0 or compressed_slots <= 0 or staging_bytes < 0:
        raise ValueError("Invalid explicit Mamba pool allocation")
    return int(
        (full_slots + 1) * params.mamba_cache_per_req
        + compressed_slots * compressed_state_bytes(params, rank)
        + staging_bytes
    )
