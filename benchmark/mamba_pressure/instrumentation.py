"""Benchmark-only observation hooks; do not change cache policy or scheduling."""

import threading
import weakref
from collections import OrderedDict, defaultdict


def install():
    import torch
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
    from sglang.srt.managers.scheduler import Scheduler

    if getattr(MambaRadixCache, "_pressure_instrumented", False):
        return
    MambaRadixCache._pressure_instrumented = True
    original_init = MambaRadixCache.__init__

    def track(tree, tensor):
        # Charge the owner: a queued view can retain an entire batched result.
        while tensor._base is not None:
            tensor = tensor._base
        key = id(tensor)
        with tree._pressure_lock:
            if key in tree._pressure_storages:
                return
            size = tensor.untyped_storage().nbytes()
            tree._pressure_storages[key] = size
            tree._pressure["staging_live_bytes"] += size
            tree._pressure["staging_peak_bytes"] = max(
                tree._pressure["staging_peak_bytes"], tree._pressure["staging_live_bytes"])

        def release():
            with tree._pressure_lock:
                tree._pressure["staging_live_bytes"] -= tree._pressure_storages.pop(key)

        weakref.finalize(tensor, release)

    def init(tree, *args, **kwargs):
        tree._pressure = defaultdict(int)
        tree._pressure_lock = threading.Lock()
        tree._pressure_storages = {}
        original_init(tree, *args, **kwargs)
        if not tree.enable_svd_compression:
            return

        class ObservedLRU(OrderedDict):
            def __setitem__(self, key, value):
                tree._pressure["compression_committed"] += 1
                super().__setitem__(key, value)

        tree._compressed_lru = ObservedLRU(tree._compressed_lru)
        for name, counter in [("_compression_queue", "compression_enqueued"),
                              ("_compression_done_queue", "compression_completed")]:
            queue = getattr(tree, name)
            original_put = queue.put

            def put(item, *a, _put=original_put, _counter=counter, **kw):
                if item is not None:
                    track(tree, item[1])
                    tree._pressure[_counter] += 1
                    tree._pressure["compression_pending_peak"] = max(
                        tree._pressure["compression_pending_peak"], len(tree._pending_compression))
                return _put(item, *a, **kw)

            queue.put = put

    MambaRadixCache.__init__ = init

    def wrap(name, observe):
        original = getattr(MambaRadixCache, name)

        def wrapped(tree, *args, **kwargs):
            observe(tree, *args, **kwargs)
            return original(tree, *args, **kwargs)

        setattr(MambaRadixCache, name, wrapped)

    def leaf(tree, node, *args, **kwargs):
        tree._pressure["evicted_entries"] += 1
        tree._pressure["evicted_kv_tokens"] += len(node.value)
        tree._pressure["evicted_compressed_entries"] += int(node.mamba_compressed)

    def tombstone(tree, node):
        tree._pressure["evicted_entries"] += 1

    def free_state(tree, node):
        if node.mamba_compressed and node.children:
            parent = node.parent
            if parent is not None and parent.children.get(tree.get_child_key_fn(node.key)) is node:
                tree._pressure["evicted_entries"] += 1
                tree._pressure["evicted_compressed_entries"] += 1

    def delete_tombstone(tree, node):
        tree._pressure["evicted_kv_tokens"] += len(node.value)

    def decompress(tree, node, *args, **kwargs):
        tree._pressure["decompression_hits"] += int(node.mamba_compressed)

    wrap("_evict_leaf_node", leaf)
    wrap("_tombstone_internal_node", tombstone)
    wrap("_free_mamba_state", free_state)
    wrap("_delete_tombstone_leaf", delete_tombstone)
    wrap("_decompress_from_pool", decompress)
    original_batch = MambaRadixCache._process_compression_batch

    def batch(tree, items):
        tree._pressure["svd_batches"] += 1
        tree._pressure["svd_batch_items"] += len(items)
        tree._pressure["svd_batch_max"] = max(tree._pressure["svd_batch_max"], len(items))
        try:
            return original_batch(tree, items)
        except Exception:
            tree._pressure["compression_failed"] += len(items)
            raise

    MambaRadixCache._process_compression_batch = batch
    original_reset = MambaRadixCache.reset

    def reset(tree):
        result = original_reset(tree)
        with tree._pressure_lock:
            live = tree._pressure["staging_live_bytes"]
            tree._pressure.clear()
            tree._pressure.update(staging_live_bytes=live, staging_peak_bytes=live)
        if torch.device(tree.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(tree.device)
        return result

    MambaRadixCache.reset = reset
    original_info = Scheduler.get_internal_state

    def info(scheduler, request):
        result = original_info(scheduler, request)
        tree = scheduler.tree_cache
        if not hasattr(tree, "_pressure"):
            return result
        pool = tree.req_to_token_pool.mamba_pool
        kv_bytes = tree.token_to_kv_pool_allocator.get_kvcache().get_kv_size_bytes()
        if isinstance(kv_bytes, tuple):
            kv_bytes = sum(kv_bytes)
        state_bytes = pool.mamba_cache.mem_usage_bytes()
        compressed_bytes = 0
        if tree.enable_svd_compression:
            compressed_bytes = sum(t.numel() * t.element_size() for t in
                                   [tree.compressed_temporal, *tree.compressed_conv])
        with tree._pressure_lock:
            metrics = dict(tree._pressure)
        admission = getattr(tree, "compression_admission", None)
        metrics["defer_prefill"] = admission is not None
        if admission is not None:
            metrics.update(admission.metrics())
        metrics.update(
            svd_worker_batch=tree.svd_worker_batch if tree.enable_svd_compression else 0,
            kv_pool_bytes=kv_bytes, mamba_pool_bytes=state_bytes,
            full_state_bytes=state_bytes // (pool.size + 1),
            temporal_shape=list(pool.mamba_cache.temporal.shape),
            temporal_element_bytes=pool.mamba_cache.temporal.element_size(),
            compressed_pool_bytes=compressed_bytes, full_state_slots=pool.size,
            compressed_slots=tree.compressed_temporal.shape[0] if tree.enable_svd_compression else 0,
            full_free_slots=int(pool.available_size()),
            compressed_entries=len(tree._compressed_lru) if tree.enable_svd_compression else 0,
            compression_pending=len(tree._pending_compression) if tree.enable_svd_compression else 0,
            compression_queue_depth=tree._compression_queue.qsize() if tree.enable_svd_compression else 0,
            cuda_allocated_bytes=torch.cuda.memory_allocated(tree.device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(tree.device),
        )
        metrics["persistent_cache_bytes"] = kv_bytes + state_bytes + compressed_bytes
        metrics["peak_cache_state_bytes"] = metrics["persistent_cache_bytes"] + metrics.get("staging_peak_bytes", 0)
        result.internal_state = {**result.internal_state, "cache_observations": metrics}
        return result

    Scheduler.get_internal_state = info
