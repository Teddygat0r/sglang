"""Isolated allocation/policy experiment; production sources are unchanged."""
import inspect
import os
from pathlib import Path
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache


def replace_method(name, old, new):
    method = getattr(MambaRadixCache, name)
    source = textwrap.dedent(inspect.getsource(method))
    assert source.count(old) == 1, (name, "source changed")
    namespace = dict(method.__globals__)
    exec(compile(source.replace(old, new), __file__ + ":" + name, "exec"), namespace)
    setattr(MambaRadixCache, name, namespace[name])


slots = int(os.environ.get("PRESSURE_COMPRESSED_SLOTS", "0"))
policy = os.environ.get("PRESSURE_POLICY", "lru")
assert policy in ("lru", "retain_full", "reuse")
if slots:
    replace_method("_init_compression_state", "max(1, pool.size // 2)", str(slots))
if policy == "retain_full":
    # Keep completed nodes full-rank while the dense pool has headroom. Under
    # dense pressure, use normal compressed eviction so requests can progress.
    replace_method("drain_compression_completions",
                   "if not self._compressed_free_slots:",
                   "if not self._compressed_free_slots and len(pool.free_slots) > 4:\n"
                   "            continue\n"
                   "        if not self._compressed_free_slots:")
if policy == "reuse":
    original_decompress = MambaRadixCache._decompress_from_pool
    def decompress(self, node, *args, **kwargs):
        node._pressure_reuses = getattr(node, "_pressure_reuses", 0) + 1
        return original_decompress(self, node, *args, **kwargs)
    MambaRadixCache._decompress_from_pool = decompress
    original_evict = MambaRadixCache._evict_compressed_lru
    def evict(self):
        # Stable LFU ordering, preserving LRU order among equal reuse counts.
        ordered = sorted(self._compressed_lru,
                         key=lambda key: getattr(self._compressed_lru[key], "_pressure_reuses", 0))
        for key in ordered:
            self._compressed_lru.move_to_end(key)
        return original_evict(self)
    MambaRadixCache._evict_compressed_lru = evict

# This imports telemetry after the policy overrides, including in spawned workers.
import server as observed_server
if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
