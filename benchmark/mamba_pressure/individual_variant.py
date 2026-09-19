"""Isolated benchmark-only optimizations on the race-fixed production baseline."""

import inspect
import textwrap

VARIANTS = ('baseline', 'logging', 'singleton', 'restore_nozero')


def rewrite(module, cls, method, substitutions):
    source = textwrap.dedent(inspect.getsource(getattr(cls, method)))
    for old, new, count in substitutions:
        if source.count(old) != count:
            raise RuntimeError(f'Unexpected source for {cls.__name__}.{method}: {old!r}')
        source = source.replace(old, new)
    namespace = {}
    exec(compile(source, f'<individual-variant:{method}>', 'exec'), module.__dict__, namespace)
    setattr(cls, method, namespace[method])


def install(variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown optimization variant: {variant}')
    if variant == 'baseline':
        return
    from sglang.srt.mem_cache import mamba_radix_cache as radix
    from sglang.srt.mem_cache import memory_pool as pools
    tree, pool = radix.MambaRadixCache, pools.MambaPool
    if variant == 'logging':
        rewrite(radix, tree, '_decompress_from_pool', [
            ('logger.info(f"Full-rank mamba state from {src_node.id} to {dst_index}")',
             'logger.debug("Restoring full-rank mamba state from node %s", src_node.id)', 1),
            ('logger.info(f"Decompressing compressed mamba state from {src_node.id} to {dst_index}")',
             'logger.debug("Restoring compressed mamba state from node %s", src_node.id)', 1),
        ])
    elif variant == 'singleton':
        rewrite(radix, tree, '_process_compression_batch', [
            ('torch.stack(snapshots, dim=0)',
             '(snapshots[0].unsqueeze(0) if K == 1 else torch.stack(snapshots, dim=0))', 1),
        ])
    elif variant == 'restore_nozero':
        rewrite(pools, pool, 'alloc', [
            ('def alloc(self, need_size: int)', 'def alloc(self, need_size: int, *, zero_initialize: bool = True)', 1),
            ('    # clear at alloc time', '    if not zero_initialize:\n        return select_index\n    # clear at alloc time', 1),
        ])
        rewrite(radix, tree, '_match_post_processor', [
            ('.mamba_pool.alloc(1)', '.mamba_pool.alloc(1, zero_initialize=False)', 2),
        ])
