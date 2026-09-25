"""Diagnose saved replay prefixes and reproduce endpoint matching on CPU.

Usage: PYTHONPATH=python .venv/bin/python benchmark/mamba_pressure/diagnose_prefix_replay.py --results /external/run
Does not launch a model, modify cache policy, or restart the experiment.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache, TreeNode
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    get_child_key,
)

from spark_run import ROOT, save


def endpoint_match(prompt, followup, *, compressed=False, checkpoint=None):
    # Use the real matcher and node-splitting implementation with CPU KV indices.
    # Recurrent state values are irrelevant to choosing a restore endpoint.
    tree = MambaRadixCache.__new__(MambaRadixCache)
    tree.enable_svd_compression = compressed
    tree.key_match_fn = _key_match_page_size1
    tree.get_child_key_fn = get_child_key
    tree.reset()
    parent, start = tree.root_node, 0
    endpoints = ([checkpoint] if checkpoint else []) + [len(prompt)]
    for end in endpoints:
        node = TreeNode()
        node.parent = parent
        node.key = RadixKey(prompt[start:end])
        node.value = torch.arange(start, end)
        node.mamba_compressed = compressed
        node.mamba_value = None if compressed else torch.tensor([1])
        parent.children[get_child_key(node.key)] = node
        tree.full_lru_list.insert_mru(node)
        if not compressed:
            tree.mamba_lru_list.insert_mru(node)
        parent, start = node, end
    values, _, best = tree._match_prefix_helper(RadixKey(followup))
    return sum(len(v) for v in values[:best])


def diagnose(root):
    root = root.resolve()
    if root.is_relative_to(ROOT):
        raise ValueError("Results must be outside the checkout")
    trace = json.loads((root / "trace_pilot.json").read_text())
    run = root / "pilot/deferral_cap2_r0"
    measured = {
        row["index"]: row
        for row in map(json.loads, (run / "requests.jsonl").read_text().splitlines())
    }
    previous, rows = {}, []
    for index, item in enumerate(trace):
        old = previous.get(item["group"])
        if old is not None:
            common = 0
            for a, b in zip(old["tokens"], item["tokens"]):
                if a != b:
                    break
                common += 1
            assert common == item["prior_input_tokens"]
            counts = {
                str(compressed): endpoint_match(
                    old["tokens"], item["tokens"], compressed=compressed
                )
                for compressed in (False, True)
            }
            # Counterfactual: an earlier state at a shared 64-token boundary is
            # matchable in both forms. This does not assert that it was stored.
            checkpoint = common // 64 * 64
            if checkpoint and checkpoint < len(old["tokens"]):
                for compressed in (False, True):
                    assert (
                        endpoint_match(
                            old["tokens"],
                            item["tokens"],
                            compressed=compressed,
                            checkpoint=checkpoint,
                        )
                        == checkpoint
                    )
            rows.append(
                dict(
                    index=index,
                    group=item["group"],
                    turn=item["round"],
                    prior_input_tokens=len(old["tokens"]),
                    shared_tokens=common,
                    changed_suffix_tokens=len(old["tokens"]) - common,
                    cached_tokens=measured[index]["cached_tokens"],
                    endpoint_only_match=counts,
                )
            )
        previous[item["group"]] = item
    summary = dict(
        followups=len(rows),
        hit_requests=sum(r["cached_tokens"] > 0 for r in rows),
        changed_suffix_histogram=dict(
            Counter(r["changed_suffix_tokens"] for r in rows)
        ),
        cached_tokens_histogram=dict(Counter(r["cached_tokens"] for r in rows)),
        dense_endpoint_misses=sum(r["endpoint_only_match"]["False"] == 0 for r in rows),
        compressed_endpoint_misses=sum(
            r["endpoint_only_match"]["True"] == 0 for r in rows
        ),
        earlier_checkpoint_match_assertions="passed for dense and compressed nodes",
        scope="CPU matcher reproduction, not a GPU compression-on/off comparison",
    )
    save(root / "prefix_diagnosis.json", dict(summary=summary, requests=rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    diagnose(parser.parse_args().results)
