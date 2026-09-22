"""Small checks for trace pairing and metric definitions."""

from argparse import Namespace
import unittest

from run import summarize_requests, trace


class AnalysisTests(unittest.TestCase):
    def test_trace_pairing_and_shared_boundary(self):
        args = Namespace(seed=17, groups=8, prefix=2048, rounds=3)
        first = trace(args, 0)
        self.assertEqual(first, trace(args, 0))
        self.assertNotEqual(first, trace(args, 1))
        for group in range(args.groups):
            requests = [r for r in first if r["group"] == group]
            self.assertEqual(len(requests), 3)
            self.assertEqual(len({tuple(r["tokens"][:args.prefix]) for r in requests}), 1)
            self.assertEqual(len({r["tokens"][args.prefix] for r in requests}), 3)

    def test_recomputation_excludes_compulsory_first_access(self):
        args = Namespace(prefix=100)
        rows = [dict(cached_tokens=cached, round=round_index, input_tokens=110,
                     output_tokens=8, ttft_ms=ttft, tpot_ms=5)
                for cached, round_index, ttft in [(0, 0, 30), (100, 1, 10), (40, 2, 20)]]
        before = {"evicted_entries": 7}
        after = dict(evicted_entries=9, persistent_cache_bytes=1024,
                     peak_cache_state_bytes=2048, cuda_peak_allocated_bytes=4096)
        result = summarize_requests(rows, 2, before, after, args)
        self.assertEqual(result["recomputed_prefix_tokens"], 60)
        self.assertAlmostEqual(result["token_cache_hit_rate"], 140/330)
        self.assertEqual(result["evicted_entries"], 2)
        self.assertEqual(result["request_throughput_rps"], 1.5)
        self.assertEqual(result["output_throughput_tps"], 12)
        self.assertEqual(result["mean_ttft_ms"], 20)
        self.assertEqual(result["p50_ttft_ms"], 20)


if __name__ == "__main__":
    unittest.main()
