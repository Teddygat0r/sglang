"""CPU checks for production allocation and the complete sweep supervisor."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "test/registered/unit/mem_cache")
)
from test_mamba_compression_allocation import (
    NativeAllocationTests as NativeAllocationTests,
    arguments as arguments,
    production_geometry as geometry,
)

import allocation_native as sweep
from sglang.srt.mem_cache.mamba_allocation import (
    compressed_state_bytes,
)


class NativeAllocationSweepTests(unittest.TestCase):
    def test_budget_and_order(self):
        g = geometry()
        for c in sweep.configs():
            charged = 8589967360 + (c["full_slots"] + 1) * g.mamba_cache_per_req
            charged += (
                c["compressed_slots"] * compressed_state_bytes(g, 16)
                + c["staging_reserve_bytes"]
            )
            self.assertLessEqual(charged, c["budget_bytes"])
            self.assertLess(c["budget_bytes"] - charged, compressed_state_bytes(g, 16))
        calls = []

        async def fake(args, config, rep, target, workload):
            calls.append((args.pilot, config["label"], rep, workload))
            target.mkdir()
            sweep.save(
                target / "result.json",
                dict(config=config, repetition=rep, metrics={"test": 1.0}),
            )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(sweep, "one_run", fake):
                asyncio.run(sweep.experiment(root))
            self.assertEqual(
                json.loads((root / "status.json").read_text())["state"], "complete"
            )
            self.assertEqual(len(calls), 24)
            for rep in range(5):
                measured = [r for r in calls if not r[0] and r[2] == rep]
                expected = ["off", "on_f16", "on_f32", "on_f64"]
                self.assertEqual(
                    [r[1] for r in measured],
                    expected if rep % 2 == 0 else expected[::-1],
                )
                self.assertTrue(all(r[3] == measured[0][3] for r in measured))
            report = json.loads((root / "summary.json").read_text())
            self.assertEqual(report["paired"]["on_f16"]["test"]["ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
