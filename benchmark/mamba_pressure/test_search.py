"""Exercise the complete supervisor startup and phase orchestration without GPUs."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import search


class SearchTests(unittest.TestCase):
    def test_complete_supervisor_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = root / "prior"
            prior.mkdir()
            (prior / "supervisor.json").write_text('{"pid": 123456789}')
            (prior / "status.json").write_text('{"state": "complete"}')
            calls = []
            async def phase(folder, name, configs, repetitions, seed):
                calls.append((name, len(configs), repetitions, seed))
                for key in configs:
                    for rep in range(repetitions):
                        target = folder / name / f"{key}_r{rep}"
                        target.mkdir(parents=True)
                        (target / "audit.json").write_text('{"passed": true}')
                return next(iter(configs.values()))
            with patch.object(search, "PRIOR", prior), patch.object(search, "phase", phase):
                asyncio.run(search.experiment(root))
            self.assertEqual(json.loads((root / "status.json").read_text())["audited_runs"], 42)
            protocol = json.loads((root / "protocol.json").read_text())
            self.assertIn("python/sglang/srt/mem_cache/mamba_radix_cache.py", protocol["hashes"])
            self.assertEqual(calls, [("allocation", 3, 3, 20261001), ("policy", 3, 3, 20261101), ("validation", 4, 6, 20261201)])

    def test_allocation_ceilings(self):
        for budget, slots in [(192, 16), (192, 32), (192, 64), (24, 8), (24, None), (192, None)]:
            c = search.config(budget, slots)
            charged = search.G["kv_pool_bytes"] + (c["full_slots"] + 1) * search.FULL + c["compressed_slots"] * search.COMPRESSED + c["staging_reserve_bytes"]
            self.assertLessEqual(charged, c["budget_bytes"])


if __name__ == "__main__":
    unittest.main()
