import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sharegpt_sweep as sweep
from test_benchmark_utils import use_synthetic_geometry


class Tokenizer:
    def encode(self, text):
        return list(text.encode())


class ShareGPTTests(unittest.TestCase):
    def setUp(self):
        use_synthetic_geometry(self)

    def test_filter_distinct_and_deterministic(self):
        def row(prompt, answer, role="human"):
            return {
                "id": prompt,
                "conversations": [
                    {"from": role, "value": prompt},
                    {"from": "gpt", "value": answer},
                ],
            }

        records = [
            row("prompt a", "answer"),
            row("prompt a", "other"),
            row("prompt b", "longer answer"),
            row("x", "too short"),
            row("prompt c", "x" * 1025),
            row("bad role", "answer", "gpt"),
            row("z" * 4090, "y" * 20),
        ]
        rows = sweep.sample(records, Tokenizer(), 2, 3)
        self.assertEqual(rows, sweep.sample(records, Tokenizer(), 2, 3))
        self.assertEqual(len({tuple(r["tokens"]) for r in rows}), 2)
        for r in rows:
            self.assertEqual(
                r["output"],
                len(records[r["dataset_index"]]["conversations"][1]["value"]),
            )
            self.assertEqual(r["round"], 0)
        with self.assertRaises(ValueError):
            sweep.sample(records, Tokenizer(), 3, 3)

    def test_equal_roomy_budgets(self):
        off, on = sweep.configs()
        geometry = sweep.load_geometry()
        for cfg in (off, on):
            self.assertEqual(cfg["budget_bytes"], 64 * 2**30)
            used = (
                cfg["kv_pool_bytes"]
                + (cfg["full_slots"] + 1) * geometry["full_state_bytes"]
                + cfg["compressed_slots"] * cfg["compressed_state_bytes"]
                + cfg["staging_reserve_bytes"]
            )
            self.assertLessEqual(used, cfg["budget_bytes"])
            self.assertLess(cfg["budget_bytes"] - used, geometry["full_state_bytes"])
            self.assertTrue(cfg["require_no_evictions"])
        self.assertEqual(on["full_slots"], 512)
        self.assertGreater(off["full_slots"], 1000)

    def test_complete_runner_pairs_and_reports(self):
        calls = []

        async def fake(args, cfg, rep, target, workload):
            calls.append((args, cfg, rep, workload))
            target.mkdir(parents=True)
            sweep.save(
                target / "result.json",
                dict(
                    config=cfg,
                    repetition=rep,
                    metrics={"ttft": float(cfg["compression"])},
                    validation={"ok": True},
                ),
            )

        records = [
            {
                "id": str(i),
                "conversations": [
                    {"from": "human", "value": f"prompt {i}"},
                    {"from": "gpt", "value": "answer"},
                ],
            }
            for i in range(300)
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            root.mkdir()
            data = Path(temp) / "data.json"
            data.write_text(json.dumps(records))
            original = Path.read_text

            def read(path, *a, **kw):
                if str(path) == "/proc/meminfo":
                    return "MemAvailable: 110000000 kB\n"
                return original(path, *a, **kw)

            with (
                patch.object(sweep, "DATASET", data),
                patch.object(sweep, "one_run", fake),
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=Tokenizer(),
                ),
                patch.object(Path, "read_text", read),
            ):
                asyncio.run(sweep.experiment(root))
            self.assertEqual(len(calls), 12)
            for rep in range(5):
                rows = [r for r in calls if not r[0].pilot and r[2] == rep]
                self.assertEqual(
                    [r[1]["compression"] for r in rows],
                    [True, False] if rep % 2 else [False, True],
                )
                self.assertEqual(rows[0][3], rows[1][3])
                self.assertEqual(len(rows[0][3]), 256)
            self.assertEqual(
                json.loads((root / "status.json").read_text())["state"], "complete"
            )


if __name__ == "__main__":
    unittest.main()
