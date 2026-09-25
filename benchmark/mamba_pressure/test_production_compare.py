import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import production_compare as sweep
import spark_run
from test_benchmark_utils import use_synthetic_geometry


class ProductionCompareTests(unittest.TestCase):
    def setUp(self):
        use_synthetic_geometry(self)

    def test_schedule_and_paired_report(self):
        calls = []

        async def fake(args, cfg, rep, target, workload):
            calls.append((args, cfg, rep, workload))
            target.mkdir(parents=True)
            metrics = {"ttft": float(cfg["compression"])}
            if cfg["compression"]:
                metrics["svd_batch_max"] = 2
            sweep.save(
                target / "result.json",
                dict(
                    config=cfg, repetition=rep, metrics=metrics, validation={"ok": True}
                ),
            )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(sweep, "one_run", fake):
                asyncio.run(sweep.experiment(root))
            self.assertEqual(len(calls), 12)
            for rep in range(5):
                rows = [r for r in calls if not r[0].pilot and r[2] == rep]
                self.assertEqual(
                    [r[1]["compression"] for r in rows],
                    [True, False] if rep % 2 else [False, True],
                )
                self.assertEqual(rows[0][3], rows[1][3])
                self.assertEqual(rows[0][1]["budget_bytes"], rows[1][1]["budget_bytes"])
            result = json.loads((root / "summary.json").read_text())
            self.assertEqual(
                result["paired_vs_off"]["deferral_cap2"]["ttft"]["ci95"], [1.0, 1.0]
            )

    def test_launcher_uses_unmodified_production_defaults(self):
        args = SimpleNamespace(port=31037, concurrency=4, seed=1)
        with tempfile.TemporaryDirectory() as temp:
            for cfg in (c for c in sweep.configs() if c["full_slots"] in (128, 32)):

                def capture(command, **kw):
                    self.assertEqual(Path(command[1]).name, "server.py")
                    self.assertNotIn("--disable-mamba-svd-prefill-deferral", command)
                    self.assertNotIn("--mamba-svd-worker-batch", command)
                    self.assertEqual(
                        "--mamba-svd-compression" in command, cfg["compression"]
                    )
                    self.assertNotIn("--context-length", command)
                    kw["stdout"].close()
                    raise RuntimeError("captured")

                with patch.object(spark_run.subprocess, "Popen", side_effect=capture):
                    with self.assertRaisesRegex(RuntimeError, "captured"):
                        asyncio.run(
                            spark_run.one_run(
                                args,
                                {
                                    **cfg,
                                    "production_defaults": True,
                                    "context_length": None,
                                },
                                0,
                                Path(temp) / cfg["label"],
                                [],
                            )
                        )


if __name__ == "__main__":
    unittest.main()
