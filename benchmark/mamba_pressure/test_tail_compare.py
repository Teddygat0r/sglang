import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import spark_run
import tail_compare as sweep
from test_benchmark_utils import use_synthetic_geometry


def detail(index, tail, value):
    return dict(
        index=index,
        round=0,
        tail=tail,
        ttft_ms=value,
        client_to_first_forward_ms=value / 2,
        cache_path="miss",
    )


class TailCompareTests(unittest.TestCase):
    def setUp(self):
        use_synthetic_geometry(self)

    def test_matched_cohorts(self):
        before = [detail(0, True, 10), detail(1, False, 2)]
        after = [detail(1, True, 5), detail(0, False, 4)]
        result = sweep.matched_metrics(before, after)
        self.assertEqual(result["eager_tail/ttft_ms/delta"], -6)
        self.assertEqual(result["union_tail/ttft_ms/delta"], -1.5)
        with self.assertRaises(ValueError):
            sweep.matched_metrics(before, after[:1])
        with self.assertRaises(ValueError):
            sweep.matched_metrics(before, [after[0], after[0]])

    def test_launcher_flags(self):
        args = SimpleNamespace(port=31037, concurrency=4, seed=1)
        with tempfile.TemporaryDirectory() as temp:
            for cfg in sweep.design():

                def capture(command, **kw):
                    self.assertEqual(
                        Path(command[1]).name,
                        "profile_server.py" if cfg["tail_profile"] else "server.py",
                    )
                    self.assertEqual(
                        "PRESSURE_TAIL_PROFILE" in kw["env"], cfg["tail_profile"]
                    )
                    self.assertEqual(
                        "--disable-mamba-svd-prefill-deferral" in command,
                        not cfg["production_defaults"],
                    )
                    if cfg["production_defaults"]:
                        self.assertNotIn("--mamba-svd-worker-batch", command)
                    else:
                        self.assertEqual(
                            command[command.index("--mamba-svd-worker-batch") + 1], "8"
                        )
                    kw["stdout"].close()
                    raise RuntimeError("captured")

                with patch.object(spark_run.subprocess, "Popen", side_effect=capture):
                    with self.assertRaisesRegex(RuntimeError, "captured"):
                        asyncio.run(
                            spark_run.one_run(
                                args, cfg, 0, Path(temp) / cfg["label"], []
                            )
                        )

    def test_schedule_report_and_abort(self):
        calls = []

        async def fake(args, cfg, rep, target, workload):
            calls.append((args, cfg, rep, workload))
            target.mkdir(parents=True)
            sweep.save(
                target / "result.json",
                dict(
                    config=cfg,
                    repetition=rep,
                    metrics={"ttft_ms": 2 if cfg["production_defaults"] else 3},
                    validation={"ok": True},
                ),
            )
            sweep.save(
                target / "tail_requests.json",
                [detail(0, True, 2 if cfg["production_defaults"] else 3)],
            )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (
                patch.object(sweep, "one_run", fake),
                patch.object(sweep, "analyze") as analyze,
            ):
                asyncio.run(sweep.experiment(root))
                self.assertEqual(analyze.call_count, 12)
            self.assertEqual(len(calls), 22)
            self.assertTrue(all(c[0].pilot for c in calls[:2]))
            for rep in range(5):
                rows = [c for c in calls if not c[0].pilot and c[2] == rep]
                expected = sweep.design()[:: (-1 if rep % 2 else 1)]
                self.assertEqual([r[1] for r in rows], expected)
                self.assertTrue(all(r[3] == rows[0][3] for r in rows))
                self.assertEqual(len(rows[0][3]), 192)
                self.assertEqual(len({r[1]["budget_bytes"] for r in rows}), 1)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(
                summary["after_control minus eager_control (n=5)"]["ttft_ms"]["ci95"],
                [-1, -1],
            )
            self.assertEqual(
                summary["Matched profile cohorts (n=5)"]["eager_tail/ttft_ms/delta"][
                    "mean"
                ],
                -1,
            )
            self.assertEqual(
                json.loads((root / "status.json").read_text())["state"], "complete"
            )
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(
                sweep, "one_run", side_effect=RuntimeError("pilot failed")
            ) as run:
                with self.assertRaisesRegex(RuntimeError, "pilot failed"):
                    asyncio.run(sweep.experiment(Path(temp)))
                self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
