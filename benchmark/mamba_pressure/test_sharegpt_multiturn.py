import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sharegpt_multiturn as sweep
from test_benchmark_utils import use_synthetic_geometry


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


def records(count):
    return [
        dict(
            id=str(i),
            conversations=[
                {"from": role, "value": f"{role} text for {i}, turn {turn}"}
                for turn in range(3)
                for role in ("human", "gpt")
            ],
        )
        for i in range(count)
    ]


class MultiTurnTests(unittest.TestCase):
    def setUp(self):
        use_synthetic_geometry(self)

    def test_exact_prefixes_and_causality(self):
        selected = sweep.sample_sessions(records(12), Tokenizer(), 8, 7)
        trace = sweep.interleave(selected, 7)
        self.assertEqual(len(trace), 24)
        self.assertEqual(trace, sweep.interleave(selected, 7))
        seen = {}
        for row in trace:
            prior = seen.get(row["group"])
            if row["round"]:
                self.assertEqual(row["tokens"][: len(prior["tokens"])], prior["tokens"])
                self.assertEqual(row["prior_input_tokens"], len(prior["tokens"]))
                self.assertEqual(row["round"], prior["round"] + 1)
            else:
                self.assertIsNone(prior)
            seen[row["group"]] = row
        stats = sweep.describe(selected)
        self.assertGreater(stats["prior_input_prefix_fraction"], 0.1)

    def test_filter_does_not_truncate(self):
        dataset = records(4)
        dataset[0]["conversations"][3]["value"] = "x" * 1025
        dataset[1]["conversations"][4]["from"] = "gpt"
        dataset[2]["conversations"] = dataset[2]["conversations"][:2]
        selected = sweep.sample_sessions(dataset, Tokenizer(), 1, 0)
        self.assertEqual(selected[0][0]["dataset_index"], 3)
        with self.assertRaises(ValueError):
            sweep.sample_sessions(dataset, Tokenizer(), 2, 0)

    def test_complete_schedule(self):
        calls = []

        async def fake(args, cfg, rep, target, workload):
            calls.append((args, cfg, rep, workload))
            target.mkdir(parents=True)
            sweep.save(
                target / "result.json",
                dict(
                    config=cfg,
                    repetition=rep,
                    metrics={"hit": 0.5},
                    validation={"ok": True},
                ),
            )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            root.mkdir()
            data = Path(temp) / "data.json"
            data.write_text(json.dumps(records(80)))
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
            self.assertTrue(all(r[0].pilot for r in calls[:2]))
            for rep in range(5):
                rows = [r for r in calls if not r[0].pilot and r[2] == rep]
                self.assertEqual(rows[0][3], rows[1][3])
                self.assertEqual(len(rows[0][3]), 192)
                self.assertEqual(
                    [r[1]["compression"] for r in rows],
                    [True, False] if rep % 2 else [False, True],
                )
            self.assertEqual(
                json.loads((root / "status.json").read_text())["state"], "complete"
            )


if __name__ == "__main__":
    unittest.main()
