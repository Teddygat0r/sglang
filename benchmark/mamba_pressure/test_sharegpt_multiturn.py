import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sharegpt_multiturn as sweep
import spark_run
from test_benchmark_utils import use_synthetic_geometry


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(
            "".join(m["role"] + ":" + m["content"] for m in messages) + "assistant:"
        )


def records(count, turns=3):
    return [
        dict(
            id=str(i),
            conversations=[
                {"from": role, "value": f"{role} text for {i}, turn {turn}"}
                for turn in range(turns)
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
        self.check_schedule(pressure=False)

    def test_pressure_schedule(self):
        self.check_schedule(pressure=True)

    def test_pressure_launcher_and_startup_require_checkpointing(self):
        args = SimpleNamespace(port=31037, concurrency=8, seed=1)
        with tempfile.TemporaryDirectory() as temp:
            for config in sweep.pressure_configs():

                def capture(command, **kwargs):
                    self.assertNotIn("--context-length", command)
                    for option, value in (
                        ("--mamba-scheduler-strategy", "extra_buffer"),
                        ("--mamba-track-interval", "256"),
                        ("--max-running-requests", "8"),
                    ):
                        self.assertEqual(command[command.index(option) + 1], value)
                    kwargs["stdout"].close()
                    raise RuntimeError("captured")

                with patch.object(spark_run.subprocess, "Popen", side_effect=capture):
                    with self.assertRaisesRegex(RuntimeError, "captured"):
                        asyncio.run(
                            spark_run.one_run(
                                args, config, 0, Path(temp) / config["label"], []
                            )
                        )
                initial = dict(
                    mamba_extra_buffer=True,
                    mamba_track_interval=256,
                    max_running_requests=8,
                )
                spark_run.validate_checkpoint_config(initial, config, 8)
                for key, invalid in (
                    ("mamba_extra_buffer", False),
                    ("mamba_track_interval", 512),
                    ("max_running_requests", 4),
                ):
                    with self.assertRaises(RuntimeError):
                        spark_run.validate_checkpoint_config(
                            {**initial, key: invalid}, config, 8
                        )

    def test_ten_turns_preserve_long_inputs_and_outputs(self):
        data = records(1, turns=10)
        data[0]["conversations"][0]["value"] = "x" * 5000
        data[0]["conversations"][1]["value"] = "y" * 2000
        selected = sweep.sample_sessions(
            data, Tokenizer(), 1, 0, turns=10, uncapped=True, native_chat=True
        )
        self.assertEqual(len(selected[0]), 10)
        self.assertGreater(len(selected[0][0]["tokens"]), 5000)
        self.assertEqual(selected[0][0]["output"], 2000)
        self.assertGreater(len(selected[0][-1]["tokens"]), 7000)
        self.assertGreater(
            sweep.describe(selected, require_roomy=False)["max_request_tokens"], 7000
        )

    def check_schedule(self, pressure):
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
            data.write_text(json.dumps(records(80, turns=10 if pressure else 3)))
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
                patch(
                    "transformers.AutoConfig.from_pretrained",
                    return_value=SimpleNamespace(
                        get_text_config=lambda: SimpleNamespace(
                            max_position_embeddings=262144
                        )
                    ),
                ),
                patch.object(Path, "read_text", read),
            ):
                asyncio.run(sweep.experiment(root, pressure=pressure))
            self.assertEqual(len(calls), 12)
            self.assertTrue(all(r[0].pilot for r in calls[:2]))
            if pressure:
                for args, cfg, _, _ in calls:
                    self.assertEqual(args.concurrency, 8)
                    self.assertEqual(cfg["concurrency"], 8)
                    self.assertEqual(cfg["mamba_scheduler_strategy"], "extra_buffer")
                    self.assertEqual(cfg["mamba_track_interval"], 256)
                    self.assertFalse(cfg.get("require_no_evictions", False))
                    self.assertTrue(cfg["prefix_reuse_pilot_only"])
                    self.assertIsNone(cfg["context_length"])
                    self.assertEqual(cfg["replay_schedule"], "causal_sessions")
                    self.assertEqual(
                        cfg["full_slots"], 32 if cfg["compression"] else 128
                    )
                self.assertEqual(
                    calls[0][1]["budget_bytes"], calls[1][1]["budget_bytes"]
                )
            for rep in range(5):
                rows = [r for r in calls if not r[0].pilot and r[2] == rep]
                self.assertEqual(rows[0][3], rows[1][3])
                self.assertEqual(len(rows[0][3]), 640 if pressure else 192)
                self.assertEqual(
                    [r[1]["compression"] for r in rows],
                    [True, False] if rep % 2 else [False, True],
                )
            self.assertEqual(
                json.loads((root / "status.json").read_text())["state"], "complete"
            )


if __name__ == "__main__":
    unittest.main()
