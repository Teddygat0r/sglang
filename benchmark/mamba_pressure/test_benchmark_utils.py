"""Portable benchmark configuration and launcher regressions."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_utils as utils


def synthetic_geometry():
    # Small invented model geometry, independent of any captured experiment.
    return dict(
        full_state_bytes=2**20,
        kv_pool_bytes=2**30,
        temporal_shape=[2, 17, 4, 128, 128],
        temporal_element_bytes=4,
    )


def use_synthetic_geometry(test):
    temporary = tempfile.TemporaryDirectory()
    test.addCleanup(temporary.cleanup)
    path = Path(temporary.name) / "geometry.json"
    path.write_text(json.dumps(synthetic_geometry()))
    environment = patch.dict(os.environ, SGLANG_MAMBA_BENCHMARK_GEOMETRY=str(path))
    environment.start()
    test.addCleanup(environment.stop)


class BenchmarkUtilityTests(unittest.TestCase):
    def test_geometry_can_be_external_server_info(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "geometry.json"
            path.write_text(
                json.dumps(
                    {"internal_states": [{"cache_observations": synthetic_geometry()}]}
                )
            )
            with patch.dict(os.environ, SGLANG_MAMBA_BENCHMARK_GEOMETRY=str(path)):
                self.assertEqual(utils.load_geometry(), synthetic_geometry())
                for config in utils.configs():
                    charged = (
                        config["kv_pool_bytes"]
                        + (config["full_slots"] + 1)
                        * synthetic_geometry()["full_state_bytes"]
                        + config["compressed_slots"] * config["compressed_state_bytes"]
                        + config["staging_reserve_bytes"]
                    )
                    self.assertLessEqual(charged, config["budget_bytes"])

    def test_rejects_results_inside_source_checkout(self):
        with patch.object(
            sys,
            "argv",
            [
                "benchmark",
                "--results",
                str(utils.ROOT / "benchmark/mamba_pressure/results/should-not-exist"),
            ],
        ):
            with self.assertRaises(SystemExit) as error:
                utils.main(None, __file__)
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(
                (
                    utils.ROOT / "benchmark/mamba_pressure/results/should-not-exist"
                ).exists()
            )

    def test_real_detached_launcher_preserves_inputs(self):
        # Exercise the same launcher path with a tiny stand-in experiment, no GPU.
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            geometry = temp / "geometry.json"
            geometry.write_text(json.dumps(synthetic_geometry()))
            script = temp / "smoke.py"
            script.write_text(
                "from benchmark_utils import main, load_geometry, save\n"
                "async def experiment(root):\n"
                "    save(root / 'status.json', {'state': 'complete', 'geometry': load_geometry()})\n"
                "if __name__ == '__main__': main(experiment, __file__)\n"
            )
            env = dict(os.environ, PYTHONPATH=str(Path(utils.__file__).parent))
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--launch",
                    "--geometry",
                    str(geometry),
                    "--results",
                    str(temp / "output"),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            status = temp / "output/status.json"
            deadline = time.monotonic() + 10
            while not status.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(
                status.exists(), (temp / "output/supervisor.log").read_text()
            )
            self.assertEqual(
                json.loads(status.read_text()),
                dict(state="complete", geometry=synthetic_geometry()),
            )
            self.assertEqual(json.loads(result.stdout)["results"], str(temp / "output"))


if __name__ == "__main__":
    unittest.main()
