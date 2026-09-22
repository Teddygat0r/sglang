"""Regression checks for normal versus explicit profiling entry points."""

import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import spark_run
from benchmark_utils import configs
from test_benchmark_utils import use_synthetic_geometry


class IsolationTests(unittest.TestCase):
    def setUp(self):
        use_synthetic_geometry(self)

    def test_entrypoints_in_fresh_process(self):
        code = """
import sys
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
enqueue = MambaRadixCache._enqueue_compression
import server
from sglang.srt.managers.scheduler import Scheduler
assert 'tail_profile' not in sys.modules
assert 'pressure_admission' not in sys.modules
assert 'individual_variant' not in sys.modules
assert 'optimization_baseline' not in sys.modules
assert not getattr(Scheduler, '_tail_profile_installed', False)
assert MambaRadixCache._enqueue_compression is enqueue
import profile_server
assert Scheduler._tail_profile_installed
assert MambaRadixCache._enqueue_compression is enqueue
"""
        with tempfile.TemporaryDirectory() as temp:
            env = dict(
                os.environ,
                PRESSURE_TAIL_PROFILE=temp,
                PRESSURE_ADMISSION="pressure",
                PRESSURE_LOW_WATERMARK="invalid",
                PRESSURE_HIGH_WATERMARK="invalid",
                PRESSURE_INDIVIDUAL_VARIANT="removed-variant",
                PRESSURE_PRE_OPTIMIZATION="1",
                PYTHONPATH=str(spark_run.ROOT / "python")
                + os.pathsep
                + str(spark_run.HERE),
            )
            profile_only = """
import profile_server
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.managers.scheduler import Scheduler
assert MambaRadixCache._pressure_instrumented
assert Scheduler._tail_profile_installed
"""
            for entry in (code, profile_only):
                result = subprocess.run(
                    [sys.executable, "-c", entry],
                    env=env,
                    cwd=spark_run.ROOT,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_launch_routing_and_retired_policy(self):
        config = configs()[0]
        args = SimpleNamespace(port=31037, concurrency=4, seed=1)
        with tempfile.TemporaryDirectory() as temp:
            for profile in (False, True):

                def capture(command, **kw):
                    self.assertEqual(
                        Path(command[1]).name,
                        "profile_server.py" if profile else "server.py",
                    )
                    self.assertEqual("PRESSURE_TAIL_PROFILE" in kw["env"], profile)
                    self.assertNotIn("PRESSURE_ADMISSION", kw["env"])
                    kw[
                        "stdout"
                    ].close()  # This fake never starts a child to own the log lifecycle.
                    raise RuntimeError("captured launcher")

                with (
                    patch.dict(
                        os.environ,
                        PRESSURE_TAIL_PROFILE="stale",
                        PRESSURE_ADMISSION="pressure",
                    ),
                    patch.object(spark_run.subprocess, "Popen", side_effect=capture),
                ):
                    with self.assertRaisesRegex(RuntimeError, "captured launcher"):
                        asyncio.run(
                            spark_run.one_run(
                                args,
                                {**config, "tail_profile": profile},
                                0,
                                Path(temp) / str(profile),
                                [],
                            )
                        )
            with self.assertRaisesRegex(ValueError, "retired"):
                asyncio.run(
                    spark_run.one_run(
                        args,
                        {**config, "admission": "pressure"},
                        0,
                        Path(temp) / "retired",
                        [],
                    )
                )
            self.assertFalse((Path(temp) / "retired").exists())


if __name__ == "__main__":
    unittest.main()
