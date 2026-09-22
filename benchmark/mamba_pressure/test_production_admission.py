"""Compatibility entrypoint; application regressions now live in registered CI."""

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "test/registered/unit/mem_cache")
)
from test_mamba_compression_admission import *  # noqa: E402,F403

if __name__ == "__main__":
    unittest.main()
