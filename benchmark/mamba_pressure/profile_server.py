"""Explicit profiling-only server; normal server.py never installs these hooks."""

import os
import sys

import server  # noqa: F401 -- install base telemetry before profiling hooks
from tail_profile import install

if not os.environ.get("PRESSURE_TAIL_PROFILE"):
    raise ValueError(
        "profile_server.py requires PRESSURE_TAIL_PROFILE output directory"
    )
install()

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
