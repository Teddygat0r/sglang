"""Normal SGLang server with benchmark-only telemetry in spawned workers."""

import os
import sys
from numbers import Integral

from instrumentation import install

install()

# Pool byte-count helpers can return numpy integers; normalize at the API edge.
from sglang.srt.managers.scheduler import Scheduler

_get_internal_state = Scheduler.get_internal_state


def get_internal_state(scheduler, request):
    result = _get_internal_state(scheduler, request)
    metrics = result.internal_state.get("cache_observations")
    if metrics is not None:
        result.internal_state["cache_observations"] = {
            key: int(value) if isinstance(value, Integral) else value
            for key, value in metrics.items()
        }
    return result


Scheduler.get_internal_state = get_internal_state

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
