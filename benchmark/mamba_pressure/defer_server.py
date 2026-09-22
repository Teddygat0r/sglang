"""Explicit nonblocking prefill-aware experiment; ordinary server is unchanged."""
import os
import sys
import server
from defer_prefill import install
install()

if __name__ == '__main__':
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    try:
        args = prepare_server_args(sys.argv[1:])
        if not args.disable_overlap_schedule or not args.disable_cuda_graph:
            raise ValueError('Deferral experiment requires graphs and overlap scheduling disabled')
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
