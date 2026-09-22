import argparse
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from sglang.srt.mem_cache.compression_admission import CompressionAdmission
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import ServerArgs
from test_mamba_compression_jobs import cache

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")


class AdmissionTests(unittest.TestCase):
    def test_cache_initialization_defaults_optout_and_fallback(self):
        from sglang.srt.mem_cache import mamba_radix_cache as module

        allocator = Mock(spec=module.TokenToKVPoolAllocator)
        allocator.device = "cuda"
        params = NS(
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=NS(mamba_pool=object()),
            page_size=1,
            disable=False,
            enable_mamba_extra_buffer=False,
            enable_metrics=False,
        )
        cases = [
            ({}, True),
            ({"mamba_svd_compression": False}, False),
            ({"disable_mamba_svd_prefill_deferral": True}, False),
            ({"pp_size": 2}, False),
            ({"disaggregation_mode": "prefill"}, False),
            ({"speculative_algorithm": "EAGLE"}, False),
            ({"dllm_algorithm": "test"}, False),
            ({"enable_pdmux": True}, False),
        ]
        for overrides, enabled in cases:
            with self.subTest(overrides=overrides):
                args = NS(**{"mamba_svd_compression": True, **overrides})
                with (
                    patch.object(module, "get_global_server_args", return_value=args),
                    patch.object(module.MambaRadixCache, "_init_compression_state"),
                    patch.object(module.MambaRadixCache, "reset"),
                ):
                    tree = module.MambaRadixCache(params)
                self.assertEqual(tree.compression_admission is not None, enabled)
                self.assertEqual(tree.svd_worker_batch, 2)

    def test_default_and_optout_cli(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(["--model-path", "dummy"])
        self.assertEqual(args.mamba_svd_worker_batch, 2)
        self.assertFalse(args.disable_mamba_svd_prefill_deferral)
        args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--disable-mamba-svd-prefill-deferral",
                "--mamba-svd-worker-batch",
                "8",
            ]
        )
        self.assertTrue(args.disable_mamba_svd_prefill_deferral)
        self.assertEqual(args.mamba_svd_worker_batch, 8)

    def test_cancel_and_shutdown_while_deferred(self):
        for reason in ("cancel", "stop"):
            admission = CompressionAdmission()
            admission.refresh(waiting=True)
            stop, cancel = threading.Event(), threading.Event()
            result = []
            worker = threading.Thread(
                target=lambda: result.append(admission.wait(stop, cancel.is_set))
            )
            worker.start()
            (cancel if reason == "cancel" else stop).set()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result, [False])

    def test_pending_selected_and_completion(self):
        admission = CompressionAdmission()
        done = threading.Event()
        event = NS(query=done.is_set)
        admission.begin_prefill()
        admission.finish_prefill(event)
        admission.refresh(False)
        self.assertFalse(admission._ready.is_set())
        done.set()
        admission.refresh(True)
        self.assertFalse(admission._ready.is_set())
        admission.refresh(False, selected=True)
        self.assertFalse(admission._ready.is_set())
        admission.refresh(False, selected=False)
        self.assertTrue(admission.wait(threading.Event(), lambda: False))

    def test_scheduler_stream_selection_exception_and_decode(self):
        for overlap in (False, True):
            for raises in (False, True):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.compression_admission = CompressionAdmission()
                scheduler.waiting_queue = []
                scheduler.chunked_req = None
                scheduler.enable_overlap = overlap
                scheduler.forward_stream = object()
                current_stream = object()
                event = NS(
                    record=lambda stream: recorded.append(stream), query=lambda: False
                )
                recorded = []
                batch = NS(
                    forward_mode=NS(is_extend_or_draft_extend_or_mixed=lambda: True)
                )

                def forward(*args):
                    self.assertFalse(scheduler.compression_admission._ready.is_set())
                    if raises:
                        raise RuntimeError("injected")
                    return "ok"

                with (
                    patch.object(scheduler, "_run_batch_impl", side_effect=forward),
                    patch.object(torch.cuda, "Event", return_value=event),
                    patch.object(
                        torch.cuda, "current_stream", return_value=current_stream
                    ),
                ):
                    if raises:
                        with self.assertRaisesRegex(RuntimeError, "injected"):
                            scheduler.run_batch(batch)
                    else:
                        self.assertEqual(scheduler.run_batch(batch), "ok")
                self.assertEqual(
                    recorded, [scheduler.forward_stream if overlap else current_stream]
                )
                batch.forward_mode.is_extend_or_draft_extend_or_mixed = lambda: False
                with (
                    patch.object(scheduler, "_run_batch_impl", return_value="decode"),
                    patch.object(
                        torch.cuda, "Event", side_effect=AssertionError("decode event")
                    ),
                ):
                    self.assertEqual(scheduler.run_batch(batch), "decode")

    def test_reset_cancels_worker_wait_without_admitting_stale_job(self):
        tree, node = cache()
        tree.compression_admission = CompressionAdmission()
        tree.compression_admission.refresh(True)
        tree._enqueue_compression(node)
        job = tree._compression_jobs[node.id]
        worker = threading.Thread(target=tree._compression_worker, daemon=True)
        with patch.object(tree, "_process_compression_batch") as process:
            worker.start()
            try:
                tree._reset_compression_state()
                self.assertTrue(job.cancelled.is_set())
            finally:
                tree._compression_stop_event.set()
                tree._compression_queue.put(None)
                worker.join(2)
            self.assertFalse(worker.is_alive())
            process.assert_not_called()

    def test_cuda_completion_event_is_nonblocking(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        admission = CompressionAdmission()
        stream = torch.cuda.Stream()
        admission.begin_prefill()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(20_000_000)
            event = torch.cuda.Event()
            event.record(stream)
        admission.finish_prefill(event)
        admission.refresh(False)
        self.assertFalse(admission._ready.is_set())
        # Synchronize only in this test, never in the production controller.
        event.synchronize()
        admission.refresh(False)
        self.assertTrue(admission.wait(threading.Event(), lambda: False))


if __name__ == "__main__":
    unittest.main()
