import asyncio
import unittest

from spark_run import replay_causal_sessions


class CausalReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_causality_concurrency_and_no_global_barrier(self):
        workload = [
            {"group": group, "round": turn} for turn in range(10) for group in range(3)
        ]
        fast_followup = asyncio.Event()
        completed = {g: -1 for g in range(3)}
        active = peak = 0

        async def execute(index, item):
            nonlocal active, peak
            group, turn = item["group"], item["round"]
            self.assertEqual(completed[group], turn - 1)
            active += 1
            peak = max(peak, active)
            if group == 0 and turn == 0:
                # A global turn barrier would deadlock here.
                await asyncio.wait_for(fast_followup.wait(), timeout=2)
            if group == 1 and turn == 1:
                fast_followup.set()
            await asyncio.sleep(0)
            completed[group] = turn
            active -= 1

        await replay_causal_sessions(workload, 2, execute)
        self.assertEqual(completed, {0: 9, 1: 9, 2: 9})
        self.assertEqual(peak, 2)

    async def test_rejects_missing_predecessor(self):
        async def execute(*args):
            self.fail("Invalid trace must not run")

        with self.assertRaisesRegex(ValueError, "contiguous"):
            await replay_causal_sessions([{"group": 0, "round": 1}], 1, execute)
