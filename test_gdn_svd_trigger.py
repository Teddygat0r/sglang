"""Drive Qwen3_5ForCausalLM._compute_gdn_svd_trigger_mask through a simulated
request lifecycle and check the state machine behaves as specified:

  - Extend steps return None and clear baselines for every active req_pool_idx.
  - First decode after prefill fires SVD and sets baseline = current seq_len.
  - Subsequent decode steps don't re-fire until seq_len - baseline >= 1024.
  - When a req_pool_idx is reused for a new request, the next extend step
    wipes its baseline so the next decode fires again.
"""
import torch

from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM


class FakeMode:
    def __init__(self, kind):
        self.kind = kind

    def is_extend(self):
        # MIXED is covered by is_extend() in the real ForwardMode enum.
        return self.kind in ("extend", "mixed")

    def is_decode(self):
        return self.kind == "decode"

    def is_mixed(self):
        return self.kind == "mixed"


class FakeBatch:
    def __init__(
        self,
        mode,
        req_pool_indices,
        seq_lens_cpu=None,
        extend_seq_lens_cpu=None,
    ):
        self.forward_mode = FakeMode(mode)
        self.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.long)
        self.batch_size = len(req_pool_indices)
        self.seq_lens_cpu = (
            None if seq_lens_cpu is None else torch.tensor(seq_lens_cpu, dtype=torch.long)
        )
        self.extend_prefix_lens_cpu = None
        self.extend_seq_lens_cpu = extend_seq_lens_cpu


class FakeLM:
    pass


def run_test(name, fn):
    print(f"  {name} ...", end=" ")
    try:
        fn()
    except AssertionError as e:
        print(f"FAIL: {e}")
        raise
    print("ok")


compute = Qwen3_5ForCausalLM._compute_gdn_svd_trigger_mask


def t_extend_clears_and_returns_none():
    lm = FakeLM()
    lm._gdn_svd_baseline = {7: 2048, 9: 4096}
    # Both active reqs are in extend. Baselines for both must be cleared.
    mask = compute(lm, FakeBatch("extend", [7, 9]))
    assert mask is None, mask
    assert lm._gdn_svd_baseline == {}, lm._gdn_svd_baseline


def t_first_decode_fires_for_all():
    lm = FakeLM()
    lm._gdn_svd_baseline = {}
    # Two fresh requests hitting their first decode step.
    mask = compute(lm, FakeBatch("decode", [3, 11], [500, 1200]))
    assert mask.tolist() == [True, True], mask
    assert lm._gdn_svd_baseline == {3: 500, 11: 1200}, lm._gdn_svd_baseline


def t_second_decode_within_period_does_not_fire():
    lm = FakeLM()
    lm._gdn_svd_baseline = {3: 500}
    mask = compute(lm, FakeBatch("decode", [3], [501]))
    assert mask.tolist() == [False], mask
    assert lm._gdn_svd_baseline == {3: 500}, lm._gdn_svd_baseline


def t_1024_boundary_fires():
    lm = FakeLM()
    lm._gdn_svd_baseline = {3: 500}
    mask = compute(lm, FakeBatch("decode", [3], [500 + 1024]))
    assert mask.tolist() == [True], mask
    assert lm._gdn_svd_baseline == {3: 1524}, lm._gdn_svd_baseline


def t_1023_after_baseline_does_not_fire():
    lm = FakeLM()
    lm._gdn_svd_baseline = {3: 500}
    mask = compute(lm, FakeBatch("decode", [3], [500 + 1023]))
    assert mask.tolist() == [False], mask


def t_mixed_batch_only_eligible_fire():
    lm = FakeLM()
    lm._gdn_svd_baseline = {3: 500, 11: 2000}  # 11 already past 1024 from baseline
    # req 3: seq=600 → 100 since baseline → no fire
    # req 11: seq=3024 → 1024 since baseline → fire
    # req 42: first time → fire
    mask = compute(lm, FakeBatch("decode", [3, 11, 42], [600, 3024, 9]))
    assert mask.tolist() == [False, True, True], mask
    assert lm._gdn_svd_baseline == {3: 500, 11: 3024, 42: 9}, lm._gdn_svd_baseline


def t_slot_reuse_resets_baseline_on_next_extend():
    lm = FakeLM()
    lm._gdn_svd_baseline = {3: 9000}  # stale from a previous request on req_slot 3
    # Old request finished, req_slot=3 was freed and reused by a new request
    # which is now doing an extend.
    compute(lm, FakeBatch("extend", [3]))
    assert lm._gdn_svd_baseline == {}, lm._gdn_svd_baseline
    # First decode of the new request fires.
    mask = compute(lm, FakeBatch("decode", [3], [128]))
    assert mask.tolist() == [True], mask
    assert lm._gdn_svd_baseline == {3: 128}, lm._gdn_svd_baseline


def t_idle_mode_is_inert():
    lm = FakeLM()
    lm._gdn_svd_baseline = {3: 500}
    batch = FakeBatch("decode", [3], [501])
    batch.forward_mode = FakeMode("idle")
    mask = compute(lm, batch)
    assert mask is None, mask
    assert lm._gdn_svd_baseline == {3: 500}, lm._gdn_svd_baseline


def t_missing_seq_lens_cpu_returns_none():
    lm = FakeLM()
    lm._gdn_svd_baseline = {}
    mask = compute(lm, FakeBatch("decode", [3], None))
    assert mask is None, mask
    assert lm._gdn_svd_baseline == {}


def t_mixed_batch_extender_cleared_decoder_fires():
    """In a MIXED batch (chunked prefill + decode): extending requests (ext_len > 1)
    have their baseline cleared and do not fire; decoding requests (ext_len == 1)
    follow the normal boundary rule."""
    lm = FakeLM()
    # Stale baselines: req 7 is about to extend (should get cleared), req 11 is
    # decoding within the mixed batch and has crossed 1024 since its last fire,
    # req 42 is decoding for the first time after prefill.
    lm._gdn_svd_baseline = {7: 9999, 11: 2000}
    batch = FakeBatch(
        "mixed",
        req_pool_indices=[7, 11, 42],
        seq_lens_cpu=[500, 3024, 77],
        # ext_len > 1 means "still prefill-extending this step"; ext_len == 1 means "decoding"
        extend_seq_lens_cpu=[256, 1, 1],
    )
    mask = compute(lm, batch)
    assert mask.tolist() == [False, True, True], mask
    # Req 7: baseline popped (was 9999 → gone)
    assert 7 not in lm._gdn_svd_baseline
    # Req 11: fired, baseline updated to current seq_len
    assert lm._gdn_svd_baseline[11] == 3024
    # Req 42: first fire, baseline set
    assert lm._gdn_svd_baseline[42] == 77


def t_mixed_batch_only_extenders_no_fire():
    """Edge case: mixed mode where no request is actually decoding (all chunks).
    Baselines get cleared for every request, no SVD fires."""
    lm = FakeLM()
    lm._gdn_svd_baseline = {3: 500, 11: 1200}
    batch = FakeBatch(
        "mixed",
        req_pool_indices=[3, 11],
        seq_lens_cpu=[700, 1500],
        extend_seq_lens_cpu=[200, 300],  # both still prefill-chunking
    )
    mask = compute(lm, batch)
    assert mask.tolist() == [False, False], mask
    assert lm._gdn_svd_baseline == {}, lm._gdn_svd_baseline


def t_needs_eager_forward_stashes_mask_and_returns_bool():
    import types

    def make_fake():
        lm = FakeLM()
        lm._gdn_svd_baseline = {}
        lm._compute_gdn_svd_trigger_mask = types.MethodType(
            Qwen3_5ForCausalLM._compute_gdn_svd_trigger_mask, lm
        )
        return lm

    lm = make_fake()
    batch = FakeBatch("decode", [3], [5])
    out = Qwen3_5ForCausalLM.needs_eager_forward(lm, batch)
    assert out is True
    assert hasattr(batch, "gdn_svd_trigger_mask")
    assert batch.gdn_svd_trigger_mask.tolist() == [True]

    # Step with no fires — shouldn't stash a mask.
    lm2 = make_fake()
    lm2._gdn_svd_baseline = {3: 500}
    batch2 = FakeBatch("decode", [3], [501])
    out2 = Qwen3_5ForCausalLM.needs_eager_forward(lm2, batch2)
    assert out2 is False
    assert not hasattr(batch2, "gdn_svd_trigger_mask")


tests = [
    ("extend clears baselines and returns None", t_extend_clears_and_returns_none),
    ("first decode fires for every request", t_first_decode_fires_for_all),
    ("second decode within period does not fire", t_second_decode_within_period_does_not_fire),
    ("seq_len - baseline == 1024 fires", t_1024_boundary_fires),
    ("seq_len - baseline == 1023 does not fire", t_1023_after_baseline_does_not_fire),
    ("mixed batch — only eligible requests fire", t_mixed_batch_only_eligible_fire),
    ("slot reuse — extend wipes stale baseline, next decode fires", t_slot_reuse_resets_baseline_on_next_extend),
    ("mixed batch — extender cleared, decoder fires per boundary rule", t_mixed_batch_extender_cleared_decoder_fires),
    ("mixed batch — all extending, no fire, baselines cleared", t_mixed_batch_only_extenders_no_fire),
    ("idle mode inert", t_idle_mode_is_inert),
    ("missing seq_lens_cpu in decode returns None", t_missing_seq_lens_cpu_returns_none),
    ("needs_eager_forward stashes mask only on fire", t_needs_eager_forward_stashes_mask_and_returns_bool),
]

print("trigger-mask state machine")
for name, fn in tests:
    run_test(name, fn)
print("all trigger tests passed")
