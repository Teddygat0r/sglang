import json
import tempfile
import unittest
from pathlib import Path

from profile_analysis import analyze, overlap_ms


class TailAnalysisTests(unittest.TestCase):
    def test_overlap_union_and_clip(self):
        spans = [dict(start_ns=a, end_ns=b) for a, b in [(0, 10), (5, 20), (25, 40)]]
        self.assertEqual(overlap_ms(spans, 7, 30), 18 / 1e6)

    def test_analysis_joins_request_and_forward(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = dict(
                meta_info={"id": "abc"},
                index=0,
                round=0,
                cached_tokens=0,
                ttft_ms=1,
                start_ns=0,
                first_ns=1000000,
            )
            (root / "requests.jsonl").write_text(json.dumps(row) + "\n")
            span = dict(
                kind="forward", rids=["abc"], start_ns=1, end_ns=900000, gpu_ms=0.8
            )
            (root / "spans-1.jsonl").write_text(json.dumps(span) + "\n")
            summary = analyze(root)
            self.assertEqual(summary["all"]["n"], 1)
            self.assertEqual(summary["miss"]["n"], 1)
            details = json.loads((root / "tail_requests.json").read_text())
            self.assertEqual(details[0]["forward_gpu_span_ms"], 0.8)
            span["rids"] = ["unrelated"]
            (root / "spans-1.jsonl").write_text(json.dumps(span) + "\n")
            with self.assertRaisesRegex(RuntimeError, "No forward spans"):
                analyze(root)


if __name__ == "__main__":
    unittest.main()
