"""Request-level summaries for explicitly profiled runs."""

import json
import statistics

from spark_run import save


def overlap_ms(spans, start, end):
    intervals = sorted(
        (max(start, s["start_ns"]), min(end, s["end_ns"]))
        for s in spans
        if s["start_ns"] < end and s["end_ns"] > start
    )
    total = 0
    last = start
    for lo, hi in intervals:
        total += max(0, hi - max(last, lo))
        last = max(last, hi)
    return total / 1e6


def analyze(directory):
    rows = [
        json.loads(line)
        for line in (directory / "requests.jsonl").read_text().splitlines()
    ]
    spans = [
        json.loads(line)
        for p in directory.glob("spans-*.jsonl")
        for line in p.read_text().splitlines()
    ]
    if not spans:
        raise RuntimeError("Missing profiling spans")
    threshold = sorted(r["ttft_ms"] for r in rows)[int(0.95 * (len(rows) - 1))]
    details = []
    for row in rows:
        rid = row["meta_info"]["id"]
        own = [s for s in spans if s.get("rid") == rid or rid in s.get("rids", [])]
        if not any(s["kind"] == "forward" for s in own):
            raise RuntimeError(f"No forward spans for request {rid}")
        restores = [s for s in own if s["kind"] == "restore"]
        start, end = row["start_ns"], row["first_ns"]
        detail = dict(
            index=row["index"],
            round=row["round"],
            rid=rid,
            ttft_ms=row["ttft_ms"],
            tail=row["ttft_ms"] >= threshold,
            cache_path="compressed"
            if any(s["compressed"] for s in restores)
            else ("full" if row["cached_tokens"] else "miss"),
        )
        for kind in ("cache_lookup", "restore", "forward"):
            relevant = [
                s
                for s in own
                if s["kind"] == kind and s["start_ns"] < end and s["end_ns"] > start
            ]
            detail[kind + "_host_ms"] = overlap_ms(relevant, start, end)
            detail[kind + "_gpu_span_ms"] = sum(s.get("gpu_ms", 0) for s in relevant)
        for kind in ("svd", "commit_loop"):
            detail[kind + "_host_overlap_ms"] = overlap_ms(
                [s for s in spans if s["kind"] == kind], start, end
            )
        forward = [
            s["start_ns"]
            for s in own
            if s["kind"] == "forward" and start <= s["start_ns"] <= end
        ]
        detail["client_to_first_forward_ms"] = (
            (min(forward) - start) / 1e6 if forward else None
        )
        details.append(detail)
    save(directory / "tail_requests.json", details)
    summary = {}
    for category in ("all", "tail", "non_tail", "compressed", "full", "miss"):
        subset = [
            r
            for r in details
            if category == "all"
            or (category == "tail" and r["tail"])
            or (category == "non_tail" and not r["tail"])
            or r["cache_path"] == category
        ]
        summary[category] = {
            "n": len(subset),
            "means": {
                key: statistics.mean(r[key] for r in subset if r[key] is not None)
                for key in details[0]
                if key.endswith("_ms") and any(r[key] is not None for r in subset)
            },
        }
    save(directory / "tail_summary.json", summary)
    return summary
