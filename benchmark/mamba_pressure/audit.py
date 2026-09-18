"""Check the final raw requests against traces, memory caps, and warmup isolation."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main(directory):
    args = read(directory / "arguments.json")
    expected = len(args["budgets"]) * args["repetitions"] * 2
    files = sorted(directory.glob("b*_r*_*/result.json"))
    assert len(files) == expected, (len(files), expected)
    checks = []
    for path in files:
        run = read(path)
        folder = path.parent
        before, after = read(folder / "before.json"), read(folder / "after.json")
        trace = read(directory / f"trace_{run['repetition']}.json")
        rows = [json.loads(line) for line in (folder / "requests.jsonl").read_text().splitlines()]
        config = run["config"]
        assert len(rows) == len(trace) == args["groups"] * args["rounds"]
        assert before["full_free_slots"] == config["full_slots"]
        assert before["compressed_entries"] == before["compression_pending"] == 0
        assert before.get("evicted_entries", 0) == 0
        assert after["peak_cache_state_bytes"] <= config["budget_bytes"]
        assert after["mamba_pool_bytes"] == after["full_state_bytes"] * (config["full_slots"] + 1)
        assert after["persistent_cache_bytes"] == sum(after[key] for key in
            ["kv_pool_bytes", "mamba_pool_bytes", "compressed_pool_bytes"])
        assert after.get("staging_peak_bytes", 0) <= config["staging_reserve_bytes"]
        for index, (row, item) in enumerate(zip(rows, trace)):
            assert row["index"] == index
            assert (row["group"], row["round"]) == (item["group"], item["round"])
            assert row["input_tokens"] == len(item["tokens"])
            assert row["output_tokens"] == args["output"]
            assert row["meta_info"]["total_retractions"] == 0
            assert 0 <= row["cached_tokens"] <= args["prefix"]
            if row["round"] == 0:
                assert row["cached_tokens"] == 0
        metrics = run["metrics"]
        assert metrics["recomputed_prefix_tokens"] == sum(
            args["prefix"] - row["cached_tokens"] for row in rows if row["round"] > 0)
        assert abs(metrics["token_cache_hit_rate"] - sum(r["cached_tokens"] for r in rows)
                   / sum(r["input_tokens"] for r in rows)) < 1e-12
        checks.append({"run": folder.name, "passed": True, "requests": len(rows),
                       "cache_budget_bytes": config["budget_bytes"],
                       "peak_cache_state_bytes": after["peak_cache_state_bytes"]})
    output = {"passed": True, "runs": len(checks), "requests": sum(c["requests"] for c in checks),
              "checks": checks}
    (directory / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "checks"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    main(parser.parse_args().results)
