#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
OUTDIR=${OUTDIR:-/home/joshuaz/sglang/ordered_svd_qwen35_0602}
SEED=${SEED:-20260602}
mkdir -p "$OUTDIR"
export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}

BENCH_ARGS=(
  --backend sglang
  --host "$HOST" --port "$PORT"
  --dataset-name generated-shared-prefix
  --gsp-num-groups 50
  --gsp-prompts-per-group 10
  --gsp-system-prompt-len 10240
  --gsp-question-len 256
  --gsp-output-len 128
  --gsp-ordered
  --max-concurrency 5
  --seed "$SEED"
  --flush-cache
  --disable-tqdm
)

launch_server() {
  local extra="$1" logf="$2"
  # shellcheck disable=SC2086
  "$PY" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --mem-fraction-static 0.85 \
    --max-running-requests 32 \
    --disable-cuda-graph \
    --disable-piecewise-cuda-graph \
    --disable-overlap-schedule \
    --enable-metrics \
    --random-seed 1234 \
    $extra > "$logf" 2>&1 &
  echo $!
}

wait_ready() {
  local pid="$1" logf="$2"
  for _ in $(seq 1 240); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "SERVER DIED during startup; tail of log:"
      tail -80 "$logf"
      return 1
    fi
    if curl -s "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "TIMEOUT waiting for server"
  tail -80 "$logf"
  return 1
}

capture_metrics() {
  local name="$1"
  curl -s "http://$HOST:$PORT/metrics" > "$OUTDIR/metrics_${name}.txt" || true
  grep -E 'sglang.*cache_hit_rate|cache_hit_rate|prefix_cache' "$OUTDIR/metrics_${name}.txt" > "$OUTDIR/cache_metrics_${name}.txt" || true
}

run_case() {
  local name="$1" extra="$2"
  local logf="$OUTDIR/server_${name}.log"
  local outf="$OUTDIR/bench_${name}.jsonl"
  local stdoutf="$OUTDIR/bench_${name}.stdout.log"
  rm -f "$outf" "$stdoutf" "$logf" "$OUTDIR/metrics_${name}.txt" "$OUTDIR/cache_metrics_${name}.txt"
  echo "=================================================================="
  echo "CASE: $name extra=[$extra]"
  echo "=================================================================="
  local pid
  pid=$(launch_server "$extra" "$logf")
  trap 'kill "$pid" 2>/dev/null || true' RETURN
  wait_ready "$pid" "$logf"
  "$PY" -m sglang.bench_serving "${BENCH_ARGS[@]}" --output-file "$outf" 2>&1 | tee "$stdoutf"
  capture_metrics "$name"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - RETURN
  sleep 5
}

run_case off ""
run_case on "--mamba-svd-compression --mamba-svd-rank 16"

"$PY" - <<'PY'
import json
from pathlib import Path
outdir = Path('/home/joshuaz/sglang/ordered_svd_qwen35_0602')
rows = []
for name in ['off','on']:
    path = outdir / f'bench_{name}.jsonl'
    if not path.exists():
        continue
    for line in path.read_text().splitlines():
        d = json.loads(line)
        si = d.get('server_info') or {}
        internal = (si.get('internal_states') or [{}])[0]
        rows.append({
            'case': name,
            'mamba_svd_compression': internal.get('mamba_svd_compression', si.get('mamba_svd_compression')),
            'mamba_svd_rank': internal.get('mamba_svd_rank', si.get('mamba_svd_rank')),
            'max_mamba_cache_size': internal.get('max_mamba_cache_size', si.get('max_mamba_cache_size')),
            'duration_s': d.get('duration'),
            'completed': d.get('completed'),
            'request_throughput': d.get('request_throughput'),
            'input_throughput': d.get('input_throughput'),
            'output_throughput': d.get('output_throughput'),
            'total_throughput': d.get('total_throughput'),
            'mean_ttft_ms': d.get('mean_ttft_ms'),
            'median_ttft_ms': d.get('median_ttft_ms'),
            'p99_ttft_ms': d.get('p99_ttft_ms'),
            'mean_tpot_ms': d.get('mean_tpot_ms'),
            'median_tpot_ms': d.get('median_tpot_ms'),
            'p99_tpot_ms': d.get('p99_tpot_ms'),
            'mean_itl_ms': d.get('mean_itl_ms'),
            'median_itl_ms': d.get('median_itl_ms'),
            'p95_itl_ms': d.get('p95_itl_ms'),
            'p99_itl_ms': d.get('p99_itl_ms'),
            'concurrency': d.get('concurrency'),
            'max_output_tokens_per_s': d.get('max_output_tokens_per_s'),
            'max_concurrent_requests': d.get('max_concurrent_requests'),
            'total_input_tokens': d.get('total_input_tokens'),
            'total_output_tokens': d.get('total_output_tokens'),
        })
(outdir / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows, indent=2))
PY
