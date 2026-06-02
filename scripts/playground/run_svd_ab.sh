#!/usr/bin/env bash
# A/B e2e benchmark: mamba SVD compression OFF vs ON.
# Launches Nemotron-H-8B (hybrid SSM -> MambaRadixCache), runs a shared-prefix
# workload that exercises compression (insert) + decompression (prefix hit),
# and records TTFT/ITL/throughput for each config.
set -u

PY=.venv/bin/python
MODEL=nvidia/Nemotron-H-8B-Base-8K
PORT=31007
HOST=127.0.0.1
OUTDIR=/tmp/svd_ab
mkdir -p "$OUTDIR"

# GB10 (sm_121a): bundled Triton ptxas (CUDA 12.8) only targets up to sm_120a.
# Point Triton at the system CUDA 13 ptxas which supports sm_121a.
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Shared-prefix workload: 16 groups x 8 prompts = 128 reqs, ~1k-token shared
# system prompt per group -> heavy radix reuse so the compressed path is hot.
BENCH_ARGS=(
  --backend sglang
  --host "$HOST" --port "$PORT"
  --dataset-name generated-shared-prefix
  --gsp-num-groups 16 --gsp-prompts-per-group 8
  --gsp-system-prompt-len 1024 --gsp-question-len 256 --gsp-output-len 128
  --max-concurrency 16
  --seed 1234
  --disable-tqdm
)

launch_server() {
  local extra="$1" logf="$2"
  # shellcheck disable=SC2086
  $PY -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --mem-fraction-static 0.85 \
    --max-running-requests 32 \
    --disable-cuda-graph \
    --disable-piecewise-cuda-graph \
    --random-seed 1234 \
    $extra > "$logf" 2>&1 &
  echo $!
}

wait_ready() {
  local pid="$1" logf="$2"
  for _ in $(seq 1 180); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "SERVER DIED during startup; tail of log:"; tail -30 "$logf"; return 1
    fi
    if curl -s "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "TIMEOUT waiting for server"; tail -30 "$logf"; return 1
}

run_case() {
  local name="$1" extra="$2"
  local logf="$OUTDIR/server_$name.log"
  local outf="$OUTDIR/bench_$name.jsonl"
  echo "=================================================================="
  echo "CASE: $name   (extra args: '$extra')"
  echo "=================================================================="
  rm -f "$outf"
  local pid; pid=$(launch_server "$extra" "$logf")
  if ! wait_ready "$pid" "$logf"; then
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 1
  fi
  grep -i "MambaRadix\|SVD compression\|svd-worker\|mamba-svd" "$logf" | head -5
  echo "--- warmup run (fills cache) ---"
  $PY -m sglang.bench_serving "${BENCH_ARGS[@]}" --num-prompts 32 >/dev/null 2>&1
  echo "--- measured run ---"
  $PY -m sglang.bench_serving "${BENCH_ARGS[@]}" --num-prompts 128 \
    --output-file "$outf" 2>&1 | grep -iE \
    "Successful|Mean TTFT|Median TTFT|P99 TTFT|Mean ITL|Median ITL|P99 ITL|Mean TPOT|Output token throughput|Total token throughput|Request throughput|cache|Benchmark duration"
  echo "--- compression-related log lines ---"
  grep -i "SVD compression enabled\|svd-worker thread started\|Async SVD failed" "$logf" | head -5
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 3
}

run_case "off" ""
run_case "on"  "--mamba-svd-compression --mamba-svd-rank 16"
echo "DONE. JSONL results in $OUTDIR/"
