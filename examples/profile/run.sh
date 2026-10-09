#!/usr/bin/env bash
# Reproduce the experiments in REPORT.md on one local GPU.
#
#   ./run.sh [-g GPU_ID] [-p PORT] [STEP...]      (defaults: GPU 0, port 8765, all steps)
#
#   overhead     Q1-Q4  overhead.py for 3 sizes x {fresh, reused} workers  (~40 min)
#   throughput   Q5     throughput.py for {fresh, reused} workers          (~8 min)
#   discussion          exit_time.py and reuse_state.py                    (~2 min)
#   figures             report.py: REPORT.md figures and tables from out/
#
# Results (JSON), server logs and env.txt go to out/, which is not committed.
set -euo pipefail

GPU=0
PORT=8765
while getopts g:p: opt; do
  case "$opt" in
    g) GPU="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    *) exit 1 ;;
  esac
done
shift $((OPTIND - 1))
STEPS=" ${*:-overhead throughput discussion figures} "
URL="http://127.0.0.1:$PORT"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/out"
mkdir -p "$OUT"
cd "$REPO"

uv sync --locked --group server --python 3.12
PY=(.venv/bin/python)

# Local runs and the server must use the same physical GPU (nvidia-smi index).
export CUDA_DEVICE_ORDER=PCI_BUS_ID

{
  echo "== $(date -u +%FT%TZ) steps:$STEPS"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.current --format=csv
  lscpu | grep -E 'Model name|^CPU\(s\)'
  grep PRETTY_NAME /etc/os-release
  uname -r
  git rev-parse HEAD
} >>"$OUT/env.txt"

SERVER=""
CACHE=""
stop_server() {
  [[ -n "$SERVER" ]] && { kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null || true; }
  [[ -n "$CACHE" ]] && rm -rf "$CACHE"
  SERVER="" CACHE=""
}
trap stop_server EXIT

# start_server NAME [server options...]: server defaults plus a private disk
# cache, so earlier runs cannot pre-populate it.
start_server() {
  local name="$1"
  shift
  CACHE="$(mktemp -d)"
  .venv/bin/kcoral server --gpus "$GPU" --host 127.0.0.1 --port "$PORT" \
    --disk-cache-dir "$CACHE" --log-dir "$OUT/logs" "$@" >"$OUT/server_$name.log" 2>&1 &
  SERVER=$!
  until curl -fs "$URL/health" >/dev/null; do
    kill -0 "$SERVER" || { tail -n 30 "$OUT/server_$name.log"; exit 1; }
    sleep 1
  done
  echo "$name: $(grep -h pool_ready "$OUT/server_$name.log")" >>"$OUT/env.txt"
}

# --max-requests-per-worker: 1 (server default) gives every request a fresh
# worker process; 0 reuses workers.
declare -A MRPW=([fresh]=1 [reused]=0)

if [[ "$STEPS" == *" overhead "* ]]; then
  for MODE in fresh reused; do
    start_server "$MODE" --max-requests-per-worker "${MRPW[$MODE]}"
    for N in 1024 1048576 16777216; do  # 4 KiB, 4 MiB, 64 MiB per float32 tensor
      CUDA_VISIBLE_DEVICES="$GPU" "${PY[@]}" "$HERE/overhead.py" --url "$URL" --n "$N" \
        --out "$OUT/overhead_${MODE}_n$N.json"
    done
    stop_server
  done
fi

if [[ "$STEPS" == *" throughput "* ]]; then
  for MODE in fresh reused; do
    start_server "$MODE" --max-requests-per-worker "${MRPW[$MODE]}"
    "${PY[@]}" "$HERE/throughput.py" --url "$URL" --out "$OUT/throughput_$MODE.json"
    stop_server
  done
fi

if [[ "$STEPS" == *" discussion "* ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "${PY[@]}" "$HERE/exit_time.py" --out "$OUT/exit_time.json"
  start_server reuse_probe --workers-per-gpu 1 --max-requests-per-worker 0
  "${PY[@]}" "$HERE/reuse_state.py" --url "$URL" --out "$OUT/reuse_state.json"
  stop_server
fi

if [[ "$STEPS" == *" figures "* ]]; then
  uv run --no-project --with matplotlib==3.10.7 python "$HERE/report.py"
fi
