#!/usr/bin/env bash
# Full reproduction: install the locked environment, then for each server
# configuration start a local KCoral GPU server, run bench.py for each vector
# size, and stop the server.
#
#   ./run.sh [GPU_ID] [PORT]        (defaults: GPU 0, port 8765)
#
# Outputs go to examples/profile/out/: env.txt, mrpw<M>_n<N>.{json,txt}, summary.md
# and server logs.
set -euo pipefail

GPU="${1:-0}"
PORT="${2:-8765}"
URL="http://127.0.0.1:$PORT"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/out"
mkdir -p "$OUT"
cd "$REPO"

uv sync --locked --group server --python 3.12

# Local runs and the server must use the same physical GPU (nvidia-smi index).
export CUDA_DEVICE_ORDER=PCI_BUS_ID

{
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.current --format=csv
  lscpu | grep -E 'Model name|^CPU\(s\)'
  grep PRETTY_NAME /etc/os-release
  uname -r
  git rev-parse HEAD
} >"$OUT/env.txt"

SERVER=""
CACHE=""
stop_server() {
  [[ -n "$SERVER" ]] && { kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null || true; }
  [[ -n "$CACHE" ]] && rm -rf "$CACHE"
  SERVER="" CACHE=""
}
trap stop_server EXIT

# --max-requests-per-worker: 1 is the server default (fresh worker process per
# request); 0 reuses workers. Everything else uses server defaults, plus a
# private disk cache so earlier runs cannot pre-populate it.
for MRPW in 1 0; do
  CACHE="$(mktemp -d)"
  .venv/bin/kcoral server --gpus "$GPU" --host 127.0.0.1 --port "$PORT" \
    --max-requests-per-worker "$MRPW" --disk-cache-dir "$CACHE" \
    --log-dir "$OUT/logs" >"$OUT/server_mrpw$MRPW.log" 2>&1 &
  SERVER=$!
  until curl -fs "$URL/health" >/dev/null; do
    kill -0 "$SERVER" || { tail -n 30 "$OUT/server_mrpw$MRPW.log"; exit 1; }
    sleep 1
  done
  grep -h pool_ready "$OUT/server_mrpw$MRPW.log" >>"$OUT/env.txt"

  for N in 1024 1048576 16777216; do  # 4 KiB, 4 MiB, 64 MiB per float32 tensor
    CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python "$HERE/bench.py" --url "$URL" --n "$N" \
      --out "$OUT/mrpw${MRPW}_n$N.json" | tee "$OUT/mrpw${MRPW}_n$N.txt"
  done
  stop_server
done

.venv/bin/python "$HERE/summarize.py" >"$OUT/summary.md"
