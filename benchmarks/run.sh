#!/usr/bin/env bash
# Run the latency and throughput benchmarks against local servers on one GPU.
#
#   benchmarks/run.sh [--gpu-id ID] [--port PORT] [--name NAME]
#                     [--worker-type TYPES] [--benchmark BENCHMARKS]
#                     [--force-cache-miss] [--skip-summary]
#
#   --gpu-id ID          GPU index (nvidia-smi order), default 0
#   --port PORT          server port, default 8765
#   --name NAME          results go to benchmarks/results/NAME/,
#                        default: short commit (+ "-dirty")
#   --worker-type TYPES  comma-separated, default fresh,reused:
#                        fresh = --max-requests-per-worker 1 (server default),
#                        reused = 0
#   --benchmark BENCHMARKS
#                        comma-separated, default latency,throughput:
#     latency     latency.py per worker type, plus the baseline without KCoral
#                 (latency.py --no-kcoral, once, without a server)
#     throughput  throughput.py per worker type
#   --force-cache-miss   latency.py --force-cache-miss; skips the baseline, to
#                        which a cache miss does not apply
#   --skip-summary       do not run summary.py on the results directory
#
# Results (JSON, one file per configuration, tagged worker_type=TYPE), server
# logs and env.txt go to the results directory, which is not committed. Runs
# with the same NAME add to it, e.g. a second run with --benchmark latency
# --force-cache-miss adds cache misses next to cache hits.
# LATENCY_ARGS and THROUGHPUT_ARGS add options to each script, e.g.
# LATENCY_ARGS="--trials 5" for a quick check.
set -euo pipefail

GPU_ID=0
PORT=8765
NAME=""
WORKER_TYPE=(fresh reused)
BENCHMARK=(latency throughput)
FORCE_CACHE_MISS=""
SKIP_SUMMARY=""

# --max-requests-per-worker for each worker type: 1 (server default) gives every
# request a fresh worker process; 0 reuses workers.
declare -A MRPW=([fresh]=1 [reused]=0)

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
PY="$REPO/.venv/bin/python"
URL=""
OUT=""
SERVER=""
DISK_CACHE=""

log() { echo "[run.sh $(date +%T)] $*" >&2; }
die() { log "error: $*"; exit 1; }
usage() { sed -n '2,/^set -euo/{/^set -euo/d; s/^# \{0,1\}//; p}' "$0"; }

parse_args() {
  while (($#)); do
    case "$1" in
      --gpu-id) GPU_ID="${2:?$1 needs a value}"; shift 2 ;;
      --port) PORT="${2:?$1 needs a value}"; shift 2 ;;
      --name) NAME="${2:?$1 needs a value}"; shift 2 ;;
      --worker-type) IFS=, read -ra WORKER_TYPE <<<"${2:?$1 needs a value}"; shift 2 ;;
      --benchmark) IFS=, read -ra BENCHMARK <<<"${2:?$1 needs a value}"; shift 2 ;;
      --force-cache-miss) FORCE_CACHE_MISS=--force-cache-miss; shift ;;
      --skip-summary) SKIP_SUMMARY=1; shift ;;
      -h | --help) usage; exit ;;
      *) die "unknown option: $1 (see --help)" ;;
    esac
  done
  local type b
  for type in "${WORKER_TYPE[@]}"; do
    [[ -n "${MRPW[$type]:-}" ]] || die "unknown worker type: $type"
  done
  for b in "${BENCHMARK[@]}"; do
    [[ "$b" == latency || "$b" == throughput ]] || die "unknown benchmark: $b"
  done
  URL="http://127.0.0.1:$PORT"
  if [[ -z "$NAME" ]]; then
    NAME="$(git -C "$REPO" rev-parse --short HEAD)"
    [[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] || NAME="$NAME-dirty"
  fi
  OUT="$HERE/results/$NAME"
}

selected() { [[ " ${BENCHMARK[*]} " == *" $1 "* ]]; }

# Install the locked environment and append the machine and commit to env.txt.
setup() {
  mkdir -p "$OUT"
  log "Results go to $OUT"
  log "Installing the locked environment..."
  (cd "$REPO" && uv sync --locked --group server --python 3.12)
  # Runs without KCoral and the server must use the same physical GPU (nvidia-smi index).
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  {
    echo "== $(date -u +%FT%TZ) benchmark: ${BENCHMARK[*]} $FORCE_CACHE_MISS"
    nvidia-smi -i "$GPU_ID" --format=csv \
      --query-gpu=index,name,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.current
    lscpu | grep -E 'Model name|^CPU\(s\)'
    grep PRETTY_NAME /etc/os-release
    uname -r
    git -C "$REPO" rev-parse HEAD
    git -C "$REPO" status --porcelain --untracked-files=no
  } >>"$OUT/env.txt"
}

# start_server NAME WORKER_TYPE: server defaults plus the worker type and a
# private disk cache, so earlier runs cannot pre-populate it. Logs go to
# OUT/server_NAME.log and OUT/logs/NAME/.
start_server() {
  local name="$1" type="$2"
  log "Starting a KCoral server ($type workers) on $URL..."
  DISK_CACHE="$(mktemp -d)"
  rm -rf "$OUT/logs/$name"
  "$REPO/.venv/bin/kcoral" server --gpus "$GPU_ID" --host 127.0.0.1 --port "$PORT" \
    --max-requests-per-worker "${MRPW[$type]}" --disk-cache-dir "$DISK_CACHE" \
    --log-dir "$OUT/logs/$name" >"$OUT/server_$name.log" 2>&1 &
  SERVER=$!
  until curl -fs "$URL/health" >/dev/null; do
    kill -0 "$SERVER" 2>/dev/null || {
      tail -n 30 "$OUT/server_$name.log" >&2
      die "the server exited; see $OUT/server_$name.log"
    }
    sleep 1
  done
  local ready
  ready="$(grep -h pool_ready "$OUT/server_$name.log" || true)"
  log "Server ready: ${ready#*pool_ready }"
  echo "$name ($type workers): $ready" >>"$OUT/env.txt"
}

stop_server() {
  if [[ -n "$SERVER" ]]; then
    log "Stopping the server..."
    kill "$SERVER" 2>/dev/null || true
    wait "$SERVER" 2>/dev/null || true
  fi
  [[ -n "$DISK_CACHE" ]] && rm -rf "$DISK_CACHE"
  SERVER="" DISK_CACHE=""
}

run_latency() {
  if [[ -z "$FORCE_CACHE_MISS" ]]; then
    log "Latency: baseline without KCoral..."
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$HERE/latency.py" --no-kcoral \
      --out "$OUT/latency_no-kcoral.json" ${LATENCY_ARGS:-}
  fi
  local cache type
  cache=$([[ -n "$FORCE_CACHE_MISS" ]] && echo miss || echo hit)
  for type in "${WORKER_TYPE[@]}"; do
    start_server "latency_${type}_$cache" "$type"
    log "Latency: $type workers, cache $cache..."
    "$PY" "$HERE/latency.py" --url "$URL" $FORCE_CACHE_MISS --tag "worker_type=$type" \
      --out "$OUT/latency_${type}_$cache.json" ${LATENCY_ARGS:-}
    stop_server
  done
}

run_throughput() {
  local type
  for type in "${WORKER_TYPE[@]}"; do
    start_server "throughput_$type" "$type"
    log "Throughput: $type workers..."
    "$PY" "$HERE/throughput.py" --url "$URL" --tag "worker_type=$type" \
      --out "$OUT/throughput_$type.json" ${THROUGHPUT_ARGS:-}
    stop_server
  done
}

run_summary() {
  log "Summarizing $OUT..."
  (cd "$REPO" && uv run --locked --group server --python 3.12 --with matplotlib==3.10.7 \
    python "$HERE/summary.py" "$OUT")
}

main() {
  parse_args "$@"
  trap stop_server EXIT
  setup
  if selected latency; then run_latency; fi
  if selected throughput; then run_throughput; fi
  if [[ -z "$SKIP_SUMMARY" ]]; then run_summary; fi
  log "Done. Results are in $OUT"
}

main "$@"
