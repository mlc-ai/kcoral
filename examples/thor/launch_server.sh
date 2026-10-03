#!/usr/bin/env bash
# Start (or stop) a KCoral GPU server on a Thor host over SSH.
#
#   ./launch_server.sh <thor-host> [--port PORT] [--dir DIR]
#   ./launch_server.sh <thor-host> --stop [--dir DIR]
#
# Start copies this KCoral checkout's tracked files to <thor-host>:DIR, stops the
# server previously started there, installs the locked server environment
# (uv sync --locked --group gpu), starts `kcoral server` bound to 127.0.0.1:PORT,
# waits for /health and prints the SSH tunnel command for this machine.
#
# DIR is relative to the remote home directory; it defaults to
# .cache/kcoral-thor-example, which only this script uses and which is safe to
# delete. The server never listens on a public interface; reach it through the
# tunnel. Requires: git and rsync locally; uv on the Thor host (see README).
set -euo pipefail

usage() {
  echo "usage: $0 <thor-host> [--port PORT] [--dir DIR] [--stop]" >&2
  exit "$1"
}

HOST=""
PORT=8000
DIR=.cache/kcoral-thor-example
MODE=start
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --dir) DIR="$2"; shift 2 ;;
    --stop) MODE=stop; shift ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option: $1" >&2; usage 1 ;;
    *) HOST="$1"; shift ;;
  esac
done
[[ -n "$HOST" ]] || usage 1

if [[ "$MODE" == start ]]; then
  REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
  echo "==> copying $REPO to $HOST:$DIR"
  ssh "$HOST" mkdir -p "$DIR"
  git -C "$REPO" ls-files -z | rsync -a --from0 --files-from=- "$REPO/" "$HOST:$DIR/"
fi

ssh "$HOST" bash -s -- "$MODE" "$DIR" "$PORT" <<'REMOTE'
set -euo pipefail
MODE="$1" DIR="$2" PORT="$3"
cd "$DIR" 2>/dev/null || { echo "nothing to stop: $DIR does not exist"; exit 0; }

# Stop the server recorded in logs/server.pid, if it is running.
if [[ -f logs/server.pid ]] && kill -0 "$(cat logs/server.pid)" 2>/dev/null; then
  echo "==> stopping the server (pid $(cat logs/server.pid))"
  kill "$(cat logs/server.pid)"
  while kill -0 "$(cat logs/server.pid)" 2>/dev/null; do sleep 1; done
elif [[ "$MODE" == stop ]]; then
  echo "no server is running from $DIR"
fi
rm -f logs/server.pid
[[ "$MODE" == start ]] || exit 0

echo "==> installing the locked server environment"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  echo "uv is not installed on this host; install it first (see README)" >&2
  exit 1
fi
uv sync --locked --group gpu --python 3.12

echo "==> starting kcoral server (127.0.0.1:$PORT)"
mkdir -p logs
# TVM 0.26's NVRTC path looks for the CCCL headers under targets/$(uname -m)-linux,
# but JetPack installs the toolkit as targets/sbsa-linux. Point NVRTC at them.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ -d "$CUDA_HOME/include/cccl" ]]; then
  export TVM_CUDA_NVRTC_EXTRA_OPTS="-I$CUDA_HOME/include/cccl"
fi
setsid nohup .venv/bin/kcoral server --device gpu --gpus 0 --host 127.0.0.1 --port "$PORT" \
  --log-dir logs >logs/server.out 2>&1 </dev/null &
echo $! >logs/server.pid
for _ in $(seq 120); do
  if curl -fs --max-time 2 "http://127.0.0.1:$PORT/health"; then
    echo
    exit 0
  fi
  sleep 1
done
echo "server did not become healthy; last log lines:" >&2
tail -n 30 logs/server.out >&2
exit 1
REMOTE

if [[ "$MODE" == stop ]]; then
  echo "To remove the server's files too:  ssh $HOST 'rm -rf $DIR'"
  exit 0
fi
cat <<EOF

KCoral is running on $HOST (log: $DIR/logs/server.out).
Open the tunnel from this machine and point the example at it:

  ssh -f -N -L $PORT:127.0.0.1:$PORT $HOST
  export KCORAL_URL=http://127.0.0.1:$PORT

Stop the server with:  $0 $HOST --stop --dir $DIR
EOF
