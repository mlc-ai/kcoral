#!/usr/bin/env bash
# Start (or stop) a KCoral GPU server on a Thor host over SSH.
#
#   ./launch_server.sh <thor-host> [--port PORT] [--dir DIR]
#   ./launch_server.sh <thor-host> --stop [--dir DIR]
#
# Start:
# 1. copies this KCoral checkout's tracked files to <thor-host>:DIR
# 2. creates the locked server environment there (uv sync --locked --group gpu)
# 3. (re)starts `kcoral server` bound to 127.0.0.1:PORT on the Thor host
# 4. waits for /health and prints the SSH tunnel command for this machine
#
# DIR is relative to the remote home directory; it defaults to
# .cache/kcoral-thor-example, which only this script uses and which is safe to
# delete. The server never listens on a public interface; reach it through the
# tunnel. Requires: git and rsync locally; uv on the Thor host (see README).
set -euo pipefail

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

HOST=""
PORT=8000
DIR=.cache/kcoral-thor-example
STOP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --dir) DIR="$2"; shift 2 ;;
    --stop) STOP=1; shift ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option: $1" >&2; usage 1 ;;
    *) HOST="$1"; shift ;;
  esac
done
[[ -n "$HOST" ]] || usage 1

# Stops the server recorded in DIR/logs/server.pid, if it is running.
STOP_SERVER='
if [[ -f logs/server.pid ]] && kill -0 "$(cat logs/server.pid)" 2>/dev/null; then
  echo "stopping the server (pid $(cat logs/server.pid))"
  kill "$(cat logs/server.pid)"
  while kill -0 "$(cat logs/server.pid)" 2>/dev/null; do sleep 1; done
elif [[ -z "${QUIET:-}" ]]; then
  echo "no server is running from this directory"
fi
rm -f logs/server.pid
'

if [[ "$STOP" == 1 ]]; then
  ssh "$HOST" bash -s -- "$DIR" <<REMOTE
set -euo pipefail
cd "\$1" 2>/dev/null || { echo "nothing to stop: \$1 does not exist"; exit 0; }
$STOP_SERVER
REMOTE
  echo "To remove the server's files too:  ssh $HOST 'rm -rf $DIR'"
  exit 0
fi

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

echo "==> copying $REPO to $HOST:$DIR"
ssh "$HOST" mkdir -p "$DIR"
git -C "$REPO" ls-files -z | rsync -a --from0 --files-from=- "$REPO/" "$HOST:$DIR/"

echo "==> installing the locked server environment on $HOST"
ssh "$HOST" bash -s -- "$DIR" <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$1"
if ! command -v uv >/dev/null; then
  echo "uv is not installed on this host; install it first (see README)" >&2
  exit 1
fi
uv sync --locked --group gpu --python 3.12
REMOTE

echo "==> starting kcoral server on $HOST (127.0.0.1:$PORT)"
ssh "$HOST" bash -s -- "$DIR" "$PORT" <<REMOTE
set -euo pipefail
cd "\$1"
PORT="\$2"
mkdir -p logs
QUIET=1
$STOP_SERVER
# TVM 0.26's NVRTC path looks for the CCCL headers under targets/\$(uname -m)-linux,
# but JetPack installs the toolkit as targets/sbsa-linux. Point NVRTC at them.
CUDA_HOME="\${CUDA_HOME:-/usr/local/cuda}"
if [[ -d "\$CUDA_HOME/include/cccl" ]]; then
  export TVM_CUDA_NVRTC_EXTRA_OPTS="-I\$CUDA_HOME/include/cccl"
fi
setsid nohup .venv/bin/kcoral server --device gpu --gpus 0 --host 127.0.0.1 --port "\$PORT" \\
  --log-dir logs >logs/server.out 2>&1 </dev/null &
echo \$! >logs/server.pid
for _ in \$(seq 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:\$PORT/health" >/dev/null 2>&1; then
    curl -fsS "http://127.0.0.1:\$PORT/health"
    echo
    exit 0
  fi
  sleep 1
done
echo "server did not become healthy; last log lines:" >&2
tail -n 30 logs/server.out >&2
exit 1
REMOTE

LOCATION="$DIR"
[[ "$DIR" == /* ]] || LOCATION="~/$DIR"
cat <<EOF

KCoral is running on $HOST (log: $LOCATION/logs/server.out).
Open the tunnel from this machine and point the example at it:

  ssh -N -L $PORT:127.0.0.1:$PORT $HOST
  export KCORAL_URL=http://127.0.0.1:$PORT

Stop the server with:  $0 $HOST --stop$([[ "$DIR" == .cache/kcoral-thor-example ]] || echo " --dir $DIR")
EOF
