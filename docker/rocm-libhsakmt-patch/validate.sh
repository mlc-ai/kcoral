#!/usr/bin/env bash
# Validate the patched ROCm runtime with KCoral's bubblewrap sandbox.
# Run from the repository root, inside a kcoral:rocm-dev-patched container
# started with the bubblewrap Docker options (see README.md).
set -uo pipefail
out=validation-results/libhsakmt-patch
mkdir -p "$out"
unset KCORAL_SERVER_HOST
step() { echo; echo "== $1"; }

step "1. patched library is the one torch loads"
python - <<'PY' 2>&1 | tee "$out/1-library.log"
import hashlib, os, torch
torch.cuda.device_count()
loaded = [l.split()[-1] for l in open("/proc/self/maps") if "libhsa-runtime64" in l]
path = sorted(set(loaded))[0]
digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print("loaded:", path)
print("patched:", os.path.exists(path + ".orig") and digest(path) != digest(path + ".orig"))
PY

step "2. unsandboxed: every GPU still initializes"
python -c 'import torch; print("devices:", torch.cuda.device_count())' 2>&1 | tee "$out/2-unsandboxed.log"

step "3. KCoral sandbox probe: each GPU sees exactly itself"
python - <<'PY' 2>&1 | tee "$out/3-probe.log"
from kcoral.support import rocm
for gpu in range(len(rocm.agents())):
    print(gpu, rocm.device_identity(gpu), rocm.sandbox_problem(gpu) or "OK")
PY

step "4. server with bubblewrap: allocation and kernel execution"
python -m kcoral server --backend rocm --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 1 \
  > "$out/4-server.log" 2>&1 &
server=$!
for _ in $(seq 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 2; done
grep -E "pool_ready|cannot be used" "$out/4-server.log"
KCORAL_URL=http://127.0.0.1:8000 python examples/rocm/triton_vector_add.py 2>&1 | tee "$out/4-triton.log"
kill "$server"; wait "$server" 2>/dev/null

step "5. sandbox, isolation and multi-GPU tests (bubblewrap required)"
KCORAL_GPU_TEST=1 KCORAL_MULTI_GPU_TEST=1 KCORAL_TEST_GPUS=0,5 KCORAL_SANDBOX_GPU=5 \
  KCORAL_REQUIRE_SANDBOX_TESTS=1 PYTHONPATH=tests \
  python -m pytest -q tests/test_sandbox.py tests/test_multi_gpu_jobs.py tests/test_rocm.py \
  2>&1 | tee "$out/5-sandbox-tests.log" | tail -15

step "6. full suite on all GPUs"
KCORAL_GPU_TEST=1 KCORAL_MULTI_GPU_TEST=1 KCORAL_TEST_GPUS=0,1,2,3,4,5,6,7 PYTHONPATH=tests \
  python -m pytest -q tests 2>&1 | tee "$out/6-full.log" | tail -15
