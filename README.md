# Benchmark Server

A stateless, synchronous HTTP service for compiling, running, and benchmarking
GPU kernels. A request is a small **program** — an ordered sequence of
instructions the server executes on a GPU worker and reports results for. There
is no session state: every request is self-contained.

## Model

A program is a list of two kinds of instruction:

- **`upload`** — hand the server a typed object (a `function` source, a `tensor`,
  or a JSON `object`). It is content-addressed by `key = sha256(bytes)` and cached,
  so a repeated upload can be sent by `key` alone.
- **`run`** — call a function over earlier results. The function is either a
  **builtin** (by name) or an earlier handle. Arguments reference earlier
  instructions by `{"$ref": "<id>"}`.

Data flows straight through handles (no control flow — a kernel's own logic lives
inside the uploaded function). Results are **structural JSON**; GPU objects
(tensors, compiled modules) stay server-side and come back as `{"handle": "<id>"}`.

Builtins: `randn` / `empty` / `zeros` (allocate a tensor), `compile_tirx` (lower a
`@T.jit` kernel), `benchmark` (CUDA-event timing), `check_close` (compare, returns
a result), `assert_close` (compare, *fails* on mismatch). A compiled module handle
is itself callable, so a kernel runs in place via a `run` whose `fn` is that handle.

The server only measures — interpreting or scoring the numbers is the client's job.

See [docs/protocol.md](docs/protocol.md) for the full wire format of `upload`,
`run`, and the request/response envelope.

## Requirements

- Python 3.12 and a CUDA GPU.
- **Mandatory** (GPU worker): CUDA-enabled `torch` and `tvm_ffi`.
- **Optional:** `tvm` — only `builtin.compile_tirx` uses it; without it that builtin
  returns an `unavailable` error and everything else still runs.

Install with `pip install -e .`; `torch`/`tvm_ffi`/`tvm` come from the environment.

For a reproducible setup, the repo ships a `uv.lock`: `uv sync --extra test`
creates the CPU test environment, and `uv sync --group gpu --extra test` adds
the pinned worker stack (CUDA 13.0 torch + `apache-tvm-ffi`).

### Docker

```bash
docker compose -f docker/compose.yaml up --build
```

builds the CUDA 13.0 dev image (`docker/Dockerfile.cu130.dev`, everything
installed from `uv.lock`) and starts the server on port 8000 with all GPUs
exposed; pick workers with `BENCH_GPUS=0,1 docker compose ...`. Cache and logs
persist in named volumes, and the source tree is bind-mounted for live edits.

## Example

### 1. Launch the server

```bash
# tvm pip-installed -> no extra env needed:
benchmark-server --gpus 0

# tvm from source -> point PYTHONPATH / TVM_LIBRARY_PATH at it:
PYTHONPATH=<tvm>/python TVM_LIBRARY_PATH=<tvm>/build/lib \
benchmark-server --gpus 0
```

`python -m benchmark_server` is equivalent. Every server limit (timeouts, cache
capacity, request/response size caps, output capture caps) has a flag — see
`benchmark-server --help`.

`BENCH_GPUS` is a comma-separated list of physical GPU ids to pin (one worker
each). `BENCH_HOST` / `BENCH_PORT` default to `127.0.0.1:8000`. Structured
event logs (one JSONL file per server run: request lifecycle, per-request GPU
assignment and timings, worker restarts) go to `BENCH_LOG_DIR` (default
`logs`; set it empty to disable). `GET /health` reports per-worker status and
the current queue length. Uploaded blobs are cached on disk in
`BENCH_CACHE_DIR` (default `cache`) and survive restarts, so clients can keep
sending key-only uploads across server runs; set it empty for a private
temporary directory instead.

### 2. Send a request

The package ships a synchronous client that computes content keys, follows the
`CACHE_MISS` retry flow, and wraps errors:

```python
from benchmark_server.client import Client, ref, run, upload_function

with Client("http://127.0.0.1:8000") as client:
    outcome = client.execute(
        [
            upload_function("fn", "def main(a):\n    return a + 1\n"),
            run("y", ref("fn"), [41]),
        ]
    )
    print(outcome.status, outcome["y"].value)  # COMPLETED 42
```

`upload_tensor` uploads torch tensors or numpy arrays (`upload_tensor_bytes`
for raw bytes). A failing instruction is data on the result; a non-200
response raises `BenchmarkServerError`; connection problems raise
`TransportError`. For large tensors, `client.prepare(instructions)` pushes the
payloads as raw binary blobs first (no base64 overhead), and
`builtin.download` + `decode_tensor` bring a result tensor back.

`examples/example_client.py` uses the client for a full kernel workflow — it
uploads an input tensor, a TIRx kernel, and a torch reference, compiles the
kernel, runs it on the tensor, asserts it matches the reference, and benchmarks it:

```bash
python examples/example_client.py 8000
```

### 3. Result

For a correct kernel, every instruction is `OK` and the body `status` is
`COMPLETED` (results abbreviated):

```json
{
  "status": "COMPLETED",
  "results": [
    {"id": "kernel", "op": "upload", "status": "OK"},
    {"id": "a",   "op": "upload", "status": "OK"},
    {"id": "mod", "op": "run", "status": "OK", "value": {"handle": "mod"}},
    {"id": "run", "op": "run", "status": "OK"},
    {"id": "chk", "op": "run", "status": "OK",
     "value": {"passed": true, "max_abs_err": 0.0, "rtol": 0.01, "atol": 0.001}},
    {"id": "perf", "op": "run", "status": "OK",
     "value": {"latency_ms": 0.0073, "warmup": 10, "repeat": 50}}
  ]
}
```

`perf.value.latency_ms` is the benchmark result. Uploads and the in-place kernel
run carry no `value`; tensors/modules come back as handles.

If the kernel is wrong, `assert_close` fails and the rest is skipped — the request
still returns **HTTP 200** (the server *did* run the program), but the body says so:

```json
{
  "status": "FAILED",
  "results": [
    "... earlier instructions OK ...",
    {"id": "chk", "op": "run", "status": "FAILED",
     "error": {"kind": "correctness", "message": "outputs differ: max_abs_err=1.0 ..."}},
    {"id": "perf", "op": "run", "status": "SKIPPED",
     "error": {"reason": "predecessor_failed"}}
  ]
}
```

## Statuses

- **Body `status`**: `COMPLETED` (every instruction OK), `FAILED` (some instruction
  failed → the rest are `SKIPPED`), or `CACHE_MISS` (resend the listed uploads with
  their `inline` bytes). A per-instruction failure carries `error.kind` — one of
  `parse`, `compile`, `runtime`, `correctness`, `unavailable` (a needed optional
  dependency like tvm isn't installed), `engine`.
- **HTTP status**: `200` for any program the server ran (including instruction
  failures and cache misses); `400` malformed request; `503` all workers busy;
  `504` execution timed out; `500` worker crashed.

## Tests

```bash
python -m pytest                 # CPU-only (no GPU needed)

# GPU integration — needs a GPU + the same tvm env as launching:
BENCH_GPU_TEST=1 PYTHONPATH=<tvm>/python TVM_LIBRARY_PATH=<tvm>/build/lib \
  python -m pytest tests/test_gpu_runtime.py
```
