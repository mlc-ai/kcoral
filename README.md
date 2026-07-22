# TIRx Benchmark Server

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

## Requirements

- Python 3.12, a CUDA GPU, and CUDA-enabled `torch`.
- A TIRx-enabled `tvm`: either `pip install apache-tvm`, or a from-source build
  (put its Python tree on `PYTHONPATH` and point `TVM_LIBRARY_PATH` at the built
  library dir). The front-end process never imports either — each GPU worker does.

Install the server itself with `pip install -e .` (its own deps are just
`fastapi` + `uvicorn`; `torch`/`tvm` come from the environment above). Without an
install, run with `PYTHONPATH=src`.

## Example

### 1. Launch the server

```bash
# tvm pip-installed -> no extra env needed:
BENCH_GPUS=0 python -m benchmark_server

# tvm from source -> point PYTHONPATH / TVM_LIBRARY_PATH at it:
BENCH_GPUS=0 \
PYTHONPATH=<tvm>/python TVM_LIBRARY_PATH=<tvm>/build/lib \
python -m benchmark_server
```

`BENCH_GPUS` is a comma-separated list of physical GPU ids to pin (one worker
each). `BENCH_HOST` / `BENCH_PORT` default to `127.0.0.1:8000`.

### 2. Send a request

`examples/example_client.py` is a standalone client (only needs `httpx`). It
uploads a TIRx kernel and a torch reference, compiles the kernel, runs it, asserts
it matches the reference, and benchmarks it:

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
    {"id": "x",   "op": "run", "status": "OK", "value": {"handle": "x"}},
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
  `parse`, `compile`, `runtime`, `correctness`, `timeout`.
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
