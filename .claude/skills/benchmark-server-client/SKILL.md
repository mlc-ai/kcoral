---
name: benchmark-server-client
description: >-
  Write client code for the benchmark server's POST /execute instruction
  protocol. Use when writing, reviewing, or debugging programs that upload
  kernels (TIRx, CUDA C, CuTeDSL, Triton), create tensors, compile, check
  correctness, or benchmark on a remote GPU server, or when using the
  benchmark_server Client and Program API.
---

# Benchmark server client

Facts needed to write a protocol-conformant client. `docs/protocol.md` is the
authoritative field-level specification; where this page and that file
disagree, that file wins.

## Model

- The server exposes `POST /execute` and `GET /health`, nothing else.
- A request body is a **program**: an ordered list of instructions executed
  top to bottom on a GPU worker.
- There is no session state. Handles (`id`s) live for one request; a second
  `execute` shares nothing with the first, so every program uploads everything
  it needs.
- `run` computes a value but does not send it back. The response carries only
  what `return` selects.
- A reference has the exact form `{"$ref": "<id>"}` and must point to an
  earlier instruction.
- `GET /health` reports the GPU `target` (e.g. `{"arch": "sm_100a"}`) and the
  installed `versions` (torch, cuda, tvm, tvm_ffi, triton, cutlass).

## Python client

The canonical program shape — upload the kernel, make the tensors, compile,
check correctness, time it, return the results:

```python
import numpy as np
from benchmark_server import Client, Program

KERNEL = r"""
from __future__ import annotations
from tvm.script import tirx as T

@T.jit
def main(A: T.Buffer((N,), "float32"), B: T.Buffer((N,), "float32"), *, N: T.constexpr):
    T.device_entry()
    i = T.cta_id([N])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0
"""
REFERENCE = "def main(a):\n    return a + 1.0\n"

program = Program()
kernel = program.upload(id="kernel", kind="module", source=KERNEL)
reference = program.upload(id="reference", kind="module", source=REFERENCE)

src = program.upload(id="src", kind="tensor", value=np.arange(256, dtype=np.float32))
dst = program.run(id="dst", fn="builtin.empty", args=[{"shape": [256], "dtype": "float32"}])

compiled = program.run(id="compiled", fn="builtin.compile_tirx", args=[kernel, {"N": 256}])
program.run(id="invoke", fn=compiled, args=[src, dst])

expected = program.run(id="expected", fn=reference, args=[src])
check = program.run(id="check", fn="builtin.assert_close", args=[dst, expected])
timing = program.run(id="timing", fn="builtin.benchmark", args=[compiled, src, dst])

program.return_(key="check", value=check)
program.return_(key="timing", value=timing)

with Client("http://localhost:8000") as client:
    result = client.execute(program, timeout_seconds=120)
print(result.results["timing"]["latency_ms_median"])
```

API surface:

```python
Program.upload(id=..., kind="module", source=..., entry=None, language="python") -> Register
Program.upload(id=..., kind="tensor", value=..., dtype=None, shape=None) -> Register
Program.upload(id=..., kind="bytes", value=...) -> Register
Program.upload(id=..., kind="library", value=..., entry=...) -> Register
Program.run(id=..., fn=..., args=[]) -> Register
Program.return_(key=..., value=...) -> None

Client(base_url, *, headers=None, connect_timeout_seconds=10)
Client.execute(program, *, timeout_seconds=None, output_limit_bytes=None) -> ProgramResult
Client.health() -> dict
Client.target() -> dict   # e.g. {"arch": "sm_100a"}
Client.close() -> None
```

The client derives `blob`, `dtype`, and `shape` from a tensor `value`. It sends
no blob parts at first, retries a `CACHE_MISS` with the missing parts, and falls
back to resending every local blob if the cache changes between the two
requests. Returned tensors decode to CPU `numpy.ndarray` (`bfloat16` and
`float8_*` via `ml_dtypes`).

## Instructions

### `upload`

A field is accepted exactly for the kinds it lists:

| Field | Kinds | Required for | Notes |
|---|---|---|---|
| `id` | all | all | Unique handle name |
| `kind` | all | all | `"module"`, `"tensor"`, `"bytes"`, or `"library"` |
| `source` | module | module | UTF-8 source defining the entry object |
| `entry` | module, library | `cuda` modules, library | Identifier naming the entry object |
| `language` | module | — | `"python"` (default) or `"cuda"` |
| `blob` | tensor, bytes, library | tensor, bytes, library | SHA-256 of the raw bytes |
| `dtype` | tensor | tensor | Tensor data type |
| `shape` | tensor | tensor | Tensor shape |

A module upload binds one object out of its source: the one named by `entry`
if set, otherwise `main`, otherwise the source's single top-level `def` or
`class`. Two top-level definitions with no `main` and no `entry` fail as
ambiguous. An uploaded module is ordinary Python executed on the worker (torch
included), so a plain function works as a reference baseline.

A `bytes` upload binds the blob's bytes unchanged. They stay in CPU memory and
can be passed to uploaded Python code, which suits files and other binary
formats the server should parse.

A `library` upload is a prebuilt ELF shared object loaded with
`tvm_ffi.load_module`; the handle is directly callable, no compile step.

### `run`

`{"op": "run", "id": ..., "fn": ..., "args": [...]}` — `fn` is a builtin name
string or a `{"$ref": id}` callable handle (a `Register` in the Python
client). Arguments equal to `{"$ref": "<id>"}` resolve to handles; other JSON
values pass as literals.

### `return`

`{"op": "return", "key": ..., "value": {"$ref": id}}` — selects a handle for
the response `results` object. A `return` that already ran keeps its entry
even if a later instruction fails, so returning early checkpoints partial
work.

## Builtins

| `fn` | Arguments | Returns |
|---|---|---|
| `builtin.randn` | `spec = {shape, dtype, seed?}` | a random tensor (floating dtypes only) |
| `builtin.empty` | `spec = {shape, dtype}` | an uninitialized tensor |
| `builtin.zeros` | `spec = {shape, dtype}` | a zero tensor |
| `builtin.compile_tirx` | `(kernel, bindings?)` — `bindings` binds `T.constexpr` dimensions | a compiled module |
| `builtin.compile_cuda` | `(source, cfg?)` — `cfg = {extra_cuda_cflags?}` | the module's exported function |
| `builtin.compile_cutedsl` | `(kernel, *tensors, cfg?)` — the tensors it specializes on; `cfg = {options?}` | a compiled kernel |
| `builtin.compile_triton` | `(kernel, *args, cfg)` — the args it specializes on; `cfg = {grid, **launch keywords}` | a callable bound to that grid |
| `builtin.benchmark` | `(mod, *tensors, cfg?)` — `cfg = {warmup_ms?, repeat_ms?, warmup?, repeat?, flush_l2?}` | timing statistics |
| `builtin.check_close` | `(actual, expected, cfg?)` — `cfg = {atol?, rtol?}` | comparison statistics |
| `builtin.assert_close` | same as `check_close` | comparison statistics; fails on mismatch |

Uploaded code can call builtins directly via
`from benchmark_server import builtin` — same registry, same behaviour. The
`compile_*` builtins release the GPU lease only when run as their own
instruction, so compiles belong at the instruction level.

## Languages

All four follow the same upload-then-compile shape:

| Language | Upload | Compile call | Facts |
|---|---|---|---|
| TIRx | `source` | `compile_tirx(kernel, bindings?)` | Source must open with `from __future__ import annotations`, or `T.Buffer((N,), ...)` raises `NameError: N` at `def` time. `bindings` is for `@T.jit` constexprs; a `@T.prim_func` rejects them |
| CUDA C | `source`, `language="cuda"`, `entry` | `compile_cuda(kernel, cfg?)` | Entry is `void f(tvm::ffi::TensorView, ...)`; `entry` is required and `main` is rejected; includes and export macro come from the server; builds are disk-cached |
| CuTeDSL | `source`, `entry` (or name the `@cute.jit` entry `main`) | `compile_cutedsl(kernel, *tensors, cfg?)` | Specializes on the tensors passed, which must be the ones it will run on; nothing is cached |
| Triton | `source`, `entry` (or name the `@triton.jit` kernel `main`) | `compile_triton(kernel, *args, cfg)` | `cfg["grid"]` (1–3 positive ints) is required; scalars and constexprs pass positionally; other `cfg` keys are launch keywords (`num_warps`, `num_stages`, constexprs by name) |

A missing toolchain fails that builtin with an `unavailable` error;
`GET /health` lists installed `versions`.

## Tensors

- Created on the server (`randn`/`zeros`/`empty`) unless the exact values
  matter — uploading pays generation, hashing, and transfer.
- Upload when values matter: a locally computed reference, a reproducible
  fixed input, or content the kernel's work depends on (e.g. indptr arrays).
- `kind="tensor"` accepts a NumPy array, a torch tensor, any DLPack object,
  or raw bytes with `dtype` and `shape`.
- `randn`, `empty`, and `zeros` default to `float16` when `dtype` is omitted.
- Accepted dtypes: `bool`, `uint8`, `int8`, `int16`, `int32`, `int64`,
  `float16`, `float32`, `float64`, `bfloat16`, `float8_e4m3fn`,
  `float8_e5m2`.
- To hand a returned `bfloat16` array to torch:
  `torch.from_numpy(value.view(np.uint8)).view(torch.bfloat16)`.

## Correctness and timing

- Comparison runs on the server against an uploaded reference module; that
  avoids shipping outputs back. Defaults are `rtol=1e-2`, `atol=1e-3`.
  `check_close` reports a mismatch as data; `assert_close` stops the program
  with a `correctness` failure.
- `builtin.benchmark` reports the per-iteration GPU activity span measured by
  CUPTI, not wall time: from the start of the first kernel, copy, or memset a
  call launches to the end of the last. The L2 flush and host work outside those
  endpoints stay out, but host time *between* two activities does not. Defaults:
  `warmup_ms=25`, `repeat_ms=100`, `flush_l2=true`. The millisecond budgets
  adapt the iteration count to the kernel; explicit `warmup`/`repeat` counts are
  for runs that must be comparable.
- `flush_l2=true` gives each call a cold cache; off, a small kernel reads its
  input from L2 and reports an unrealistic latency.
- Returns `latency_ms_median`, `latency_ms_mean`, `latency_ms_min`,
  `latency_ms_max`, `activities_stable`, plus the `flush_l2`, `warmup`, `repeat`
  used. `activities_stable` is `false` when the timed iterations did not all
  launch the same activities, so the stats describe a mixture rather than one
  kernel.
- The request's GPU time is `lease_held_ms`, not `elapsed_ms`; the remainder
  is queueing (`queue_ms`, `lease_wait_ms`) and off-GPU work such as compiles.

## Outcomes

- `options`: `timeout_seconds` (default 300, maximum 3600),
  `output_limit_bytes` (default 1 MiB per stream). Values above a maximum are
  clamped. `stdout`/`stderr` come back with the response.
- `COMPLETED` — every instruction ran; `results` holds the returned values.
- `FAILED` — one instruction failed and the rest were skipped; returns that
  already ran stay in `results`. `error` carries `kind` (`parse`, `compile`,
  `runtime`, `correctness`, `serialization`, `unavailable`, `engine`),
  `message`, `instruction_index`, `instruction_id`, `traceback`.
- `CACHE_MISS` — blobs missing; the Python client retries this once
  automatically.
- Exceptions: `BenchmarkServerError` (carries `status_code` and `kind`; 503
  means no worker free, 504 means `timeout_seconds` hit), `TransportError`
  (request never reached the server), `ProtocolError` (malformed response).

## Non-Python clients

The wire format is `multipart/form-data` with a `program` part
(`application/json`) plus one `blob:<sha256>` part
(`application/octet-stream`) per tensor, bytes, or library blob, where
`<sha256>` is the lowercase hex SHA-256 of the part bytes. Blobs are cached by
hash: on `status: CACHE_MISS`, resend the program with the parts listed in
`missing_blobs`, and resend every blob if that retry misses again. Responses
containing tensors or bytes are multipart with a `result` JSON part and
`return:<index>` binary parts. The typed value encoding, blob-cache rules, and
full error table are in `docs/protocol.md`.

## References

- `docs/protocol.md` — field-level wire specification: request envelope,
  value encoding, library upload build routes (TVM FFI, TIRx
  `export_library`, CuTeDSL `--enable-tvm-ffi`) and their link flags, full
  HTTP error table.
- `docs/client_guide.md` — narrative guide: server-side compile vs prebuilt
  library trade-offs, measurement guidance.
- `examples/remote_compile_client.py` — runnable: all four languages compiled
  on the server.
- `examples/library_upload_client.py` — runnable: build on the client, upload
  the shared object.
