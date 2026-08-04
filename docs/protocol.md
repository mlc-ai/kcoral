# Instruction protocol

The server exposes one synchronous endpoint, `POST /execute` (plus `GET /health`).
A request body is a **program**: an ordered list of instructions the server runs
on a GPU worker. There is no session state; every request is self-contained, and
its handles live only for that request.

```text
POST /execute          Content-Type: multipart/form-data
```

## Request envelope

`multipart/form-data` is an HTTP body format containing multiple named parts,
each with its own content type. A boundary string separates the parts. Here it
combines the JSON program and raw tensor bytes in one request.

The request contains:

| Part | Content type | Required | Notes |
|---|---|---:|---|
| `program` | `application/json` | yes | Instructions and options |
| `blob:<sha256>` | `application/octet-stream` | no | Raw tensor bytes |

`<sha256>` is the lowercase 64-character SHA-256 of the part bytes.

The `program` part is:

```json
{
  "instructions": [ /* one or more upload / run / return instructions */ ],
  "options": { "timeout_seconds": 120 }
}
```

| Field | Type | Required | Notes |
|---|---|---:|---|
| `instructions` | array | yes | Non-empty, executed top to bottom |
| `options` | object | no | See [Options](#options) |

A multipart request can be constructed directly in Python:

```python
import hashlib
import json

import httpx
import numpy as np

input_array = np.arange(256, dtype=np.float32)
input_bytes = input_array.tobytes()
input_sha256 = hashlib.sha256(input_bytes).hexdigest()
program = {
    "instructions": [
        {
            "op": "upload",
            "id": "input",
            "kind": "tensor",
            "blob": input_sha256,
            "dtype": "float32",
            "shape": [256],
        },
        {"op": "return", "key": "input", "value": {"$ref": "input"}},
    ]
}

response = httpx.post(
    "http://server:8000/execute",
    files={
        "program": (None, json.dumps(program), "application/json"),
        f"blob:{input_sha256}": (
            None,
            input_bytes,
            "application/octet-stream",
        ),
    },
)
response.raise_for_status()
```

`httpx` generates the boundary and encodes each `files` entry as one named
multipart part. The higher-level client described below also handles caching
and response decoding.

Each `upload` and `run` has a unique string `id`. A reference has the exact form
`{"$ref": "<id>"}` and must point to an earlier instruction.

---

## `upload`

Uploads a module or tensor and binds it to a handle.

### Module

```json
{
  "op": "upload",
  "id": "kernel",
  "kind": "module",
  "source": "def main(x):\n    return x * 2\n"
}
```

`source` is UTF-8 Python source embedded in the `program` part and must define a
callable named `main`.

### Tensor

```json
{
  "op": "upload",
  "id": "input",
  "kind": "tensor",
  "blob": "<sha256>",
  "dtype": "float16",
  "shape": [32, 128]
}
```

`blob` names raw contiguous row-major bytes. Their length must equal
`product(shape) * dtype.itemsize`. The tensor is copied to the assigned GPU.

### Fields

A field is required exactly for the kinds it lists, and is rejected for the
others: a `module` upload carries `source` and no tensor fields, a `tensor`
upload carries `blob`, `dtype`, and `shape` and no `source`.

| Field | Kinds | Required for | Notes |
|---|---|---|---|
| `op` | all | all | `"upload"` |
| `id` | all | all | Unique handle name |
| `kind` | all | all | `"module"` or `"tensor"` |
| `source` | module | module | UTF-8 Python source defining `main` |
| `blob` | tensor | tensor | SHA-256 of raw tensor bytes |
| `dtype` | tensor | tensor | Tensor data type |
| `shape` | tensor | tensor | Tensor shape |

### Tensor blob cache

The server verifies supplied blobs against their part names and caches them by
hash. A tensor may reference a cached blob without supplying its multipart part.
If any blob is missing, the program does not run:

```json
{
  "status": "CACHE_MISS",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "missing_blobs": ["<sha256>"]
}
```

The client resends the program with the missing parts. Malformed names,
duplicates, hash mismatches, and unreferenced parts are invalid requests.

---

## `run`

Calls a function over earlier values and binds its result to a handle.

```json
{
  "op": "run",
  "id": "compiled",
  "fn": "builtin.compile_tirx",
  "args": [{"$ref": "kernel"}, {"N": 256}]
}
```

| Field | Type | Required | Notes |
|---|---|---:|---|
| `op` | string | yes | `"run"` |
| `id` | string | yes | Handle for the result |
| `fn` | string \| `{"$ref": id}` | yes | Builtin name or callable handle |
| `args` | array | no | Defaults to `[]` |

Each argument equal to `{"$ref": "<id>"}` resolves to that handle. Other JSON
values are passed as literals.

### Builtins

| `fn` | Arguments | Returns |
|---|---|---|
| `builtin.randn` | `spec = {shape, dtype, seed?}` | a random tensor |
| `builtin.empty` | `spec = {shape, dtype}` | an uninitialized tensor |
| `builtin.zeros` | `spec = {shape, dtype}` | a zero tensor |
| `builtin.compile_tirx` | `(kernel, bindings?)` — `bindings` binds `T.constexpr` dimensions | a compiled module |
| `builtin.benchmark` | `(mod, *tensors, cfg?)` — `cfg = {warmup_ms?, repeat_ms?, warmup?, repeat?, flush_l2?}` | timing statistics |
| `builtin.check_close` | `(actual, expected, cfg?)` — `cfg = {atol?, rtol?}` | comparison statistics |
| `builtin.assert_close` | same as `check_close` | comparison statistics; fails on mismatch |

`benchmark` returns `latency_ms_median`, `latency_ms_mean`, `latency_ms_min`,
`latency_ms_max`, `flush_l2`, `warmup`, and `repeat`.

`check_close` and `assert_close` return `passed`, `max_abs_err`, `max_rel_err`,
`rtol`, and `atol`.

---

## `return`

Selects a handle for the response:

```json
{
  "op": "return",
  "key": "timing",
  "value": {"$ref": "benchmark"}
}
```

| Field | Type | Required | Notes |
|---|---|---:|---|
| `op` | string | yes | `"return"` |
| `key` | string | yes | Unique key in the response `results` object |
| `value` | `{"$ref": id}` | yes | Earlier handle to return |

`return` has no `id` and creates no handle. Instructions run in the order given
and a `return` may appear anywhere after the instruction it references, so a
program can interleave returns with the uploads and runs that follow them. A
`return` that has already run contributes its entry to `results` even if a later
instruction fails.

---

## Options

| Field | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `timeout_seconds` | number | no | `300` | Worker execution deadline; maximum `3600` |
| `output_limit_bytes` | integer | no | `1048576` | Maximum bytes returned for each of stdout and stderr; maximum `16777216`; `0` disables capture |

A value above either maximum is clamped to it, not rejected.

---

## Response

For a successful program, HTTP status is `200`:

```json
{
  "status": "COMPLETED",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "queue_ms": 0.4,
  "elapsed_ms": 812.6,
  "results": {
    "timing": {
      "type": "object",
      "value": {
        "latency_ms_median": {"type": "number", "value": 0.0073}
      }
    }
  },
  "stdout": "",
  "stderr": "",
  "stdout_truncated": false,
  "stderr_truncated": false
}
```

`results` contains only values selected by `return`. `request_id` is also sent
in the `X-Request-ID` header. `queue_ms` is worker wait time; `elapsed_ms` is
worker execution and result serialization time.

### Fields

A 200 response carries the fields below. A non-200 carries the smaller error
body described under [Errors](#errors) instead.

| Field | Type | Present | Notes |
|---|---|---|---|
| `status` | string | always | `COMPLETED`, `FAILED`, or `CACHE_MISS` |
| `request_id` | string | always | Also sent as `X-Request-ID` |
| `queue_ms` | number | run | Worker wait time |
| `elapsed_ms` | number | run | Worker execution and serialization time |
| `results` | object | run | Entries for every `return` that ran; may be empty |
| `error` | object | `FAILED` | See [Errors](#errors) |
| `missing_blobs` | array | `CACHE_MISS` | Blob hashes the server does not hold |
| `stdout` | string | run | Captured standard output |
| `stderr` | string | run | Captured standard error |
| `stdout_truncated` | boolean | run | Whether `stdout` hit `output_limit_bytes` |
| `stderr_truncated` | boolean | run | Whether `stderr` hit `output_limit_bytes` |

"run" marks fields present whenever the worker returned an outcome, so on both
`COMPLETED` and `FAILED` but not on `CACHE_MISS`.

### Value encoding

| Type | Encoding |
|---|---|
| null | `{"type": "null"}` |
| boolean | `{"type": "boolean", "value": true}` |
| integer | `{"type": "integer", "value": 42}` |
| number | `{"type": "number", "value": 3.14}` |
| string | `{"type": "string", "value": "hello"}` |
| array | `{"type": "array", "value": [<value>, ...]}` |
| object | `{"type": "object", "value": {"<key>": <value>, ...}}` |
| bytes | `{"type": "bytes", "part": "return:0", "sha256": "<sha256>"}` |
| tensor | `{"type": "tensor", "dtype": "float16", "shape": [32, 128], "part": "return:0", "sha256": "<sha256>"}` |

Arrays and objects recursively contain encoded values. Object keys are unique
strings with no ordering semantics. Numbers must be finite. Python lists and
tuples both encode as `array`.

If no `bytes` or `tensor` appears, the response is `application/json`. Otherwise
it is `multipart/form-data`:

| Part | Content type | Required | Notes |
|---|---|---:|---|
| `result` | `application/json` | yes | Response metadata and value tree |
| `return:<index>` | `application/octet-stream` | conditional | Raw bytes for a bytes or tensor node |

Binary parts use depth-first numbering. Clients use `part` to locate data and
verify `sha256`. Tensor data is C-contiguous, row-major, and little-endian; its
length must match `dtype` and `shape`.

### Errors

An instruction failure stops the program. Every `return` that already ran keeps
its entry in `results`, so a program can checkpoint partial work by returning it
before the instructions that might fail:

```json
{
  "status": "FAILED",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "queue_ms": 0.4,
  "elapsed_ms": 12.7,
  "results": {
    "timing": {
      "type": "object",
      "value": {
        "latency_ms_median": {"type": "number", "value": 0.0073}
      }
    }
  },
  "error": {
    "kind": "correctness",
    "message": "outputs differ: max_abs_err=0.5 exceeds atol=0.001",
    "instruction_index": 6,
    "instruction_op": "run",
    "instruction_id": "check",
    "traceback": "Traceback (most recent call last):\n  ..."
  },
  "stdout": "",
  "stderr": ""
}
```

The failing instruction itself contributes nothing: a `return` that fails while
encoding adds neither a `results` entry nor binary parts.

A `FAILED` response's `error` describes that instruction:

| Field | Type | Notes |
|---|---|---|
| `kind` | string | See kinds below |
| `message` | string | Human-readable description |
| `instruction_index` | integer | Zero-based position in `instructions` |
| `instruction_op` | string | `"upload"`, `"run"`, or `"return"` |
| `instruction_id` | string \| null | The instruction's `id`; `null` for `return` |
| `traceback` | string | Server-side traceback, truncated to 8192 bytes |

Instruction error kinds are `parse`, `compile`, `runtime`, `correctness`,
`serialization`, `unavailable`, and `engine`.

| HTTP | Body | Meaning |
|---:|---|---|
| 200 | `status: COMPLETED` | Program completed |
| 200 | `status: FAILED` | An instruction failed; `results` holds the returns that ran |
| 200 | `status: CACHE_MISS` | Tensor blobs are missing; program did not run |
| 400 | `status: ERROR` | Malformed request or program, including duplicate JSON keys and NaN/Infinity |
| 413 | `status: ERROR` | Request body exceeds the server's size limit |
| 503 | `status: ERROR` | No worker is available; includes `Retry-After` |
| 504 | `status: ERROR`, `error.kind: timeout` | Execution timed out |
| 500 | `status: ERROR`, `error.kind: engine` | Worker or server failure |
| 500 | `status: ERROR`, `error.kind: response_too_large` | Results exceed the server's response-size limit |

`ERROR` is not a program outcome, so its body is much smaller: `status`,
`request_id`, and an `error` of `kind` and `message` only, with no `results`,
timings, or captured output.

```json
{
  "status": "ERROR",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "error": {"kind": "busy", "message": "server saturated"}
}
```

Its `kind` is `parse`, `request_too_large`, `busy`, `timeout`, `engine`, or
`response_too_large` — a separate set from the instruction kinds above.

The dividing line is whether the worker returned an outcome, not whether the
program ran: a timed-out or crashed program reaches a worker but is killed, so
it answers `ERROR` rather than `FAILED`.

---

## Example

```json
{
  "instructions": [
    {
      "op": "upload",
      "id": "kernel",
      "kind": "module",
      "source": "<TIRx source defining main>"
    },
    {
      "op": "upload",
      "id": "input",
      "kind": "tensor",
      "blob": "<input_sha256>",
      "dtype": "float32",
      "shape": [256]
    },
    {
      "op": "run",
      "id": "output",
      "fn": "builtin.empty",
      "args": [{"shape": [256], "dtype": "float32"}]
    },
    {
      "op": "run",
      "id": "compiled",
      "fn": "builtin.compile_tirx",
      "args": [{"$ref": "kernel"}, {"N": 256}]
    },
    {
      "op": "run",
      "id": "invoke",
      "fn": {"$ref": "compiled"},
      "args": [{"$ref": "input"}, {"$ref": "output"}]
    },
    {
      "op": "run",
      "id": "timing",
      "fn": "builtin.benchmark",
      "args": [
        {"$ref": "compiled"},
        {"$ref": "input"},
        {"$ref": "output"},
        {"warmup": 10, "repeat": 50}
      ]
    },
    {"op": "return", "key": "timing", "value": {"$ref": "timing"}},
    {"op": "return", "key": "output", "value": {"$ref": "output"}}
  ],
  "options": {"timeout_seconds": 120}
}
```

The multipart request includes `blob:<input_sha256>` with the raw input tensor.

## Python client

The Python client constructs the multipart body, tensor hash, and boundary
automatically:

```python
import numpy as np

from benchmark_server import Client, Program

kernel_source = """
from __future__ import annotations
from tvm.script import tirx as T

@T.jit
def main(A: T.Buffer((N,), "float32"), B: T.Buffer((N,), "float32"), *, N: T.constexpr):
    T.device_entry()
    i = T.cta_id([N])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0
"""
input_array = np.arange(256, dtype=np.float32)

program = Program()
kernel = program.upload(
    id="kernel",
    kind="module",
    source=kernel_source,
)
input_tensor = program.upload(id="input", kind="tensor", value=input_array)
output = program.run(
    id="output",
    fn="builtin.empty",
    args=[{"shape": [256], "dtype": "float32"}],
)
compiled = program.run(
    id="compiled",
    fn="builtin.compile_tirx",
    args=[kernel, {"N": 256}],
)
program.run(id="invoke", fn=compiled, args=[input_tensor, output])
timing = program.run(
    id="timing",
    fn="builtin.benchmark",
    args=[compiled, input_tensor, output, {"warmup": 10, "repeat": 50}],
)
program.return_(key="timing", value=timing)

with Client("http://server:8000") as client:
    result = client.execute(program, timeout_seconds=120)

print(result.results["timing"])
print(result.stdout, result.stderr)
```

```python
Program.upload(id=..., kind="module", source=...) -> Register
Program.upload(id=..., kind="tensor", value=..., dtype=None, shape=None) -> Register
Program.run(id=..., fn=..., args=[]) -> Register
Program.return_(key=..., value=...) -> None

Client(base_url, *, headers=None, connect_timeout_seconds=10)
Client.execute(program, *, timeout_seconds=None, output_limit_bytes=None) -> ProgramResult
Client.health()
Client.close() -> None
```

For tensors, the client derives `blob`, `dtype`, and `shape` from `value`, retries
one `CACHE_MISS` with the missing parts, and decodes returned tensors to CPU
`tvm_ffi.Tensor`. Server errors, transport failures, and malformed responses use
`BenchmarkServerError`, `TransportError`, and `ProtocolError`.
