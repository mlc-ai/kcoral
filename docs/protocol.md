# Instruction protocol

The server exposes two synchronous endpoints:

```text
POST /execute
GET  /health
```

An execution request contains one self-contained, ordered program. Handles and
uploaded values exist only for that request.

## Request

`POST /execute` uses `multipart/form-data`.

| Part | Content type | Required | Contents |
|---|---|---:|---|
| `program` | `application/json` | yes | Instructions and options |
| `blob:<sha256>` | `application/octet-stream` | no | Raw bytes for a tensor upload |

`<sha256>` is the lowercase 64-character SHA-256 hash of the part bytes. A
request contains exactly one `program` part and at most one part for each hash.

The `program` part has this shape:

```json
{
  "instructions": [
    {
      "op": "upload",
      "id": "add",
      "kind": "module",
      "files": {
        "main.py": "def main(a, b):\n    return a + b\n"
      }
    },
    {
      "op": "run",
      "id": "answer",
      "fn": {"$ref": "add"},
      "args": [20, 22]
    },
    {
      "op": "return",
      "key": "answer",
      "value": {"$ref": "answer"}
    }
  ],
  "options": {
    "timeout_seconds": 120,
    "output_limit_bytes": 1048576
  }
}
```

Instructions execute from top to bottom. Each `upload` and `run` has a unique
string `id`. A reference has the exact form `{"$ref": "<id>"}` and must name an
earlier instruction.

## Instructions

### `upload`

#### Module

```json
{
  "op": "upload",
  "id": "kernel",
  "kind": "module",
  "files": {
    "main.py": "from utils import scale\n\ndef main(x):\n    return scale(x)\n",
    "utils.py": "def scale(x):\n    return x * 2\n"
  },
  "entry": "main.py:main"
}
```

| Field | Required | Description |
|---|---:|---|
| `op` | yes | `"upload"` |
| `id` | yes | Handle bound to the entry callable |
| `kind` | yes | `"module"` |
| `files` | yes | Map from relative paths to UTF-8 text |
| `entry` | no | `"<path>:<attribute>"`; defaults to `"main.py:main"` |

Files are materialized in a private directory and loaded through Python's normal
import machinery. Paths use `/`, are relative, contain no empty, `.`, or `..`
segments, and contain no NUL or `:`. `entry` must name a file in `files` and a
callable attribute in that module.

#### Tensor

```json
{
  "op": "upload",
  "id": "input",
  "kind": "tensor",
  "blob": "81d4...<64 lowercase hexadecimal characters>",
  "dtype": "float16",
  "shape": [32, 128]
}
```

| Field | Required | Description |
|---|---:|---|
| `op` | yes | `"upload"` |
| `id` | yes | Handle bound to the GPU tensor |
| `kind` | yes | `"tensor"` |
| `blob` | yes | SHA-256 of the raw tensor bytes |
| `dtype` | yes | Tensor data type |
| `shape` | yes | Row-major tensor shape |

The blob contains contiguous row-major bytes and is copied to the worker's
assigned GPU. Its byte length must equal `product(shape) * dtype.itemsize`.

#### Tensor blob cache

The server verifies every supplied blob against its part name and caches it by
hash. A tensor may reference a cached blob without including its multipart part.

If any referenced tensor blob is absent, the program does not run:

```json
{
  "status": "CACHE_MISS",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "missing_blobs": [
    "81d4...<64 lowercase hexadecimal characters>"
  ]
}
```

The client may resend the same program with parts for the missing hashes.
Malformed names, duplicate parts, hash mismatches, and unreferenced parts are
invalid requests.

### `run`

```json
{
  "op": "run",
  "id": "compiled",
  "fn": "builtin.compile_tirx",
  "args": [
    {"$ref": "kernel"},
    {"N": 256}
  ]
}
```

```json
{
  "op": "run",
  "id": "output",
  "fn": {"$ref": "compiled"},
  "args": [
    {"$ref": "input"},
    {"$ref": "output_buffer"}
  ]
}
```

| Field | Required | Description |
|---|---:|---|
| `op` | yes | `"run"` |
| `id` | yes | Handle bound to the call result |
| `fn` | yes | Builtin name or reference to a callable handle |
| `args` | no | Positional arguments; defaults to `[]` |

An argument equal to `{"$ref": "<id>"}` resolves to that handle. Other JSON
values are passed as literals. Call results remain server-side until selected by
a `return` instruction.

#### Builtins

| `fn` | Arguments | Result |
|---|---|---|
| `builtin.randn` | `{shape, dtype, seed?}` | Random tensor |
| `builtin.empty` | `{shape, dtype}` | Uninitialized tensor |
| `builtin.zeros` | `{shape, dtype}` | Zero tensor |
| `builtin.compile_tirx` | `(kernel, bindings?)` | Compiled callable module |
| `builtin.benchmark` | `(module, *tensors, config?)` | Timing statistics |
| `builtin.check_close` | `(actual, expected, config?)` | Comparison statistics |
| `builtin.assert_close` | `(actual, expected, config?)` | Comparison statistics; fails the program on mismatch |

`compile_tirx` accepts an uploaded `@T.jit` or `@T.prim_func`. Its optional
bindings map `T.constexpr` names to values.

`benchmark` accepts:

```json
{
  "warmup_ms": 25,
  "repeat_ms": 100,
  "warmup": 10,
  "repeat": 50,
  "flush_l2": true
}
```

Explicit `warmup` and `repeat` counts take precedence over time budgets. The
result contains `latency_ms_median`, `latency_ms_mean`, `latency_ms_min`,
`latency_ms_max`, `flush_l2`, `warmup`, and `repeat`.

`check_close` and `assert_close` accept `rtol` and `atol`. Their result contains
`passed`, `max_abs_err`, `max_rel_err`, `rtol`, and `atol`.

### `return`

```json
{
  "op": "return",
  "key": "timing",
  "value": {"$ref": "benchmark"}
}
```

| Field | Required | Description |
|---|---:|---|
| `op` | yes | `"return"` |
| `key` | yes | Unique key in the response `results` object |
| `value` | yes | Reference to an earlier handle |

`return` has no `id` and creates no handle. All `return` instructions follow the
`upload` and `run` instructions. Values not selected by `return` are released at
the end of the request.

## Options

| Field | Default | Description |
|---|---:|---|
| `timeout_seconds` | `300` | Worker execution deadline; maximum `3600` |
| `output_limit_bytes` | `1048576` | Maximum bytes returned for each of stdout and stderr; `0` disables capture |

## Response

### Success

```json
{
  "status": "COMPLETED",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "queue_ms": 0.4,
  "elapsed_ms": 812.6,
  "results": {
    "correctness": {
      "type": "boolean",
      "value": true
    },
    "timing": {
      "type": "object",
      "value": {
        "latency_ms_median": {
          "type": "number",
          "value": 0.0073
        }
      }
    }
  },
  "stdout": "",
  "stderr": "",
  "stdout_truncated": false,
  "stderr_truncated": false
}
```

`results` contains only values selected by `return`. `queue_ms` measures the wait
for a GPU worker. `elapsed_ms` measures execution and result serialization on the
worker. Every response includes the same `request_id` in the `X-Request-ID`
header.

### Value encoding

Every returned value is a recursively tagged node:

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

`array.value` and `object.value` recursively contain tagged values. Object keys
are unique strings and have no ordering semantics. Integers and finite
floating-point numbers are supported; non-finite numbers are rejected. Python
lists and tuples both encode as `array`.

If the value tree contains no `bytes` or `tensor`, the response uses
`application/json`. Otherwise it uses `multipart/form-data`:

| Part | Content type | Contents |
|---|---|---|
| `result` | `application/json` | Complete response metadata and value tree |
| `return:<index>` | `application/octet-stream` | Raw bytes for one bytes or tensor node |

Binary parts are numbered in depth-first traversal order. Clients locate parts
through the node's `part` field and verify their SHA-256. Tensor data is
C-contiguous, row-major, and little-endian; its byte length must match `dtype`
and `shape`.

### Failure

An instruction failure stops the program and returns no `results`:

```json
{
  "status": "FAILED",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "queue_ms": 0.4,
  "elapsed_ms": 12.7,
  "error": {
    "kind": "correctness",
    "message": "outputs differ",
    "instruction_index": 6
  },
  "stdout": "",
  "stderr": "",
  "stdout_truncated": false,
  "stderr_truncated": false
}
```

Instruction error kinds are `parse`, `compile`, `runtime`, `correctness`,
`serialization`, `unavailable`, and `engine`.

| HTTP | Body | Meaning |
|---:|---|---|
| 200 | `status: COMPLETED` | Program completed |
| 200 | `status: FAILED` | An instruction failed |
| 200 | `status: CACHE_MISS` | Tensor blobs are missing; program did not run |
| 400 | `status: ERROR` | Malformed request or invalid program |
| 503 | `status: ERROR` | No worker is available; includes `Retry-After` |
| 504 | `status: ERROR`, `error.kind: timeout` | Execution timed out |
| 500 | `status: ERROR`, `error.kind: engine` | Worker or server failure |

## Python client

```python
from benchmark_server import Client, Program

program = Program()

kernel = program.upload(
    id="kernel",
    kind="module",
    files={"main.py": kernel_source},
)
input_tensor = program.upload(
    id="input",
    kind="tensor",
    value=input_array,
)
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
program.run(
    id="invoke",
    fn=compiled,
    args=[input_tensor, output],
)
timing = program.run(
    id="timing",
    fn="builtin.benchmark",
    args=[compiled, input_tensor, output, {"warmup": 10, "repeat": 50}],
)
program.return_(key="timing", value=timing)
program.return_(key="output", value=output)

with Client("http://server:8000") as client:
    result = client.execute(program, timeout_seconds=120)

print(result.results["timing"])
print(result.stdout, result.stderr)
```

The public interface is:

```python
Program.upload(id=..., kind="module", files=..., entry="main.py:main") -> Register
Program.upload(id=..., kind="tensor", value=..., dtype=None, shape=None) -> Register
Program.run(id=..., fn=..., args=[]) -> Register
Program.return_(key=..., value=...) -> None

Client(base_url, *, headers=None, connect_timeout_seconds=10)
Client.execute(program, *, timeout_seconds=None, output_limit_bytes=None) -> ProgramResult
Client.health() -> Health
Client.close() -> None
```

For tensor uploads, the client derives `blob`, `dtype`, and `shape` from `value`;
raw bytes require explicit `dtype` and `shape`. The client sends the program
without tensor parts first, then retries once with only `missing_blobs` after a
`CACHE_MISS`.

`ProgramResult` exposes `status`, `request_id`, `queue_ms`, `elapsed_ms`,
`results`, `stdout`, `stderr`, `stdout_truncated`, `stderr_truncated`, and
`error`. Values decode to `None`, `bool`, `int`, `float`, `str`, `list`, `dict`,
`bytes`, or a CPU `tvm_ffi.Tensor`.

Non-200 server responses raise `BenchmarkServerError`. Transport failures raise
`TransportError`. Malformed responses raise `ProtocolError`.

## Health

`GET /health` returns:

```json
{
  "status": "ok",
  "gpu_count": 2,
  "queue_length": 0,
  "workers": [
    {"gpu_id": 0, "status": "idle", "uptime_seconds": 3600.0},
    {"gpu_id": 1, "status": "busy", "uptime_seconds": 3592.4}
  ]
}
```
