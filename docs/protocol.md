# Instruction protocol

The server exposes one synchronous endpoint, `POST /execute` (plus `GET /health`).
A request body is a **program**: an ordered list of instructions the server runs
on a GPU worker. There is no session state; every request is self-contained, and
its handles live only for that request.

```text
POST /execute          Content-Type: multipart/form-data
```

## Request envelope

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

Example multipart request:

```http
POST /execute HTTP/1.1
Host: server:8000
Content-Type: multipart/form-data; boundary=benchmark-boundary

--benchmark-boundary
Content-Disposition: form-data; name="program"
Content-Type: application/json

{"instructions":[{"op":"upload","id":"input","kind":"tensor","blob":"<input_sha256>","dtype":"float32","shape":[256]},{"op":"return","key":"input","value":{"$ref":"input"}}]}
--benchmark-boundary
Content-Disposition: form-data; name="blob:<input_sha256>"
Content-Type: application/octet-stream

<raw tensor bytes>
--benchmark-boundary--
```

The boundary is client-generated. `<input_sha256>` is the full SHA-256 of the
raw tensor bytes.

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
  "files": {
    "main.py": "from utils import scale\n\ndef main(x):\n    return scale(x)\n",
    "utils.py": "def scale(x):\n    return x * 2\n"
  },
  "entry": "main.py:main"
}
```

`files` maps relative POSIX paths to UTF-8 text. Paths contain no empty, `.`, or
`..` segments and no NUL or `:`. `entry` has the form `"<path>:<attribute>"`,
defaults to `"main.py:main"`, and must name a callable in `files`.

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

| Field | Kinds | Required | Notes |
|---|---|---:|---|
| `op` | all | yes | `"upload"` |
| `id` | all | yes | Unique handle name |
| `kind` | all | yes | `"module"` or `"tensor"` |
| `files` | module | yes | Source path to text mapping |
| `entry` | module | no | Entry callable |
| `blob` | tensor | yes | SHA-256 of raw tensor bytes |
| `dtype` | tensor | yes | Tensor data type |
| `shape` | tensor | yes | Tensor shape |

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

`return` has no `id` and creates no handle. All `return` instructions follow the
`upload` and `run` instructions.

---

## Options

| Field | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `timeout_seconds` | number | no | `300` | Worker execution deadline; maximum `3600` |
| `output_limit_bytes` | integer | no | `1048576` | Maximum bytes returned for each of stdout and stderr; `0` disables capture |

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
  "stderr": ""
}
```

Instruction error kinds are `parse`, `compile`, `runtime`, `correctness`,
`serialization`, `unavailable`, and `engine`.

| HTTP | Body | Meaning |
|---:|---|---|
| 200 | `status: COMPLETED` | Program completed |
| 200 | `status: FAILED` | An instruction failed |
| 200 | `status: CACHE_MISS` | Tensor blobs are missing; program did not run |
| 400 | `status: ERROR` | Malformed request or program |
| 503 | `status: ERROR` | No worker is available; includes `Retry-After` |
| 504 | `status: ERROR`, `error.kind: timeout` | Execution timed out |
| 500 | `status: ERROR`, `error.kind: engine` | Worker or server failure |

---

## Example

```json
{
  "instructions": [
    {
      "op": "upload",
      "id": "kernel",
      "kind": "module",
      "files": {"main.py": "<TIRx source defining main>"}
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

```python
from benchmark_server import Client, Program

program = Program()
kernel = program.upload(
    id="kernel",
    kind="module",
    files={"main.py": kernel_source},
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
Program.upload(id=..., kind="module", files=..., entry="main.py:main") -> Register
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
