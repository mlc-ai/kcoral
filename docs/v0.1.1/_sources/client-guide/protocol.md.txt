# KCoral Protocol

KCoral exposes two client HTTP endpoints. A program is an ordered list of
instructions, executed in one request with no persistent session handles.
This page describes a direct server. The [Router](../server-guide/router.md) preserves
the execution protocol while adding node selection and routing metadata.

## Endpoints

### POST /execute

Submit one program. The request body uses `multipart/form-data` with a JSON
`program` part and optional binary data parts. The request has no query fields.

See the [complete example](#example) below.

| Header | Behavior |
| --- | --- |
| `X-Request-ID` | Identifies each HTTP attempt and matches the result/error `request_id` and server events. The Router generates a UUID before admission and forwards it to Python. A direct Python request may supply exactly one canonical lowercase UUID; absent, duplicate, or invalid IDs are replaced. Cache-miss retries get separate IDs. |
| `X-KCoral-Node` | Identifies the node selected by the Router. Send it back as a cache-retry preference; a missing or unavailable preference falls back to another eligible node. |

| Part | Content type | Required | Meaning |
| --- | --- | --- | --- |
| `program` | `application/json` | yes | The program object below |
| `blob:<sha256>` | `application/octet-stream` | when not cached | Raw tensor, byte, file or library content referenced by an upload |

The outer content type must include a multipart boundary. Each part needs one
`Content-Disposition: form-data; name="..."` header and one `Content-Type`
header. Send binary content as raw bytes, not base64; nested multipart bodies
are unsupported. Encode the `program` JSON as UTF-8. Part order is unrestricted.

SHA-256 is the content hash used to identify binary data. `<sha256>` is its
lowercase 64-character hexadecimal digest over the raw bytes.

| Program field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `instructions` | array | yes | Nonempty list of operations executed in order |
| `options` | object | no | GPU count, execution timeout and captured output limits; see [Options](#options) |

Each supplied binary part must be referenced by an upload; multiple uploads
can share one part by naming the same digest. The server rejects
duplicate or malformed part names, wrong content types and hashes that do not
match the supplied bytes. JSON objects reject duplicate keys, and protocol
objects reject unknown fields. Numeric options and JSON arguments reject
non-finite numbers such as NaN and Infinity; argument objects may use arbitrary
string keys.

(options)=

**Options**

All `options` fields are optional. Timeout and output defaults and maxima are
server-configurable; values above the maxima are clamped. Booleans are not
accepted as numbers.

| Field | Type | Default | Meaning and limit |
| --- | --- | --- | --- |
| `gpu_count` | integer | omitted | Use 1–8 GPUs with instruction-level leasing; omission uses the server's ordinary GPU or CPU worker pool |
| `timeout_seconds` | number | `300` | Finite, strictly positive execution budget in seconds; maximum `900` |
| `output_limit_bytes` | integer | `1048576` | Non-negative capture limit per stdout/stderr stream; maximum `16777216`; `0` disables capture |

The execution budget excludes waiting for a worker and waiting for GPU leases.
With explicit `gpu_count`, worker initialization and interpreter teardown count
against it. Admission timeouts and body-size limits are configured by the
[server](../server-guide/launch-the-server.md#time-and-size-limits), not by request
options. Body-size limits include multipart framing.

See [Response](#response) for HTTP statuses, result fields, and value encodings.

### GET /health

Read endpoint health, request load, and the compilation environment.

```http
GET /health HTTP/1.1
Host: 127.0.0.1:8000
```

Example response from a GPU server with two workers:

```json
{
  "status": "ok",
  "instance_id": "09dc4eaa-a8b1-46cf-b5fb-a3448dcd7ca6",
  "started_at": "2026-08-29T18:42:11.019012Z",
  "gpu_count": 1,
  "load": {
    "request_capacity": 2,
    "requests_in_progress": 2,
    "requests_waiting": 3
  },
  "target": {"arch": "sm_100a"},
  "versions": {"torch": "2.14.0+cu132", "cuda": "13.2", "tvm_ffi": "0.1.13.post2"}
}
```

Version strings above are illustrative; use the values returned by your server.
The GPU runtime reports `torch`, `cuda` (the CUDA version used to build PyTorch),
and available `tvm`, `tvm_ffi`, `triton`, `cutlass` and `flashinfer` versions.
Optional entries can be absent; this is not an inventory of every installed
compiler or command-line tool. CPU servers report the available `tvm_ffi` version.

| Field | Type | Meaning |
| --- | --- | --- |
| `status` | string | Direct server: `ok` when the handler responds. Router: `ok` when a ready node has connected execution resources, otherwise `unavailable` with HTTP 503 |
| `instance_id` | string | Changes on each endpoint restart |
| `started_at` | string | Endpoint startup time in UTC (RFC 3339) |
| `gpu_count` | integer or null | Configured GPUs; `0` on CPU servers; `null` on routers |
| `load.request_capacity` | integer | Serviceable request capacity, occupied and free |
| `load.requests_in_progress` | integer | Assigned requests, including compilation, GPU waiting, and cleanup |
| `load.requests_waiting` | integer | Requests awaiting assignment at this endpoint |
| `target` | object | Compilation target, including `arch` on a GPU server; empty on a CPU server |
| `versions` | object | Runtime and toolchain version strings |

Direct-server capacity counts workers, excluding background replacements, and
becomes zero during shutdown. Router capacity counts execution connections on
eligible nodes; its request counts cover submissions through that router.

Counts can change before submission. During recovery or cleanup, in-progress
requests may exceed capacity. Assigned requests can wait for GPU access even
when `requests_waiting` is zero.

On the router, `instance_id` and `started_at` describe the router process.
For choosing a compilation target, see [Remote Compilation](../tutorials/remote-compilation.md#read-the-gpu-target).

## Operations

Every instruction is a JSON object with an `op` field. The four values are
`upload`, `get_function`, `run` and `return`.

| Common field | Meaning |
| --- | --- |
| `op` | Required operation name; determines the accepted fields |
| `id` | Required, nonempty, unique identifier when the operation produces a handle; absent for `return` |
| `{"$ref": "id"}` | A reference value naming an earlier handle; it is not a top-level instruction field |

A handle names a value inside this request. `upload`, `get_function` and `run`
produce handles. References must have exactly the `$ref`
key and cannot refer forward or cross request boundaries. Each operation below
specifies where references are resolved. Return keys are unique in a separate
namespace from instruction identifiers. Each field table is exhaustive:
unlisted fields are rejected.

### upload

Upload source or binary data and produce a handle.

```json
{
  "op": "upload",
  "id": "module",
  "kind": "module",
  "source": "def add_one(x): return x + 1"
}
```

Each field is required for the listed kinds.

| Field | Kinds | Notes |
|---|---|---|
| `op` | all | `"upload"` |
| `id` | all | Unique handle name |
| `kind` | all | `"module"`, `"tensor"`, `"bytes"`, `"file"`, or `"library"` |
| `source` | module | Python source executed to define a module |
| `blob` | tensor, bytes, file, library | SHA-256 of the raw bytes |
| `path` | file | Relative destination in the request working directory |
| `dtype` | tensor | Tensor data type |
| `shape` | tensor | Tensor shape |

| Kind | Content and result |
| --- | --- |
| <a id="module"></a>`module` | Inline Python `source`, executed to form a namespace. The handle binds the whole module. |
| <a id="tensor"></a>`tensor` | Raw contiguous row-major bytes, copied to the assigned GPU. The byte length must equal `product(shape) * dtype.itemsize`. |
| <a id="bytes"></a>`bytes` | The blob's bytes unchanged in CPU memory. Pass them to uploaded Python to parse files or other binary formats. |
| `file` | Materialize the blob at `path` in the request workspace and bind its normalized relative path as a string. Needs no GPU. See the file rules below. |
| `library` | A precompiled TVM FFI shared library for the server's platform, loaded with `tvm_ffi.load_module`. The handle binds the loaded module; see [Library](#library). |

Tensor `dtype` must be one of `bool`, `uint8`, `int8`, `int16`, `int32`,
`int64`, `float16`, `float32`, `float64`, `bfloat16`, `float8_e4m3fn`, or
`float8_e5m2`. Elements use little-endian byte order. `shape` is an array of
non-negative integers, excluding booleans: `[]` describes a scalar, and a zero
dimension describes an empty tensor. Uploads use the worker's current CUDA
device, initially logical device `0`; there is no upload field selecting a GPU.

CPU workers accept module, bytes and file uploads. Tensor and library uploads
fail at execution with `unavailable` on a CPU worker.

For each uncached binary upload, supply the bytes in a multipart part named
`blob:<sha256>`. For example:

```json
[
  {"op": "upload", "id": "input", "kind": "tensor", "blob": "<sha256>",
   "dtype": "float16", "shape": [32, 128]},
  {"op": "upload", "id": "raw_bytes", "kind": "bytes", "blob": "<sha256>"},
  {"op": "upload", "id": "file_path", "kind": "file", "blob": "<sha256>", "path": "data/tensor.bin"}
]
```

(file)=

**File upload rules**

| Rule | Behavior |
| --- | --- |
| Path | Relative POSIX path naming a file. Rejects `..` components, backslashes, and NUL. |
| Normalization | Removes `.` and repeated `/` components: `./data//tensor.bin` becomes `data/tensor.bin`. |
| Length | Each component is at most 255 UTF-8 bytes; the normalized path is at most 4096 bytes. Empty paths and the workspace root (`.`) are not valid destinations. |
| Conflicts | Normalized paths must be unique and cannot conflict as a file and directory; a program cannot upload both `data` and `data/tensor.bin`. |
| Filesystem access | Creates a regular file with mode `0600` and missing parent directories with mode `0700`. Creates and opens every component without following symbolic links. |
| Workspace | A fresh temporary working directory per request, owned by the parent process and removed after completion, failure, timeout, or worker crash. |
| Lifetime | Blob cache entries remain available after materialized files are removed. |

An upload cannot overwrite a file already created by an earlier instruction.
With bubblewrap isolation, `.kcoral` is reserved for runtime files and uploads
there fail with `runtime`. Materialization failures are instruction errors,
whereas invalid literal paths and declared upload conflicts are rejected before
execution with HTTP 400.

#### Library

A library is an already-built TVM FFI module, whether its bytes came from the
client's toolchain or a preceding CPU-server request. Uploading it requires no
compilation:

```json
{
  "op": "upload",
  "id": "kernels",
  "kind": "library",
  "blob": "<sha256>"
}
```

`blob` names the bytes of an ELF shared object for the server's platform. The
server loads it with `tvm_ffi.load_module` and binds the resulting module.
Later `get_function` instructions may bind any number of its TVM FFI exports. A library
that cannot be loaded, or a requested function that is absent, fails with a
`compile` error. Nothing else about the object is inspected, so any producer
TVM FFI can load is accepted. Three are usual:

| Producer | Export | Server requirement |
| --- | --- | --- |
| C++ / `tvm_ffi.cpp.build` | `TVM_FFI_DLL_EXPORT_TYPED_FUNC` emits `__tvm_ffi_<name>`; `tvm_ffi.cpp.build_inline(functions=...)` adds it automatically. A code generator can emit the symbol directly. | TVM FFI |
| `tvm.Executable.export_library` | Embedded module blob | TVM installed, including the loader registered by its CUDA runtime |
| CuTeDSL with `--enable-tvm-ffi` | `__tvm_ffi_<name>`, linked against `libcute_dsl_runtime.so` | A compatible CuTeDSL runtime. Check `versions.cutlass`; incompatible runtime symbols can cause loading to fail. |

The exported function determines its argument types. Build device code for
`GET /health`'s target architecture and provide compatible runtime dependencies.
KCoral does not validate the library's target architecture before loading it;
an incompatible device image can fail at launch.

For build examples, see
[local compilation](../tutorials/benchmark-kernel.md#on-the-client) and
[CPU-server compilation](../tutorials/remote-compilation.md).

### get_function

Select a named object from an earlier module or library upload.

```json
{
  "op": "get_function",
  "id": "add_one",
  "module": {"$ref": "module"},
  "name": "add_one"
}
```

| Field | Type | Required | Notes |
|---|---|---:|---|
| `op` | string | yes | `"get_function"` |
| `id` | string | yes | Handle for the callable |
| `module` | `{"$ref": id}` | yes | Earlier `module` or `library` upload |
| `name` | string | yes | Non-empty function or object name |
| `cpu_only` | boolean | no | Defaults to `false`; `true` declares that the function does not access the GPU |

| Source | Selection behavior |
| --- | --- |
| Python module | Looks up `name` in the executed namespace. |
| TVM-FFI library | Calls the loaded module's `get_function`. The selected function keeps its defining module alive. |

Module and function handles are request-local capabilities and cannot be
returned in a response.

A function declared `cpu_only` touches no GPU. A `run` of that handle releases
the worker's GPU lease first, and a CUDA runtime or driver API call from it, as
seen by CUPTI, fails the instruction with error kind `gpu_access`. The check is
best effort: it sees a call only after it has begun, and none from a child
process. The declaration applies to the handle as a `run` target only.

On GPU workers, module uploads execute their imports and top-level Python under
the GPU lease, regardless of how functions are later selected. Tensor and library
uploads, `get_function`, ordinary `run` calls and value returns also acquire the
lease. File uploads and file/folder returns release it after synchronizing;
byte uploads leave lease ownership unchanged. Final runtime cleanup reacquires
the lease even after a CPU-only tail.

### run

Call a function and bind the value it returns. Here `add_one` and `x` refer to
earlier instructions.

```json
{
  "op": "run",
  "id": "y",
  "fn": {"$ref": "add_one"},
  "args": [{"$ref": "x"}]
}
```

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `op` | string | yes | `run` |
| `id` | string | yes | Unique handle for the result |
| `fn` | reference | yes | An earlier callable handle, `{"$ref": id}` |
| `args` | array | no | Positional arguments, default `[]` |

| Value in `args` | Passed to the callable |
| --- | --- |
| Top-level `{"$ref": id}` | Value of the earlier handle |
| Reference-shaped object nested in a list or object | Literal JSON; not resolved |
| Other JSON value | Literal value |

Selecting an object with `get_function` does not prove it is callable; using a
non-callable object here fails at execution. `run` binds the computed value but
does not include it in the response; add a `return` to expose it.

Uploaded Python can perform tasks such as allocation, compilation, correctness
checks and measurement. A callable returned by a `run` can be used by a later
`run`.

### return

Select an earlier value for the response.

```json
{
  "op": "return",
  "key": "output",
  "value": {"$ref": "y"}
}
```

| Field | Type | Required | Notes |
|---|---|---:|---|
| `op` | string | yes | `"return"` |
| `key` | string | yes | Nonempty, unique key in the response `results` object |
| `value` | `{"$ref": id}` | For value returns | Earlier handle to return; omit `kind` and `path` |
| `kind` | string | For file/folder returns | `"file"` or `"folder"`; omit `value` |
| `path` | string or `{"$ref": id}` | For file/folder returns | Workspace path, supplied literally or through an earlier handle |

`return` snapshots its value at that instruction and does not stop execution.
It creates no handle. See [Errors](#errors) for partial-result behavior.

**File and folder returns**

```json
{"op": "return", "key": "report", "kind": "file", "path": "outputs/report.txt"}
```

To return a folder whose path is held in an earlier register:

```json
{"op": "return", "key": "debug", "kind": "folder", "path": {"$ref": "output_path"}}
```

| Rule | Behavior |
| --- | --- |
| Paths | Follow [file-upload rules](#file), relative to the original request workspace even if code changes cwd. |
| Snapshot | Contents are captured at that instruction, without holding the GPU lease. Folders include hidden files and empty directories; original metadata is omitted. |
| Rejected content | Symlinks, special files, repeated directories, and observable changes during reads |
| Failures | Missing paths, wrong types, invalid runtime paths, read failures, and collection limits fail that return with `serialization`. See [Errors](#errors) for partial results. |
| Size limit | Contents are buffered and count against the server response-size limit. |

## Multi-GPU execution

Set `options.gpu_count` to an integer from 1 through 8 to run the program
**once** on that many GPUs. `CUDA_VISIBLE_DEVICES` exposes the assigned set as
logical devices `0` through `gpu_count - 1`; the response's `gpu_ids` reports
physical devices. The script owns process creation, communication, and
synchronization. KCoral does not broadcast instructions or create a communication
group. Registers belong to the request's interpreter.

Single- and multi-GPU programs use the same worker and instruction-level leasing.
Before a GPU instruction, the worker acquires its complete device set atomically.
Before `cpu_only` execution it synchronizes all devices and releases the set;
a later GPU instruction reacquires the **same** devices. Waiting for devices
follows the timeout rules in [Options](#options). Releasing a lease does not
move or discard this program's tensors.
GPU subprocesses must finish before their `run` returns. Do not mark a GPU
launcher CPU-only.

On timeout or crash, the worker cleans up its process tree before abandoning
its leases. Unverified cleanup makes the affected devices unavailable and stops
the server pool from accepting new requests. Leftover descendants are terminated
and fail the request. Shutdown cancels queued allocations and drains running requests.

Explicit `gpu_count` requests use a fresh worker with the assigned set;
omitting the option uses the configured single-GPU workers. This option requires
Linux and a direct connection to a GPU server; the router does not select by GPU
count. Counts exceeding server capacity, or requests to a CPU server, fail with
HTTP 400. See [multi-GPU examples](writing-a-program.md#run-a-multi-gpu-program).

## Upload Caching

Tensor, byte and library uploads use the memory cache; file uploads use the
persistent file cache. Both are keyed by the raw content's SHA-256. Module source
travels inline and is not cached. Caches retain uploaded bytes, not execution
state. Resolved bytes remain usable by an admitted request even if evicted.

Cache capacity, eviction and storage settings are described in
[server cache configuration](../server-guide/launch-the-server.md#cache).

To use a cached upload, send its hash but omit its binary part. If required
bytes are absent, the server returns before executing any instruction:

```json
{
  "status": "CACHE_MISS",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "missing_blobs": ["<sha256>"]
}
```

Resend the same program with the listed parts. The Python client does this
automatically, then makes one final attempt with every local blob if another
cache miss occurs. Cache retention is an optimization rather than a guarantee.

The same digest may exist in either or both cache categories. A request using
that digest for both a file and a tensor, byte string or library can share the
resolved bytes; newly supplied content is offered to each referenced category.
A hit in one category does not generally guarantee a hit in the other.

## Response

| HTTP | Body | Meaning |
|---:|---|---|
| 200 | `status: COMPLETED` | Program completed |
| 200 | `status: FAILED` | Instruction or cleanup failure; see [Errors](#errors) |
| 200 | `status: CACHE_MISS` | Referenced blobs are missing; program did not run |
| 400 | `status: ERROR` | Malformed request or program, including duplicate JSON keys and NaN/Infinity |
| 413 | `status: ERROR` | Request body exceeds the server's size limit |
| 503 | `status: ERROR` | No worker is available; includes `Retry-After` |
| 504 | `status: ERROR`, `error.kind: timeout` | Execution timed out |
| 500 | `status: ERROR`, `error.kind: engine` | Worker failure outside an instruction, or server failure |
| 500 | `status: ERROR`, `error.kind: response_too_large` | Results exceed the server's response-size limit |

A 200 response carries the fields below. A non-200 carries the smaller error
body described under [Errors](#errors) instead.

| Field | Type | Present | Notes |
|---|---|---|---|
| `status` | string | always | `COMPLETED`, `FAILED`, or `CACHE_MISS` |
| `request_id` | string | always | Also sent as `X-Request-ID` |
| `queue_ms` | number | run | Worker wait time |
| `elapsed_ms` | number | run | Time after worker assignment, including GPU waits, execution, result encoding and cleanup |
| `lease_wait_ms` | number | run | Waiting for the assigned GPU set |
| `lease_held_ms` | number | run | Wall-clock time holding the GPU set, not summed GPU activity time |
| `gpu_ids` | array of integers | run with explicit `options.gpu_count` | Allocated physical devices, in logical-device order |
| `gpu_count` | integer | run with explicit `options.gpu_count` | Number of allocated devices |
| `results` | object | run | Captured returns, subject to [failure behavior](#errors) |
| `error` | object | `FAILED` | See [Errors](#errors) |
| `missing_blobs` | array | `CACHE_MISS` | Blob hashes the server does not hold |
| `stdout` | string | run | Captured standard output |
| `stderr` | string | run | Captured standard error |
| `stdout_truncated` | boolean | run | Whether captured stdout exceeded `output_limit_bytes` |
| `stderr_truncated` | boolean | run | Whether captured stderr exceeded `output_limit_bytes` |

"run" marks fields present whenever the worker returned an outcome, so on both
`COMPLETED` and `FAILED` but not on `CACHE_MISS`.

Captured streams are decoded as UTF-8 with replacement for invalid bytes. The
capture limit applies to raw bytes before decoding. With capture disabled,
both strings are empty and both truncation flags are false.

For a successful program, HTTP status is `200`:

```json
{
  "status": "COMPLETED",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "queue_ms": 0.4,
  "elapsed_ms": 812.6,
  "lease_wait_ms": 0.0,
  "lease_held_ms": 1.4,
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

`elapsed_ms` includes `lease_wait_ms` and `lease_held_ms`; the remainder is
time without a GPU lease. Holding a lease reserves the GPU and includes host
work during that period. Kernel latency is measured separately by the program.

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
| file | `{"type": "file", "size": 3, "part": "return:0", "sha256": "<sha256>"}` |
| folder | `{"type": "folder", "files": {"nested/a": <file value>}, "directories": ["empty", "nested"]}` |
| tensor | `{"type": "tensor", "dtype": "float16", "shape": [32, 128], "part": "return:0", "sha256": "<sha256>"}` |

Arrays and objects recursively contain encoded values. Object keys are unique
strings with no ordering semantics. Numbers must be finite. Python lists and
tuples both encode as `array`.

If no `bytes`, `tensor`, or `file` appears, the response is `application/json`. Otherwise
it is `multipart/form-data`:

| Part | Content type | Required | Notes |
|---|---|---:|---|
| `result` | `application/json` | yes | Response metadata and value tree |
| `return:<index>` | `application/octet-stream` | conditional | Raw bytes for a bytes, tensor, or file node |

Binary parts use depth-first numbering. Clients use `part` to locate data and
verify `sha256`. Tensor data is C-contiguous, row-major, and little-endian; its
length must match `dtype` and `shape`. A file's `size` is a non-negative integer
(not a boolean) and must match its binary part length.

Folder `files` maps sorted relative paths to file values; `directories` lists
sorted directory paths, including every ancestor. The selected root is implicit.
Paths must be canonical under file-upload rules, unique, and free of file/directory
conflicts. An empty folder has empty `files` and `directories`. Each file uses
its own binary part with the usual depth-first numbering and integrity checks.

### Errors

An ordinary instruction failure stops the program and preserves earlier returns.
A worker crash loses those returns and captured streams; HTTP-level failures
carry no partial results.

The `error` object in a `FAILED` response has this form:

```json
{
  "kind": "correctness",
  "message": "outputs differ: max_abs_err=0.5 exceeds atol=0.001",
  "instruction_index": 6,
  "instruction_op": "run",
  "instruction_id": "check",
  "traceback": "Traceback (most recent call last):\n  ..."
}
```

The failing instruction itself contributes nothing: a `return` that fails while
encoding adds neither a `results` entry nor binary parts.

| Field | Type | Notes |
|---|---|---|
| `kind` | string | See kinds below |
| `message` | string | Human-readable description |
| `instruction_index` | integer | Zero-based position in `instructions` |
| `instruction_op` | string | `"upload"`, `"get_function"`, `"run"`, or `"return"` |
| `instruction_id` | string \| null | The instruction's `id`; `null` only for `return` |
| `traceback` | string | Server-side traceback, retaining at most the final 8192 characters; can be empty for native crashes or cleanup errors |

Instruction error kinds are `parse`, `compile`, `runtime`, `gpu_access`,
`correctness`, `serialization`, `unavailable`, and `engine`. An unhandled Python
exception from a `run`, including `AssertionError`, is reported as `engine`;
`correctness` is used when the harness explicitly raises that execution error.
A `gpu_access` error means a `cpu_only` function entered the CUDA API; it adds
`cuda_call`, `location`, and `interfered_request_id`, and its `traceback` is
the stack at that call. `interfered_request_id` identifies the other request
holding the worker's GPU set at detection time, or is `null` when there is no
single identifiable holder.

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

| Failure | Outcome and recovery |
| --- | --- |
| Native uploaded code terminates its worker | `FAILED` with `runtime` for the active instruction |
| CUDA error poisons the worker's context | `FAILED` for the active instruction; only that worker process is replaced before it accepts another program |
| Non-sticky CUDA launch error found while draining the request | `FAILED`; the error is cleared, and a healthy worker can be reused if its configured request limit permits |
| Timeout, worker failure outside an instruction, or server failure | `ERROR` |

Cleanup can report a failure after the final instruction, attributed to the
last instruction observed. Worker replacement also follows the configured
request limit, which defaults to one request per process.

### Router errors

The Router forwards server outcomes and can also return its own `ERROR` body
with the same `status`, `request_id` and `error` fields:

| HTTP | `error.kind` | Meaning |
| --- | --- | --- |
| 400 | `invalid_request` | Invalid request headers or a failure reading the client's body |
| 413 | `request_too_large` | Router request-size limit exceeded |
| 503 | `router_busy` | Router queue full; includes `Retry-After: 1` |
| 503 | `no_node` | Wait for compatible node capacity expired; includes `Retry-After: 1` |
| 502 | `server_transport` | Node connection failed or returned an invalid response; execution outcome may be unknown |

For retry and disconnect behavior, see
[Router recovery](../server-guide/router.md#recover-from-failures-and-stop-nodes).

## Example

```json
{
  "instructions": [
    {
      "op": "upload",
      "id": "harness",
      "kind": "module",
      "source": "def main(x): return x + 1"
    },
    {
      "op": "get_function",
      "id": "main",
      "module": {
        "$ref": "harness"
      },
      "name": "main",
      "cpu_only": true
    },
    {
      "op": "run",
      "id": "output",
      "fn": {
        "$ref": "main"
      },
      "args": [
        41
      ]
    },
    {
      "op": "return",
      "key": "output",
      "value": {
        "$ref": "output"
      }
    }
  ],
  "options": {
    "timeout_seconds": 30
  }
}
```

This program uploads its own harness and returns `42`; it needs no binary parts.
The same upload/get_function/run shape supports GPU harnesses and compiler tools.

The Python package constructs request parts, hashes and response values for you.

| Task | Python client reference |
| --- | --- |
| Submit a first request | [Quickstart](../getting-started/quickstart.md) |
| Construct programs and manage clients | [Program guide](writing-a-program.md); [API signatures and errors](../python-api/index.rst) |
| Upload files and folders | `upload_folder` expands into ordinary file-upload instructions with no new operation. See [files used by uploaded scripts](writing-a-program.md#upload-files-and-folders). |
| Receive files and folders | Results decode to `ReturnedFile` and `ReturnedFolder`. See [client usage](writing-a-program.md#return-files-and-folders). |
