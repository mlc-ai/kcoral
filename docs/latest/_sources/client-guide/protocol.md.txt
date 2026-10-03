# KCoral Protocol

KCoral exposes two client HTTP endpoints. A program is an ordered list of
instructions, executed in one request with no persistent session handles.
This page describes a direct server. The [Router](../server-guide/router.md) preserves
the execution protocol while adding node selection and routing metadata.

## Endpoints

### POST /execute

Submit one program. The request body uses `multipart/form-data` with a JSON
`program` part and optional binary data parts. The request has no query fields.

See the [complete example](#example) below. The Router uses the same program
and response format.

| Header | Behavior |
| --- | --- |
| `X-Request-ID` | Identifies each HTTP attempt and matches the result/error `request_id` and server events. The Router generates a UUID before admission and forwards it to Python. A direct Python request may supply exactly one canonical lowercase UUID; absent, duplicate, or invalid IDs are replaced. Cache-miss retries get separate IDs. |
| `X-KCoral-Node` | Identifies the node selected by the Router. Send it back as a cache-retry preference; a missing or unavailable preference falls back to another eligible node. |

| Part | Content type | Required | Meaning |
| --- | --- | --- | --- |
| `program` | `application/json` | yes | The program object below |
| `blob:<sha256>` | `application/octet-stream` | when not cached | Raw tensor, byte, file or library content referenced by an upload |

SHA-256 is the content hash used to identify binary data. `<sha256>` is its
lowercase 64-character hexadecimal digest over the raw bytes.

| Program field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `instructions` | array | yes | Nonempty list of operations executed in order |
| `options` | object | no | Execution timeout and captured output limits; see [Options](#options) |

Each supplied binary part must be referenced by an upload. The server rejects
duplicate or malformed part names, wrong content types and hashes that do not
match the supplied bytes. JSON objects reject duplicate keys, unknown fields,
and non-finite numbers such as NaN and Infinity.

(options)=

**Options**

Both `options` fields are optional. Values above either maximum are clamped to it.

| Field | Type | Default | Meaning and limit |
| --- | --- | --- | --- |
| `timeout_seconds` | number | `300` | Worker execution deadline; maximum `900` |
| `output_limit_bytes` | integer | `1048576` | Maximum bytes returned for each of stdout and stderr; maximum `16777216`; `0` disables capture |

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
SIGTERM or Ctrl+C starts graceful shutdown.

A CPU compilation server has an empty compilation target. Read the target from
the GPU server and supply it when compiling on a CPU server. Workers on one GPU
server must agree on the target; the server rejects a mixed-target pool.

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

For modules, `get_function` selects an object from the executed Python namespace.
The object need not be directly callable: a compiler tool may consume it first.
Upload CUDA source as a `file` and pass its path and export names to your compiler.
All Python-based kernel languages use the same module upload shape; uploaded
harness code handles compilation. See the
[benchmark tutorial](../tutorials/benchmark-kernel.md#on-a-gpu-server).

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
| Conflicts | Normalized paths must be unique and cannot conflict as a file and directory; a program cannot upload both `data` and `data/tensor.bin`. |
| Filesystem access | Creates a regular file with mode `0600` and missing parent directories with mode `0700`. Creates and opens every component without following symbolic links. |
| Workspace | A fresh temporary working directory per request, owned by the parent process and removed after completion, failure, timeout, or worker crash. |
| Lifetime | Blob cache entries remain available after materialized files are removed. |

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
| CuTeDSL with `--enable-tvm-ffi` | `__tvm_ffi_<name>`, linked against `libcute_dsl_runtime.so` | `versions` must report `cutlass`. Building against a newer cutlass than the server's fails to load, naming the missing symbol. |

The function takes DLPack-compatible tensors, and its device code must be built
for the architecture `GET /health` reports. Building for another one fails later,
at launch, with `cudaErrorNoKernelImageForDevice`.

The exported function has this shape — the body does not matter, only the interface.
Exporting it from C++:

```c++
#include <tvm/ffi/container/tensor.h>

void add_one(tvm::ffi::TensorView x, tvm::ffi::TensorView y) { /* launch a kernel */ }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, add_one);
```

Exporting the same interface from TIRx, where the function's own name becomes
the exported function name and naming the target explicitly let a client build without a GPU
of its own:

```python
from tvm.script import tirx as T

@T.prim_func
def add_one(A: T.Buffer((256,), "float32"), B: T.Buffer((256,), "float32")): ...  # kernel body

target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
with target:  # the tirx pipeline reads the arch from Target.current()
    executable = tvm.compile(tvm.IRModule({"add_one": add_one}), target=target, tir_pipeline="tirx")
executable.export_library("add_one.so")
```

CuTeDSL exports a relocatable object rather than a shared one, so it is the one
route with a link step. `cute.compile` specializes on the tensors it is handed,
so they must have the shape and dtype the kernel will be called with:

```python
compiled = cute.compile(add_one, src, dst, options="--enable-tvm-ffi")
compiled.export_to_c("add_one.o", "add_one", export_only_tvm_ffi_symbols=True)
```

`aot_config` reports the link flags. `--no-undefined` is worth passing because
the alternative — linking the static runtime archive instead — succeeds while
leaving symbols that only fail later, at load:

```bash
g++ -shared -o add_one.so add_one.o -Wl,--no-undefined \
    $(tvm-ffi-config --ldflags) \
    $(python -m cutlass.cute.export.aot_config --ldflags --libs --with-tvm-ffi)
```

Every route leaves a file on disk, and the upload carries its bytes: `blob` is
their SHA-256 and the bytes themselves travel as the matching `blob:<sha256>`
part, exactly as a tensor's do.

```python
data = pathlib.Path("add_one.so").read_bytes()
upload = {
    "op": "upload",
    "id": "kernels",
    "kind": "library",
    "blob": hashlib.sha256(data).hexdigest(),
}
```

Select the exported function, then call it; no compile instruction appears.

```json
[
  {"op": "get_function", "id": "kernel",
   "module": {"$ref": "kernels"}, "name": "add_one"},
  {"op": "run", "id": "invoke", "fn": {"$ref": "kernel"},
   "args": [{"$ref": "x"}, {"$ref": "y"}]}
]
```

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

Given a library that exports `init` and `step`, the Python client writes:

```python
module = program.upload(kind="library", value=library_bytes)
init = program.get_function(module=module, name="init")
step = program.get_function(module=module, name="step")
program.run(fn=init, args=[input_tensor])
program.run(fn=step, args=[input_tensor, output_tensor])
```

Module and function handles are request-local capabilities and cannot be
returned in a response.

A function declared `cpu_only` touches no GPU. A `run` of that handle releases
the worker's GPU lease first, and a CUDA runtime or driver API call from it, as
seen by CUPTI, fails the instruction with error kind `gpu_access`. The check is
best effort: it sees a call only after it has begun, and none from a child
process. The declaration applies to the handle as a `run` target only.

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
| `key` | string | yes | Unique key in the response `results` object |
| `value` | `{"$ref": id}` | For value returns | Earlier handle to return; omit `kind` and `path` |
| `kind` | string | For file/folder returns | `"file"` or `"folder"`; omit `value` |
| `path` | string or `{"$ref": id}` | For file/folder returns | Workspace path, supplied literally or through an earlier handle |

`return` has no `id` and creates no handle. Instructions run in the order given
and a `return` may appear anywhere after the instruction it references, so a
program can interleave returns with the uploads and runs that follow them. A
`return` that has already run contributes its entry to `results` even if a later
instruction fails.

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
| Fields | Exactly `op`, `key`, `kind`, and `path`. `kind` is `file` or `folder`; `path` is a literal string or an earlier reference resolving to one. Return keys are unique across all variants. |
| Paths | Follow [file-upload rules](#file), relative to the original request workspace even if code changes cwd. |
| Snapshot | Contents are captured at that instruction, without holding the GPU lease. Folders include hidden files and empty directories; original metadata is omitted. |
| Rejected content | Symlinks, special files, repeated directories, and observable changes during reads |
| Failures | Missing paths, wrong types, invalid runtime paths, read failures, and collection limits fail that return with `serialization`. Failed returns add no result or binary parts; earlier returns survive ordinary instruction failures. |
| Size limit | Contents are buffered in the response and count against `max_response_bytes` (default 256 MiB). `output_limit_bytes` controls only stdout/stderr. |

## Upload Caching

| Property | Memory cache | File cache |
| --- | --- | --- |
| Upload kinds | `tensor`, `bytes`, `library` | `file` |
| Storage | Python server process | Persistent disk cache |
| Key | SHA-256 of raw content | SHA-256 of raw content |
| Default budget | 16 GiB | 16 GiB |
| Budget setting | `cache_capacity_bytes` / `--cache-capacity-bytes` | `disk_cache_capacity_mbytes` / `--disk-cache-capacity-mbytes` |
| Retention | Less recently used, unpinned entries may be evicted. Objects larger than one quarter of the budget are not retained by default. | Entries may be evicted, unavailable, or too large to retain. |
| Active requests | Referenced cached bytes are pinned while the request executes. | Requests retain their resolved bytes; eviction does not invalidate an admitted request. |
| Server restart | Cache is lost. | Entries can survive. |

Module source travels inline and does not use either cache. Cached bytes do not
preserve a previous tensor's mutations, a compiled module, or an execution;
instructions still create request-local values.

The file cache defaults to `$XDG_CACHE_HOME/kcoral/files` when `XDG_CACHE_HOME`
is absolute, otherwise `~/.cache/kcoral/files`. Set `disk_cache_dir` or
`--disk-cache-dir` to change it. An empty directory option (`None` in Python) or
zero capacity disables file caching; it does not move files into the memory cache.
Storage failure or content too large to cache does not prevent execution when
the request supplies the bytes.

Materialized files live in a separate workspace per request. It is removed on
completion, failure, timeout, or worker crash; cached content may remain.

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
| 200 | `status: FAILED` | An instruction failed or terminated its worker; `results` holds the returns that ran |
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
| `elapsed_ms` | number | run | Worker execution and serialization time |
| `lease_wait_ms` | number | run | Waiting for the GPU another worker held |
| `lease_held_ms` | number | run | Holding the GPU — the request's GPU time |
| `results` | object | run | Entries for every `return` that ran; may be empty |
| `error` | object | `FAILED` | See [Errors](#errors) |
| `missing_blobs` | array | `CACHE_MISS` | Blob hashes the server does not hold |
| `stdout` | string | run | Captured standard output |
| `stderr` | string | run | Captured standard error |
| `stdout_truncated` | boolean | run | Whether `stdout` hit `output_limit_bytes` |
| `stderr_truncated` | boolean | run | Whether `stderr` hit `output_limit_bytes` |

"run" marks fields present whenever the worker returned an outcome, so on both
`COMPLETED` and `FAILED` but not on `CACHE_MISS`.

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

`results` contains only values selected by `return`. `request_id` is also sent
in the `X-Request-ID` header.

The four timings decompose a request: `queue_ms` waiting for a worker, then
`elapsed_ms` of execution, of which `lease_wait_ms` was spent waiting for the GPU
and `lease_held_ms` holding it. What is left,
`elapsed_ms - lease_wait_ms - lease_held_ms`, is execution without a GPU lease.
`lease_held_ms` measures how long the request reserves the GPU, including any
host work performed while holding the lease. Kernel measurements are reported
separately by the uploaded program.

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

An instruction failure stops the program. Every `return` that already ran keeps
its entry in `results`, so a program can checkpoint partial work by returning it
before the instructions that might fail:

```json
{
  "status": "FAILED",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "queue_ms": 0.4,
  "elapsed_ms": 12.7,
  "lease_wait_ms": 0.0,
  "lease_held_ms": 10.0,
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
  "stderr": "",
  "stdout_truncated": false,
  "stderr_truncated": false
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
| `instruction_op` | string | `"upload"`, `"get_function"`, `"run"`, or `"return"` |
| `instruction_id` | string \| null | The instruction's `id`; `null` only for `return` |
| `traceback` | string | Server-side traceback, truncated to 8192 bytes |

Instruction error kinds are `parse`, `compile`, `runtime`, `gpu_access`,
`correctness`, `serialization`, `unavailable`, and `engine`. A `gpu_access`
error means a `cpu_only` function that entered the CUDA API; it adds
`cuda_call`, `location`, and `interfered_request_id`, and its `traceback` is
the stack at that call.

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
| Native uploaded code terminates its worker | `FAILED` for the active instruction after the server replaces the worker |
| CUDA error poisons the worker's context | `FAILED` for the active instruction; only that worker process is replaced before it accepts another program |
| Non-sticky CUDA launch error found while draining the request | `FAILED`; the server clears the error and keeps the healthy worker |
| Timeout, worker failure outside an instruction, or server failure | `ERROR` |

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
