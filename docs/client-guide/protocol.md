<a id="instruction-protocol"></a>

# KCoral Protocol

KCoral exposes two client HTTP endpoints. HTTP is the request-and-response transport;
JSON is the text format used for structured fields. A program is an ordered list
of instructions, executed in one request with no persistent session handles.
This page describes a direct server. The [Router](../server-guide/router.md) preserves
the execution protocol while adding node selection and routing metadata.

## Endpoints

### GET /health

Read endpoint health, request load, and the compilation environment.

```http
GET /health HTTP/1.1
Host: localhost:8000
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

Through the Rust router, `/execute` uses the same program and response format.
`X-KCoral-Node` identifies the selected node and serves as a cache-retry
preference. A missing or unavailable preference falls back to another eligible
node.

Each HTTP attempt has an `X-Request-ID` header matching the result/error
`request_id` and the server events. The router generates a UUID before admission
and forwards it to Python. A direct Python request may supply exactly one
canonical lowercase UUID; absent, duplicate, or invalid IDs are replaced.
Retries after `CACHE_MISS` are separate HTTP attempts with separate IDs.

### POST /execute

Submit one program. The request body uses `multipart/form-data`, a format that
combines named parts with individual content types. It carries a JSON `program`
part and optional binary data parts. The request has no query fields.

Example `program` part:

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

<a id="request-envelope"></a>

#### Request fields

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

#### Response fields

The response includes `status` and `request_id`. Execution outcomes also carry
`results`, request timings and captured output; `FAILED` adds an `error` object.
`CACHE_MISS` instead carries `missing_blobs` and means no instructions ran.
The full [response fields and encodings](#response) are defined below.

| Outcome | HTTP status | Body |
| --- | --- | --- |
| Program completed | 200 | `status: COMPLETED` |
| An instruction failed | 200 | `status: FAILED` |
| Referenced bytes are missing | 200 | `status: CACHE_MISS` |
| Request, capacity, timeout or server failure | 400, 413, 503, 504 or 500 | `status: ERROR`; see [Errors](#errors) |

`X-Request-ID` also carries the request identifier. Returned bytes or tensors
make the response multipart; otherwise it is `application/json`.

## Operations

Every instruction is a JSON object with an `op` field. The four values are
`upload`, `get_function`, `run` and `return`.

| Common field | Meaning |
| --- | --- |
| `op` | Required operation name; determines the accepted fields |
| `id` | Required, nonempty, unique identifier when the operation produces a handle; absent for file uploads and `return` |
| `{"$ref": "id"}` | A reference value naming an earlier handle; it is not a top-level instruction field |

A handle names a value inside this request. `upload` except file upload,
`get_function` and `run` produce handles. References are resolved recursively
inside arrays and objects, must have exactly the `$ref` key, and cannot refer
forward or cross request boundaries. Return keys are unique in a separate
namespace from instruction identifiers. The operation's field table is exhaustive:
unlisted fields are rejected.

### upload

Upload source or binary data. A file upload materializes a file; other kinds
produce a handle.

```json
{
  "op": "upload",
  "id": "module",
  "kind": "module",
  "source": "def add_one(x): return x + 1"
}
```

#### Fields

| Field | Kinds | Required for | Notes |
|---|---|---|---|
| `op` | all | all | `"upload"` |
| `id` | module, tensor, bytes, library | module, tensor, bytes, library | Unique handle name; rejected for file |
| `kind` | all | all | `"module"`, `"tensor"`, `"bytes"`, `"file"`, or `"library"` |
| `source` | module | module | UTF-8 Python or CUDA source defining a module |
| `language` | module | — | `"python"` (default) or `"cuda"` |
| `blob` | tensor, bytes, file, library | tensor, bytes, file, library | SHA-256 of the raw bytes |
| `path` | file | file | Relative destination in the request working directory |
| `dtype` | tensor | tensor | Tensor data type |
| `shape` | tensor | tensor | Tensor shape |

#### Details

##### Module

`source` is text carried directly in the program. `language` selects its loader:
`python` by default or `cuda`. Python source executes to form a namespace;
CUDA source is retained for compilation. Upload binds the whole module, and
`get_function` selects an object from it. A selected object need not be directly
callable: a compiler tool may consume it first.

All Python-based kernel languages use the same module upload shape; their
compilation is implemented by uploaded harness code. See the
[benchmark tutorial](../tutorials/benchmark-kernel.md#on-a-gpu-server).

<a id="cuda-c-modules"></a>
<a id="cutedsl-modules"></a>
<a id="triton-modules"></a>

##### Tensor

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
`product(shape) * dtype.itemsize`. Supply those bytes in a multipart part named
`blob:<sha256>` when the blob is not already cached. The tensor is copied to the
assigned GPU.

##### Bytes

```json
{
  "op": "upload",
  "id": "file",
  "kind": "bytes",
  "blob": "<sha256>"
}
```

The handle binds the blob's bytes unchanged. They stay in CPU memory and can be
passed to uploaded Python code, which makes this kind suitable for files and
other binary formats that the server should parse.

##### File

```json
{
  "op": "upload",
  "kind": "file",
  "blob": "<sha256>",
  "path": "data/tensor.bin"
}
```

The blob is copied to `path` as a regular file with mode `0600`. Missing parent
directories are created with mode `0700`. The instruction is a filesystem side
effect rather than a handle, so it has no `id` and cannot be referenced or
returned. It does not need the GPU.

Every program runs with a fresh temporary directory as its current working
directory. That directory is owned by the parent process and removed after the
program completes, fails, times out, or crashes its worker. Blob cache entries
remain available after the materialized files have been removed.

`path` uses POSIX `/` separators. It must be relative, must name a file, and must
not contain a `..` component, a backslash, or NUL. `.` and repeated `/`
components are removed, so `./data//tensor.bin` becomes `data/tensor.bin`.
Normalized paths must be unique and cannot conflict as a file and directory;
for example, one program cannot upload both `data` and `data/tensor.bin`. The
server creates and opens every component without following symbolic links.

##### Library

A library is an already-built shared object, whether its bytes came from the
client's toolchain or a preceding CPU-server request. The GPU server compiles
nothing:

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
Later `get_function` instructions may bind any number of its exports. A library
that cannot be loaded, or a requested function that is absent, fails with a
`compile` error. Nothing else about the object is inspected, so any producer
TVM FFI can load is accepted. Three are usual:

- `TVM_FFI_DLL_EXPORT_TYPED_FUNC`, which `tvm_ffi.cpp.build` applies for you,
  emitting a `__tvm_ffi_<name>` symbol. A code generator that emits that symbol
  directly works equally well;
- `tvm.Executable.export_library`, which embeds a module blob rather than a
  plain symbol. Unpacking one needs the loader the TVM CUDA runtime registers,
  so it requires a server with tvm installed;
- CuTeDSL's `--enable-tvm-ffi` export, which emits the same `__tvm_ffi_<name>`
  symbol but leaves the object linked against `libcute_dsl_runtime.so`, so it
  requires a server whose `versions` reports `cutlass`. An object built against a
  newer cutlass than the server's fails to load, naming the symbol it wanted.

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

<a id="blob-cache"></a>

#### Memory cache

Tensor, byte-string and library uploads share a server-process memory cache
keyed by the raw content's SHA-256. Module source is carried in the program and
does not use this blob cache. Reusing bytes does not reuse a previous tensor,
compiled module or execution: instructions still create request-local values.

The memory budget is `cache_capacity_bytes` (`--cache-capacity-bytes`, default
16 GiB). GiB means 1024 cubed bytes. Less recently used, unpinned entries may
be evicted. Referenced cached bytes are pinned while requests execute. An object
larger than one quarter of the budget is not retained by default, but supplied
bytes still work for that request. Restarting the Python server loses this cache.

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

#### File cache

File uploads use a separate persistent disk cache. Its default directory is
`$XDG_CACHE_HOME/kcoral/files` when `XDG_CACHE_HOME` is absolute, otherwise
`~/.cache/kcoral/files`. The default budget is 16384 MiB (16 GiB); MiB means
1024 squared bytes. Configure `disk_cache_dir` and
`disk_cache_capacity_mbytes`, or the corresponding server flags.

An empty directory option (`None` in Python) or zero capacity disables file
caching. It does not move files into the memory cache. Entries can survive a
server restart but may be evicted, unavailable or too large to retain. Storage
failure does not prevent execution when the request supplies the bytes.

Cached content and materialized files have different lifetimes. Each request
gets its own working directory; it is removed on completion, failure, timeout
or worker crash. The content cache may remain. Requests retain their own resolved
bytes, so disk eviction does not invalidate an admitted request.

The same digest may exist in either or both cache categories. A request using
that digest for both a file and a tensor, byte string or library can share the
resolved bytes; newly supplied content is offered to each referenced category.
A hit in one category does not generally guarantee a hit in the other.

<a id="get_function"></a>

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

#### Fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `op` | string | yes | `"get_function"` |
| `id` | string | yes | Handle for the callable |
| `module` | `{"$ref": id}` | yes | Earlier `module` or `library` upload |
| `name` | string | yes | Non-empty function or object name |
| `cpu_only` | boolean | no | Defaults to `false`; `true` declares that the function does not access the GPU |

#### Details

For Python source, the name indexes the executed namespace. For CUDA source it
returns an object containing the uploaded text as `source` and the selected
function name as `name`. For example, uploading `"void add() {}"` and selecting
`"add"` gives an object whose `source == "void add() {}"` and `name == "add"`.
A compiler can read those fields to build the function. C++ `main` and
non-identifiers are rejected. For a
TVM-FFI library it calls the loaded module's `get_function`; the function keeps
its defining module alive.

Given a library that exports `init` and `step`, the Python client writes:

```python
module = program.upload(id="kernels", kind="library", value=library_bytes)
init = program.get_function(id="init", module=module, name="init")
step = program.get_function(id="step", module=module, name="step")
program.run(id="initialize", fn=init, args=[input_tensor])
program.run(id="invoke", fn=step, args=[input_tensor, output_tensor])
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

#### Fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `op` | string | yes | `run` |
| `id` | string | yes | Unique handle for the result |
| `fn` | reference | yes | An earlier callable handle, `{"$ref": id}` |
| `args` | array | no | Positional arguments, default `[]` |

#### Details

Top-level reference values in `args` resolve to earlier handles. References
nested inside lists or objects remain literals. Other JSON values pass as
literals. Selecting an object with `get_function` does not prove it is callable;
using a non-callable object here fails at execution. `run` binds the computed
value but does not include it in the response; add a `return` to expose it.

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

#### Fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `op` | string | yes | `"return"` |
| `key` | string | yes | Unique key in the response `results` object |
| `value` | `{"$ref": id}` | For value returns | Earlier handle to return; omit `kind` and `path` |
| `kind` | string | For file/folder returns | `"file"` or `"folder"`; omit `value` |
| `path` | string or `{"$ref": id}` | For file/folder returns | Workspace path, supplied literally or through an earlier handle |

#### Details

`return` has no `id` and creates no handle. Instructions run in the order given
and a `return` may appear anywhere after the instruction it references, so a
program can interleave returns with the uploads and runs that follow them. A
`return` that has already run contributes its entry to `results` even if a later
instruction fails.

#### File and folder selection

```json
{"op": "return", "key": "report", "kind": "file", "path": "outputs/report.txt"}
```

To return a folder whose path is held in an earlier register:

```json
{"op": "return", "key": "debug", "kind": "folder", "path": {"$ref": "output_path"}}
```

These variants accept exactly `op`, `key`, `kind`, and `path`. `kind` is `file`
or `folder`; `path` is a literal string or an earlier reference resolving to one.
Paths follow file-upload rules and are relative to the original request workspace,
even if code changes cwd. Return keys are unique across all variants.

Each return snapshots contents at that instruction, without holding the GPU lease.
Folders include hidden files and empty directories; original metadata is omitted.
Symlinks, special files, repeated directories, and observable changes during reads
are rejected. Missing paths, wrong types, invalid runtime paths, read failures,
and collection limits fail that return with `serialization`. Failed returns add
no result or binary parts; earlier returns survive ordinary instruction failures.

Contents are buffered in the execution response. The existing `max_response_bytes`
limit (default 256 MiB) applies. `output_limit_bytes` controls only stdout/stderr.

## Options

| Field | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `timeout_seconds` | number | no | `300` | Worker execution deadline; maximum `900` |
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

<a id="fields-1"></a>

### Fields

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
| `instruction_id` | string \| null | The instruction's `id`; `null` for `return` and file upload |
| `traceback` | string | Server-side traceback, truncated to 8192 bytes |

Instruction error kinds are `parse`, `compile`, `runtime`, `gpu_access`,
`correctness`, `serialization`, `unavailable`, and `engine`. A `gpu_access`
error means a `cpu_only` function that entered the CUDA API; it adds
`cuda_call`, `location`, and `interfered_request_id`, and its `traceback` is
the stack at that call.

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

The dividing line is whether the failure can be attributed to an instruction.
A worker terminated by native uploaded code answers `FAILED` for the active
instruction after the server replaces it. A CUDA error that poisons a worker's
context also answers `FAILED` for the active instruction; the server replaces
only that worker process before it accepts another program. A non-sticky CUDA
launch error found while draining the request also answers `FAILED`, but the
server clears the error and keeps that healthy worker. A timeout, a worker
failure outside an instruction, or a server failure answers `ERROR`.

---

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

## Python client

The Python package constructs request parts, hashes and response values for you.
Follow the [quickstart](../getting-started/quickstart.md) for a first request, the
[program guide](writing-a-program.md) for client construction and lifecycle, and the
[Python interface reference](../python-api/index.rst) for signatures and errors.

`upload_folder` expands into ordinary file-upload instructions and introduces
no new protocol operation. File uploads return no register; see
[files used by uploaded scripts](writing-a-program.md#files-used-by-uploaded-scripts).

File/folder results decode to `ReturnedFile` and `ReturnedFolder`; see
[client usage](writing-a-program.md#returning-files-and-folders).
