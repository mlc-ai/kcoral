# Instruction protocol

The server exposes one synchronous endpoint, `POST /execute` (plus `GET /health`).
A request body is a **program**: an ordered list of instructions the server runs
on a GPU worker. There is no session state; every request is self-contained, and
its handles live only for that request.

```text
POST /execute          Content-Type: multipart/form-data
GET  /health
```

`GET /health` reports readiness, the `target` an uploaded library must be built
for, and the `versions` a client may want to match:

```json
{
  "status": "ok",
  "gpu_count": 1,
  "queue_length": 0,
  "target": {"arch": "sm_100a"},
  "versions": {"torch": "2.13.0+cu130", "cuda": "13.0",
               "tvm": "0.26.0", "tvm_ffi": "0.1.13.post2",
               "triton": "3.6.0", "cutlass": "4.7.0"},
  "gpus": [{"gpu_id": 0, "lease_depth": 1}],
  "workers": [{"gpu_id": 0, "status": "busy", "uptime_seconds": 12.4},
              {"gpu_id": 0, "status": "idle", "uptime_seconds": 12.4}]
}
```

Every worker in a pool shares one target — a server whose GPUs disagree refuses
to start, so run one server per GPU model.

Several workers share each GPU (`--workers-per-gpu`), so one can compile while
another measures on the GPU it is not using. They take turns through a per-GPU
lease and never run on it at once, so a measurement is unaffected by what else
the server is doing. `lease_depth` is how many workers hold or are queued for
that GPU.

By default each worker generation serves one request
(`--max-requests-per-worker 1`) and is replaced before its slot returns to the
idle pool. This gives every request a fresh CUDA context and allocator state.
Setting the option to `0` enables unlimited worker reuse for trusted kernels and
higher throughput, at the cost of allowing undefined CUDA behaviour to depend on
the process's prior allocation/module history.

## Request envelope

`multipart/form-data` is an HTTP body format containing multiple named parts,
each with its own content type. A boundary string separates the parts. Here it
combines the JSON program and binary blobs in one request.

The request contains:

| Part | Content type | Required | Notes |
|---|---|---:|---|
| `program` | `application/json` | yes | Instructions and options |
| `blob:<sha256>` | `application/octet-stream` | no | Tensor, byte, or library data |

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

### GPU placement

Every `run` instruction accepts an optional `gpu` field:

| Value | Behaviour |
|---|---|
| `"auto"` | Use the server's safe default; this is also the default when omitted |
| `"required"` | Acquire the worker's GPU lease before the instruction |
| `"none"` | Synchronize and release a held GPU lease before the instruction |

`auto` preserves the behaviour of programs written before this field existed.
Known host-only compile builtins release the lease, while other calls acquire
it. A compile may reacquire the lease for a short deferred driver or
module-loading phase.

`required` and `none` override those defaults. A `required` host-only builtin
keeps the lease while it runs; a `none` instruction promises that all of its
work is GPU-free. The server trusts the promise; `none` is a scheduling
declaration rather than a security boundary.

Uploads and returns do not accept `gpu`; they always acquire and retain the
lease. The Python client exposes the field on `Program.run()` and omits it from
the wire when its value is `auto`.

---

## `upload`

Uploads a module, tensor, byte string, or library and binds it to a handle.

### Module

```json
{
  "op": "upload",
  "id": "kernel",
  "kind": "module",
  "source": "def main(x):\n    return x * 2\n"
}
```

`source` is UTF-8 Python source embedded in the `program` part. The handle binds
to one object defined there, chosen in this order:

1. the name given by `entry`, if the upload sets one;
2. `main`, if the source defines it;
3. the source's only top-level `def` or `class`.

Several top-level definitions and no `main` is ambiguous: the upload fails and
the error names the candidates. Only definitions count, so a module-level
constant beside one kernel keeps the entry unambiguous.

```json
{
  "op": "upload",
  "id": "kernel",
  "kind": "module",
  "source": "def tile(x):\n    ...\n\ndef matmul(...):\n    ...\n",
  "entry": "matmul"
}
```

The entry object need not be callable: a decorator may bind a handle that a
builtin consumes rather than one `run` calls directly. Using a non-callable
handle as a `run` `fn` fails at run time.

#### CUDA C modules

`language` selects how `source` is read. It defaults to `"python"`; `"cuda"`
makes `source` CUDA C that `builtin.compile_cuda` builds into a callable:

```json
{
  "op": "upload",
  "id": "kernel",
  "kind": "module",
  "language": "cuda",
  "source": "void add_one(tvm::ffi::TensorView x, tvm::ffi::TensorView y) { ... }",
  "entry": "add_one"
}
```

A CUDA upload executes nothing, so it binds the source text rather than an
object, and `entry` is required — there is no namespace to infer it from. The
entry is exported through TVM FFI, so it takes `tvm::ffi::TensorView` parameters
and returns `void`; the includes and the export macro are supplied by the server.
`main` is rejected, because C++ reserves it as the program entry point.

The kernel is built for the worker GPU's arch-specific target (`sm_100a` on
Blackwell, `sm_90a` on Hopper), so instructions gated behind those targets —
tcgen05, wgmma — are available. Set `TVM_FFI_CUDA_ARCH_LIST` on the server to
override. Builds are cached on disk by source and flags.

#### CuTeDSL modules

CuTeDSL needs no `language` of its own: a `@cute.jit` kernel is ordinary Python,
so it uploads as one and `builtin.compile_cutedsl` compiles it. Name the
`@cute.jit` entry `main` or set `entry` — a kernel and its launcher are two
top-level definitions, which is otherwise ambiguous.

```json
{
  "op": "upload",
  "id": "kernel",
  "kind": "module",
  "source": "import cutlass.cute as cute\n\n@cute.kernel\ndef add_kernel(...):\n    ...\n\n@cute.jit\ndef add(...):\n    ...\n",
  "entry": "add"
}
```

CuTeDSL specializes on the tensors it is compiled against, so `compile_cutedsl`
takes them alongside the handle and they must be the ones the kernel will run
on. What comes back is callable with plain tensors:

```json
{"op": "run", "id": "compiled", "fn": "builtin.compile_cutedsl",
 "args": [{"$ref": "kernel"}, {"$ref": "x"}, {"$ref": "y"}]}
{"op": "run", "id": "invoke", "fn": {"$ref": "compiled"},
 "args": [{"$ref": "x"}, {"$ref": "y"}]}
```

Nothing is cached, so resubmitting a kernel recompiles it. Uploading a prebuilt
library instead skips the server-side compile entirely — see
[Library](#library).

#### Triton modules

A `@triton.jit` kernel is ordinary Python too, and `builtin.compile_triton`
compiles it. What Triton needs beyond the other languages is a launch grid: it is
normally computed by the caller at `kernel[grid](...)`, and the server will not
evaluate a client expression to get one, so it travels as `cfg.grid` — one to
three positive ints.

```json
{"op": "run", "id": "compiled", "fn": "builtin.compile_triton",
 "args": [{"$ref": "kernel"}, {"$ref": "x"}, {"$ref": "y"}, 4096, 256,
          {"grid": [16], "num_warps": 4}]}
{"op": "run", "id": "invoke", "fn": {"$ref": "compiled"},
 "args": [{"$ref": "x"}, {"$ref": "y"}, 4096, 256]}
```

Triton specializes on the arguments — dtypes, `tl.constexpr` values, pointer
alignment — so compiling takes the ones the kernel will be launched on, with
scalars and constexprs positional alongside the tensors. Every other `cfg` key is
a launch keyword: `num_warps`, `num_stages`, or a constexpr by name. The compiled
callable replays them, because Triton keys its cache on them and a launch that
differs recompiles — back on the GPU's time. The grid is not part of that key;
the callable carries it because a launch has nowhere else to get one.

The server caches nothing, but Triton's own on-disk cache makes a resubmitted
kernel much cheaper.

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

### Bytes

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

### Library

A library is a shared object the client already built, so the server compiles
nothing:

```json
{
  "op": "upload",
  "id": "kernel",
  "kind": "library",
  "blob": "<sha256>",
  "entry": "add_one"
}
```

`blob` names the bytes of an ELF shared object for the server's platform. The
server loads it with `tvm_ffi.load_module` and checks that it exports `entry`;
one that cannot be loaded, or that has no such function, fails the upload with a
`compile` error. Nothing else about the object is inspected, so any producer TVM
FFI can load is accepted. Three are usual:

- `TVM_FFI_DLL_EXPORT_TYPED_FUNC`, which `tvm_ffi.cpp.build` applies for you,
  emitting a `__tvm_ffi_<entry>` symbol. A code generator that emits that symbol
  directly works equally well;
- `tvm.Executable.export_library`, which embeds a module blob rather than a
  plain symbol. Unpacking one needs the loader the TVM CUDA runtime registers,
  so it requires a server with tvm installed;
- CuTeDSL's `--enable-tvm-ffi` export, which emits the same `__tvm_ffi_<entry>`
  symbol but leaves the object linked against `libcute_dsl_runtime.so`, so it
  requires a server whose `versions` reports `cutlass`. An object built against a
  newer cutlass than the server's fails to load, naming the symbol it wanted.

The function takes DLPack-compatible tensors, and its device code must be built
for the architecture `GET /health` reports. Building for another one fails later,
at launch, with `cudaErrorNoKernelImageForDevice`.

The exported entry has this shape — the body does not matter, only the interface.
Exporting it from C++:

```c++
#include <tvm/ffi/container/tensor.h>

void add_one(tvm::ffi::TensorView x, tvm::ffi::TensorView y) { /* launch a kernel */ }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, add_one);
```

Exporting the same interface from TIRx, where the function's own name becomes
`entry` and naming the target explicitly is what lets a client build without a GPU
of its own:

```python
from tvm.script import tirx as T

@T.prim_func
def add_one(A: T.Buffer((256,), "float32"), B: T.Buffer((256,), "float32")):
    ...  # kernel body

target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
with target:  # the tirx pipeline reads the arch from Target.current()
    executable = tvm.compile(
        tvm.IRModule({"add_one": add_one}), target=target, tir_pipeline="tirx"
    )
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
upload = {"op": "upload", "id": "kernel", "kind": "library",
          "blob": hashlib.sha256(data).hexdigest(), "entry": "add_one"}
```

Nothing then stands between the upload and the call: the handle is already the
callable, so no compile instruction appears.

```json
{"op": "run", "id": "invoke", "fn": {"$ref": "kernel"},
 "args": [{"$ref": "x"}, {"$ref": "y"}]}
```

### Fields

A field is accepted exactly for the kinds it lists, and is rejected for the
others: a `module` upload carries `source` and an optional `entry` and
`language`, a `tensor` upload carries `blob`, `dtype`, and `shape`, a `bytes`
upload carries `blob`, and a `library` upload carries `blob` and `entry`.

| Field | Kinds | Required for | Notes |
|---|---|---|---|
| `op` | all | all | `"upload"` |
| `id` | all | all | Unique handle name |
| `kind` | all | all | `"module"`, `"tensor"`, `"bytes"`, or `"library"` |
| `source` | module | module | UTF-8 source defining the entry object |
| `entry` | module, library | `cuda` modules, library | Identifier naming the entry object |
| `language` | module | — | `"python"` (default) or `"cuda"` |
| `blob` | tensor, bytes, library | tensor, bytes, library | SHA-256 of the raw bytes |
| `dtype` | tensor | tensor | Tensor data type |
| `shape` | tensor | tensor | Tensor shape |

### Blob cache

The server verifies supplied blobs against their part names and caches them by
hash. A tensor, byte string, or library may reference a cached blob without
supplying its multipart part, so unchanged data is uploaded once and later
requests cost only its hash. If any blob is missing, the program does not run:

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
| `builtin.compile_cuda` | `(source, cfg?)` — `cfg = {extra_cuda_cflags?}` | the module's exported function |
| `builtin.compile_cutedsl` | `(kernel, *tensors, cfg?)` — the tensors it specializes on; `cfg = {options?}` | a compiled kernel |
| `builtin.compile_triton` | `(kernel, *args, cfg)` — the args it specializes on; `cfg = {grid, **launch keywords}` | a callable bound to that grid |
| `builtin.benchmark` | `(mod, *tensors, cfg?)` — `cfg = {warmup_ms?, repeat_ms?, warmup?, repeat?, flush_l2?}` | timing statistics |
| `builtin.check_close` | `(actual, expected, cfg?)` — `cfg = {atol?, rtol?}` | comparison statistics |
| `builtin.assert_close` | same as `check_close` | comparison statistics; fails on mismatch |

`benchmark` returns `latency_ms_median`, `latency_ms_mean`, `latency_ms_min`,
`latency_ms_max`, `activities_stable`, `flush_l2`, `warmup`, and `repeat`.
Each latency is the CUPTI span from the earliest to the latest GPU activity of
one call — kernels, copies, and memsets, plus host time between them; the flush
and host work outside those endpoints are excluded. `activities_stable` is
`false` when the iterations did not all launch the same activities.

`check_close` and `assert_close` return `passed`, `max_abs_err`, `max_rel_err`,
`rtol`, and `atol`.

The four `compile_*` builtins are registered `cpu_only`, so a worker drops its GPU
lease while their host compilation runs and another worker measures meanwhile.
CUDA C compilation is split at that boundary: nvcc and linking run without the
lease, then the worker reacquires it before loading the shared object and
registering its CUDA module. A new builtin should declare `cpu_only` only when its
off-lease phase touches no GPU at all. Any driver/module-loading finalization must
be deferred until the engine reacquires the lease; otherwise it can perturb a
neighbouring worker's kernel or timing.

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
`elapsed_ms - lease_wait_ms - lease_held_ms`, is work done off the GPU. Only
`lease_held_ms` is GPU time, so it, not `elapsed_ms`, is what a caller should
divide by to cost a benchmark in GPU-seconds.

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
| 200 | `status: FAILED` | An instruction failed or terminated its worker; `results` holds the returns that ran |
| 200 | `status: CACHE_MISS` | Tensor blobs are missing; program did not run |
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

By default, a worker is also replaced after every completed request, whether or
not an error was detected. The fresh-process isolation covers all CUDA-backed
code, including Triton, and should remain enabled for untrusted submissions and
error corpora. `max_requests_per_worker: 0` reuses workers and is intended only
for validated throughput benchmarks. Record the setting with throughput results:
process replacement has a measurable cost for short requests, while reuse lets
undetected CUDA undefined behaviour influence a later request.

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
Program.upload(id=..., kind="module", source=..., entry=None, language="python") -> Register
Program.upload(id=..., kind="tensor", value=..., dtype=None, shape=None) -> Register
Program.upload(id=..., kind="bytes", value=...) -> Register
Program.upload(id=..., kind="library", value=..., entry=...) -> Register
Program.run(id=..., fn=..., args=[]) -> Register
Program.return_(key=..., value=...) -> None

Client(base_url, *, headers=None, connect_timeout_seconds=10)
Client.execute(program, *, timeout_seconds=None, output_limit_bytes=None) -> ProgramResult
Client.health() -> dict
Client.target() -> dict          # the health response's `target`, e.g. {"arch": "sm_100a"}
Client.close() -> None
```

For tensors, the client derives `blob`, `dtype`, and `shape` from `value`. It
starts without blob parts, retries a `CACHE_MISS` with the missing parts, and
falls back to all local blobs if the cache changes between requests. Returned
tensors decode to CPU `numpy.ndarray` (`bfloat16` and `float8_*` via
`ml_dtypes`). Server errors, transport failures, and malformed responses use
`BenchmarkServerError`, `TransportError`, and `ProtocolError`, which
`benchmark_server` exports alongside `Client` and `Program`.
