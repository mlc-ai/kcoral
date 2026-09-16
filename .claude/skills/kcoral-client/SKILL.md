---
name: kcoral-client
description: >-
  Write client code for KCoral's POST /execute instruction
  protocol. Use when writing, reviewing, or debugging programs that upload
  kernels (TIRx, CUDA C, CuTeDSL, Triton), create tensors, compile, check
  correctness, or benchmark on a remote GPU server, or when using the
  kcoral Client and Program API.
---

# KCoral client

Facts needed to write a protocol-conformant client. `docs/client-guide/protocol.md` is the
authoritative field-level specification; where this page and that file
disagree, that file wins.

## Model

- The client endpoints are `POST /execute` and `GET /health`.
- A request body is a **program**: an ordered list of instructions executed
  top to bottom on a GPU worker.
- There is no session state. Handles (`id`s) live for one request; a second
  `execute` shares nothing with the first, so every program uploads everything
  it needs.
- `run` computes a value but does not send it back. The response carries only
  what `return` selects.
- A reference has the exact form `{"$ref": "<id>"}` and must point to an
  earlier instruction.
- The `GET /health` response includes the GPU `target` (e.g. `{"arch": "sm_100a"}`),
  installed `versions`, and `load`: `request_capacity` (occupied + free capacity),
  `requests_in_progress` (assigned requests), and `requests_waiting`
  (requests awaiting assignment).

## Python client

For simple tasks, use `@client.function(timeout=30)` and call `.remote()` for the
decoded value, or `.execute()` for the full `ProgramResult`. Define the function
in a file, import dependencies inside it, and pass inputs explicitly. Configure
the server on `Client` and keep it open for remote calls. See
[Remote functions](../../../docs/client-guide/writing-a-program.md#remote-functions)
for supported inputs and execution boundaries.

Upload a harness, select its entry point, run it and return what you want to inspect:

```python
from kcoral import Client, Program

program = Program()
module = program.upload(id="harness", kind="module", source="""
import torch

def main(n):
    x = torch.arange(n, dtype=torch.float32, device="cuda")
    y = x + 1
    torch.testing.assert_close(y, x + 1)
    return {"ok": True, "output": y}
""")
main = program.get_function(id="main", module=module, name="main")
report = program.run(id="report", fn=main, args=[256])
program.return_(key="report", value=report)
with Client("http://localhost:8000") as client:
    result = client.execute(program, timeout_seconds=120)
print(result.results)
```

Uploaded Python can perform tasks such as compilation, tensor allocation,
validation, measurement, or invoking scripts and CLI tools. `run.fn` references
an earlier callable; a callable returned by a `run` can be used by a later call.

API surface:

```python
Program.upload(id=..., kind="module", source=..., language="python") -> Register
Program.upload(id=..., kind="tensor", value=..., dtype=None, shape=None) -> Register
Program.upload(id=..., kind="bytes", value=...) -> Register
Program.upload(id=..., kind="library", value=...) -> Register
Program.upload_file(blob=..., path=...) -> None
Program.upload_folder(folder, *, path=...) -> None
Program.get_function(id=..., module=..., name=..., cpu_only=False) -> Register
Program.run(id=..., fn=..., args=[]) -> Register
Program.return_(key=..., value=...) -> None
Program.return_file(key=..., path=...) -> None    # str or Register resolving to str
Program.return_folder(key=..., path=...) -> None  # str or Register resolving to str

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

`upload_folder` snapshots a local directory into ordinary file upload
instructions; retries send the same instructions with different blob parts.
It includes hidden files, rejects links, special files and repeated directories,
and omits empty directories and original permissions/timestamps. A failed call
leaves the program unchanged. Caching is automatic and best-effort.

`return_file` and `return_folder` capture workspace-relative paths at that
instruction; `path` may be an earlier register containing a path string.
Use `result[key].save(destination)` to save locally. File results also provide
`read_bytes()`; folder results provide `files` and `directories`. Folders include
hidden files and empty directories. Missing paths, symlinks, and special files
fail collection. Earlier returns survive later ordinary instruction failures.
Saves require an existing parent; file replacement needs `overwrite=True`, and
folder destinations must be new. Transfers are buffered and subject to server
limits. See [client usage](../../../docs/client-guide/writing-a-program.md#returning-files-and-folders).

## Instructions

### `upload`

A field is accepted exactly for the kinds it lists:

| Field | Kinds | Required for | Notes |
|---|---|---|---|
| `id` | module, tensor, bytes, library | module, tensor, bytes, library | Unique handle name; rejected for file |
| `kind` | all | all | `"module"`, `"tensor"`, `"bytes"`, `"file"`, or `"library"` |
| `source` | module | module | UTF-8 Python or CUDA source defining a module |
| `language` | module | — | `"python"` (default) or `"cuda"` |
| `blob` | tensor, bytes, file, library | tensor, bytes, file, library | SHA-256 of the raw bytes |
| `path` | file | file | Relative path in the request working directory |
| `dtype` | tensor | tensor | Tensor data type |
| `shape` | tensor | tensor | Tensor shape |

A module upload binds its full source namespace. Use `get_function` to select a
named Python object or CUDA source function. Uploaded Python is ordinary code
executed on the worker (torch included), so a plain function works as a
reference baseline.

A `bytes` upload binds the blob's bytes unchanged. They stay in CPU memory and
can be passed to uploaded Python code, which suits files and other binary
formats the server should parse.

A `library` upload is a prebuilt ELF shared object loaded with
`tvm_ffi.load_module`; use `get_function` to bind one of its exported functions.

A `file` upload returns no register. Use `upload_file(blob=..., path=...)` in the
Python client, where `blob` is bytes-like;
the wire field holds its SHA-256. The server copies it into the request's private
working directory with mode `0600` and removes that directory after execution.
Its content can remain in the disk cache across requests and server restarts.

### `get_function`

Select an object from an uploaded module with `get_function(id=..., module=..., name=...)`.
The returned `Register` stores its instruction ID and can be passed as `fn` to `run`.

`get_function(..., cpu_only=True)` declares that later calls to this function
use no GPU. Before its `run`, a GPU worker synchronizes and releases the lease.
A detected CUDA call fails with `gpu_access`. This is a best-effort check.
The flag applies to the selected function's calls; the Python module upload
executes top-level code under the GPU lease.

### `run`

`{"op": "run", "id": ..., "fn": {"$ref": "function_id"}, "args": [...]}`
calls an earlier callable. In Python, pass the `Register` returned by
`get_function` (or a `run` that returned a callable); the client encodes its ID
as `{"$ref": "function_id"}`. Top-level argument references resolve to values;
other JSON values pass as literals.

### `return`

`{"op": "return", "key": ..., "value": {"$ref": id}}` — selects a handle for
the response `results` object. A `return` that already ran keeps its entry
even if a later instruction fails, so returning early checkpoints partial
work.

## Compilation and measurement helpers

For convenience, uploaded Python can import `compile_tirx` and `benchmark` from
`kcoral.builtins`. You can also use your own compilation and measurement harness.
See the [API reference](../../../docs/python-api/index.rst#gpu-utilities)
for signatures and options, and the
[compilation tutorial](../../../docs/tutorials/benchmark-kernel.md#where-to-compile)
for GPU-server, CPU-server and local compilation examples.

## Tensors

Two ways to provide tensors are to upload Python that initializes them remotely,
or to upload local tensors with `kind="tensor"`.

- `kind="tensor"` accepts a NumPy array, a torch tensor, any DLPack object,
  or raw bytes with `dtype` and `shape`.
- Accepted dtypes: `bool`, `uint8`, `int8`, `int16`, `int32`, `int64`,
  `float16`, `float32`, `float64`, `bfloat16`, `float8_e4m3fn`,
  `float8_e5m2`.
- To hand a returned `bfloat16` array to torch:
  `torch.from_numpy(value.view(np.uint8)).view(torch.bfloat16)`.

## Correctness and timing

Your uploaded code can configure, for example, tolerances, assertions, profiler
invocation, warmup, and measurement methodology. Return reports and relevant
artifacts explicitly. Synchronize and wait for GPU subprocesses before
returning or releasing a lease.

## Request timing

The execution response contains top-level `queue_ms`, `elapsed_ms`,
`lease_wait_ms`, and `lease_held_ms` fields. The Python client exposes them as
attributes of the `ProgramResult` returned by `Client.execute`, for example
`result.lease_held_ms`.

`queue_ms` measures the wait for a worker. `elapsed_ms` starts once a worker is
assigned and includes waiting for the GPU lease (`lease_wait_ms`), holding it
(`lease_held_ms`), and other worker work. Holding the lease reserves the GPU;
it does not imply continuous GPU activity. Kernel latency is measured separately
by `benchmark` or your own measurement code.

## Outcomes

- `options`: `timeout_seconds` (default 300, maximum 3600),
  `output_limit_bytes` (default 1 MiB per stream). Values above a maximum are
  clamped. `stdout`/`stderr` come back with the response.
- `COMPLETED` — every instruction ran; `results` holds the returned values.
- `FAILED` — one instruction failed and the rest were skipped; returns that
  already ran stay in `results`. `error` carries `kind` (`parse`, `compile`,
  `runtime`, `gpu_access`, `correctness`, `serialization`, `unavailable`,
  `engine`), `message`, `instruction_index`, `instruction_id`, `traceback`.
- `CACHE_MISS` — blobs missing; the Python client retries this once
  automatically.
- Exceptions: `KCoralError` (carries `status_code` and `kind`; 503
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
full error table are in `docs/client-guide/protocol.md`.

## References

- `docs/client-guide/protocol.md` — field-level wire specification: request envelope,
  value encoding, library upload build routes (TVM FFI, TIRx
  `export_library`, CuTeDSL `--enable-tvm-ffi`) and their link flags, full
  HTTP error table.
- `docs/client-guide/writing-a-program.md` — narrative guide: server-side compile vs prebuilt
  library trade-offs, measurement guidance.
- `examples/first_program.py` — runnable: upload a function and tensor, then
  execute and return the result.
- `examples/benchmark_kernel.py` — TIRx compilation and measurement.
- `examples/remote_compile_client.py` — remote compilation in four languages.
- `examples/library_upload_client.py` — client build and library upload.
- `examples/cpu_compile_gpu_execute.py` — CPU build and GPU execution.
