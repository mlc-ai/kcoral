<a id="writing-a-benchmark-program"></a>
<a id="writing-a-program"></a>

# Write a client program

`Client` connects to a KCoral server. `Program` describes work to run there.
Building a program does not execute it: `Client.execute()` submits the ordered
instructions and decodes the selected results. The
[first GPU program](../getting-started/quickstart.md) shows a complete runnable example.

## Create and close a client

Use a context manager so the client's HTTP connections are closed when you finish.
HTTP is the request-and-response protocol used between the client and server.
One client can submit many programs to the same server.

```python
from kcoral import Client

with Client("http://localhost:8000", connect_timeout_seconds=10) as client:
    health = client.health()
    target = client.target()
    print(health["versions"])
    print(target["arch"])
```

`health()` reports endpoint status, load, and compilation metadata.
`target()` reads the GPU architecture an uploaded compiled library must match; ask the GPU
server for it, not a CPU compilation server. Optional `headers` are sent with
every request. If you do not use `with`, call `client.close()` explicitly.

`connect_timeout_seconds` limits connection establishment. It does not limit
execution. Pass `timeout_seconds` to `execute()` for a server-side execution
deadline, and `output_limit_bytes` to limit captured output per stream.

## Remote functions

Use `@client.function()` for a self-contained Python function. Configure the server
address on `Client`; decorated functions reuse its connections and headers:

```python
from kcoral import Client

with Client("http://localhost:8000") as client:

    @client.function(timeout=30)
    def gpu_sum(n):
        import torch

        return torch.arange(n, device="cuda").sum().item()

    print(gpu_sum.remote(4))  # 6
```

Save the function in a Python file, import dependencies inside it, and install
them on the server. Closures, external globals, additional decorators, async
functions and generators are unsupported. Arguments accept JSON values, bytes
and tensors; bytes and tensors must be whole arguments, not nested in containers.
Returned tensors are local NumPy arrays. Each remote call has no retained state.
See `examples/remote_function.py` for a tensor upload/run/download example.

Ordinary calls such as `gpu_sum(4)` execute locally. `.remote()` raises
`RemoteExecutionError` on an instruction failure; its `.result` retains the error,
traceback and captured output. `.execute()` returns the full `ProgramResult`,
including failed outcomes. `.build_program()` builds the equivalent `Program`
without executing it. Keep the client open for remote calls.

## Build instructions with Program

The builder returns a `Register` when an instruction produces a value. A register
is a local Python reference to a server-side value identified by its instruction
name; it is not the value itself.

| Method | Purpose | Returns |
| --- | --- | --- |
| `upload(id=..., kind=..., ...)` | Upload module source, a tensor, bytes or a compiled library | `Register` |
| `upload_file(blob=..., path=...)` | Snapshot bytes as a file in the request workspace | `None` |
| `upload_folder(folder, path=...)` | Snapshot a local directory as file uploads | `None` |
| `get_function(id=..., module=..., name=..., cpu_only=False)` | Select a function or object from an earlier module or library | `Register` |
| `run(id=..., fn=..., args=None)` | Call a selected or computed callable | `Register` |
| `return_(key=..., value=...)` | Select an earlier value for the response | `None` |
| `return_file(key=..., path=...)` | Select a workspace file for the response | `None` |
| `return_folder(key=..., path=...)` | Select a workspace folder for the response | `None` |
| `instructions` | Inspect a shallow copy of the wire instruction list | `list[dict]` |

Instruction identifiers must be nonempty and unique. The name `return_` has a
trailing underscore because `return` is a Python keyword. Use the
[Python API](../python-api/index.rst) for complete signatures and parameter types.

### Upload and select a function

```python
import numpy as np

from kcoral import Program

program = Program()
module = program.upload(
    id="module", kind="module", source="def scale(x, factor): return x * factor"
)
scale = program.get_function(id="scale", module=module, name="scale")
values = np.zeros(4, dtype=np.float32)
x = program.upload(id="x", kind="tensor", value=values)
y = program.run(id="y", fn=scale, args=[x, 2])
program.return_(key="output", value=y)
```

`upload` accepts `module`, `tensor`, `bytes` and `library`. Its file counterpart
is `upload_file`; `upload(kind="file")` is not a supported Python call. A module
binds a namespace; `get_function` explicitly selects an object from it. A
precompiled library follows the same selection step.

### Pass values and references

Pass the value returned by `get_function` as the `fn` argument to `run`:

```python
scale = program.get_function(id="scale", module=module, name="scale")
y = program.run(id="scaled", fn=scale, args=[x, 2])
```

`scale` is a `Register`, a Python object holding the ID `"scale"`. The client
serializes it as `{"$ref": "scale"}` in the request. The same applies to a
`Register` returned by a `run` that produced another callable.

Top-level `Register` arguments in `args` are encoded the same way. Ordinary
numbers, strings, lists and dictionaries pass as JSON literals. An explicit
`{"$ref": "id"}` in a top-level argument also resolves to that earlier value;
reference-shaped dictionaries nested inside lists or objects remain literals.
All references must point to earlier instructions in the same request.

## Submit and read results

```python
with Client("http://localhost:8000") as client:
    result = client.execute(program, timeout_seconds=120, output_limit_bytes=65536)

if result.completed:
    output = result.results["output"]
    # result["output"] is the same lookup.
else:
    print(result.error)
print(result.stdout, result.stderr)
```

The returned `ProgramResult` includes `status`, `request_id`, `results`, `error`,
captured `stdout` and `stderr`, and flags showing whether either stream was
truncated. Tensor results are NumPy arrays in client CPU memory; byte results
are Python `bytes`. Modules and callable handles cannot be returned.

`queue_ms` measures waiting for a worker. `elapsed_ms` is worker execution time,
including `lease_wait_ms` waiting for exclusive GPU access and `lease_held_ms`
holding that access. These request-level durations are different from a kernel's
measurement reported by your harness or profiler.

## Request lifecycle

1. **Construct locally.** Builder calls append instructions; binary uploads
   snapshot input at the time of the call.
2. **Resolve uploads.** `execute()` first sends a cache-only request. Missing
   cached blobs cause a retry with the missing bytes; a further miss causes
   one last request containing every local blob.
3. **Execute in order.** One worker runs the instructions. There is no retained
   program session between submissions.
4. **Return selected values.** Only return instructions contribute result
   entries. Returning early preserves that entry if a later instruction fails.
5. **Clean up.** The request's registers, GPU values and temporary files expire.
   Closing the client closes connections; it does not erase server caches.

You may submit the same `Program` again. Its binary snapshots are reused, but
its instructions execute again and get fresh remote values. You cannot pass a
register from an earlier request to a new program. The default server replaces
a worker after each request; opting into worker reuse does not extend the
protocol lifetime of a register.

File and memory caches retain uploaded bytes as an optimization. A cache hit
does not preserve a previous tensor's mutations, a compiled callable, or an
execution's output. Learn the separate rules in the protocol's
[memory cache](protocol.md#memory-cache) and [file cache](protocol.md#file-cache) sections.

## Tensors

You can initialize tensors remotely by uploading Python, or upload tensors
created locally.

For remote initialization:

```python
module = program.upload(id="allocator", kind="module", source="""
import torch

def make_input():
    return torch.randn((4096, 4096), dtype=torch.bfloat16, device="cuda")
""")
make_input = program.get_function(id="make_input", module=module, name="make_input")
x = program.run(id="x", fn=make_input)
```

For a local tensor:

```python
import numpy as np

values = np.zeros((4, 4), dtype=np.float32)
x = program.upload(id="x", kind="tensor", value=values)
```

`kind="tensor"` accepts a NumPy array, a torch tensor, any object supporting
DLPack, or raw bytes together with `dtype` and `shape`.

Accepted dtypes: `bool`, `uint8`, `int8`, `int16`, `int32`, `int64`, `float16`,
`float32`, `float64`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2`.

Returned tensors arrive as CPU `numpy.ndarray`, with `bfloat16` and the float8
types carried by `ml_dtypes`. To hand one to torch, reinterpret the raw bytes:

```python
torch.from_numpy(value.view(np.uint8)).view(torch.bfloat16)
```

## Files used by uploaded scripts

Use `upload_file` when uploaded Python code expects a relative file path:

```python
program = Program()
program.upload_file(blob=tensor_bytes, path="./inputs/tensor.bin")
module = program.upload(id="reader_module", kind="module", source=READER_SOURCE)
reader = program.get_function(id="reader", module=module, name="main")
result = program.run(id="result", fn=reader)
```

The module can use `open("inputs/tensor.bin", "rb")` unchanged. File uploads
return no register. Put them before any instruction that reads the files,
including a module upload whose top-level code opens them.

To snapshot a local directory at the current program position:

```python
program.upload_folder("./assets", path="inputs")
```

`assets/a` becomes `inputs/a`; `assets/sub/b` becomes `inputs/sub/b`. Both helpers
snapshot content when called, so later changes to the supplied bytes or local
files do not affect execution or retries. Folder uploads include hidden files
and reject symbolic links (including the source directory), repeated directories,
and special files such as FIFOs. Empty directories and original permissions and
timestamps are omitted; an empty folder adds no instructions.

Destinations must be relative POSIX paths without `..` components. Duplicate
paths and file/directory conflicts are rejected; parent directories are created
automatically. A traversal, read, or destination-validation failure leaves the
program unchanged.

Each execution gets a fresh working directory, removed after completion,
failure, timeout, or worker crash. Caching is automatic and best-effort.
The workspace is not a sandbox for uploaded Python code.

## Returning files and folders

After uploaded code creates outputs in the request workspace:

```python
program.return_file(key="report", path="outputs/report.txt")
program.return_folder(key="debug", path="outputs/debug")
result = client.execute(program)
result["report"].save("report.txt")
result["debug"].save("debug")
```

`path` accepts a relative POSIX path or an earlier `Register` containing one.
Each return captures contents at that instruction; later writes do not change
it. Ordinary `return_` of a path string returns only the string. Earlier returns
remain available if a later instruction fails; a failed collection adds nothing.

`ReturnedFile.read_bytes()` reads the received contents. `ReturnedFolder.files`
maps relative paths to `ReturnedFile`; `.directories` lists its directories.
Folders include hidden files and empty directories, but omit original metadata.
Missing paths, symlinks, special files, and files changing during collection fail
the return. Finish writing outputs before returning them.

`.save()` writes the exact local destination and requires an existing parent.
File replacement needs `overwrite=True`; folder destinations must be new.
Saving rejects symlink traversal and `..` and preserves existing files on failure.
Folders are visible while being written. Failed saves remove partial folders;
an abrupt process exit can leave them behind.
Results remain usable after closing the client. Filesystem operations target Linux.

Contents arrive in the same buffered response and are subject to server limits;
`execute()` does not save files automatically.

## Handling failures

A submission ends in one of three ways, and the difference between them matters:

```python
result = client.execute(program)
if result.status == "FAILED":
    error = result.error          # kind, message, instruction_index, instruction_id, traceback
```

- **`COMPLETED`** — every instruction ran.
- **`FAILED`** — one instruction failed, and the instructions after it were
  skipped. Returns that already ran are still in `results`, so putting a
  `return_` before a risky instruction preserves the work up to that point.
  `error["kind"]` is one of `parse`, `compile`, `runtime`, `gpu_access`,
  `correctness`, `serialization`, `unavailable` or `engine`, and
  `error["instruction_id"]` names the instruction that failed.
- **An exception** — the client could not obtain a valid program outcome.
  `KCoralError` carries `status_code` and `kind`: `503` with a
  `Retry-After` header means no worker was free, and `504` means the program hit
  `timeout_seconds` (default 300 s, maximum 900). `TransportError` means no HTTP
  response could be obtained; the server may already have executed the program,
  so check whether repeating its effects is acceptable before retrying.
  `ProtocolError` means the response did not follow the protocol.


## Next steps

<a id="the-shape-of-a-program"></a>
<a id="where-to-compile"></a>
<a id="languages-supported-by-remote-compilation"></a>
<a id="measuring"></a>
<a id="checking-correctness"></a>
<a id="running-your-own-code-off-the-gpu"></a>

- [Benchmark a Kernel with KCoral](../tutorials/benchmark-kernel.md) explains
  compilation choices, supported languages, correctness checks, measurement,
  uploaded harnesses and functions that release the GPU.
- [KCoral Protocol](protocol.md) defines endpoints and instruction fields.
- [Agent Integration Guide](../tutorials/agent-integration.md) shows how to give a
  coding agent the repository skill and a concrete execution task.
