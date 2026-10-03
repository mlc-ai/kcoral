# Write a client program

To submit a benchmark to KCoral, you describe the work in a program. Each program
represents one request: it specifies the code and data to upload, the functions
to run, and the results to return. You build the program locally, send it to the
server, and receive the results after the server executes its instructions in
order. The [first GPU program](../getting-started/quickstart.md) walks through
this workflow with a complete runnable example.

## Preparation

Before submitting a program, [install the server](../getting-started/installation.md#install-the-server)
on a Linux machine with an NVIDIA GPU and a compatible driver, then start it:

```bash
kcoral server --device gpu --gpus 0 --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete.` and leave the server running on a
trusted network accessible only to trusted clients. When connecting from
another machine, replace `127.0.0.1` in the examples below with the server's
reachable hostname or IP address. For details, see
[Launch the server](../server-guide/launch-the-server.md).

## Build programs with instructions

A program is a sequence of instructions that one worker on the server executes
in order. Each instruction describes one step, such as uploading data or calling
a function. The `Program` builder provides methods for composing these steps
into a request. There are four core instructions:

| Instruction | Python method | Purpose |
| --- | --- | --- |
| `upload` | `upload(kind=..., ...)` | Upload code, tensors, bytes, or compiled libraries |
| `get_function` | `get_function(module=..., name=...)` | Select a function from an uploaded module or library |
| `run` | `run(fn=..., args=...)` | Call a function with the supplied arguments |
| `return` | `return_(key=..., value=...)` | Select a value to send back to the client |

The builder also provides helpers for working with files: `upload_file()` and
`upload_folder()` add uploads to the program, while `return_file()` and
`return_folder()` select files or folders to send back. These use the same
`upload` and `return` instructions. The sections on
[uploading files](#upload-files-and-folders) and
[returning files](#return-files-and-folders) cover these helpers in detail.

When an instruction produces a value, the builder returns a `Register`: a local
Python reference to the value that will exist on the server when the instruction
runs. Pass registers to later instructions to select a function to call, supply
its arguments, or choose a result to return. This lets you pass data through
several function calls within one program. Each register must come from an
earlier instruction in the same program.

With `Program` imported from `kcoral`, this snippet builds a program that
multiplies `3` by `2` and returns the result as `"output"`:

```python
program = Program()
module = program.upload(
    kind="module", source="def scale(x, factor): return x * factor"
)
scale = program.get_function(module=module, name="scale")
y = program.run(fn=scale, args=[3, 2])
program.return_(key="output", value=y)
```

`upload()` supplies the code, and `get_function()` selects `scale` from it.
The `scale` register tells `run()` which function to call; `args` supplies its
inputs. Finally, `return_()` selects the value referenced by `y` to send back
under the key `"output"`. After submission, that value will be `6`.

See the [Python API](../python-api/index.rst) for complete method signatures and
parameter types.

## Create a client and submit programs

The Python client's `Client` class connects to a KCoral server. Create a client
with the server's address; you can use the same client to submit many programs.

```python
from kcoral import Client

with Client("http://127.0.0.1:8000", connect_timeout_seconds=10) as client:
    health = client.health()
    target = client.target()
    print(health["versions"])
    print(target["arch"])
```

The client provides `health()` to check the server's status and software versions,
and `target()` to check its GPU architecture.

The first argument is the server URL. `connect_timeout_seconds` controls how long
the client waits to open a connection to that server, in seconds. For example,
if the server's machine is unreachable and the connection attempt receives no
response, the setting above stops that attempt after 10 seconds. This setting
does not limit how long a submitted program can run. See the
[Python API](../python-api/index.rst) for all client options.

### Submit a simple program

To demonstrate `client.execute()` and its options, this example uploads one line
of Python that prints `2` on the server. The call submits the program and returns
its results, including the captured output:

```python
from kcoral import Client, Program

print_program = Program()
print_program.upload(kind="module", source="print(1 + 1)")

with Client("http://127.0.0.1:8000") as client:
    result = client.execute(print_program, timeout_seconds=30, output_limit_bytes=1024)

if not result.completed:
    raise RuntimeError(result.error)
print(result.stdout, end="")  # 2
```

Here, `timeout_seconds=30` gives the server a 30-second execution limit, and
`output_limit_bytes=1024` captures up to 1 KiB each of standard output and
standard error. The server runs `print(1 + 1)` after `client.execute(print_program, ...)`
submits the program, rather than when `print_program.upload(...)` adds the instruction.

Each submission is independent: the server keeps no program session between
requests. When a request finishes, its temporary files and values are discarded,
and its registers can no longer be used to access them. You can submit the same
`Program` again, but its instructions will run again with fresh values. Results
already returned to the client remain available.

### Read results

For the multiplication program built earlier, `return_(key="output", value=y)`
selects the value to retrieve. After submitting that program, check whether it
completed and read `"output"` from the response:

```python
with Client("http://127.0.0.1:8000") as client:
    result = client.execute(program, timeout_seconds=120, output_limit_bytes=65536)

if result.completed:
    output = result.results["output"]
    # result["output"] is the same lookup.
else:
    print(result.error)
print(result.stdout, result.stderr)
```

The returned `ProgramResult` contains the values you selected in `results`, along
with the request's `status`, `request_id`, and any `error`. It also includes
captured `stdout` and `stderr` and flags indicating whether either was truncated.
Returned tensors arrive as NumPy arrays in client CPU memory, and byte results
arrive as Python `bytes`. Modules and callable handles cannot be returned.

A `COMPLETED` result means every instruction ran successfully. If an instruction
fails, the result has status `FAILED`, and `result.error` describes what went
wrong. The server skips the remaining instructions, but values selected by
`return_()` before the failure remain in `result.results`. Returning an
intermediate result can therefore preserve useful work if a later step fails.

Some failures prevent the client from obtaining a program result. In those
cases, `execute()` raises an exception: `KCoralError` for an HTTP error from the
server, `TransportError` for a connection or transfer problem, or `ProtocolError`
for an invalid response. A lost connection does not tell you whether the server
ran the program, so consider whether repeating its work is safe before retrying.
See the {ref}`Python API error reference <python-errors>` for details.

Alongside these results, the response includes timing metrics to help you
understand where the request spent its time. `queue_ms` measures how long it
waited for a worker, while `elapsed_ms` measures time spent on the worker.
Within that worker time, `lease_wait_ms` records the wait for exclusive GPU
access, and `lease_held_ms` records how long the request held that access.
These metrics can help distinguish a busy server or GPU from a slow program.
They describe the request as a whole; use the measurements from your benchmark
harness or profiler to assess the kernel's execution time.

## Work with tensors and files

A program may need tensor inputs or files for its code to read, and it may
produce tensors or files you want to retrieve. You can upload data from the
client or create it on the server, then select the outputs to return. The
following snippets illustrate separate uses of a `Program`; start with a fresh
`program = Program()` for each example.

### Create and upload tensors

Tensors hold the inputs and outputs of most machine learning kernels. KCoral
lets you create them on the server or upload them from the client, then pass
them to functions through registers.

When you only need generated inputs, such as random values for a benchmark, you
can create them directly on the GPU by uploading a Python function:

```python
module = program.upload(kind="module", source="""
import torch

def make_input():
    return torch.randn((4096, 4096), dtype=torch.bfloat16, device="cuda")
""")
make_input = program.get_function(module=module, name="make_input")
x = program.run(fn=make_input)
```

When you already have input data on the client, upload it with `kind="tensor"`:

```python
import numpy as np

values = np.zeros((4, 4), dtype=np.float32)
x = program.upload(kind="tensor", value=values)
```

`kind="tensor"` accepts a NumPy array, a torch tensor, any object supporting
DLPack, or raw bytes together with `dtype` and `shape`.

Supported dtypes include `bool`, `uint8`, `int8`, `int16`, `int32`, `int64`, `float16`,
`float32`, `float64`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2`.

To retrieve a tensor, select its register with `program.return_(key="output", value=x)`.
After execution, `result.results["output"]` is a NumPy array in client CPU memory.
The `bfloat16` and float8 dtypes use `ml_dtypes`. With NumPy imported as `np` and
PyTorch as `torch` on the client, convert a returned `bfloat16` array named `value`
to a PyTorch tensor by reinterpreting its bytes:

```python
torch.from_numpy(value.view(np.uint8)).view(torch.bfloat16)
```

### Upload files and folders

Use `upload_file()` to make a file available to code running on the server.
Supply its contents as bytes and choose a destination in the request's working
directory:

```python
program.upload_file(blob=tensor_bytes, path="inputs/tensor.bin")
```

To upload an existing local file, read its contents and pass them as `blob`:

```python
from pathlib import Path

program.upload_file(blob=Path("./data/tensor.bin").read_bytes(), path="inputs/tensor.bin")
```

Here, `./data/tensor.bin` is the local source file, and `inputs/tensor.bin` is its
destination on the server. Use either upload above; both create the same remote
path.

Code that runs after this instruction can read the file with
`open("inputs/tensor.bin", "rb")`. The upload also returns a register containing
the relative path, which you can pass as an argument to a function.

To upload a local folder and its contents, use `upload_folder()`:

```python
program.upload_folder("./assets", path="inputs")
```

This places `assets/a` at `inputs/a` and `assets/sub/b` at `inputs/sub/b`.
Both helpers capture the contents when called, so later changes to local files
do not change a program you have already prepared. Folder uploads include hidden
files, but omit empty directories and original file permissions. Symbolic links
and special files are not supported.

Upload destinations are relative to the request's working directory, which is
removed when the request finishes. See the {ref}`file upload rules <file>` for
path requirements and restrictions.

### Return files and folders

A program may produce files you want to keep, such as a profiling report or a
folder of debugging output. Use `return_file()` or `return_folder()` to retrieve
these outputs. Add the instructions after the code that writes the files, then
save the returned contents on the client:

```python
program.return_file(key="report", path="outputs/report.txt")
program.return_folder(key="debug", path="outputs/debug")
with Client("http://127.0.0.1:8000") as client:
    result = client.execute(program)

if not result.completed:
    raise RuntimeError(result.error)
result["report"].save("report.txt")
result["debug"].save("debug")
```

The `path` identifies a file or folder in the request's working directory. It
can be a relative path string or a register containing one. Each return captures
the contents at that point in the program, so finish writing the files first.

Calling `execute()` retrieves the selected contents, and `.save()` writes them
to a local destination. The destination's parent directory must already exist.
To replace an existing file, pass `overwrite=True`; a folder destination must
be new. Returned contents remain available after the client is closed.

You can also inspect the contents without saving them: `ReturnedFile.read_bytes()`
returns the file's bytes, while `ReturnedFolder.files` and `.directories` expose
the folder's files and directory paths. Returned folders include hidden files
and empty directories, but do not preserve original file metadata. See the
[Python API](../python-api/index.rst) for the full return and save options.

## Next steps

For complete programs, browse the repository's
[examples directory](https://github.com/mlc-ai/kcoral/tree/main/examples).
The tutorials below explain how to use these programs for common workflows.

- [Benchmark a Kernel with KCoral](../tutorials/benchmark-kernel.md): follow
  `examples/benchmark_kernel.py` to compile a kernel, check its output, and
  measure its execution time.
- [Remote Compilation](../tutorials/remote-compilation.md): compile on a CPU
  server and run the resulting library on a GPU server.
- [Agent Integration Guide](../tutorials/agent-integration.md): give a coding
  agent the context it needs to write and submit KCoral programs.
- [Python API](../python-api/index.rst) and [KCoral Protocol](protocol.md): look up
  method signatures or the underlying request and response formats.
