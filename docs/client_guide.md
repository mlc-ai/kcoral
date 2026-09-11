# Writing a benchmark program

This guide shows how to get a kernel timed on the server, and which choices
matter along the way. [`protocol.md`](protocol.md) is the field-level
specification — every instruction, every error code — but you should not need it
to write a first program.

## The shape of a program

Every benchmark follows the same six steps: upload the kernel, make the tensors,
compile, check correctness, time it, return the results.

```python
import numpy as np
from kcoral import Client, Program

KERNEL = r"""
from __future__ import annotations
from tvm.script import tirx as T

@T.jit
def main(A: T.Buffer((N,), "float32"), B: T.Buffer((N,), "float32"), *, N: T.constexpr):
    T.device_entry()
    i = T.cta_id([N])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0
"""
REFERENCE = "def main(a):\n    return a + 1.0\n"

program = Program()
kernel_module = program.upload(id="kernel_module", kind="module", source=KERNEL)
kernel = program.get_function(id="kernel", module=kernel_module, name="main")
reference_module = program.upload(id="reference_module", kind="module", source=REFERENCE)
reference = program.get_function(id="reference", module=reference_module, name="main")

src = program.upload(id="src", kind="tensor", value=np.arange(256, dtype=np.float32))
dst = program.run(id="dst", fn="builtin.empty", args=[{"shape": [256], "dtype": "float32"}])

compiled = program.run(id="compiled", fn="builtin.compile_tirx", args=[kernel, {"N": 256}])
program.run(id="invoke", fn=compiled, args=[src, dst])

expected = program.run(id="expected", fn=reference, args=[src])
check = program.run(id="check", fn="builtin.assert_close", args=[dst, expected])
timing = program.run(id="timing", fn="builtin.benchmark", args=[compiled, src, dst])

program.return_(key="check", value=check)
program.return_(key="timing", value=timing)

with Client("http://localhost:8000") as client:
    result = client.execute(program, timeout_seconds=120)
print(result.results["timing"]["latency_ms_median"])
```

Four rules the example relies on:

- **Handles live for one request.** `kernel`, `compiled` and the rest are names
  inside this one program. A second `client.execute()` call shares nothing with
  the first, so every program has to upload everything it needs.
- **`run` computes a value but does not send it back.** It binds the result to a
  handle that later instructions can use; the response carries only what
  `return_` selects. Above, `invoke` runs the kernel and the client never sees
  its result, while `timing` comes back because a `return_` names it.
- **Functions are selected explicitly.** A module upload binds its Python or
  CUDA source namespace. `get_function` selects a named object from it; the same
  instruction selects an exported function from a prebuilt TVM-FFI library.
- **An uploaded module need not be a kernel.** `REFERENCE` above is ordinary
  Python executed on the worker, torch included, so `def main(a, b): return a @ b`
  is a perfectly good baseline to measure against.

Runnable versions of all this live in [`../examples`](../examples):
`remote_compile_client.py` (four languages, compiled on the server) and
`library_upload_client.py` (built on the client, uploaded prebuilt).
`cpu_compile_gpu_execute.py` compiles on a CPU server and uploads the returned
library to a separate GPU server.

## Where to compile

A kernel can be built by the GPU server, by a separate CPU instance of the same
server, or by the client and uploaded as a finished shared object.

**Use a CPU server when the compiler and GPU should scale independently.** The
client reads `arch` from the GPU server, sends source and
`builtin.compile_cuda_binary` to the CPU server with a normal `/execute`
request, then uploads the returned bytes as a `library` in a second normal
request:

```python
arch = gpu_client.target()["arch"]
compiled = cpu_client.execute(build_compile_program(source, arch))
library = compiled.results["library"]
result = gpu_client.execute(build_benchmark_program(library))
```

The CPU server starts with `--device cpu --num-workers N`. It has no target of its
own, reports `gpu_count: 0`, and never initializes PyTorch or a CUDA context.
Only CUDA C compilation is supported in CPU mode; tensor creation, library
loading, correctness checks, and timing belong in the GPU request. See
[`../examples/cpu_compile_gpu_execute.py`](../examples/cpu_compile_gpu_execute.py)
for the complete programs.

**Compiling on the server is the recommended starting point.** Upload the kernel
as source text and let one of the `builtin.compile_*` builtins build it. The
server compiles for its own GPU architecture, and all four compile builtins are
registered `cpu_only`, meaning the worker gives up the GPU while compiling — so
your compile does not consume benchmark time, and another worker measures
meanwhile.

**Upload a prebuilt library** when your kernel needs custom compilation that the
server's `compile_*` builtins cannot support. You compile the kernel and upload
the library; the server only loads the compiled object and calls its exported
functions.

For CUDA C, nvcc and linking run with the GPU lease released; the server
reacquires it for the short module-loading phase before `compile_cuda` completes.
Compilation overlaps a neighbouring benchmark, its driver activity does not.

| | Server-side compile | Prebuilt library |
|---|---|---|
| Use when | The build is a standard one | Custom compilation the `compile_*` builtins cannot support |
| Client needs | Nothing | A CUDA toolchain and the right arch |
| Build surface | `extra_cuda_cflags` (CUDA C), `options` (CuTeDSL) — nothing else | Anything you can run |
| Repeat cost | Recompiled, except CUDA C (disk-cached) and Triton (its own cache) | Cached by hash — a resubmission re-sends nothing |
| Fails as | A `compile` error naming the diagnostic | `cudaErrorNoKernelImageForDevice` at launch, if built for the wrong arch |

In practice, reach for a library when the build spans several translation units,
comes out of a code generator, needs flags the builtins do not expose, or is a
CUTLASS-heavy recipe you already have working. Anything else is cheaper to
compile on the server.

A TVM-FFI library may export several launchers. Upload it, bind the functions
once, then use the returned registers anywhere a compiled kernel is accepted:

```python
module = program.upload(id="kernels", kind="library", value=library_bytes)
initialize = program.get_function(id="initialize", module=module, name="initialize")
step = program.get_function(id="step", module=module, name="step")

program.run(id="initialize_input", fn=initialize, args=[x])
program.run(id="invoke", fn=step, args=[x, y])
timing = program.run(
    id="timing",
    fn="builtin.benchmark",
    args=[step, x, y, {"warmup": 10, "repeat": 50}],
)
program.return_(key="timing", value=timing)
```

The module and its derived functions remain request-local handles. They can be
passed to later `run` instructions but cannot be returned to the client.

A library must be built for the GPU the server runs, not the one sitting in your
client machine:

```python
arch = client.target()["arch"]   # e.g. "sm_100a"
```

`protocol.md` covers the three producers a library may come from
(`TVM_FFI_DLL_EXPORT_TYPED_FUNC`, `tvm.Executable.export_library`, and CuTeDSL's
`--enable-tvm-ffi`) and the link flags each one needs.

## Languages supported by remote compilation

All four follow the same upload-then-compile shape; what differs is what the
compile call has to be told.

| Language | Upload | Compile call | Watch for |
|---|---|---|---|
| TIRx | `source`, then `get_function(name)` | `compile_tirx(kernel, bindings?)` | `bindings` supplies a `@T.jit` kernel's `T.constexpr` values; a `@T.prim_func` is already concrete and rejects them. The source must open with `from __future__ import annotations`, or a shape annotation like `T.Buffer((N,), dtype)` evaluates at `def` time and raises `NameError: N` |
| CUDA C | `source`, `language="cuda"`, then `get_function(name)` | `compile_cuda(kernel, cfg?)` | `void f(tvm::ffi::TensorView, ...)`; `main` is rejected; builds are cached on disk, so recompiling the same source is much cheaper than the first build |
| CUDA C on a CPU server | `source`, `language="cuda"`, then `get_function(name)` | `compile_cuda_binary(kernel, {"arch": arch, ...})` | `arch` must come from the GPU server; returns shared-object bytes rather than loading them |
| CuTeDSL | `source`, then `get_function(name)` | `compile_cutedsl(kernel, *tensors, cfg?)` | Specializes on the tensors, so pass the ones it will run on; nothing is cached |
| Triton | `source`, then `get_function(name)` | `compile_triton(kernel, *args, cfg)` | `cfg["grid"]` is required; pass scalars and constexprs positionally; other `cfg` keys are launch keywords |

If a server lacks the toolchain a builtin needs, that builtin fails with
`unavailable` and the rest of the server keeps working. `GET /health` lists the
`versions` actually installed, so check there before assuming a language is
available.

## Measuring

`builtin.benchmark(mod, *tensors, cfg?)` reports the **per-iteration GPU activity
span measured by CUPTI**: from the start of the first kernel, copy, or memset a
call launches to the end of the last. The L2 flush and host work outside those
endpoints stay out, but host time *between* two activities does not — several
kernels with Python in between measures that too. Iterations are drained like
flashinfer's `bench_gpu_time_with_cupti`, timing a kernel in isolation.

| `cfg` key | Default | Meaning |
|---|---|---|
| `warmup_ms` | `25` | How long to warm up. The server times 5 calls, then runs as many iterations as fit the budget |
| `repeat_ms` | `100` | How long to spend on timed iterations, converted to a count the same way |
| `warmup` | — | An explicit warmup iteration count, used instead of the budget |
| `repeat` | — | An explicit timed iteration count, used instead of the budget |
| `flush_l2` | `true` | Zero a buffer twice the size of L2 before every call, outside the timed span, so each call starts with a cold cache |

It returns `latency_ms_median`, `latency_ms_mean`, `latency_ms_min` and
`latency_ms_max`, along with `activities_stable` and the `flush_l2`, `warmup`
and `repeat` it actually used.

- **Check `activities_stable`.** It is `false` when the timed iterations did not
  all launch the same activities — a data-dependent kernel, say — so the stats
  describe a mixture rather than one kernel.
- **Prefer the millisecond budgets to explicit counts.** `warmup_ms` and
  `repeat_ms` adapt to the kernel, so a microsecond kernel and a millisecond
  kernel both get a sensible number of iterations. Set `warmup`/`repeat` when
  two runs have to use identical counts to be comparable; setting both of them
  also skips the 5-call estimate.
- **Leave `flush_l2` on** unless you specifically mean to measure a cache-hot
  kernel. With it off, a small kernel reads its input straight out of L2 and
  reports a latency it would never reach in a real pipeline.
- **Account for GPU time with `lease_held_ms`, not `elapsed_ms`.** Workers take
  turns on a GPU through a lease, and `lease_held_ms` is the part of the request
  that actually occupied the card. The remainder,
  `elapsed_ms - lease_wait_ms - lease_held_ms`, happened off the GPU — compiling,
  mostly.

## Checking correctness

`builtin.check_close(actual, expected, cfg?)` compares two tensors and returns
`passed`, `max_abs_err`, `max_rel_err`, `rtol` and `atol`. A mismatch is data
rather than a failure, so the program carries on. `builtin.assert_close` runs
the same comparison but raises a `correctness` failure when it does not pass,
which stops the program — use it when there is no point timing a kernel that
computes the wrong answer. Both default to `rtol=1e-2` and `atol=1e-3`.

Compare on the server against a Python reference module, the way the first
example does, rather than returning the output and comparing on the client. That
avoids shipping the output tensor back, and it keeps working for tensors too
large to return at all.

## Tensors

**Create them on the server** unless the client needs to specify their exact
values:

```python
program.run(id="x", fn="builtin.randn", args=[{"shape": [4096, 4096], "dtype": "bfloat16", "seed": 0}])
program.run(id="y", fn="builtin.zeros", args=[{"shape": [4096, 4096], "dtype": "float32"}])
```

Uploading instead pays for local generation, hashing, and the transfer, which
makes it markedly more expensive than creating the tensor on the server, and
increasingly so as the tensor grows. Re-uploading identical bytes hits the
server's blob cache and skips the transfer, but the cached bytes are still
copied to the GPU on every request, so the gap narrows rather than closes.

Upload when the values themselves matter: a reference you computed locally, a
fixed input your results have to stay reproducible against, or a tensor whose
contents the kernel's work depends on, such as the `q_indptr` and `kv_indptr`
arrays telling a paged-attention kernel where each sequence begins. `kind="tensor"`
accepts a NumPy array, a torch tensor, any object supporting DLPack, or raw
bytes together with `dtype` and `shape`.

`randn` requires a floating-point dtype and takes an optional `seed`. It,
`empty` and `zeros` all fall back to `float16` when `dtype` is omitted, so it is
worth stating explicitly.

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

## Calling builtins from uploaded code

An uploaded module executes in the worker process, where the server package
itself is importable, so builtins can also be called directly rather than
through `run` instructions:

```python
SOURCE = r"""
from kcoral import builtin

def main(x):
    y = builtin.randn({"shape": [256], "dtype": "float32", "seed": 0})
    return builtin.check_close(x, y)
"""
```

`kcoral.builtin` resolves attributes through the same registry a
`run` instruction uses, so `builtin.check_close` above is exactly the function
`"builtin.check_close"` names on the wire — same behaviour, same error kinds.
`dir()` on the module lists every registered name.

One caveat: the `compile_*` builtins give the GPU up only when they run as
their own instruction. Called from inside uploaded code, a compile runs while
the worker holds the GPU, so its time counts against `lease_held_ms` and
blocks other workers' measurements. Create tensors, compare, and time freely
from code; keep compiles at the instruction level.

## Running your own code off the GPU

The compile builtins give the GPU up while they run. Uploaded code can do the
same when it needs no GPU — a reference computed on the CPU, a custom build
step, parsing an uploaded file — by declaring the function `cpu_only` when
selecting it:

```python
CPU_REFERENCE = r"""
import torch

def main(n):
    x = torch.arange(n, dtype=torch.float32)  # CPU tensors throughout
    return x + 1.0
"""

module = program.upload(id="reference_module", kind="module", source=CPU_REFERENCE)
reference = program.get_function(id="reference", module=module, name="main", cpu_only=True)
expected = program.run(id="expected", fn=reference, args=[256])
check = program.run(id="check", fn="builtin.assert_close", args=[dst, expected])
```

A `run` of a `cpu_only` handle releases the GPU lease first, so its time lands
outside `lease_held_ms`. Anything that touches CUDA — creating a tensor on the
device, reading a GPU tensor back with `.cpu()` — fails the instruction with a
`gpu_access` error naming the call and the source line, so give such a function
CPU data. `check_close` and `assert_close` accept its CPU result as `expected`.
The check is best effort: it cannot see a child process, and it catches a call
only after it reached the GPU. The flag applies to the handle as a `run` target
only; passed to `benchmark`, the function runs on the GPU's time.
[`../examples/remote_compile_client.py`](../examples/remote_compile_client.py)
checks every kernel against such a reference.

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
- **An exception** — the request never produced a program outcome at all.
  `KCoralError` carries `status_code` and `kind`: `503` with a
  `Retry-After` header means no worker was free, and `504` means the program hit
  `timeout_seconds` (default 300 s, maximum 3600). `TransportError` means the
  request never reached the server, and `ProtocolError` means the response did
  not follow the protocol.
