# Benchmark a Kernel with KCoral

This tutorial compiles a vector-add kernel, checks its output against a Python
reference and returns GPU timing statistics. A kernel is a function executed on
the graphics processing unit (GPU). The client builds a request; the server
owns compilation, device data and measurement.

## Prerequisites

Install the [GPU worker environment](../getting-started/installation.md#running-gpu-programs)
and [launch the server](../server-guide/launch-the-server.md). The first example uses TIRx,
TVM's Python-embedded kernel language. TVM is a tensor compiler. Later sections
explain CUDA C, NVIDIA's GPU extension to C++, CuTeDSL, NVIDIA's Python language
for CuTe kernels, and Triton, a GPU kernel language and compiler.

## The shape of a program

Every benchmark has six steps: upload the kernel, make the tensors, compile,
check correctness, measure it, and explicitly return the results.

```bash
KCORAL_URL=http://localhost:8000 python examples/benchmark_kernel.py
```

```{literalinclude} ../../examples/benchmark_kernel.py
:language: python
```

{download}`Download the complete benchmark <../../examples/benchmark_kernel.py>`.

The script prints a correctness report with `passed: True` and timing statistics
including `latency_ms_median`. Latency varies by GPU and workload; there is no
fixed expected number. A failed compilation, check or measurement stops the
program and is printed as an error rather than presented as a timing result.

The output `dst` is explicitly populated by the kernel before comparison.
`assert_close` stops the request on a mismatch, so timing only follows a
successful correctness check. Registers last for this request; read
[Writing a Program](../client-guide/writing-a-program.md) for their lifecycle.

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
loading, correctness checks, and timing belong in the GPU request.

**Compiling on the server is the recommended starting point.** Upload the kernel
as source text and let one of the `builtin.compile_*` builtins build it. The
server compiles for its own GPU architecture, and the compilation builtins are
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

[KCoral Protocol](../client-guide/protocol.md#library) covers the three producers a library may come from
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
span measured by CUPTI (NVIDIA's CUDA Profiling Tools Interface)**: from the start of the first kernel, copy, or memset a
call launches to the end of the last. The L2 (second-level cache) flush and host work outside those
endpoints stay out, but host time *between* two activities does not — several
kernels with Python in between measures that too. Iterations are drained like
flashinfer's `bench_gpu_time_with_cupti`, timing a kernel in isolation.

See [measurement configuration](../client-guide/builtin-tools.md#measurement-configuration)
for all options and defaults.

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
The [Remote Compilation tutorial](remote-compilation.md) uses this approach
to check kernels in each supported language.
