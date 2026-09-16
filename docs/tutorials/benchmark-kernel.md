# Benchmark a Kernel with KCoral

Upload Python that runs your kernel, checks the output, and returns a timing
report. For convenience, KCoral provides `compile_tirx` and `benchmark` in
`kcoral.builtins` for your program to use directly. You can also use your own
compilation and measurement code.

## Prerequisites

Install the [GPU worker environment](../getting-started/installation.md#gpu-server)
and [launch the server](../server-guide/launch-the-server.md). The first example
uses TIRx, TVM's Python-embedded kernel language.

## Compile, check and measure

```bash
KCORAL_URL=http://localhost:8000 python examples/benchmark_kernel.py
```

```{literalinclude} ../../examples/benchmark_kernel.py
:language: python
```

{download}`Download the example <../../examples/benchmark_kernel.py>`.

`compile_tirx(kernel, bindings)` specializes a TIRx kernel and compiles it for
CUDA. `bindings` supplies constexpr values; a `PrimFunc` can be compiled without
bindings. Compiled executables are cached by structural hash, up to 32 entries
per process. The helper requires TVM in the worker environment.

`benchmark(compiled, src, dst)` measures GPU activity with CUPTI. Both functions
use the GPU access of the Python call that invokes them. Importing them into a
module also allows selecting them with `get_function` and calling them in
separate instructions.

## Where to compile

### On a GPU server

Upload your kernel and Python compilation code. For TIRx, import
`compile_tirx` from `kcoral.builtins`. CUDA C, CuTeDSL and Triton can use their own
compiler APIs. The example below demonstrates all four languages.
`Client.health()` reports installed versions.

Compilation can initialize or query CUDA and load GPU modules. Such calls run
under the GPU lease, alongside any later kernel execution.

### On a CPU server

A CPU server executes uploaded Python compilation code and returns library bytes
for a subsequent request to a GPU server. Supply the target architecture from
`Client.target()` on the GPU server. CPU workers also accept CUDA source uploads;
selecting a name returns the source text and name for your compiler to consume.

[Remote Compilation](remote-compilation.md) walks through building CUDA C
on a CPU server, then uploading the resulting library to a GPU server for execution
and measurement.

### On the client

Build a library locally and upload it with `kind="library"`. Build for the
architecture reported by the server's `Client.target()`. The
[library protocol](../client-guide/protocol.md#library) describes export and linking requirements,
and the example below demonstrates the build and submission.

## Checking correctness

Uploaded code can use, for example, `torch.testing.assert_close` to compare
output against a reference. Return any reports you want to inspect. An assertion
failure stops subsequent instructions; results already selected by `return_`
remain available.

## Measuring GPU activity

`kcoral.builtins.benchmark` measures each call from its first GPU activity to its last,
including kernels, copies and memsets. Host work before and after those endpoints
is excluded; gaps between GPU activities are included. This supports functions
that launch multiple GPU operations.

Pass an optional configuration dict after the callable's arguments:

```python
from kcoral.builtins import benchmark

timing = benchmark(kernel, src, dst, {"warmup_ms": 25, "repeat_ms": 100, "flush_l2": True})
```

Those are the defaults. The time budgets determine iteration counts from an
initial estimate. Explicit `warmup` and `repeat` counts override their respective
budgets. L2 flushing happens before each call and outside its measured span.

The report contains `latency_ms_median`, `latency_ms_mean`, `latency_ms_min`,
`latency_ms_max`, `warmup`, `repeat`, `flush_l2` and `activities_stable`. A false
`activities_stable` means calls did not all launch the same GPU activities.
The helper requires PyTorch and cupti-python in the worker environment.

Your Python can also invoke measurement tools such as NCU or Compute Sanitizer
and return their reports with `return_file` or `return_folder`. Wait for GPU
subprocesses to finish before returning or releasing the GPU lease.

## Running host work without the GPU lease

`get_function(..., cpu_only=True)` declares that calls to the selected function
use no GPU. Before its `run`, a GPU worker synchronizes outstanding work and
releases the lease, allowing another request to use the GPU. Detected CUDA calls
fail with `gpu_access`; this is a best-effort check.

The flag applies to the selected function's `run` calls. The module's top-level
Python still executes under the lease when uploaded. To separate host compilation
from GPU loading, return a path or bytes from a CPU-only build function, then
load the compiled result in a function with GPU access.

## Request timing

The execution response includes these timing fields, also exposed as attributes
on the Python client's `ProgramResult`. `queue_ms` measures waiting for a worker. After assignment, `elapsed_ms` includes
GPU-lease waiting (`lease_wait_ms`), holding the lease (`lease_held_ms`), and other
worker work. Holding a lease reserves the GPU but does not imply continuous GPU
activity. Use the benchmark report for kernel latency.

## Compilation examples

### Compile on the GPU server

This client compiles, checks and measures TIRx, CUDA C, CuTeDSL and Triton kernels.
Set `KCORAL_URL` to the GPU server address.

```bash
KCORAL_URL=http://localhost:8000 python examples/remote_compile_client.py
```

```{literalinclude} ../../examples/remote_compile_client.py
:language: python
```

{download}`Download the client <../../examples/remote_compile_client.py>`.

### Build and upload a library

This client builds a CUDA C library locally for the server's target architecture,
then uploads, checks and measures it. The client machine needs the CUDA toolkit,
a host C++ compiler and TVM FFI from the
[compiler environment](../getting-started/installation.md#cpu-compilation-server).

```bash
KCORAL_URL=http://localhost:8000 python examples/library_upload_client.py
```

```{literalinclude} ../../examples/library_upload_client.py
:language: python
```

{download}`Download the client <../../examples/library_upload_client.py>`.

For a separate compilation server, follow [Remote Compilation](remote-compilation.md).
