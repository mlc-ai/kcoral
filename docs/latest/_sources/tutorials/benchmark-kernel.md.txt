# Benchmark a Kernel with KCoral

Evaluating a kernel means answering two questions: does it produce the right
output, and how long does it take to run? In this tutorial, you will send a
small add-one kernel to a KCoral GPU server, check its output against a reference,
and retrieve a timing report. The client describes the experiment; compilation,
correctness checks, and measurement happen on the server.

The first example uses KCoral's {py:func}`~kcoral.builtins.compile_tirx` and
{py:func}`~kcoral.builtins.benchmark` helpers. Once you
have followed that flow, you can substitute your own compiler or measurement
code and use the same program structure for other kernel languages.

## Prerequisites

On the server machine, install the
[GPU worker environment](../getting-started/installation.md#server-system-requirements)
and [launch the server](../server-guide/launch-the-server.md). This example
uses TIRx, TVM's Python-embedded kernel language, so the server needs TVM for
compilation, PyTorch for tensors and correctness checks, and CUPTI for measurement.

On the client machine, install the
[KCoral client](../getting-started/installation.md#install-the-client), which
includes NumPy for preparing the input array. This example needs no local GPU,
CUDA toolkit, or TVM installation: the client sends source and data to the server.

## Run a complete example

Start with `examples/benchmark_kernel.py`, which adds one to each of 256
`float32` values. From the repository checkout, run it against your GPU server:

```bash
KCORAL_URL=http://127.0.0.1:8000 python examples/benchmark_kernel.py
```

<div class="code-example">
<div class="code-example-preview">

```{literalinclude} ../../examples/benchmark_kernel.py
:language: python
:lines: 1-24
```

</div>
<details>
<summary><span class="code-example-expand">Show full source</span><span class="code-example-collapse">Show less</span>: <code>benchmark_kernel.py</code></summary>

```{literalinclude} ../../examples/benchmark_kernel.py
:language: python
```

</details>
</div>

{download}`Download the example <../../examples/benchmark_kernel.py>`.

The client uploads the kernel and an `evaluate` function, uploads the input
tensor, and calls `evaluate` on the server. That function allocates an output
tensor, compiles the kernel, and checks its result against `src + 1`. It only
benchmarks the kernel after that check passes. On success, the script prints
`{'passed': True}` followed by the timing report; otherwise it reports the
program error. The successful output has this shape, with the timing dictionary
abridged here and the latency depending on your GPU:

```text
{'passed': True}
{'latency_ms_median': <latency>, ...}
```

Inside `evaluate`, `compile_tirx(kernel, bindings)` specializes a TIRx kernel and compiles it for
CUDA. `bindings` supplies constexpr values; a `PrimFunc` can be compiled without
bindings. Compiled executables are cached by structural hash, up to 32 entries
per process. The helper requires TVM in the worker environment.

`benchmark(compiled, src, dst)` measures GPU activity with CUPTI. Both functions
use the GPU access of the Python call that invokes them. Importing them into a
module also allows selecting them with `get_function` and calling them in
separate instructions.

The rest of this tutorial looks at the choices behind that example: where to
compile, how to overlap CPU and GPU work, and how to check and measure the kernel.

## Choose where to compile

The first example compiles on the GPU server, keeping the experiment in one
request. You can also compile elsewhere when you want to reuse an existing
build environment or scale compilation separately from GPU execution.

### On a GPU server

Compiling on the GPU server keeps the client lightweight and gives the compiler
access to the device it is targeting. Upload your kernel and Python compilation
code. For TIRx, import
`compile_tirx` from `kcoral.builtins`. CUDA C, CuTeDSL and Triton can use their own
compiler APIs. `Client.health()` reports installed versions.

For a complete example that compiles, checks, and measures kernels in all four
languages, run this client from the repository checkout:

```bash
KCORAL_URL=http://127.0.0.1:8000 python examples/remote_compile_client.py
```

{download}`Download remote_compile_client.py <../../examples/remote_compile_client.py>`.

Compilation can initialize CUDA or load GPU modules. Calls that do this need
exclusive access to the GPU, just as kernel execution does, so another request
cannot use that GPU until the call gives up access.

### On the client

If you already have the compiler on your client machine, you can keep the build
local and use the remote GPU for evaluation. The
`examples/library_upload_client.py` example does this for a
CUDA C add-one kernel. It needs the CUDA toolkit, a host C++ compiler, and TVM FFI
with its C++ build dependencies on the client. See
[local compilation setup](../getting-started/installation.md#prepare-for-local-compilation-optional).

Its `build_library` function writes the CUDA source to a temporary directory,
compiles it for the supplied GPU architecture, and reads the resulting shared
library as bytes:

```{literalinclude} ../../examples/library_upload_client.py
:language: python
:pyobject: build_library
```

Here, `SOURCE` is the example's CUDA code, including a TVM FFI export for
`add_one`. The client reads the remote GPU's architecture before building:

```{literalinclude} ../../examples/library_upload_client.py
:language: python
:start-at:         arch = client.target()
:end-at:             library = build_library(arch, directory)
:dedent: 8
```

The program then uploads the compiled bytes and selects the exported function:

```{literalinclude} ../../examples/library_upload_client.py
:language: python
:start-at:     module = program.upload(kind="library"
:end-at:     kernel = program.get_function(module=module, name="add_one")
:dedent: 4
```

Later instructions call `kernel` with the input and output tensors, check the
result, and measure it. Compilation has already finished on the client, so the
server only needs to load and execute the library. See the
[library protocol](../client-guide/protocol.md#library) for export and linking
requirements. Run the complete client from the repository checkout:

```bash
KCORAL_URL=http://127.0.0.1:8000 python examples/library_upload_client.py
```

{download}`Download library_upload_client.py <../../examples/library_upload_client.py>`.

### On a CPU server

If a CPU server is available and the build does not require a GPU, you can use
it for compilation. It executes uploaded Python compilation code and returns
library bytes for a subsequent request to the GPU server. Supply the target architecture from
`Client.target()` on the GPU server. Upload CUDA source with `upload_file` and pass
its path and export names to your compilation function.

[Remote Compilation](remote-compilation.md) walks through building CUDA C
on a CPU server, then uploading the resulting library to a GPU server for execution
and measurement.

## Overlap CPU work with GPU execution

Some parts of an experiment, such as preparing source files or running a
CPU-only compiler, do not need the GPU. While one request does that work,
another could use the GPU to run a kernel.

KCoral coordinates this by granting one worker exclusive GPU access at a time,
called a *GPU lease*. By default, an uploaded function keeps that access for
its entire call. If the function does no GPU work, select it with
`get_function(..., cpu_only=True)`. Before running it, the worker waits for its
outstanding GPU work to finish and releases its exclusive access, allowing
another request to use the GPU. Detected CUDA calls in the CPU-only function
fail with `gpu_access`; this is a best-effort check.

The `cpu_only=True` option applies only when `run` invokes the selected
function. Uploading a Python module is a separate step: the server executes
its imports and other statements outside function bodies to create the module.
For example, a tensor allocation written outside a function runs during upload.
That code may initialize CUDA or use the GPU, so KCoral gives the module upload
exclusive GPU access even if you later select one of its functions as CPU-only.

To let compilation overlap with another request's GPU work, put the build in
a function that does not access the GPU and select it with `cpu_only=True`.
For example, it could run `nvcc` to produce a shared library and return the
library's file path. A later function can take that path, load the library,
and launch the kernel. Leave `cpu_only` at its default of `False` for this
second function so KCoral gives it exclusive GPU access. Keeping the two steps
separate lets other requests use the GPU while the compiler runs.

## Check correctness

Before trusting a timing result, check that the kernel computes the expected
answer. The first example uses `torch.testing.assert_close` to compare its
output against a reference; your uploaded code can do the same with tolerances
appropriate to your workload. You can also upload your own correctness-checking
functions or scripts to validate outputs against a reference or test properties
specific to your kernel. Return any reports you want to inspect. An assertion
failure stops subsequent instructions; results already selected by `return_`
remain available.

## Measure GPU activity

Once correctness passes, measure how long the kernel's GPU work takes.
`kcoral.builtins.benchmark` measures each call from its first GPU activity to
its last, including kernels, copies and memsets. Host work before and after those endpoints
is excluded; gaps between GPU activities are included. This supports functions
that launch multiple GPU operations.

To adjust the measurement, pass an optional configuration dict after the
callable's arguments in your uploaded Python. Here, `kernel` is the callable
to measure, and `src` and `dst` are its input and output tensors:

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

Use this report to assess kernel latency. The request timing fields on
`ProgramResult` include other costs, such as waiting and compilation; see
[reading results](../client-guide/writing-a-program.md#read-results) when
investigating a slow request.

For profiling with NCU or run-iket, or checking GPU memory access with Compute
Sanitizer, use the [builtin CLI tools](../client-guide/builtin-cli-tools.md).
They run the tool on the server and retrieve its reports for you.
