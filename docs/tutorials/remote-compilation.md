# Remote Compilation

A GPU server can compile and evaluate a kernel in one request, as shown in
[Benchmark a Kernel with KCoral](benchmark-kernel.md). When compilation takes
much longer than execution, however, moving builds to CPU-only machines lets
you scale compilation capacity independently of your GPU machines.

In this tutorial, you will compile a CUDA C kernel on a CPU server and execute
it on a separate GPU server. The first request returns the compiled shared
library as bytes to the client. The client then uploads those bytes to the GPU
server in a second request, which checks the kernel's output and measures its
execution time.

This workflow requires a compiler that can build without accessing a GPU.
Compiler APIs that query CUDA or load GPU modules during compilation still
need a GPU server.

The client carries the compiled artifact between the requests. Each request
has its own `Program`:

| Request | Destination | Inputs | Returned values |
| --- | --- | --- | --- |
| 1. Compile | CPU server | Kernel source and the execution GPU's target architecture | Shared-library bytes |
| 2. Execute | GPU server | Those library bytes and the workload | Correctness and timing reports |

You will follow this two-request workflow with a client that compiles a CUDA C
add-one kernel and then checks and benchmarks it.

## Prepare the two servers

The CPU server needs build tools, while the GPU server needs the
GPU runtime and measurement tools. Prepare each environment for its role.
Install the [client](../getting-started/installation.md#install-the-client) on the
machine coordinating the requests. It needs no local compiler or GPU.
The two server roles have different requirements:

| Role | Environment |
| --- | --- |
| CPU compiler | The [compiler environment](../getting-started/installation.md#server-system-requirements), the CUDA toolkit with `nvcc`, and a host C++ compiler |
| GPU executor | The [GPU worker environment](../getting-started/installation.md#server-system-requirements), including PyTorch, TVM FFI and CUPTI for benchmarking |

The compiled library must be compatible with the GPU server's operating system,
CPU architecture, GPU architecture, and runtime dependencies. Keep the two
servers' TVM FFI and CUDA components compatible. See the
[library upload protocol](../client-guide/protocol.md#library) for loading and
export requirements.

This example uploads Python that builds CUDA C through TVM FFI, then uses
CUPTI to collect GPU activity timestamps.

```{warning}
KCoral allows clients to execute arbitrary code on its workers. Only allow
trusted clients to access your KCoral server or Router. Deploy on a trusted,
isolated network and never expose these endpoints to the public internet.
Run workers in a sandbox with restricted permissions and access to host resources.
```

For a local demonstration, start a CPU server in one terminal:

```bash
kcoral server --device cpu --num-workers 8 --host 127.0.0.1 --port 8000
```

In another terminal on the same machine, start a GPU server:

```bash
kcoral server --device gpu --gpus 0 --host 127.0.0.1 --port 8001
```

Both commands listen only on the local machine. To run the servers on separate
machines, use `--host 0.0.0.0` on each server to accept connections over a trusted
network, and use their reachable addresses in the client command below. See
[Launch the server](../server-guide/launch-the-server.md) for configuration.

## Run the complete client

First run the whole workflow to see the two servers working together. The
sections that follow walk through how the client constructs each request.
From the repository checkout, run the client with both servers on the local
machine, or replace `127.0.0.1` with each server's reachable address:

```bash
KCORAL_CPU_URL=http://127.0.0.1:8000 \
KCORAL_GPU_URL=http://127.0.0.1:8001 \
python examples/cpu_compile_gpu_execute/main.py
```

{download}`Download the complete client <../../examples/cpu_compile_gpu_execute/main.py>`.
The downloaded file can be run directly with the same environment variables.
If both servers run on the client machine, the defaults are
`http://127.0.0.1:8000` for compilation and `http://127.0.0.1:8001` for execution.

On success, the script prints three lines with this format; the size,
architecture, and GPU timings depend on your environment:

```text
compiled <size> KiB for <arch>
CPU request held a GPU lease for 0 ms
GPU request held its lease for <time> ms; kernel median <latency> us
```

The CPU request compiles without using a GPU. The GPU request checks correctness
before measuring the kernel, so reaching this output means both requests
completed and the correctness check passed. A
[GPU lease](benchmark-kernel.md#overlap-cpu-work-with-gpu-execution) is exclusive access to
the device; the GPU request's lease time covers more than the kernel measurement.
If either program returns `FAILED`, the script instead reports
`compile failed: ...` or `benchmark failed: ...` with the error details.
HTTP and connection failures raise client exceptions.

Now that you have seen the complete workflow, the following sections trace how
the client chooses a compilation target, builds the library, and submits it to
the GPU server.

## Read the GPU target

The compiler needs to know what GPU architecture to build for. The CPU server has
no GPU from which to determine that architecture, so the client asks the GPU
server before building:

```python
arch = gpu_client.target()["arch"]
```

The client passes `arch` to the uploaded `compile_cuda_binary` function so the
library is built for the GPU that will execute it.

## Compile and return the library

The first request turns source code into a library that the client can carry
to the GPU server. The source defines a GPU kernel and a host function,
`add_one`, which launches
it. The host function uses TVM FFI's `TensorView` parameters to access tensors.
The uploaded compiler uses `tvm_ffi.cpp.build_inline` to supply the required
includes and export wrapper.

```{literalinclude} ../../examples/cpu_compile_gpu_execute/main.py
:language: python
:start-at: CUDA_SOURCE =
:end-before: REFERENCE =
```

The example's `OPERATIONS` string defines the Python compiler and execution
helpers uploaded by each program:

<div class="code-example">
<div class="code-example-preview">

```{literalinclude} ../../examples/cpu_compile_gpu_execute/main.py
:language: python
:start-at: OPERATIONS =
:end-before: def compile_cuda_binary
```

</div>
<details>
<summary><span class="code-example-expand">Show full source</span><span class="code-example-collapse">Show less</span>: <code>cpu_compile_gpu_execute/main.py</code></summary>

```{literalinclude} ../../examples/cpu_compile_gpu_execute/main.py
:language: python
:start-at: OPERATIONS =
:end-before: def compile_program
```

</details>
</div>

The compilation program uploads that Python and the CUDA source file, selects
`compile_cuda_binary` with `get_function(..., cpu_only=True)`, passes the file path
and configuration (`functions=["add_one"]`, `arch`), and returns the library:

```{literalinclude} ../../examples/cpu_compile_gpu_execute/main.py
:language: python
:pyobject: compile_program
```

The uploaded `compile_cuda_binary` function builds a shared library without loading it or
launching the kernel. `return_(key="library", ...)` selects the bytes for the
response. After `cpu_client.execute()` succeeds,
`compiled.results["library"]` is a Python `bytes` object containing the shared
library, equivalent to the contents of a `.so` file.

## Upload and benchmark the library

With compilation complete, the second request can focus on checking and
measuring the kernel. Its program receives those bytes as its `library` argument. It uploads
them with `kind="library"`, selects the exported function, creates input and
output tensors, and runs the kernel:

```{literalinclude} ../../examples/cpu_compile_gpu_execute/main.py
:language: python
:pyobject: benchmark_program
```

The GPU server loads the uploaded library, and `get_function` selects its
exported `add_one` function. Calling it with `program.run` launches the kernel
that was compiled on the CPU server. The program then checks the output and
benchmarks the kernel only if that check passes, following the
[benchmark tutorial](benchmark-kernel.md#check-correctness).

## Submit both programs

The client's `main` function brings these steps together. It connects to both
servers, runs the compilation program, and uses the returned library to run
the benchmark program:

```{literalinclude} ../../examples/cpu_compile_gpu_execute/main.py
:language: python
:pyobject: main
```

The client checks each result before proceeding: a failed compilation stops it
before the GPU submission, and a failed GPU program stops it before printing a
successful timing report. Inspect the result's `error` and `request_id` to
identify the failing instruction and find the corresponding server logs. See
[Read results](../client-guide/writing-a-program.md#read-results) for handling
program failures and connection exceptions.

After the GPU server finishes, the client reads the timing report from
`result.results["timing"]` and prints the kernel's median duration. The
correctness report is also available in `result.results["check"]`. For the
meaning of the timing statistics, see
[Measure GPU activity](benchmark-kernel.md#measure-gpu-activity).
