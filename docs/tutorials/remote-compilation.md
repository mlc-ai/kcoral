# Remote Compilation

In large-scale kernel evaluation, compilation can take much longer than kernel
execution. Build steps that need only a CPU (central processing unit) can run
separately from execution on a GPU (graphics processing unit). Running that
host work on expensive GPU servers can increase cost and leave GPUs underutilized.
Some compiler APIs query CUDA or load GPU modules during compilation; those
steps need GPU access.

A more cost-effective approach is to compile on inexpensive CPU-only machines,
download the compiled result, and upload it to a GPU server for execution.
KCoral supports this workflow through file upload and download: one request
returns a compiled shared library, and the next uploads it for remote execution.

The two requests can use different servers. A single GPU server with the
compilation tools installed can also serve both roles.

The client carries the compiled artifact between the requests. Each request
has its own `Program` and its own handles:

| Request | Destination | Inputs | Returned values |
| --- | --- | --- | --- |
| 1. Compile | Compilation server | Kernel source and the execution GPU's target architecture | Shared-library bytes |
| 2. Execute | Execution server | Those library bytes and the workload | Correctness and timing reports |

Both programs use `POST /execute`. The complete client below compiles a CUDA C
add-one kernel and then checks and benchmarks it over 1,048,576 `float32`
elements. CUDA C is NVIDIA's GPU extension to C++.

## Prepare the two servers

Install the [client](../getting-started/installation.md#install-the-client) on the
machine coordinating the requests. It needs no local compiler or GPU.
The two server roles have different requirements:

| Role | Environment |
| --- | --- |
| CPU compiler | The [compiler environment](../getting-started/installation.md#cpu-compilation-server), the CUDA toolkit with `nvcc`, and a host C++ compiler |
| GPU executor | The [GPU worker environment](../getting-started/installation.md#gpu-server), including PyTorch, TVM FFI and CUPTI for benchmarking |

`nvcc` compiles CUDA source. TVM FFI is a foreign-function interface for calling
compiled code and exchanging tensors. CUPTI is NVIDIA's CUDA Profiling Tools
Interface, used to collect GPU activity timestamps. This example uploads Python
that builds CUDA C through TVM FFI.

On the compilation host, start a CPU server:

```bash
kcoral server --device cpu --num-workers 8 --host 0.0.0.0 --port 8000
```

On the execution host, start a GPU server:

```bash
kcoral server --device gpu --gpus 0 --workers-per-gpu 8 --host 0.0.0.0 --port 8001
```

The hosts may be different machines. See
[Launch the server](../server-guide/launch-the-server.md) for binding and worker
configuration.

## Run the complete client

From the repository checkout, replace these addresses with your servers:

```bash
KCORAL_CPU_URL=http://compiler.example.com:8000 \
KCORAL_GPU_URL=http://gpu.example.com:8001 \
python examples/cpu_compile_gpu_execute.py
```

{download}`Download the complete client <../../examples/cpu_compile_gpu_execute.py>`.
The downloaded file can be run directly with the same environment variables.
If both servers run on the client machine, the defaults are
`http://localhost:8000` for compilation and `http://localhost:8001` for execution.

`KCORAL_CPU_URL` selects the compilation endpoint. To use one GPU server for
both roles, install the compilation tools there and set both URL variables to
that server's address. The client still submits two independent programs.

## Read the execution target

Before compiling, ask the **execution server** which GPU architecture the
library must support:

```python
arch = gpu_client.target()["arch"]
```

`target()` reads `GET /health`. The CPU server has no GPU target of its own;
pass the execution server's `arch` explicitly to the uploaded `compile_cuda_binary`
function.
For example, a server might report `sm_100a`; use its actual reported value.

## Request 1: compile and return the library

The source defines a GPU kernel and a host function, `add_one`, which launches
it. The host function uses TVM FFI's `TensorView` parameters to access tensors.
The uploaded compiler uses `tvm_ffi.cpp.build_inline` to supply the required
includes and export wrapper.

```{literalinclude} ../../examples/cpu_compile_gpu_execute.py
:language: python
:start-at: CUDA_SOURCE =
:end-before: REFERENCE =
```

The example's `OPERATIONS` string defines the Python compiler and execution
helpers uploaded by each program:

```{literalinclude} ../../examples/cpu_compile_gpu_execute.py
:language: python
:start-at: OPERATIONS =
:end-before: def compile_program
```

The compilation program uploads that Python and the CUDA source, selects
`compile_cuda_binary` with `get_function(..., cpu_only=True)`, selects the CUDA
source name `add_one`, compiles for `arch`, and explicitly returns the library:

```{literalinclude} ../../examples/cpu_compile_gpu_execute.py
:language: python
:pyobject: compile_program
```

The uploaded `compile_cuda_binary` function builds a shared library without loading it or
launching the kernel. `return_(key="library", ...)` selects the bytes for the
response. After `cpu_client.execute()` succeeds,
`compiled.results["library"]` is a Python `bytes` object containing the shared
library, equivalent to the contents of a `.so` file.

The compilation request creates no GPU tensors. It can finish and release its
worker before the execution request begins. When using a GPU server for this
request, `cpu_only=True` releases its lease during the compiler function's `run`.
The Python module upload still executes its top-level code under the lease.

## Request 2: upload the library and execute

The next program receives those bytes as its `library` argument. It uploads
them with `kind="library"`, selects the exported function, creates input and
output tensors, and runs the kernel:

```{literalinclude} ../../examples/cpu_compile_gpu_execute.py
:language: python
:pyobject: benchmark_program
```

Loading the library recreates a module and function handle on the execution
server. This program contains no compilation operation. It compares the kernel
output against an uploaded Python reference on that server, then benchmarks
only if the uploaded `assert_close` function passes. That function uses
`torch.testing.assert_close`; measurement uses the importable
`kcoral.builtins.benchmark` helper.

The response returns `check` and `timing`. Tensor data stays on the execution
server; the client receives the correctness report and timing statistics.

## Submit the requests in order

The client connects the two programs by passing the returned bytes to the
second builder. It checks that compilation completed and returned binary data
before submitting anything to the execution server:

```{literalinclude} ../../examples/cpu_compile_gpu_execute.py
:language: python
:pyobject: main
```

There are two `Client.execute()` calls. Each is an independent program
submission; automatic blob-cache negotiation may add HTTP attempts to a
submission. The second call uploads the library through the normal client
cache protocol.

On success, the script prints the library size in KiB (1024 bytes), the target
architecture, each request's GPU lease time, and the kernel's median duration
in microseconds. A lease gives a worker exclusive access to its GPU; the CPU
compilation request holds no GPU lease. The execution request's
`result.results["check"]` contains the correctness report and
`result.results["timing"]` contains the measurement statistics. See
[Benchmark a Kernel with KCoral](benchmark-kernel.md#measuring-gpu-activity) to interpret them.

## Reuse the artifact and handle failures

The returned library bytes can be saved to a file and supplied to another
execution request later. The execution server may be a different instance;
no handle or session from the compilation server is needed. Each execution
still uploads the bytes and selects its own function handle.

The library must match the execution host's platform, GPU architecture and
runtime dependencies. Keep the compiler and executor's TVM FFI and CUDA
components compatible. See the [library upload protocol](../client-guide/protocol.md#library)
for loading and export requirements.

A failed compilation stops the client before the second submission. A failed
execution stops it before reporting a successful benchmark. Inspect each
result's `error` and `request_id` to identify which server and instruction
failed. Request or transport exceptions are distinct from a `FAILED` result;
the [failure guide](../client-guide/writing-a-program.md#handling-failures)
explains how to handle them.
