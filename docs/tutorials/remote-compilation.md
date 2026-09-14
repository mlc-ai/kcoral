# Remote Compilation

Send kernel source to a KCoral server and compile it there, then run it, check
the output and measure its execution. The client needs only KCoral and NumPy,
a Python array library. The compiler and GPU (graphics processing unit) live
on the server.

This tutorial follows an add-one kernel over 256 `float32` values. Its complete
client builds a separate program for each of four languages: TIRx, TVM's kernel
language; CUDA C, NVIDIA's GPU extension to C++; CuTeDSL, NVIDIA's Python language
for CuTe kernels; and Triton, a GPU kernel language and compiler. TVM is a tensor
compiler. Each program follows the same upload, compile, run and check sequence.

## Prepare the server and client

Install the [client](../getting-started/installation.md#the-client) locally and
use a server with the [GPU worker environment](../getting-started/installation.md#running-gpu-programs).
Running all four programs requires the full environment, including the CUDA
toolkit, its `nvcc` compiler, a host C++ compiler and `ninja`, a build tool.
See [Launch the server](../server-guide/launch-the-server.md) for startup options.

Check the server's target and installed tools before submitting:

```python
import os
from kcoral import Client

with Client(os.environ.get("KCORAL_URL", "http://localhost:8000")) as client:
    print(client.target())
    print(client.health()["versions"])
```

The GPU server selects the compilation architecture for this workflow. The
client does not need a GPU, CUDA installation or local compiler.

## Run the complete client

From the repository checkout, set the server address and run:

```bash
KCORAL_URL=http://localhost:8000 python examples/remote_compile_client.py
```

{download}`Download the complete client <../../examples/remote_compile_client.py>`.
The downloaded file can also be run directly with the same `KCORAL_URL` setting.
The script submits one program per language and prints request and kernel
timings for each successful result. The following sections walk through its
CUDA C program; the other languages use the same structure.

## Define the kernel

The uploaded source contains a GPU kernel and an exported host function,
`add_one`, which launches it. TVM FFI is TVM's foreign-function interface for
calling compiled code and exchanging tensors; its `TensorView` parameters give
the host function access to the input and output tensors. KCoral supplies the
required includes and export wrapper when compiling the selected function.

```{literalinclude} ../../examples/remote_compile_client.py
:language: python
:start-at: CUDA_KERNEL =
:end-before: TRITON_KERNEL =
```

## Build the program

This function builds instructions locally. Nothing is compiled or run until
`Client.execute()` submits the resulting `Program`.

```{literalinclude} ../../examples/remote_compile_client.py
:language: python
:pyobject: cuda_program
```

1. **Upload and select.** `upload(kind="module", language="cuda")` sends the
   source. `get_function(name="add_one")` selects the host function to compile.
   These steps describe code; they do not compile or launch it.
2. **Provide tensors.** Upload the exact input `[0, 1, ..., 255]` as `float32`
   data and allocate an output tensor with `builtin.empty`. The output starts
   uninitialized and must be written before comparison.
3. **Compile.** `builtin.compile_cuda` builds the selected function for the
   worker's GPU and returns a callable handle. Keep compilation in its own
   `run` instruction so the worker can release its exclusive GPU lease while
   the host compiler runs. A lease gives one worker exclusive use of a GPU.
4. **Run and check.** Invoke the compiled function with `src` and `dst`, then
   compare the output against a CPU (central processing unit) reference.
5. **Measure and return.** `builtin.benchmark` measures the compiled function
   after correctness passes. `return_` selects the timing report and output
   tensor for the response. The client checks the returned tensor again with
   NumPy.

The callable, tensors and other handles exist only within this request.
The server's compilation cache may reuse a CUDA build, but a later request
still needs its own upload, function selection and compile instructions.

## Check against a CPU reference

The reference constructs its input on the CPU and never reads a GPU tensor.
Selecting it with `cpu_only=True` allows other workers to use the GPU while it
runs. `builtin.assert_close` compares its CPU result with the GPU output and
stops the program on a mismatch, before the benchmark instruction can run.

```{literalinclude} ../../examples/remote_compile_client.py
:language: python
:start-at: CPU_REFERENCE =
:end-before: def tirx_program
```

## Choose another kernel language

The complete client defines `tirx_program()`, `cutedsl_program()`,
`cuda_program()` and `triton_program()`. Each uses the same inputs and reference.
The compilation call changes with the selected language:

| Language | Compilation call | What to supply |
| --- | --- | --- |
| TIRx | `builtin.compile_tirx(kernel, {"N": N})` | Bind the `@T.jit` kernel's compile-time dimension `N`; begin its source with `from __future__ import annotations` |
| CUDA C | `builtin.compile_cuda(kernel)` | Select a host function from a `language="cuda"` module; optional configuration adds `extra_cuda_cflags` |
| CuTeDSL | `builtin.compile_cutedsl(kernel, src, dst)` | Supply the tensors the compiled callable will use |
| Triton | `builtin.compile_triton(kernel, src, dst, N, 256, {"grid": [1], "num_warps": 4})` | Supply tensor and scalar arguments, the block size, and the launch grid; invoke the result with the same arguments |

These are server function signatures. With `Program.run`, pass the function
name as `fn` and its arguments as the `args` list, as the CUDA builder above
does. See [Builtin Tools](../client-guide/builtin-tools.md#compilation) for
configuration details and required dependencies.

## Interpret results and failures

| Result | Meaning |
| --- | --- |
| `result.completed` | Every instruction, including the correctness check, succeeded |
| `result.results["dst"]` | A CPU NumPy array containing `[1, 2, ..., 256]` |
| `result.results["timing"]["latency_ms_median"]` | Median per-call GPU activity span in milliseconds; the script prints it in microseconds |
| `result.results["timing"]["activities_stable"]` | Whether measured calls launched the same GPU activities |
| `result.elapsed_ms` | Worker execution duration, including compilation and waiting for GPU access |
| `result.lease_held_ms` | Time the request held exclusive GPU access |

The script's `total - GPU` display includes time waiting for GPU access as well
as compilation and CPU work; it is not a compiler-only measurement. Request
durations are also different from the kernel's timing report. See
[Benchmark a Kernel with KCoral](benchmark-kernel.md#measuring) for measurement
settings and how to interpret `activities_stable`.

An instruction failure produces `FAILED` and `result.error`. The script prints
that failure and continues to the next language. A missing toolchain produces
`unavailable`; compiler errors produce `compile`; an output mismatch produces
`correctness`. Request and transport errors raise exceptions instead. Use the
[failure guide](../client-guide/writing-a-program.md#handling-failures) to
distinguish them before retrying.

To compile on a separate CPU server, use `builtin.compile_cuda_binary` with
the GPU server's target, then upload its returned library bytes in a GPU
request. The [two-server workflow](benchmark-kernel.md#where-to-compile)
explains this variation.
