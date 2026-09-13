# Installation

Run these commands from a checkout of the KCoral repository. Python 3.10 or
newer runs the package; building this documentation uses Python 3.12.

The client sends programs to an existing server. GPU means graphics processing
unit; CPU means central processing unit. Only the server's workers need the
GPU libraries or compilation tools listed below. CUDA is NVIDIA's GPU programming
platform; `nvcc` is its compiler. TVM is a tensor compiler, and TVM FFI is its
foreign-function interface for calling compiled code. PyTorch provides tensors,
`ninja` runs compilation tasks, and CUPTI (CUDA Profiling Tools Interface) provides
GPU activity timestamps. The [language guide](../tutorials/benchmark-kernel.md#languages-supported-by-remote-compilation)
describes TIRx, CuTeDSL and Triton kernel compilation.

## The client

```bash
pip install .
```

This gives the Python client on its own, which is all that sending programs to
a running server needs. It does not depend on a GPU, so it can install on a
machine with no GPU, no CUDA and no compiler.

## The server

The `server` extra adds the HTTP front-end, FastAPI and uvicorn, which the
client never imports:

```bash
pip install '.[server]'
```

Neither the client nor the front-end touches a GPU. Running programs needs a
worker environment holding more, and how much depends on what the programs use:

| To run | The worker environment needs |
|---|---|
| a CPU compilation server | TVM FFI, `nvcc`, `ninja`, and a host C++ compiler |
| any GPU program | PyTorch and TVM FFI |
| `compile_tirx` | TVM as well |
| `compile_cuda` | `nvcc`, a host C++ compiler, and `ninja` as well |
| `compile_cutedsl`, or a CuTeDSL library upload | `nvidia-cutlass-dsl` as well |
| `compile_triton` | `triton`, which the CUDA PyTorch wheels already carry |
| `benchmark`, or a `cpu_only` function | `cupti-python` as well |
| a program importing FlashInfer | `flashinfer-python` 0.6.17 or newer as well |

A builtin whose requirement is absent answers `unavailable` and the rest of the
server is unaffected, so a partial environment is a usable deployment.

Build whichever of the environments below matches the work.

## Front-end, engine, and client

The lockfile builds this one, CPU-only, with no GPU or compiler needed:

```bash
uv sync --no-editable
```

`uv sync --no-editable --no-default-groups` narrows it to the client's dependencies alone.

## Running GPU programs

The `gpu` group adds PyTorch, TVM, TVM FFI, CuTeDSL, and the CUPTI Python
bindings, which together cover every builtin and every kind of library upload,
plus other dependencies for programs whose reference calls it:

```bash
uv sync --no-editable --group gpu
```

`nvcc` and a host C++ compiler still come from the system; everything else is a
wheel, and no environment variables are needed.

## Running CPU compilation workers

The `compiler` group adds TVM FFI and `ninja` without installing PyTorch or
other GPU runtimes. The CUDA toolkit and a host C++ compiler still come from the
system:

```bash
uv sync --no-editable --group compiler
```

## Next steps

If you have a server address, follow the [quickstart](quickstart.md).
To run your own server, continue to [deployment](../server-guide/launch-the-server.md).
