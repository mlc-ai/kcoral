# Installation

Install the client on the machine that submits programs. If you also host a
KCoral server, install a server environment on each machine that runs it.
The steps below install KCoral from source.

## Get the source

You need Git and Python 3.10 or newer. The server commands below use Python 3.12
on Linux.

```bash
git clone https://github.com/mlc-ai/kcoral.git
cd kcoral
```

Run the remaining commands from this repository directory.

<a id="the-client"></a>

## Install the client

Create and activate a virtual environment, then install KCoral:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

This installs the client and its dependencies. The client machine needs no GPU
(graphics processing unit), CUDA toolkit, or compiler. CUDA is NVIDIA's GPU
programming platform.

Verify that the client imports successfully:

```bash
python -c "from kcoral import Client, Program; print('KCoral client is ready')"
```

If you already have a server address, continue to
[Your First Program](quickstart.md).

<a id="the-server"></a>

## Install the server

On the server machine, get the source as above and install
[uv](https://docs.astral.sh/uv/getting-started/installation/), a Python package
and environment manager. Choose one of the environments below. Each command
creates `.venv`, installs KCoral with the client and server dependencies, and
uses the versions recorded in `uv.lock`. uv downloads Python 3.12 if needed.

For [filesystem isolation](../server-guide/launch-the-server.md#isolate-worker-files-with-bubblewrap),
install [bubblewrap](https://github.com/containers/bubblewrap) separately on Linux.
It needs `--disable-userns` support and permission to create unprivileged user
namespaces, including inside containers. If unavailable, the server warns and
runs without isolation.

<a id="running-gpu-programs"></a>

### GPU server

Use this environment to execute and benchmark GPU kernels. Before installing:

- Install an NVIDIA driver compatible with the CUDA 13.2 PyTorch packages
  selected by the repository. Confirm that `nvidia-smi` lists your GPU.
- To compile CUDA C kernels on this server, also install the
  [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
  and a supported host C++ compiler. Confirm that `nvcc --version` and
  `c++ --version` work in your shell.

Install the Python packages and activate the environment:

```bash
uv sync --locked --no-editable --group gpu --python 3.12
source .venv/bin/activate
```

The `gpu` dependency group includes the tensor, compilation and profiling
libraries available to uploaded Python programs. It does not install the system driver or
host C++ compiler.

Check that PyTorch, the tensor library used by workers, can access the GPU:

```bash
python -c "import torch, tvm_ffi; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
kcoral server --help
```

The first command should print your GPU's name. The second should display the
server's command-line options. Continue to
[Launch the server](../server-guide/launch-the-server.md) to start it.

<a id="running-cpu-compilation-workers"></a>

### CPU compilation server

Use this environment to compile CUDA C on a CPU (central processing unit), then
send the compiled library to a GPU server for execution. This machine needs the
[CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/),
including `nvcc`, and a supported host C++ compiler; it does not need a GPU.
Install the toolkit's compiler components without the GPU driver on this host.

```bash
uv sync --locked --no-editable --group compiler --python 3.12
source .venv/bin/activate
```

The `compiler` group installs TVM FFI (a foreign-function interface for compiled
code) and Ninja (a build tool). Verify the Python package and compiler tools:

```bash
python -c "import tvm_ffi; print('KCoral compiler dependencies are ready')"
nvcc --version
c++ --version
ninja --version
kcoral server --help
```

Each command should succeed. Follow
[Remote Compilation](../tutorials/remote-compilation.md) to launch the CPU and
GPU servers and pass a compiled library between them.

<a id="front-end-engine-and-client"></a>

### Server without worker libraries

If you only need the server package, for example to develop the request-handling
code, install the default environment:

```bash
uv sync --locked --no-editable --python 3.12
source .venv/bin/activate
kcoral server --help
```

This installs the client and server packages. Add the `gpu` or `compiler` group
above before running the corresponding worker workloads. For an existing Python
environment managed with pip, `python -m pip install '.[server]'` installs the
same server extra; it does not install worker libraries.

## Use the environment

In a new terminal, return to the repository and run `source .venv/bin/activate`
before invoking `python` or `kcoral`. Repeat the appropriate installation command
after updating the source to reinstall KCoral and its dependencies.
