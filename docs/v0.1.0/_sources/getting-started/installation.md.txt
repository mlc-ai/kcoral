# Installation

KCoral can be installed from a prebuilt package or built from source. Install
the client on the machine that submits programs. If you also host a KCoral
server, install the server dependencies on each machine that runs it.

Python 3.10 or newer is required.

## Method 1: Install a prebuilt package

Prebuilt wheels are available on [PyPI](https://pypi.org/project/kcoral/).

<a id="the-client"></a>

### Install the client

```bash
python -m pip install kcoral
```

This installs the package with client dependencies. Running the clients needs no GPU.

Verify that the client imports successfully:

```bash
python -c "from kcoral import Client, Program; print('KCoral client is ready')"
```

<a id="the-server"></a>

### Install the server

Install the `server` extra:

<a id="front-end-engine-and-client"></a>
<a id="install-the-server-with-pip"></a>

```bash
python -m pip install 'kcoral[server]'
```

The `server` extra supports both GPU execution and CPU compilation. See
[system requirements](#server-system-requirements).

## Method 2: Build from source

Use a source installation to modify KCoral or install a specific revision.

### Get the source

Rust 1.87 or newer is required.

```bash
git clone https://github.com/mlc-ai/kcoral.git
cd kcoral
```

Run the remaining source installation commands from this repository directory.

### Install in editable mode

Install the client:

```bash
python -m pip install -e .
```

To include the server dependencies, use:

```bash
python -m pip install -e '.[server]'
```

(gpu-server)=
(cpu-compilation-server)=
## Server system requirements

<a id="running-gpu-programs"></a>
<a id="running-cpu-compilation-workers"></a>

| Server mode | Purpose | Hardware |
| --- | --- | --- |
| GPU (`--device gpu`) | Compile, execute, and benchmark kernels | NVIDIA GPU and driver |
| CPU (`--device cpu`) | Compile CUDA C for execution on a GPU server | No GPU or GPU driver required |

Compiling CUDA C in either mode requires the
[CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
and a compatible C++ compiler.

Install [bubblewrap](https://github.com/containers/bubblewrap) 0.8.0 or newer for
[filesystem isolation](../server-guide/launch-the-server.md#isolate-worker-files-with-bubblewrap).
On Debian/Ubuntu:

```bash
sudo apt install bubblewrap
```

For deployment, see [Launch the server](../server-guide/launch-the-server.md)
and [Remote Compilation](../tutorials/remote-compilation.md).
