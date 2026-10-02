# Installation

KCoral can be installed from a prebuilt package or built from source. Install
the client on the machine that submits programs. If you also host a KCoral
server, install the server dependencies on each machine that runs it.

Python 3.10 or newer is required. Use a virtual environment so the installation does not modify your system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Run the installation commands below with this environment active.

## Method 1: Install a prebuilt package

Prebuilt wheels are available on [PyPI](https://pypi.org/project/kcoral/).
They support Linux on x86-64 and AArch64 with glibc 2.28 or newer.

<a id="the-client"></a>

### Install the client

```bash
python -m pip install kcoral
```

This installs the package with client dependencies. Running the client needs no GPU.

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

Install Git, Rust 1.87 or newer with Cargo, and a C/C++ build toolchain.
The package build compiles the Rust router and node supervisor, including when
installing only the client dependencies.

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

The server runs on Linux. Its Python dependencies are installed by the `server`
extra; the GPU driver and system tools below must be installed separately.

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

The host or container must also permit unprivileged user namespaces. If the
server's startup check cannot launch bubblewrap, it warns and runs without
filesystem isolation; see the isolation guide above for configuration.

For deployment, see [Launch the server](../server-guide/launch-the-server.md)
and [Remote Compilation](../tutorials/remote-compilation.md).
