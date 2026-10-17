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

### Install the client

```bash
python -m pip install kcoral
```

This installs the package with client dependencies. Running the client needs no GPU.

Verify that the client imports successfully:

```bash
python -c "from kcoral import Client, Program; print('KCoral client is ready')"
```

### Install the server

The server requires Linux on x86-64 or AArch64 with glibc 2.28 or newer.
Install the `server` extra:

```bash
python -m pip install 'kcoral[server]'
```

The `server` extra supports both GPU execution and CPU compilation. See
[system requirements](#server-system-requirements).

### Install the server for AMD GPUs (experimental)

The `server` extra installs CUDA builds of PyTorch and NVIDIA packages, so do
not use it on an AMD machine. Install the ROCm build of PyTorch first, from the
PyTorch index matching your ROCm release, then the `rocm` extra, which never
installs or replaces PyTorch:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/rocm7.0 torch
python -m pip install 'kcoral[rocm]'
```

```{important}
An unpatched ROCm runtime (checked through 7.0.2) does not work inside KCoral's
[bubblewrap sandbox](../server-guide/launch-the-server.md#isolate-worker-files-with-bubblewrap),
which is on by default, when the machine or container has more than one GPU. A
sandboxed worker sees only its own GPU's device files, ROCm then fails on the
GPUs it cannot open and finds no GPU at all, and the server fails to start with
"HIP found no GPU inside the bubblewrap sandbox". Choose one:

- **Patch the ROCm runtime.** Rebuild ROCm's HSA runtime with the three-line
  patch in `docker/rocm-libhsakmt-patch/`, as `docker/Dockerfile.rocm.dev`
  does. The patch kit's README explains the build. This keeps per-worker
  isolation, as on NVIDIA.
- **Use the ROCm image.** `docker/Dockerfile.rocm.dev` always includes the
  patched runtime.
- **Turn the sandbox off.** Start the server with `--sandbox none`. Workers then
  share the server's filesystem view.

Bubblewrap itself also needs the host or container to allow it; see
[system requirements](#server-system-requirements).
```

See [AMD GPUs](../server-guide/launch-the-server.md#amd-gpus-experimental) for
how ROCm workers differ from CUDA workers.

## Method 2: Build from source

Use a source installation to modify KCoral or install a specific revision.

### Get the source

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

On AMD GPUs, install ROCm PyTorch first and use the `rocm` extra instead
(`-e '.[rocm]'`); see [Install the server for AMD GPUs](#install-the-server-for-amd-gpus-experimental).

These builds are pure Python. A [Router](../server-guide/router.md) deployment
also needs the Rust router and node supervisor, which run on Linux. Install Rust
1.87 or newer with Cargo and a C/C++ build toolchain, then build them into the
package:

```bash
KCORAL_BUILD_RUST=1 python -m pip install -e '.[server]'
```

## Server system requirements

The server runs on Linux. Its Python dependencies are installed by the `server`
extra; the GPU driver and system tools below must be installed separately.

| Server mode | Purpose | Hardware |
| --- | --- | --- |
| GPU (`--device gpu`) | Compile, execute, and benchmark kernels | NVIDIA GPU and driver, or AMD Instinct GPU with ROCm (experimental) |
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

## Prepare for local compilation (optional)

If you want to compile CUDA C libraries on the client, as in the
[benchmark tutorial](../tutorials/benchmark-kernel.md#on-the-client), install
the CUDA toolkit and a compatible C++ compiler on that machine. Add TVM FFI
and its C++ build dependencies to the active client environment:

```bash
python -m pip install 'apache-tvm-ffi[cpp]>=0.1.14.post0'
```

These additional dependencies are needed only when compiling locally. The
KCoral `server` extra already includes the TVM FFI build dependencies for
server-side compilation.
