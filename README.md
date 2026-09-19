<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/brand/kcoral-logo-dark.png" />
    <img src="docs/_static/brand/kcoral-logo-light.png" alt="KCoral" width="480" />
  </picture>
</p>

KCoral executes GPU benchmark programs over HTTP. It can also run as a CPU
compilation service: upload code and data, call functions, and explicitly return
the results you need.

**[Documentation](https://kcoral.mlc.ai/docs/)** · [Quickstart](docs/getting-started/quickstart.md) ·
[Write a client program](docs/client-guide/writing-a-program.md) · [KCoral Protocol](docs/client-guide/protocol.md)

## Install

<a id="the-client"></a>
<a id="the-server"></a>
<a id="front-end-engine-and-client"></a>
<a id="running-gpu-programs"></a>
<a id="running-cpu-compilation-workers"></a>

Python 3.10 or newer is required. Install the prebuilt package from PyPI:

```bash
pip install kcoral
```

This installs the package with client dependencies. Use `pip install "kcoral[server]"`
for both GPU execution and CPU compilation dependencies. See
[installation](docs/getting-started/installation.md) for source build and system requirements.

## Run remote tools

```bash
export KCORAL_URL='http://127.0.0.1:8000'
kcoral run python --send experiment -- experiment/check.py
kcoral run compute-sanitizer --send experiment -- python experiment/check.py
kcoral run ncu --send experiment --out artifacts/ncu \
  -- --set basic --launch-count 1 -- python experiment/capture.py
kcoral run run-iket --send experiment --out artifacts/iket \
  -- profile --postprocess json -- python experiment/capture.py
kcoral run shell --send experiment -- bash experiment/setup.sh
```

Any remote tool also accepts `--host HOST --port PORT`, for example
`kcoral run python --host 127.0.0.1 --port 8000 --send experiment -- experiment/check.py`.
These options override `KCORAL_URL`, with a warning showing the address used.

See [Builtin CLI Tools](docs/client-guide/builtin-cli-tools.md) for common options and a
separate guide to each tool, including dependencies, examples, and returned files.

## Run the server

> [!WARNING]
> KCoral allows clients to execute arbitrary code on its workers. Only allow
> trusted clients to access your KCoral server or Router. Deploy on a trusted,
> isolated network and never expose these endpoints to the public internet.
> Run workers in a sandbox with restricted permissions and access to host resources.

The server checks [bubblewrap filesystem isolation](docs/server-guide/launch-the-server.md#isolate-worker-files-with-bubblewrap)
at startup by default. If unavailable, it warns and runs without isolation.
Use `--sandbox none` to disable it explicitly.

After installing the worker environment:

```bash
kcoral server --host 127.0.0.1 --port 8000
```

See [deployment](docs/server-guide/launch-the-server.md) and [configuration](docs/server-guide/launch-the-server.md#configuration)
for CPU compilation, GPU selection, worker settings and limits.

### File upload cache

[File cache configuration](docs/server-guide/launch-the-server.md#cache) covers
the persistent cache directory, capacity and disabling caching.

## Route across nodes

A Router accepts client requests and chooses an available compute node. Each
node runs `kcoral server --router URL --node-id NAME` to start a supervisor
that manages its Python server; both node processes connect outward to the
Router. Clients use the same execution API.

See [Router deployment](docs/server-guide/router.md) for setup, scheduling and failure
handling, and the [Rust package guide](rust/kcoral/README.md) for implementation
and integration test commands.

## Logs

Follow requests and worker events using the [logging guide](docs/server-guide/logging.md).

## Python client

For simple tasks, use [`@client.function()`](docs/client-guide/writing-a-program.md#call-remote-functions)
and call `.remote()` to run a Python function on the server.

With a running GPU server, the first program uploads a tensor, adds one on the
GPU, returns it and checks the values:

```bash
KCORAL_URL=http://127.0.0.1:8000 python examples/first_program.py
```

See [Your First Program](docs/getting-started/quickstart.md),
[Write a client program](docs/client-guide/writing-a-program.md) and the
[Python API](docs/python-api/index.rst) for the complete example and interfaces.

Use `Program.return_file()` or `return_folder()` to receive workspace outputs,
then call `.save(destination)` on the result. See
[returning files and folders](docs/client-guide/writing-a-program.md#return-files-and-folders).

## Protocol summary

Programs contain `upload`, `get_function`, `run` and `return` instructions.
The [protocol](docs/client-guide/protocol.md) defines requests, results, caching and errors.
[Tutorials](docs/tutorials/benchmark-kernel.md) cover remote compilation, library
uploads and separate compilation/execution servers.

### Multi-GPU kernels and scripts

Reserve 1–8 GPUs for a complete program with
`client.execute(program, gpu_count=N)`. The program runs once with the full
GPU set visible and can execute its existing Python or `torchrun` command.
KCoral holds the set through process-tree cleanup; the script owns communication.
Two standalone examples call NCCL all-reduce through PyTorch. Each file includes
the kernel call, correctness checks, and the client Program. Uploads load
definitions and files; `get_function` selects the entry point and `run` invokes it:

- [One process controlling all GPUs](examples/multi_gpu_kernel.py).
- [One worker process per GPU](examples/multi_gpu_kernel_multiprocess.py).
