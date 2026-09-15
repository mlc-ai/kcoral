# KCoral

KCoral executes GPU benchmark programs over HTTP. It can also run as a CPU
compilation service: upload code and data, call functions, and explicitly return
the results you need.

**[Documentation](docs/index.md)** · [Quickstart](docs/getting-started/quickstart.md) ·
[Writing a Program](docs/client-guide/writing-a-program.md) · [KCoral Protocol](docs/client-guide/protocol.md)

## Install

<a id="the-client"></a>
<a id="the-server"></a>
<a id="front-end-engine-and-client"></a>
<a id="running-gpu-programs"></a>
<a id="running-cpu-compilation-workers"></a>

Python 3.10 or newer is required. From this checkout:

```bash
pip install .
```

This installs only the client. See [installation](docs/getting-started/installation.md)
for server, GPU and compiler dependencies.

## Run the server

After installing the worker environment:

```bash
kcoral --host 127.0.0.1 --port 8000
```

See [deployment](docs/server-guide/launch-the-server.md) and [configuration](docs/server-guide/launch-the-server.md#configuration)
for CPU compilation, GPU selection, worker settings and limits.

### File upload cache

[File cache configuration](docs/server-guide/launch-the-server.md#file-upload-cache) covers
the persistent cache directory, capacity and disabling caching.

## Route across nodes

A Router accepts client requests and chooses an available compute node. Each
node runs `kcoral-node` to supervise its Python server; both node processes
connect outward to the Router. Clients use the same execution API.

See [Router deployment](docs/server-guide/router.md) for setup, scheduling and failure
handling, and the [Rust package guide](rust/kcoral/README.md) for implementation
and integration test commands.

## Logs

Follow requests and worker events using the [logging guide](docs/server-guide/logging.md).

## Python client

With a running GPU server, the first program uploads a tensor, adds one on the
GPU, returns it and checks the values:

```bash
KCORAL_URL=http://localhost:8000 python examples/first_program.py
```

See [Your First Program](docs/getting-started/quickstart.md),
[Writing a Program](docs/client-guide/writing-a-program.md) and the
[Python API](docs/python-api/index.rst) for the complete example and interfaces.

Use `Program.return_file()` or `return_folder()` to receive workspace outputs,
then call `.save(destination)` on the result. See
[returning files and folders](docs/client-guide/writing-a-program.md#returning-files-and-folders).

## Protocol summary

Programs contain `upload`, `get_function`, `run` and `return` instructions.
The [protocol](docs/client-guide/protocol.md) defines requests, results, caching and errors.
[Tutorials](docs/tutorials/benchmark-kernel.md) cover remote compilation, library
uploads and separate compilation/execution servers.

## Development

See [development](docs/development-guide/build-the-docs.md) for tests and formatting checks.
[Build the documentation locally](docs/development-guide/build-the-docs.md#build-and-preview-the-documentation)
to browse the full site with navigation and search at `http://127.0.0.1:8008`.
