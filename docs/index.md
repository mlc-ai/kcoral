# KCoral

KCoral runs programs on remote GPU workers. Upload code and data,
call functions, check correctness and return measurements. A GPU is a graphics
processing unit; a CPU is a central processing unit. CPU compilation services
can build kernels for a separate GPU server.

## Start here

| Your task | Read |
| --- | --- |
| Run your first GPU program | [Installation](getting-started/installation.md) and [Your First Program](getting-started/quickstart.md) |
| Run a Python function remotely | [Remote Python Functions](client-guide/remote-functions.md) |
| Write a client program | [Writing a Program](client-guide/writing-a-program.md), [KCoral Protocol](client-guide/protocol.md) |
| Operate a service | [Launch the server](server-guide/launch-the-server.md), [Logging](server-guide/logging.md) and [Router](server-guide/router.md) |
| Compile remotely, then execute the returned library in another request | [Remote Compilation](tutorials/remote-compilation.md) |
| Measure or automate a workload | [Benchmark a Kernel with KCoral](tutorials/benchmark-kernel.md) and [Agent Integration Guide](tutorials/agent-integration.md) |
| Extend the project | [Build the Docs](development-guide/build-the-docs.md) and [Python API](python-api/index.rst) |

```{toctree}
:caption: Getting Started
:maxdepth: 1
:hidden:

getting-started/installation
getting-started/quickstart
```

```{toctree}
:caption: Client Guide
:maxdepth: 1
:hidden:

client-guide/writing-a-program
client-guide/remote-functions
client-guide/protocol
```

```{toctree}
:caption: Server Guide
:maxdepth: 1
:hidden:

server-guide/launch-the-server
server-guide/logging
server-guide/router
```

```{toctree}
:caption: Tutorials
:maxdepth: 1
:hidden:

tutorials/benchmark-kernel
tutorials/remote-compilation
tutorials/agent-integration
```

```{toctree}
:caption: Development Guide
:maxdepth: 1
:hidden:

development-guide/build-the-docs
```

```{toctree}
:caption: Python API
:maxdepth: 1
:hidden:

python-api/index
```
