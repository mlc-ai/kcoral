# KCoral

KCoral runs programs on remote GPU workers. Upload code and data,
call functions, check correctness and return measurements. CPU compilation
services can build kernels for a separate GPU server.

## Start here

| Your task | Read |
| --- | --- |
| Run your first GPU program | [Installation](getting-started/installation.md) and [Your First Program](getting-started/quickstart.md) |
| Run a Python function with a decorator | [Remote functions](client-guide/remote-functions.md) |
| Write a client program | [Write a client program](client-guide/writing-a-program.md), [KCoral Protocol](client-guide/protocol.md) |
| Run Python, checks, profilers, or shell commands | [Builtin CLI Tools](client-guide/builtin-cli-tools.md) |
| Operate a service | [Launch the server](server-guide/launch-the-server.md), [Logging](server-guide/logging.md) and [Launch the router](server-guide/router.md) |
| Compile remotely, then execute the returned library in another request | [Remote Compilation](tutorials/remote-compilation.md) |
| Measure or automate a workload | [Benchmark a Kernel with KCoral](tutorials/benchmark-kernel.md) and [Agent Integration Guide](tutorials/agent-integration.md) |
| Look up Python interfaces | [Python API](python-api/index.rst) |

```{toctree}
:caption: Getting Started
:maxdepth: 1
:hidden:

getting-started/installation
getting-started/quickstart
```

```{toctree}
:caption: Protocol
:maxdepth: 1
:hidden:

client-guide/protocol
```

```{toctree}
:caption: Client Guide
:maxdepth: 1
:hidden:

client-guide/remote-functions
client-guide/writing-a-program
client-guide/builtin-cli-tools
```

```{toctree}
:caption: Server Guide
:maxdepth: 1
:hidden:

server-guide/launch-the-server
server-guide/router
server-guide/logging
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
:caption: Python API
:maxdepth: 1
:hidden:

python-api/index
```
