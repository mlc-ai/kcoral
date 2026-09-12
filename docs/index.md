# KCoral

KCoral runs self-contained programs on remote GPU workers. Upload code and data,
call functions, check correctness and return measurements. A GPU is a graphics
processing unit; a CPU is a central processing unit. CPU compilation services
can build kernels for a separate GPU server.

## Start here

| Your task | Read |
| --- | --- |
| Run your first GPU program | [Installation](getting-started/installation.md) and [Your First Program](getting-started/quickstart.md) |
| Write a client program | [Writing a Program](client_guide.md), [KCoral Protocol](protocol.md) and [Builtin Tools](reference/builtins.md) |
| Operate a service | [Launch the server](server/deployment.md), [Logging](server/logging.md) and [Router](server/router.md) |
| Measure or automate a workload | [Benchmark a Kernel with KCoral](tutorials/benchmark-kernel.md) and [Agent Integration Guide](tutorials/agent-integration.md) |
| Extend the project | [Build the Docs](development.md) and [Python API](reference/python-api.rst) |

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

client_guide
protocol
reference/builtins
```

```{toctree}
:caption: Server Guide
:maxdepth: 1
:hidden:

server/deployment
server/logging
server/router
```

```{toctree}
:caption: Tutorials
:maxdepth: 2
:hidden:

tutorials/benchmark-kernel
tutorials/agent-integration
```

```{toctree}
:caption: Development Guide
:maxdepth: 1
:hidden:

development
```

```{toctree}
:caption: Python API
:maxdepth: 1
:hidden:

reference/python-api
```
