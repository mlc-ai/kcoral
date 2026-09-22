# Builtin tool definitions

The `kcoral.tools` subpackage contains every built-in remote tool. Each tool has
one definition file with its client entry point, argument validation, and remote
execution code:

| Command | Definition file |
| --- | --- |
| `kcoral run python` | `python/kcoral/tools/python.py` |
| `kcoral run compute-sanitizer` | `python/kcoral/tools/compute_sanitizer.py` |
| `kcoral run ncu` | `python/kcoral/tools/ncu.py` |
| `kcoral run run-iket` | `python/kcoral/tools/run_iket.py` |
| `kcoral run bench` | `python/kcoral/tools/bench.py` |
| `kcoral run shell` | `python/kcoral/tools/shell.py` |

See the [client reference](../client-guide/builtin-cli-tools.md) for command
syntax, requirements, and examples.

## Dispatch and shared code

`tools/__init__.py` maps CLI names to definition modules. `tools/cli.py` handles
`kcoral run` and imports only the selected definition. The top-level command
entry point delegates to this dispatcher.

The helper modules hold behavior shared by the tools:

- `_common.py` parses common options, validates profiler argument boundaries,
  constructs subprocess-tool requests, and handles captured output and downloads.
- `_inputs.py` snapshots uploads while preserving each selected directory's name
  and executable bits.
- `_worker.py` unpacks inputs, runs subprocesses in their request working directory,
  and collects selected files. Tool-specific command construction and report
  validation stay in the corresponding definition file.

Each definition exposes `parse_args(argv)` and `main(argv)` for the client, plus
`run(...)` for the worker. Subprocess definitions receive a shared execution
callback that handles executable lookup, environment, working directory, and
closed stdin. `bench.py` builds its own per-case programs and runs correctness
checks and CUPTI measurements in the worker process.

## Uploaded definitions

The client uploads the selected definition file's source and selects its `run`
function through the instruction protocol. Subprocess programs also upload
`_worker.py`; benchmark programs use its file-transfer helpers. Requests contain
the definitions they execute, so the remote worker does not need to discover
modules from a client checkout.

A definition must import without client-side package context when uploaded.
Keep imports of sibling helpers and the client API inside `main`, `parse_args`,
or request-building functions. Imports needed only for GPU execution belong
inside the remote functions. This also keeps local command help usable without
the server or GPU dependencies installed.

When adding a tool, create its definition module, register its CLI name, and add
its command reference. Reuse shared helpers and keep its native argument rules
and report handling in its own file. Run the tool tests and check a normal
installation: definitions and helper sources must be present in the installed
package because request construction reads those files.
