# Agent Integration Guide

A coding agent can use KCoral to test generated kernels on a remote GPU and
use the results to guide its next revision. To make that loop useful, the agent
needs both the programming interface and a clear definition of success. This
tutorial shows how to provide that context, ask for a runnable experiment, and
review the results before iterating.

## Provide the KCoral skill

Start by giving the agent the `kcoral-client` skill, which explains how to
construct and submit programs. In a repository checkout, ask the agent to read
`.agents/skills/kcoral-client/SKILL.md`. If you are working outside the repository,
download the skill and provide it as task context or install it in your agent's
supported skill location.

{download}`Download the skill <../../.claude/skills/kcoral-client/SKILL.md>`.

The agent can also refer to [Write a client program](../client-guide/writing-a-program.md)
and the [Python API](../python-api/index.rst) for examples and method details.
Use documentation that matches the installed KCoral version.

## Define the experiment

The agent needs enough information to build an experiment whose results you
can assess. Provide the server address, kernel source or function to implement,
tensor shapes and types, correctness tolerance, and the outputs you want. Say
whether compilation should happen on the GPU server, on a separate CPU server,
or locally.

For example, give the agent this prompt from the repository checkout:

```text
Read .agents/skills/kcoral-client/SKILL.md before writing the client.

Write and run a Python client for the KCoral server at http://127.0.0.1:8000.
Implement add-one for 4096 float32 elements using CUDA C. Compile on the
GPU server, compare the output with a Python reference using rtol=1e-2 and
atol=1e-3, and benchmark only after correctness passes. Return the correctness
report and timing statistics. Report any failure instead of a successful timing.
Keep the program in a file that can be rerun.

Use the supplied server; do not launch another server or change its configuration.
Report the exact command, target architecture, runtime versions and results.
```

Replace the workload, tolerances, and endpoint with yours. Supply any required
input files. The example authorizes the agent to run the experiment; if you want
to review the program first, ask it to write the client without submitting it.

## Guide the agent through the workflow

With the task and interface established, the agent can turn the experiment into
a program. Ask it to check the server's GPU architecture and installed tools,
then prepare the inputs, compile the kernel, check its output, and measure it
only after correctness passes.

The [benchmark tutorial](benchmark-kernel.md) walks through this flow. If the
agent needs to compile on a separate CPU server, point it to
[Remote Compilation](remote-compilation.md). For profiling or running an existing
script, the [builtin CLI tools](../client-guide/builtin-cli-tools.md) may be enough.

## Review the generated program

Before relying on the results, check that the program tests the intended
workload and only reports timings after correctness passes:

| Check | Reason |
| --- | --- |
| Inputs, shapes, dtypes, and tolerances match your task | The experiment should measure the workload you intended |
| The kernel output is checked against the reference before benchmarking | Incorrect output should not produce a successful timing report |
| Compilation and execution match the GPU server's architecture and installed tools | A local build or assumption may not match the remote environment |
| The client reports failures and returns the requested results | Missing or partial results must not be presented as success |
| The agent provides a runnable file and its invocation | You should be able to reproduce the experiment |

For a closer review of how the program submits work and reads results, use
[Write a client program](../client-guide/writing-a-program.md).

## Use the results to guide the next revision

Once the agent has run the experiment, use the result to give it a focused
follow-up task. If correctness fails, provide the error and ask it to fix the
kernel before optimizing. If correctness passes but the kernel is slow, ask it
to investigate the bottleneck and compare its next version against this run.
Keep the input shapes, dtypes, tolerances, and GPU environment consistent so
the comparison measures the effect of the code change.

Save the runnable client and results from each version you want to compare.
When an error needs more context, use the request ID to find the
[server logs](../server-guide/logging.md) and share the relevant entries with
the agent. This gives the next revision a concrete starting point.
