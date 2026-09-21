# Agent Integration Guide

A coding agent can write KCoral programs using the repository's `kcoral-client`
skill. A skill is a Markdown instruction file that gives an agent a reusable
workflow and project-specific facts. It is not a Python dependency and does not
run the server or submit requests by itself.

## Give the agent the skill

This repository exposes the skill at `.agents/skills/kcoral-client/SKILL.md`.
That directory links to `.claude/skills/kcoral-client`, the underlying source.
An agent that supports repository skills can discover it there. For another
agent, explicitly provide the file as task context or put it in that agent's
supported skill location. Merely mentioning the skill name does not guarantee
that an agent has read it.

```bash
cat .agents/skills/kcoral-client/SKILL.md
```

{download}`Download the skill <../../.claude/skills/kcoral-client/SKILL.md>`.

Give the agent access to the same revision of the skill, the
[KCoral Protocol](../client-guide/protocol.md) and
[Write a client program](../client-guide/writing-a-program.md). The protocol is the authority for field
validation when a summary and the protocol disagree. The Python API supplies
the actual method signatures.

## State a concrete task

Provide the server address, kernel source or function to implement, tensor
shapes and types, correctness tolerance, and the outputs you want. Say whether
compilation should happen on the GPU server, on a separate CPU server, or locally.
GPU means graphics processing unit; CPU means central processing unit.

For example, give the agent this prompt from the repository checkout:

```text
Read .agents/skills/kcoral-client/SKILL.md before writing the client.
Use docs/client-guide/protocol.md for field validation. Upload your own Python harness
for compilation, allocation, correctness checks, and measurement.

Write a runnable Python client for the KCoral server at http://localhost:8000.
Implement add-one for 4096 float32 elements using CUDA C. Compile on the
GPU server, create or upload input tensors, compare the output with a Python
reference, and benchmark only after correctness passes. Return the correctness
report and all timing statistics. Handle FAILED outcomes separately from
request/transport exceptions. Keep the program in a file that can be rerun.

Use the supplied server; do not launch another server or change its configuration.
Report the exact command, target architecture, runtime versions and results.
```

Replace the task's workload and endpoint with yours. Supply any required files
and their expected relative paths. If the agent may execute submissions, make
that scope explicit; otherwise ask it to produce the program for review.

## Have the agent build a program

The skill guides the agent through this sequence:

1. Inspect [`Client.health()`](../client-guide/protocol.md#get-health) and, when
   compiling a library, the GPU server's `Client.target()` so it uses an available
   toolchain and the right architecture.
2. Upload the module and use `get_function()` to select the kernel or launcher.
3. Allocate device inputs in uploaded Python, or upload data whose exact values
   matter. Use `upload_file()` or `upload_folder()` for scripts that read files.
4. Compile with an uploaded harness, or upload a prebuilt library.
5. Run the kernel and compare it against a reference with your harness’s assertions.
6. Run your measurement harness only after the comparison succeeds.
7. Add `return_()` instructions for the correctness and timing reports, then
   submit with `Client.execute()` and inspect the outcome.

The [benchmark tutorial](benchmark-kernel.md) explains compilation and GPU
lease handling. Use the Python client to construct multipart requests: the client handles hashing, missing-blob negotiation and
decoding tensor results.

## Review the generated program

| Check | Reason |
| --- | --- |
| Every reference points to an earlier instruction in this request | Handles do not survive another submission |
| Output tensors are written before comparison | An uninitialized allocation is not a computed answer |
| The result is explicitly selected by `return_()` | `run()` alone does not send a value back |
| File uploads use the dedicated helper methods | `Program.upload(kind="file")` is not supported |
| Host-only work is selected with `get_function(..., cpu_only=True)` and invoked in its own `run` | The worker releases the GPU lease and checks for CUDA access |
| `assert_close` precedes `benchmark` | Incorrect output should not produce a successful timing report |
| The program checks `result.completed` before reading expected keys | Failures can return only partial results |
| Target and version information accompanies results | Generated code must match the actual server environment |

Uploaded Python can use its own libraries, CLI tools and profiler scripts.
`cpu_only=True` applies only to the selected function's own `run` instruction.
Compilation that loads a CUDA module must keep the lease or split host building
and GPU loading into separate instructions. Nested reference-shaped arguments
remain JSON literals; pass handles as top-level arguments.

## Preserve evidence and iterate

Keep the generated client, source code, exact command, input shapes and types,
server target and versions, correctness report, timing report and `request_id`.
A program with `status == "FAILED"` is a valid response with an instruction
error; a transport exception may leave execution outcome unknown. Do not turn
either into a successful timing number.

Feed the specific error, failing instruction and relevant server version back
to the agent for the next revision. On an uncertain transport outcome, check
logs before replaying code with external side effects. Each revision should
still be a complete program and should pass correctness before it is measured.
