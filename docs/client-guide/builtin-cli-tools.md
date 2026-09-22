# Builtin CLI Tools

KCoral provides command line tools for running experiments on remote workers.
Use `kcoral run TOOL` to upload inputs, execute a program, and retrieve its
output through an existing KCoral server or Router.

| Tool | Use it to |
| --- | --- |
| [python](#python) | Run a Python script, module, or inline command with the worker's interpreter. |
| [compute-sanitizer](#compute-sanitizer) | Find CUDA memory-access and synchronization errors. |
| [ncu](#ncu) | Collect NVIDIA Nsight Compute kernel performance reports. |
| [run-iket](#run-iket) | Collect instrumented kernel execution timelines. |
| [bench](#bench) | Check correctness and measure a candidate using a checkout's benchmark definitions. |
| [shell](#shell) | Run a shell script, uploaded executable, or program installed on the worker. |

The common reference below defines KCoral's options and execution behavior.
Each tool section then describes its command format, dependencies, native
arguments, returned files, and failure handling.

## Common reference

### Command structure and help

The five subprocess tools use this structure:

```text
kcoral run TOOL [KCoral options] -- [native arguments]
```

Put connection, upload, environment, and download options **before** the first
`--`. Everything after it belongs to the selected tool. For `ncu` and
`run-iket`, a second `--` separates profiler options from the application.
`bench` has its own workload and version arguments and does not require a
separator.

```bash
kcoral run --help
kcoral run python --help
kcoral run bench --help
```

These commands display KCoral's help locally without contacting a server.
For the native executable's options, use the commands in each tool section.
Native options are interpreted by the version installed on the worker.

### Connection and execution options

KCoral does not select a server unless a connection option or a nonempty
`KCORAL_URL` is supplied.

| Option | Default | Meaning and accepted values |
| --- | --- | --- |
| `-h`, `--help` | — | Show this tool's KCoral options and exit. |
| `--url URL` | `KCORAL_URL` | Complete server URL, including an optional HTTPS scheme or path prefix. Cannot be combined with `--host` or `--port`. |
| `--host HOST` | `127.0.0.1` when only `--port` is supplied | HTTP server hostname, IPv4 address, or IPv6 address. Supply the host without a scheme, port, or path; IPv6 brackets are optional. |
| `--port PORT` | `8000` when only `--host` is supplied | HTTP server port, an integer from 1 through 65535. |
| `--timeout SECONDS` | `300` | Positive integer execution deadline for each request, capped by the server's configured maximum. For `bench`, this applies separately to each workload request. |
| `--output-limit-bytes N` | `268435456` (256 MiB) | Positive integer capture limit for **each** of stdout and stderr, capped by the server. Zero is not accepted by these CLI tools. |

Select a server with an environment variable:

```bash
export KCORAL_URL='http://gpu.example.com:8000'
kcoral run python -- -c 'print("hello from the worker")'
```

Or specify the address on the invocation:

```bash
kcoral run python --host gpu.example.com --port 8000 --send experiment -- check.py
kcoral run bench kda/decode v0 --host gpu.example.com --port 8000
```

If either `--host` or `--port` is present, KCoral constructs `http://HOST:PORT`
using the defaults in the table for any omitted component. It does not inherit
any component from `KCORAL_URL`. If that environment variable is also nonempty,
KCoral warns on stderr and prints the address it will use:

```text
kcoral: warning: --host/--port override KCORAL_URL; using http://gpu.example.com:8000
```

An explicit `--url` also overrides `KCORAL_URL`. Use it when the endpoint needs
HTTPS or a path prefix. Supplying `--url` with either host/port option is a
local argument error.

### Input, environment, and output options

These options control uploaded files, subprocess variables, and returned files.
Use them before the first `--`; the tool sections describe how each command
prepares its inputs and collects its results.

| Option | Default | Meaning |
| --- | --- | --- |
| `--send PATH` | No uploaded files | Upload a file or a directory's contents. Repeat for additional inputs. |
| `-e NAME[=VALUE]`, `--env NAME[=VALUE]` | Worker environment | Set one subprocess variable, or copy one local variable by name. Repeat for additional variables. |
| `--fetch PATH` | No selected outputs | Return a file or directory relative to the remote working directory. Repeat for additional outputs; requires `--out`. |
| `--out DIRECTORY` | No download directory | New local destination for returned files. Profilers require it for their reports; explicit file selection pairs it with `--fetch`. |

### Input layout and working directory

`--send` snapshots local inputs before execution. The worker receives regular
files in a fresh working directory; `KCORAL_DIR` points to that directory.

| Local selection | Remote path |
| --- | --- |
| `--send experiment/check.py` | `check.py` |
| `--send experiment` containing `check.py` | `check.py` |
| `--send experiment` containing `data/input.json` | `data/input.json` |

A directory's own name is not included. Multiple `--send` selections are
combined into the same working directory, so overlapping destination names
are rejected. Symbolic links and special files are rejected. Python
`__pycache__` entries are skipped; other hidden files are included. Empty input
directories are not uploaded. Executable bits are preserved, but full original
permission modes are not.

Send the files your program needs, including local modules and configuration.
Sending a script does not discover its imports or upload its parent directory.
Without `--send`, the subprocess starts in an empty working directory. Relative
paths in its arguments are resolved there; absolute paths refer to the worker's
filesystem, subject to the server's isolation settings.

### Environment variables and executable lookup

Environment overrides are optional. Without them, the subprocess inherits the
worker's environment, including its assigned GPU. The worker's Python executable
directory is prepended to `PATH` when locating programs.

Use one `-e` or `--env` per variable:

```bash
kcoral run python --send experiment --env MODE=debug --env MY_LOCAL_VARIABLE -- check.py
kcoral run shell --send experiment -e MODE=debug -e LABEL='trial one' -- sh setup.sh
```

| Form | Behavior |
| --- | --- |
| `--env MODE=debug` | Set `MODE` to `debug` in the remote subprocess. |
| `--env MY_LOCAL_VARIABLE` | Read this variable from the client's environment and send its value. An unset local variable is an argument error. |
| `--env EMPTY=` | Set a variable to an empty string. |
| Repeated assignments to the same name | The last assignment wins. |

Variable names must start with a letter or underscore and contain only letters,
digits, or underscores. `CUDA_VISIBLE_DEVICES` and `KCORAL_DIR` are managed by
KCoral and cannot be overridden. Setting `PATH` with `--env` still leaves the
worker's Python executable directory first in the subprocess search path.

Local variables are not copied automatically. For example, putting
`MODE=debug` before the local `kcoral` command sets the client's environment;
it reaches the worker only if selected with `--env MODE`. Your local shell
also expands expressions such as `$HOME` before invoking KCoral unless they
are quoted appropriately.

### Standard input and output

These tools run without an interactive terminal. The launched subprocess starts
with closed standard input; local stdin is not forwarded, and interactive prompts
are unsupported. Upload input files and pass their paths, or invoke a remote
shell to redirect an uploaded file into a child program's stdin. No interactive
session is kept between requests.

Output is captured during execution and replayed after each request finishes.
Normal UTF-8 text preserves its content, line breaks, and order within each
stream. KCoral writes the captured stdout to local stdout and the captured
stderr to local stderr. The following differences from direct local execution
apply:

- Output is not streamed. Python's `-u` flag does not make the client display it
  before the request completes.
- Stdout and stderr are collected separately and replayed in that order; their
  original interleaving is not preserved.
- Output is decoded as UTF-8 with invalid bytes replaced. Use returned files for
  binary data or for logs whose exact original bytes matter.
- The capture limit applies independently to each stream. Truncation produces
  a warning on stderr and does not by itself change the subprocess exit code.
- Programs may change colors or progress displays when they detect that they
  are not connected to a terminal.

KCoral does not add a prefix to ordinary subprocess stdout. Its artifact
messages and warnings use stderr. `bench` additionally prints workload progress,
worker information, and a summary to stdout. Local redirection and pipelines
still work, for example:

```bash
kcoral run python --send experiment -- check.py > run.log 2> run.err
```

### Returned files, limits, and exit status

`--fetch` paths must be relative, without `..` components. Returned files and
directories retain their relative layout, including empty output directories.
Symbolic links and special files cannot be returned. `--out` must name a new
local directory; parent directories are created automatically, and existing
output is never overwritten.

Files are collected after the subprocess finishes, even when it exits with an
error. A hard timeout, worker failure, or response-size error can prevent their
return. Each request has its own workspace; a later invocation is not a
continuation of the earlier one. Keep related setup and execution steps in one
invocation, or explicitly download and resend the needed files.

The server caps execution time, captured output, and serialized responses.
Its default maximum requested capture is 256 MiB per stream, and its default
response limit is 1 GiB. A server may configure lower limits. See
[transfer limits](protocol.md) and [server configuration](../server-guide/launch-the-server.md#configuration).

| Outcome | CLI exit behavior |
| --- | --- |
| Subprocess completes and requested files are available | Preserve its exit code. |
| Subprocess is terminated by signal N | Return `128 + N`. |
| Requested output is missing | Return nonzero; preserve an existing nonzero subprocess exit code. |
| Transport, worker execution, or artifact-return failure | Return `1` with a diagnostic on stderr. |
| Invalid KCoral command line | Return `2` without running the tool. |
| Benchmark run | Use the correctness and failure rules in the `bench` section. |

## python

### Purpose and requirements

Use `python` for scripts, module entry points, or inline Python commands. It
runs the **worker's Python interpreter**, not the client's interpreter. The
worker must have the imported packages and any required GPU libraries installed.
Upload your own modules and data along with the entry-point script.

### Command format and arguments

```text
kcoral run python [KCoral options] -- [Python options] SCRIPT [script arguments]
kcoral run python [KCoral options] -- [Python options] -m MODULE [module arguments]
kcoral run python [KCoral options] -- [Python options] -c CODE [code arguments]
```

This tool supports all [common options](#common-reference), including `--send`,
`--env`, and `--fetch` with `--out`.

| Argument | Meaning |
| --- | --- |
| `SCRIPT` | Script path on the worker, normally relative to the uploaded working directory. |
| `-m MODULE` | Run a module available from uploaded files or the worker's installed packages. |
| `-c CODE` | Execute an inline command; quote it as one local shell argument. |
| Python options | Forwarded to the worker's interpreter, for example `-O`, `-W ignore`, or `-X dev`. Place them before the script, `-m`, or `-c`. |
| Script/module arguments | Forwarded unchanged to the program after its entry point. |

A script, module, or inline command is required. The Python prompt, `-i`, and
stdin execution with `-` are not supported. Options after the first KCoral `--`
are Python or program arguments, even if they have names such as `--host`.

### Run a script or module

Given this local directory:

```text
experiment/
  check.py
  data/input.json
  my_package/
    __init__.py
    check.py
```

Run the script with its own arguments, or run the package module:

```bash
kcoral run python --send experiment -- check.py --input data/input.json
kcoral run python --send experiment -- -m my_package.check
kcoral run python --send experiment -- -W ignore -X dev check.py
kcoral run python -- -c 'import sys; print(sys.version)'
```

### Return results and interpret failures

If `check.py` writes `results/report.json`, download its containing directory:

```bash
kcoral run python --send experiment --fetch results --out artifacts/python -- check.py
```

The local result is `artifacts/python/results/report.json`. For exact binary
output, write a remote file and fetch it instead of writing binary bytes to
stdout. Results are still collected after `sys.exit(N)` or an uncaught exception,
provided execution reaches normal subprocess completion.

`sys.exit(N)` becomes the CLI exit code. An uncaught exception normally produces
a traceback on stderr and a nonzero code. A missing module usually means it was
neither uploaded nor installed on the worker; installing it only on the client
does not make it available remotely. A missing script often means the command
used `experiment/check.py` after uploading the **contents** of `experiment/`.

To inspect native Python help on the worker, use
`kcoral run shell -- python --help`. Python's
[command line reference](https://docs.python.org/3/using/cmdline.html) describes
the native interpreter options.

## compute-sanitizer

### Purpose and requirements

NVIDIA Compute Sanitizer runs a CUDA application under a selected correctness
checker. The worker needs the `compute-sanitizer` executable, a compatible CUDA
driver and GPU, and the application's dependencies. KCoral invokes the installed
executable; it does not install the checker or compile the application for you.

### Command format and checker selection

```text
kcoral run compute-sanitizer [KCoral options] -- [sanitizer options] APPLICATION [arguments]
```

All common subprocess options are available. There is one KCoral separator;
the remaining arguments follow Compute Sanitizer's native syntax.

| Native option | Purpose |
| --- | --- |
| `--tool memcheck` | Check memory accesses; this is Compute Sanitizer's default checker. |
| `--tool racecheck` | Check shared-memory access hazards. |
| `--tool initcheck` | Check uninitialized memory accesses; the installed version controls supported address spaces. |
| `--tool synccheck` | Check synchronization usage. |
| `--error-exitcode N` | Request a nonzero status when the checker reports errors. Use this when findings must fail automation. |
| `--log-file PATH` | Write the checker log to a remote file. Pair it with KCoral `--fetch PATH --out DIRECTORY` to retain it locally. |

These are native options placed **after** `--`. Other options supported by the
installed Compute Sanitizer are also forwarded. The application can be a program
installed on the worker or an uploaded executable such as `./check`.

### Check a Python or compiled application

Run the default checker, select a race check, or check an uploaded executable:

```bash
kcoral run compute-sanitizer --send experiment -- python check.py
kcoral run compute-sanitizer --send experiment \
  -- --tool racecheck --error-exitcode 1 python check.py
kcoral run compute-sanitizer --send experiment \
  -- --tool memcheck --error-exitcode 1 ./check
```

A compiled program must target the worker's platform and preserve its executable
bit in the upload. For useful source locations, build it with the line
information recommended by Compute Sanitizer.

### Save findings and handle failures

```bash
kcoral run compute-sanitizer --send experiment \
  --fetch sanitizer.log --out artifacts/check \
  -- --tool memcheck --error-exitcode 1 --log-file sanitizer.log python check.py
```

The returned log is `artifacts/check/sanitizer.log`. Selecting `--log-file`
changes where the native tool writes its diagnostics; do not assume the same
text will also appear on stdout.

KCoral preserves **Compute Sanitizer's** exit code. It does not parse the log to
turn findings into failure, so set `--error-exitcode` explicitly when needed.
A missing requested log also makes the invocation fail. If instrumented execution
exceeds the timeout, increase `--timeout` within the server's configured maximum;
a hard timeout may prevent log collection.

Use `kcoral run shell -- compute-sanitizer --help` to inspect the installed
version. See NVIDIA's [Compute Sanitizer manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
for checker coverage and native options.

## ncu

### Purpose and requirements

NVIDIA Nsight Compute measures GPU kernel performance and produces a report for
later inspection. The worker needs `ncu`, an application compatible with the
worker's GPU, and permission to collect the required GPU performance counters.
The application must actually launch kernels selected by the profiling options.

### Command format and options

```text
kcoral run ncu [KCoral options] --out DIRECTORY -- [ncu options] -- APPLICATION [arguments]
```

This tool supports the common connection, execution, `--send`, and `--env`
options. `--out` is required and `--fetch` is not supported. The first `--`
starts Nsight Compute arguments; the second starts the application command.
Both separators are required, even when no native profiler options are supplied.

| Native option | Purpose |
| --- | --- |
| `--set basic` | Collect the basic section set. Use a set available in the installed version. |
| `--set full` | Request the full section set, which can require more profiling passes and time. |
| `--launch-count N` | Limit the number of profiled launches. |
| `--launch-skip N` | Skip matching launches before profiling. |
| `--kernel-name FILTER` | Select kernels using Nsight Compute's name-filter syntax. |
| `--section IDENTIFIER` | Select a metric section supported by the installed version. |

KCoral sets the export path and launches the application in capture mode.
It rejects native `--export`/`-o`, `--import`/`-i`, `--mode`, `--config-file`,
and `--config-file-path` options, including abbreviations that conflict with
these managed options. Use the local report for later import or inspection.

Implicit Nsight configuration files are disabled. The subprocess receives
`NCU_PROFILE=1` unless you explicitly supply another value with `--env`; this
allows timing helpers that honor the variable to avoid a competing profiler
subscription.

### Capture a report

Collect one launch with the basic set:

```bash
kcoral run ncu --send experiment --out artifacts/ncu \
  -- --set basic --launch-count 1 -- python capture.py
```

Use the native defaults, or select a kernel and skip warmup launches:

```bash
kcoral run ncu --send experiment --out artifacts/ncu-default \
  -- -- python capture.py
kcoral run ncu --send experiment --out artifacts/ncu-filtered --timeout 600 \
  -- --kernel-name 'regex:my_kernel.*' --launch-skip 5 --launch-count 1 \
  -- python capture.py
```

Native filtering and replay behavior are controlled by the installed profiler.
Profiling time is not the same as an ordinary benchmark run: collecting more
metrics may require repeated executions of the kernel.

### Report location and failure handling

For `--out artifacts/ncu`, the report is always:

```text
artifacts/ncu/capture.ncu-rep
```

The local destination must be new. Open the downloaded report with a compatible
Nsight Compute installation, or print it with a local CLI:

```bash
ncu --import artifacts/ncu/capture.ncu-rep
```

KCoral returns the profiler's exit status and attempts to download any report
that exists, even when the profiler or application fails. If no report was
created, the command fails with a missing-artifact message even if `ncu` exited
with zero. Common causes include filters that match no kernels, no CUDA kernel
launches, unavailable performance counters, or failure before report creation.
A hard timeout can prevent the report from being returned.

For native help without creating a report, run `kcoral run shell -- ncu --help`.
See the [Nsight Compute CLI manual](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
for section sets, filters, replay, and platform requirements.

## run-iket

### Purpose and requirements

The instrumented kernel execution timeline profiler (IKET) records execution
traces from supported, instrumented kernels. The worker needs `run-iket`, a
compatible CuTe domain-specific language (DSL) distribution with IKET support,
and the GPU and driver required by that distribution. The application's kernel
must contain suitable instrumentation for the timeline you want to collect;
KCoral does not add instrumentation to uploaded source.

### Command format and options

```text
kcoral run run-iket [KCoral options] --out DIRECTORY -- profile [profile options] -- APPLICATION [arguments]
```

Common connection, execution, upload, and environment options apply. `--out`
is required; `--fetch` is not supported. The first `--` starts native profiler
arguments. The native `profile` command and the second `--` before the
application are required.

| Native option | Purpose |
| --- | --- |
| `--postprocess json` | Request JSON trace postprocessing. |
| `--postprocess perfetto` | Request output for a compatible Perfetto trace viewer. |
| `--keep` | Retain intermediate profiling outputs when supported by the installed version. |
| `--use-config PATH` | Use a profiler configuration available on the worker; upload it with the experiment when needed. |

Available options and postprocessing formats depend on the installed IKET
version. KCoral forwards them to `run-iket`. It owns the profiler's output and
working directories and rejects native `--output-dir`/`-o` and `--working-dir`
options. The wrapper supports the `profile` workflow, not standalone native
postprocessing subcommands.

### Collect and retrieve a timeline

```bash
kcoral run run-iket --send experiment --out artifacts/iket \
  -- profile --postprocess json -- python capture.py
```

To retain intermediate files as well as the processed trace:

```bash
kcoral run run-iket --send experiment --out artifacts/iket-debug --timeout 600 \
  -- profile --postprocess json --keep -- python capture.py
```

All files left in the profiler's managed output directory are downloaded under
`--out`, preserving their layout. Trace filenames and intermediate subdirectories
are determined by the native profiler version; KCoral does not rename them.
Inspect the returned directory and use a viewer compatible with the selected
postprocessing format.

### Exit behavior and troubleshooting

The command returns the native profiler's exit code. Available output files are
returned after an unsuccessful subprocess exit as well. If the managed output
directory contains no files, KCoral reports missing output and returns nonzero.
The existence of output files alone does not guarantee a useful timeline; check
that the returned trace contains the expected instrumented kernel activity.

If the executable is missing, install IKET in the worker environment. If the
application runs but the trace is empty, check kernel instrumentation and native
profiler diagnostics. Upload any files referenced by a custom configuration.
When inspecting a native option or format mismatch, run:

```bash
kcoral run shell -- run-iket profile --help
```

NVIDIA's [IKET profiling guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/iket_profiling.html)
describes kernel instrumentation and version-specific requirements.

## bench

### Purpose and requirements

`bench` checks and measures a candidate using a benchmark adapter in a local
TIRx-kernel-agent checkout. The adapter selects the task, input shapes, reference
implementation, and iteration defaults. KCoral sends that checkout's benchmark
code and inputs to the worker for execution.

The client needs a checkout containing `kernel-evolution/bench_adapter.py`, its
initialized `thirdparty/flashinfer-bench-evolve` submodule, the adapter's Python
dependencies, and access to the selected input tensor files. PyTorch is used
locally to load tensor data on the CPU. The worker needs the task's execution
dependencies and compatible GPU, including any libraries imported by the candidate.

### Command format and parameters

```text
kcoral run bench WORKLOAD [VERSION] [connection and execution options]
    [--repo PATH] [--warmup N] [--repeat N]
```

There is no required native-argument separator. Use the
[connection and execution options](#connection-and-execution-options) to select
the server, timeout, and capture limit. Code and input preparation are automatic,
as described below.

| Argument or option | Default | Meaning |
| --- | --- | --- |
| `WORKLOAD` | Required | Workload registered by the selected adapter, for example `kda/decode`. It is not an arbitrary Python filename. |
| `VERSION` | `baseline` | Candidate directory name such as `v0`, or `baseline` for the adapter's baseline run. |
| `--repo PATH` | Search current directory and parents | Checkout containing `kernel-evolution/bench_adapter.py`, or the directory containing `bench_adapter.py` directly. |
| `--warmup N` | Adapter's workload default | Nonnegative number of warmup iterations. |
| `--repeat N` | Adapter's workload default | Positive number of measured iterations. |

The registered workload list, shape selection, timing method, and defaults
belong to the checkout's adapter and may change with its revision. Inspect
that adapter's `PACKAGED` mapping for its workload keys and defaults. KCoral
does not maintain a separate fixed workload list.

### Checkout discovery and candidate selection

Without `--repo`, KCoral searches the current directory and each parent for
`kernel-evolution/bench_adapter.py` or `bench_adapter.py`. With `--repo`, it checks
only the supplied directory in those two forms.

For the standard adapter, this layout makes `kda/decode v0` select the shown
candidate:

```text
TIRx-kernel-agent/
  kernel-evolution/
    bench_adapter.py
    kda/decode/v0/lowered.py
  thirdparty/
    flashinfer-bench-evolve/
```

Run the baseline, a candidate, or a checkout elsewhere on your machine:

```bash
kcoral run bench kda/decode
kcoral run bench kda/decode v0 --warmup 3 --repeat 50
kcoral run bench kda/decode v0 --repo /path/to/TIRx-kernel-agent \
  --host gpu.example.com --port 8000
```

No explicit upload is needed. The adapter provides the candidate source,
benchmark harness, task definition, and tensor inputs. The candidate's additional
runtime imports must be available on the worker.

### Requests, output, and exit status

One invocation can cover multiple selected input shapes. KCoral sends a separate
request for each workload entry, with the required code and tensors included in
**every request**. Through a Router, successive requests may execute on different
workers. The timeout applies separately to each request, not to the total command.

Stdout includes the selected server and workload progress, worker information
when it changes, the harness's captured output, and the combined summary. Summary
columns and timing units are defined by the adapter's benchmark harness. The
command returns `1` if any returned correctness row has `passed=false`, and `0`
when execution completes without such a failure. Setup and execution errors are
nonzero. A correctness failure is different from an execution error: completed
rows can be summarized even when a candidate fails a correctness check.

`bench` does not write a results directory. To retain its text output, redirect
it locally:

```bash
kcoral run bench kda/decode v0 > bench.log 2> bench.err
```

If the adapter is not found, run from the checkout or pass `--repo`. If the
benchmark package or tensor files are missing, complete the checkout's dependency
and data setup. For an unknown workload, use a key registered by that adapter.
For a missing candidate, check `WORKLOAD/VERSION/lowered.py`. For remote import
or GPU errors, check the worker environment; the client's installed packages
are not automatically transferred.

## shell

### Purpose and requirements

`shell` runs the command after `--` directly. It is not limited to Bash and does
not automatically insert a shell. Use it for shell scripts, uploaded executables,
or programs installed on the worker. The selected executable and any interpreter
it needs must be available in the worker environment or uploaded working directory.

### Command format and argument handling

```text
kcoral run shell [KCoral options] -- EXECUTABLE [arguments]
```

All common connection, execution, upload, environment, and file-return options
are supported. `EXECUTABLE` is required; this command does not open a prompt.

| Command form | How it is executed |
| --- | --- |
| `sh setup.sh` | Find `sh` in the worker's executable search path and pass it the uploaded script. |
| `python setup.py` | Find `python` in the worker environment and run the script. |
| `./setup.sh` | Execute the uploaded file directly; it needs an executable bit and a valid interpreter declaration. |
| `/path/to/program` | Execute that path on the worker, if visible under its filesystem isolation settings. |
| `bash -c 'COMMANDS'` | Let the remote Bash process interpret pipelines, redirections, variable expansion, or multiple commands. |

Arguments are forwarded as separate arguments, preserving local quoting.
KCoral does not expand wildcards or interpret shell operators on its own.
A bare program name is searched on the worker's `PATH`; use `./program` to
select an uploaded executable in the current directory.

### Run scripts and return their files

These forms use different interpreters or direct execution:

```bash
kcoral run shell --send experiment -- bash setup.sh
kcoral run shell --send experiment -- sh setup.sh
kcoral run shell --send experiment -- python setup.py
kcoral run shell --send experiment -- ./setup.sh
```

The same `--fetch` and `--out` options work for all of them. If the script creates
`results/summary.json`, this invocation saves it as
`artifacts/setup/results/summary.json`:

```bash
kcoral run shell --send experiment --fetch results --out artifacts/setup \
  -- sh setup.sh
```

For direct execution, make the script executable before uploading it and use a
first line such as `#!/bin/sh` that selects an interpreter available remotely.
Uploaded compiled binaries must be compatible with the worker platform.

### Shell syntax, environment, and setup steps

Use an explicit remote shell when commands need shell syntax. Keep expressions
quoted so the local shell does not expand them first:

```bash
kcoral run shell --send experiment --env MODE=debug \
  -- bash -c 'printf "%s\n" "$MODE"; python check.py > check.log; cat check.log'
kcoral run shell --send experiment \
  -- sh -c 'sh setup.sh && python check.py'
```

The second example keeps setup and execution in the same request. For a batch
program that expects stdin, upload its input file and redirect it remotely:

```bash
kcoral run shell --send experiment -- sh -c './process < input.txt'
```

A later `kcoral run` invocation receives a fresh workspace; variables exported or files
created by an earlier script are not automatically carried into it. Package
installation or writes outside the workspace depend on the server's permissions
and isolation settings; `shell` is not a persistent remote login session.

Local shell operators outside the quoted remote command operate on the client.
For example, `kcoral run shell -- echo hello > result.txt` writes a **local**
file after receiving the remote output. To create a remote file, use a command
such as `sh -c 'echo hello > result.txt'` and select it with `--fetch result.txt`.

### Exit behavior and troubleshooting

The direct executable's exit code becomes the CLI exit code, subject to the
common artifact and execution failure rules. With `bash -c` or `sh -c`, the
shell determines the status. For example, `setup && check` prevents `check`
from running if setup fails, whereas a successful final command can hide an
earlier failure in a semicolon-separated sequence. Use the shell's own failure
handling when writing multi-command scripts.

A missing executable means its path was not found on the worker. An executable
permission or format error commonly means the uploaded file lacks an executable
bit, has an unavailable interpreter, or targets a different platform. A command
that expects input may exit or report end-of-file because stdin is closed.
Select output files with `--fetch` before the request ends; output collection
cannot turn an interactive program into a supported batch command.
