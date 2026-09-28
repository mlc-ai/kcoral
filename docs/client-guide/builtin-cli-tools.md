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
| [shell](#shell) | Run a shell script, uploaded executable, or program installed on the worker. |

The common reference below defines KCoral's options and execution behavior.
Each tool section then describes its command format, dependencies, native
arguments, returned files, and failure handling.

## Common reference

### Command structure and help

All tools use this structure:

```text
kcoral run TOOL [KCoral options] -- [native arguments]
```

Put connection, upload, environment, and download options **before** the first
`--`. Everything after it belongs to the selected tool. For `ncu` and
`run-iket`, a second `--` separates profiler options from the application.

```bash
kcoral run --help
kcoral run python --help
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
| `--timeout SECONDS` | `300` | Positive integer execution deadline for each request, capped by the server's configured maximum. |
| `--output-limit-bytes N` | `268435456` (256 MiB) | Positive integer capture limit for **each** of stdout and stderr, capped by the server. Zero is not accepted by these CLI tools. |

Select a server with an environment variable:

```bash
export KCORAL_URL='http://gpu.example.com:8000'
kcoral run python -- -c 'print("hello from the worker")'
```

Or specify the address on the invocation:

```bash
kcoral run python --host gpu.example.com --port 8000 --send experiment -- experiment/check.py
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

These options control uploaded files, remote environment variables, and returned files.
Use them before the first `--`; the tool sections describe how each command
prepares its inputs and collects its results.

| Option | Default | Meaning |
| --- | --- | --- |
| `--send PATH` | No uploaded files | Upload a file or directory, preserving its name. Repeat for additional inputs. |
| `-e NAME[=VALUE]`, `--env NAME[=VALUE]` | Worker environment | Set one remote environment variable, or copy one local variable by name. Repeat for additional variables. |
| `--fetch PATH` | No selected outputs | Return a file or directory relative to the remote working directory. Repeat for additional outputs; requires `--out`. |
| `--out DIRECTORY` | No download directory | New local destination for returned files. Required by profilers; other tools require `--fetch` and `--out` together. |

### Input layout and working directory

`--send` snapshots local inputs before execution. The worker receives regular
files in a fresh working directory; `KCORAL_DIR` points to that directory.

| Local selection | Remote path |
| --- | --- |
| `--send experiment/check.py` | `check.py` |
| `--send experiment` containing `check.py` | `experiment/check.py` |
| `--send experiment` containing `data/input.json` | `experiment/data/input.json` |
| `--send /local/path/experiment` containing `check.py` | `experiment/check.py` |
| `--send .` from a directory named `experiment` | `experiment/...` |

A directory keeps its own name and internal layout; local parent directories
are not included. A trailing slash does not change this behavior. Selecting `.`
uses the current directory's name. The filesystem root cannot be selected
because it has no directory name. A selected file uses its filename alone.
Multiple `--send` selections share one remote working directory: differently
named directories can contain identically named files, while duplicate or
conflicting remote file paths are rejected. Symbolic links and special files
are rejected. Python
`__pycache__` entries are skipped; other hidden files are included. Empty input
directories are not uploaded. Executable bits are preserved, but full original
permission modes are not.

The command runs from the remote working directory, **not** from inside the
uploaded directory. For `--send experiment`, pass `experiment/check.py` to Python
and `experiment/data/input.json` for a data path relative to the working
directory. `KCORAL_DIR` points to the parent of `experiment/`. Running a script
does not automatically change the working directory to the script's location.
Files written as `results/report.json` are still selected with `--fetch results`;
files written as `experiment/results/report.json` need `--fetch experiment/results`.

Send the files your program needs, including local modules and configuration.
Sending a script does not discover its imports or upload its parent directory.
Without `--send`, the tool starts in an empty working directory. Relative paths
in tool arguments are resolved in that directory; absolute paths refer to the worker's
filesystem, subject to the server's isolation settings.

### Environment variables and executable lookup

Environment overrides are optional. Without them, execution uses the
worker's environment, including its assigned GPU. The worker's Python executable
directory is prepended to `PATH` when locating programs.

Use one `-e` or `--env` per variable:

```bash
kcoral run python --send experiment --env MODE=debug --env MY_LOCAL_VARIABLE -- experiment/check.py
kcoral run shell --send experiment -e MODE=debug -e LABEL='trial one' -- sh experiment/setup.sh
```

| Form | Behavior |
| --- | --- |
| `--env MODE=debug` | Set `MODE` to `debug` for remote execution. |
| `--env MY_LOCAL_VARIABLE` | Read this variable from the client's environment and send its value. An unset local variable is an argument error. |
| `--env EMPTY=` | Set a variable to an empty string. |
| Repeated assignments to the same name | The last assignment wins. |

Variable names must start with a letter or underscore and contain only letters,
digits, or underscores. `CUDA_VISIBLE_DEVICES` and `KCORAL_DIR` are managed by
KCoral and cannot be overridden. Setting `PATH` with `--env` still leaves the
worker's Python executable directory first in the executable search path.
Overrides apply only to this invocation's subprocess environment.

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
messages and warnings use stderr. Local redirection and pipelines still work,
for example:

```bash
kcoral run python --send experiment -- experiment/check.py > run.log 2> run.err
```

### Returned files, limits, and exit status

`--fetch` paths must be relative, without `..` components. Returned files and
directories retain their relative layout, including empty output directories.
Symbolic links and special files cannot be returned. `--out` must name a new
local directory; parent directories are created automatically, and existing
output is never overwritten.

Files are collected after execution, including a subprocess failure. A hard
timeout, worker failure, or response-size error can prevent their return.
Each request has its own workspace; a later invocation is not a
continuation of the earlier one. Keep related setup and execution steps in one
invocation, or explicitly download and resend the needed files.

The server caps execution time, captured output, and serialized responses.
Its default maximum requested capture is 16 MiB per stream, and its default
response limit is 256 MiB. A server may configure different limits. See
[transfer limits](protocol.md) and [server configuration](../server-guide/launch-the-server.md#configuration).

| Outcome | CLI exit behavior |
| --- | --- |
| Subprocess completes and requested files are available | Preserve its exit code. |
| Subprocess is terminated by signal N | Return `128 + N`. |
| Requested output is missing | Return nonzero; preserve an existing nonzero subprocess exit code. |
| Transport, worker execution, or artifact-return failure | Return `1` with a diagnostic on stderr. |
| Invalid KCoral command line | Return `2` without running the tool. |

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
kcoral run python --send experiment -- experiment/check.py --input experiment/data/input.json
kcoral run python --send experiment -- -m experiment.my_package.check
kcoral run python --send experiment -- -W ignore -X dev experiment/check.py
kcoral run python -- -c 'import sys; print(sys.version)'
```

### Return results and interpret failures

If `check.py` writes `results/report.json`, download its containing directory:

```bash
kcoral run python --send experiment --fetch results --out artifacts/python -- experiment/check.py
```

The local result is `artifacts/python/results/report.json`. For exact binary
output, write a remote file and fetch it instead of writing binary bytes to
stdout. Results are still collected after `sys.exit(N)` or an uncaught exception,
provided execution reaches normal subprocess completion.

`sys.exit(N)` becomes the CLI exit code. An uncaught exception normally produces
a traceback on stderr and a nonzero code. A missing module usually means it was
neither uploaded nor installed on the worker; installing it only on the client
does not make it available remotely. A missing script often means the command
used `check.py` instead of `experiment/check.py` after `--send experiment`.

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
installed on the worker or an uploaded executable such as `./experiment/check`.

### Check a Python or compiled application

Run the default checker, select a race check, or check an uploaded executable:

```bash
kcoral run compute-sanitizer --send experiment -- python experiment/check.py
kcoral run compute-sanitizer --send experiment \
  -- --tool racecheck --error-exitcode 1 python experiment/check.py
kcoral run compute-sanitizer --send experiment \
  -- --tool memcheck --error-exitcode 1 ./experiment/check
```

A compiled program must target the worker's platform and preserve its executable
bit in the upload. For useful source locations, build it with the line
information recommended by Compute Sanitizer.

### Save findings and handle failures

```bash
kcoral run compute-sanitizer --send experiment \
  --fetch sanitizer.log --out artifacts/check \
  -- --tool memcheck --error-exitcode 1 --log-file sanitizer.log python experiment/check.py
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
  -- --set basic --launch-count 1 -- python experiment/capture.py
```

Use the native defaults, or select a kernel and skip warmup launches:

```bash
kcoral run ncu --send experiment --out artifacts/ncu-default \
  -- -- python experiment/capture.py
kcoral run ncu --send experiment --out artifacts/ncu-filtered --timeout 600 \
  -- --kernel-name 'regex:my_kernel.*' --launch-skip 5 --launch-count 1 \
  -- python experiment/capture.py
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

IKET records execution traces from supported, instrumented kernels. The worker
needs `run-iket`, a compatible CuTeDSL distribution with IKET support,
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
  -- profile --postprocess json -- python experiment/capture.py
```

To retain intermediate files as well as the processed trace:

```bash
kcoral run run-iket --send experiment --out artifacts/iket-debug --timeout 600 \
  -- profile --postprocess json --keep -- python experiment/capture.py
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
| `sh experiment/setup.sh` | Find `sh` in the worker's executable search path and pass it the uploaded script. |
| `python experiment/setup.py` | Find `python` in the worker environment and run the script. |
| `./experiment/setup.sh` | Execute the uploaded file directly; it needs an executable bit and a valid interpreter declaration. |
| `/path/to/program` | Execute that path on the worker, if visible under its filesystem isolation settings. |
| `bash -c 'COMMANDS'` | Let the remote Bash process interpret pipelines, redirections, variable expansion, or multiple commands. |

Arguments are forwarded as separate arguments, preserving local quoting.
KCoral does not expand wildcards or interpret shell operators on its own.
A bare program name is searched on the worker's `PATH`; use `./program` to
select an uploaded executable in the current directory.

### Run scripts and return their files

These forms use different interpreters or direct execution:

```bash
kcoral run shell --send experiment -- bash experiment/setup.sh
kcoral run shell --send experiment -- sh experiment/setup.sh
kcoral run shell --send experiment -- python experiment/setup.py
kcoral run shell --send experiment -- ./experiment/setup.sh
```

The same `--fetch` and `--out` options work for all of them. If the script creates
`results/summary.json`, this invocation saves it as
`artifacts/setup/results/summary.json`:

```bash
kcoral run shell --send experiment --fetch results --out artifacts/setup \
  -- sh experiment/setup.sh
```

For direct execution, make the script executable before uploading it and use a
first line such as `#!/bin/sh` that selects an interpreter available remotely.
Uploaded compiled binaries must be compatible with the worker platform.

### Shell syntax, environment, and setup steps

Use an explicit remote shell when commands need shell syntax. Keep expressions
quoted so the local shell does not expand them first:

```bash
kcoral run shell --send experiment --env MODE=debug \
  -- bash -c 'printf "%s\n" "$MODE"; python experiment/check.py > check.log; cat check.log'
kcoral run shell --send experiment \
  -- sh -c 'sh experiment/setup.sh && python experiment/check.py'
```

The second example keeps setup and execution in the same request. For a batch
program that expects stdin, upload its input file and redirect it remotely:

```bash
kcoral run shell --send experiment -- sh -c './experiment/process < experiment/input.txt'
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
