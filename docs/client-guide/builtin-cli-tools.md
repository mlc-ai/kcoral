# Builtin CLI Tools

KCoral's built-in command line tools upload experiments and run them on an
existing server through `kcoral run TOOL`. They also work through a KCoral Router.

| Tool | Purpose |
| --- | --- |
| [python](#python) | Run a Python script, module, or inline command. |
| [compute-sanitizer](#compute-sanitizer) | Check CUDA memory accesses and synchronization. |
| [ncu](#ncu) | Collect NVIDIA Nsight Compute performance reports. |
| [run-iket](#run-iket) | Collect instrumented kernel execution timelines. |
| [bench](#bench) | Check and benchmark a candidate from a TIRx-kernel-agent checkout. |
| [shell](#shell) | Run a script or another executable. |

## Common usage

List the available tools and inspect a tool's options:

```bash
kcoral run --help
kcoral run python --help
```

`kcoral run --help` lists the tools; `kcoral run TOOL --help` shows the selected
tool's options. Install the programs and packages used by your experiment on
the **server**. The `bench` section also describes its local dependencies.

### Select a server

All tools accept the same connection options. Set a default address through
the environment:

```bash
export KCORAL_URL='http://SERVER_HOST:PORT'
```

Or specify an HTTP address directly on a command:

```bash
kcoral run python --host gpu.example.com --port 8000 --send experiment -- check.py
kcoral run bench kda/decode v0 --host gpu.example.com --port 8000
```

`--host` and `--port` take precedence over `KCORAL_URL`. If either option is
supplied, the address is `http://HOST:PORT`; an omitted host defaults to
`127.0.0.1` and an omitted port to `8000`. Neither value is inherited from
`KCORAL_URL`. Hostnames, IPv4, and IPv6 addresses are supported. If `KCORAL_URL`
is also set, the command writes a warning to stderr with the effective address:

```text
kcoral: warning: --host/--port override KCORAL_URL; using http://gpu.example.com:8000
```

Use `--url URL` for a complete URL, including HTTPS or a path prefix; it also
overrides `KCORAL_URL`. Do not combine `--url` with `--host` or `--port`.

### Timeouts and output

Every tool accepts these request options:

| Option | Default | Behavior |
| --- | --- | --- |
| `--timeout SECONDS` | `300` | Execution deadline for each request, capped by the server's configured maximum. |
| `--output-limit-bytes N` | `268435456` (256 MiB) | Capture limit per output stream, also capped by the server. |

Remote output is replayed when a request completes. The client reports output
truncation. A timeout kills the worker and its subprocess group, and can prevent
artifact collection. A transport failure or failed server instruction produces
a nonzero exit code.

Commands run as the configured worker user, with the server's filesystem
isolation settings. Each request has a fresh workspace; files needed afterward
must be returned explicitly. Upload and result sizes are subject to the server's
[transfer limits](protocol.md).

### Uploads and subprocess arguments

`python`, `compute-sanitizer`, `ncu`, `run-iket`, and `shell` share the following
upload and subprocess options. `bench` prepares its code and inputs from a
benchmark checkout, as described in its section below.

```bash
kcoral run TOOL [KCoral options] -- [native tool arguments]
```

The first `--` separates KCoral options from native tool arguments. Arguments
containing spaces remain single arguments. Standard input is closed.

`--send PATH` accepts a file or a directory and can be repeated. A file is
uploaded under its basename; a directory's **contents** become the remote
working directory, so `experiment/check.py` is run as `check.py`. Conflicting
paths, symbolic links, and special files are rejected. Python `__pycache__`
directories are skipped; other hidden files are included. Send a focused
experiment directory instead of a checkout containing environments, caches,
or repository metadata. Executable bits are preserved. Without `--send`, the
command starts in an empty directory.

`KCORAL_DIR` points to this working directory, and the worker's Python
environment is first on `PATH`. Commands inherit the worker's GPU assignment.
Local files and local environment variables are not sent implicitly. To pass
variables to the subprocess:

```bash
kcoral run python --send experiment -e MODE=debug -e MY_LOCAL_VARIABLE -- check.py
```

`-e NAME` copies that variable's local value; `-e NAME=VALUE` sets it explicitly.
Both forms can be repeated. `CUDA_VISIBLE_DEVICES` and `KCORAL_DIR` are managed
by KCoral and cannot be overridden with `-e`.

### Returned files and exit codes

For `python`, `compute-sanitizer`, and `shell`, use `--fetch PATH` with
`--out DIRECTORY` to download selected files or directories. `--fetch` is
repeatable and selects paths relative to the uploaded working directory.
Returned paths keep their relative names, including nested directories and
empty directories.

The profilers `ncu` and `run-iket` require `--out DIRECTORY` and collect their
reports automatically. They do not accept `--fetch`.

In all cases, `--out` must name a **new** local directory; parent directories
are created automatically. Existing output is never overwritten. Reports and
selected files are downloaded even when the subprocess exits unsuccessfully.
A missing requested output produces a nonzero exit code.

These five tools return the subprocess's exit code; termination by signal N
returns 128 + N. `bench` reports failure when a correctness check fails.

## python

Run a script, inline command, or module with the worker's Python interpreter:

```bash
kcoral run python --send experiment -- check.py
kcoral run python -- -c 'print("hello from the worker")'
kcoral run python --send experiment -- -m my_package.check
```

Install the script's Python dependencies in the worker environment. Interactive
Python and execution from standard input are unsupported.

To return a file produced by the script:

```bash
kcoral run python --send experiment --fetch result.bin --out artifacts/python \
  -- check.py
```

## compute-sanitizer

NVIDIA Compute Sanitizer checks CUDA memory accesses and synchronization.
Install `compute-sanitizer` on the server, then pass its native options before
the application:

```bash
kcoral run compute-sanitizer --send experiment -- python check.py
kcoral run compute-sanitizer --send experiment \
  -- --tool racecheck python check.py
```

Compute Sanitizer's native exit behavior is preserved. Pass `--error-exitcode 1`
when findings should fail your command. To return its log:

```bash
kcoral run compute-sanitizer --send experiment \
  --fetch sanitizer.log --out artifacts/check \
  -- --error-exitcode 1 --log-file sanitizer.log python check.py
```

## ncu

NVIDIA Nsight Compute collects GPU kernel performance measurements. Install
`ncu` on the server with the GPU, driver, and profiling permissions it requires.

```bash
kcoral run ncu --send experiment --out artifacts/ncu \
  -- --set basic --launch-count 1 -- python capture.py
```

The first `--` starts Nsight Compute options; the second starts the application
command. KCoral downloads the report to `artifacts/ncu/capture.ncu-rep` for
local inspection. `--out` is required.

KCoral manages the report path and capture mode. Do not pass native export,
import, mode, or configuration-file options. Implicit Nsight configuration
files are disabled. `NCU_PROFILE=1` is set unless explicitly overridden with
`-e`, allowing TIRx's Proton timing helper to avoid a conflicting profiler
subscription.

## run-iket

The instrumented kernel execution timeline profiler (`run-iket`) records
kernel execution traces. Install a CuTe DSL distribution with IKET support on
the server and use an instrumented kernel that can produce a timeline.

```bash
kcoral run run-iket --send experiment --out artifacts/iket \
  -- profile --postprocess json -- python capture.py
```

The first `--` starts profiler arguments, including the required `profile`
command; the second starts the application. `--out` is required. KCoral returns
the profiler's output directory, including JSON traces and retained
intermediate files, to `artifacts/iket`.

KCoral manages the output and working directories. Do not pass native
`--output-dir` or `--working-dir` options.

## bench

Check and benchmark a TIRx candidate from a current TIRx-kernel-agent checkout
with its `thirdparty/flashinfer-bench-evolve` submodule initialized:

```bash
kcoral run bench kda/decode v0
kcoral run bench kda/decode baseline --warmup 3 --repeat 50
kcoral run bench kda/decode v0 --repo /path/to/TIRx-kernel-agent
```

`bench` accepts a workload and an optional candidate version directly, without
a native-argument `--` separator. It discovers `kernel-evolution/bench_adapter.py`
from the current directory or its parents, or from `--repo`. The adapter defines
the registered workloads, default iteration counts, shape selection, pinned
benchmark sources, and input tensors. A version such as `v0` selects
`kernel-evolution/kda/decode/v0/lowered.py`; omitting it runs the baseline.
`--warmup` and `--repeat` override the workload's iteration counts.

The local environment needs the adapter's dependencies, including PyTorch to
load input tensors on the CPU. The remote environment needs the task's GPU
dependencies. KCoral ships the pinned harness, task definition, candidate source,
and required tensors **for every workload request**, so successive requests can
run on different workers. It prints the harness's correctness and timing output
and the combined summary. Failed correctness checks produce a nonzero exit code.
Benchmark task definitions stay in the source checkout.

`bench` uses the common connection, timeout, and output-limit options. It does
not accept `--send`, `-e`, `--fetch`, or `--out`.

## shell

Run a script or executable available in the worker environment or uploaded
working directory:

```bash
kcoral run shell --send experiment -- bash setup.sh
kcoral run shell --send experiment -- sh setup.sh
kcoral run shell --send experiment -- python setup.py
kcoral run shell --send experiment -- ./setup.sh
kcoral run shell --send experiment --fetch results --out artifacts/setup \
  -- bash setup.sh
```

`shell` executes the command given to it directly and is not limited to Bash.
The same `--fetch` and `--out` options apply to each command. For shell syntax
such as pipes, redirection, or multiple commands, invoke a shell explicitly:

```bash
kcoral run shell --send experiment -- bash -c 'python check.py > check.log && cat check.log'
```

Install the selected shell and other executables on the server. Uploaded
scripts retain their executable bits, so `-- ./setup.sh` also works for an
executable script with a suitable interpreter line.
