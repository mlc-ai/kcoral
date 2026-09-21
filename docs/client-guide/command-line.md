# Remote command line tools

The `kcoral` client can upload an experiment and run Python, Compute Sanitizer
(CUDA memory and synchronization checks), NVIDIA Nsight Compute (`ncu`),
the instrumented kernel execution timeline profiler (`run-iket`), benchmarks,
or shell commands on an existing server. Install the client with `pip install .`
and select the server:

```bash
export KCORAL_URL='http://SERVER_HOST:PORT'
```

Use `--url URL` on a command to override the environment variable. The tools
use the ordinary execution protocol and work through a KCoral Router too.
Use `kcoral server` to start an execution server and `kcoral router` to start a
Router. `kcoral run --help` lists the remote tools. Server flags follow the
`server` subcommand; tool-specific options follow `run TOOL`.

## Upload and execute

Given an `experiment/` directory containing `check.py`, `capture.py`, and
`setup.sh`:

```bash
kcoral run python --send experiment -- check.py
kcoral run compute-sanitizer --send experiment -- python check.py
kcoral run compute-sanitizer --send experiment \
  -- --tool racecheck python check.py
kcoral run ncu --send experiment --out artifacts/ncu \
  -- --set basic --launch-count 1 -- python capture.py
kcoral run run-iket --send experiment --out artifacts/iket \
  -- profile --postprocess json -- python capture.py
kcoral run shell --send experiment -- bash setup.sh
```

The first `--` separates KCoral options from native tool arguments. For `ncu`
and `run-iket`, a second `--` separates profiler options from the application.
Arguments containing spaces remain single arguments. `shell` executes the
command given to it; use `bash -c '...'` for shell syntax such as pipes.

`--send` accepts a file or a directory and can be repeated. A file is uploaded
under its basename; a directory's **contents** become the remote working
directory, so `experiment/check.py` is run as `check.py`. Conflicting paths,
symbolic links, and special files are rejected. Python `__pycache__` directories
are skipped; other hidden files are included. Send a focused experiment directory
instead of a checkout containing environments, caches, or repository metadata.
Executable bits are preserved. Without `--send`, the command starts in an empty
directory. Local files and local environment variables are not sent implicitly.

Each invocation has a new working directory. `KCORAL_DIR` points to it, and the
worker's Python environment is first on `PATH`. Commands inherit the worker's
GPU assignment. To pass additional variables:

```bash
kcoral run python --send experiment -e MODE=debug -e MY_LOCAL_VARIABLE -- check.py
kcoral run python -- -c 'print("hello from the worker")'
kcoral run python --send experiment -- -m my_package.check
```

`-e NAME` copies that variable's local value; `-e NAME=VALUE` sets it explicitly.
`CUDA_VISIBLE_DEVICES` and `KCORAL_DIR` are managed by KCoral and cannot be
overridden with `-e`. Python uses the worker's interpreter. Standard input is
closed and interactive Python is unsupported. Output is replayed when the
request completes, rather than streamed live.

## Reports and exit status

`ncu` returns `artifacts/ncu/capture.ncu-rep`. It disables implicit Nsight
configuration files and sets `NCU_PROFILE=1` unless explicitly overridden, which
lets TIRx's Proton timing helper avoid a conflicting profiler subscription.
`run-iket` returns its output directory, including JSON traces and retained
intermediate files. KCoral manages profiler output paths; do not also pass native
output-directory, export, or import options. Inspect saved reports locally.

For Python, shell, and Compute Sanitizer, select additional files explicitly:

```bash
kcoral run compute-sanitizer --send experiment \
  --fetch sanitizer.log --out artifacts/check \
  -- --error-exitcode 1 --log-file sanitizer.log python check.py
kcoral run shell --send experiment --fetch results --out artifacts/setup \
  -- bash setup.sh
```

`--fetch` is repeatable and selects paths relative to the uploaded directory.
It must be paired with `--out`. Returned paths keep their relative names,
including nested directories and empty directories. `--out` must name a **new**
local directory; parent directories are created automatically. Existing output
is never overwritten. Upload and result sizes are subject to the server's
[transfer limits](protocol.md).

The client returns the remote process's exit code; termination by signal N
returns 128 + N. Reports are downloaded even when the process exits unsuccessfully.
A missing requested report, transport failure, or failed server instruction
produces a nonzero exit code. Compute Sanitizer's native exit behavior is
preserved: pass `--error-exitcode 1` when findings should fail your command.

`--timeout SECONDS` requests an execution deadline (default 300 seconds), capped
by the server's configured maximum. A server timeout kills the worker and its
subprocess group. `--output-limit-bytes N` requests a per-stream capture limit
(default 16 MiB), also capped by the server. The client reports truncation.
Hard timeouts can prevent artifact collection.

## Benchmark a TIRx candidate

From a current TIRx-kernel-agent checkout with its
`thirdparty/flashinfer-bench-evolve` submodule initialized:

```bash
kcoral run bench kda/decode v0
kcoral run bench kda/decode baseline --warmup 3 --repeat 50
kcoral run bench kda/decode v0 --repo /path/to/TIRx-kernel-agent
```

KCoral discovers `kernel-evolution/bench_adapter.py` from the current directory
or its parents, or from `--repo`. The adapter defines the registered workloads,
default iteration counts, shape selection, pinned benchmark sources, and input
tensors. A version such as `v0` selects
`kernel-evolution/kda/decode/v0/lowered.py`; omitting it runs the baseline.

The local environment needs the adapter's dependencies, including PyTorch to
load input tensors on the CPU. The remote environment needs the task's GPU
dependencies. KCoral ships the pinned harness, task definition, candidate source,
and required tensors **for every workload request**, so successive requests can
run on different workers. It prints the harness's correctness and timing output
and the combined summary. Failed correctness checks produce a nonzero exit code.
Benchmark task definitions stay in the source checkout rather than being copied
into KCoral.

## Worker dependencies

The regular remote tools need only the KCoral client on the local machine.
Install Python packages, Compute Sanitizer, Nsight Compute, and an IKET-enabled
CuTe DSL distribution on the **server** as needed by the experiment. Missing
executables are reported by name. Profiler-specific GPU and driver requirements
still apply, and `run-iket` needs an instrumented kernel to produce a timeline.
Commands run as the configured worker user, with the server's filesystem
isolation settings. Files needed after a request must be returned explicitly.

These commands adapt the remote tool runners and benchmark driver from
[TIRx-kernel-agent](https://github.com/mlc-ai/TIRx-kernel-agent/tree/e25825057c1abbc9bf8676d0e0c2bfa24a1dff59/kernel-evolution).
