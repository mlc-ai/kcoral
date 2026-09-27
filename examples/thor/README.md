# Optimizing a GEMM on a remote Jetson Thor with a coding agent

This example shows remote kernel development with KCoral. A coding agent runs
on your workstation or laptop and optimizes an fp16 GEMM written in
[TIRx](https://github.com/apache/tvm/tree/v0.26.0/docs/tirx). Every compile,
correctness check and benchmark executes on a remote NVIDIA Jetson Thor
(sm_110a) KCoral server. The agent starts from a textbook CUDA-core kernel and
works towards cuBLAS. You can re-check the correctness and the speedup it
claims with one command.

```
 your machine                                      Thor
┌──────────────────────────────┐                ┌──────────────────────────┐
│ coding agent + bench.py      │ POST /execute  │ KCoral server            │
│ (KCoral client, no GPU)      │───────────────▶│ compiles TIRx, runs and  │
│ work/*.py candidate kernels  │                │ times kernels on sm_110a │
└──────────────────────────────┘                └──────────────────────────┘
```

| File | Purpose |
|---|---|
| `initial_kernel.py` | The agent's starting point: shared-memory tiled GEMM on CUDA cores |
| `definition.json`, `workload.jsonl` | The task in FlashInfer Trace format: the GEMM and its cuBLAS reference, and the shapes |
| `bench.py` | Client CLI: checks and times kernels plus cuBLAS in one KCoral request |
| `remote_bench.py` | Server-side harness that `bench.py` uploads with every request |
| `PROMPT.md` | Self-contained instructions for the optimization agent |
| `launch_server.sh` | Optional helper that starts a KCoral server on Thor over SSH |

## The task

The agent optimizes an fp16 GEMM, `D = A @ B^T` with fp32 accumulation, on four
shapes: two squares and two LLM feed-forward projections. It starts from
`initial_kernel.py`, a simple kernel on CUDA cores, and chases cuBLAS.

`bench.py` checks every kernel for correctness before timing it, and times all
kernels together with cuBLAS under the same protocol on the server, so the
speedups it reports are comparable and repeatable. It also copes with Thor's
load-dependent clocks. If you want the details, the task is defined in
[FlashInfer Trace](https://bench.flashinfer.ai/docs/flashinfer-trace) format
(`definition.json` for the operation and reference, `workload.jsonl` for the
shapes), and the correctness rule and timing protocol are implemented in
`remote_bench.py`.

## Setup

You need:

- [uv](https://docs.astral.sh/uv/getting-started/installation/) on your machine.
- This KCoral checkout.
- One of the following:
  - access to a running Thor KCoral server;
  - SSH access to a Jetson Thor (JetPack 7, CUDA 13 driver) to start one yourself.

### 1. Client environment

```bash
cd examples/thor
uv sync --locked
```

This sets up the KCoral client, which is all your machine needs.

### 2a. Use an existing Thor server

```bash
export KCORAL_URL=http://<server>:<port>
curl -fsS "$KCORAL_URL/health"    # expect "status":"ok" and "arch":"sm_110a"
```

### 2b. Or start a server on Thor

The server uses KCoral's own lockfile (`uv sync --locked --group gpu`). It
installs torch from the PyTorch cu132 index and TVM from PyPI; both work with
the JetPack 7 driver. The server listens only on `127.0.0.1` on the Thor host.
You reach it through an SSH tunnel and never expose it on the network.

The helper script does everything in one step:

```bash
./launch_server.sh <thor-host>          # e.g. user@thor.local; --port 8000 by default
```

It copies this checkout to `~/.cache/kcoral-thor-example` on the host
(`--dir` to change), installs the environment, starts `kcoral server` in the
background and waits until it is healthy. It then prints the tunnel command.
Rerunning it restarts the server, and `./launch_server.sh <thor-host> --stop`
stops it.

Or by hand:

```bash
# on Thor, once: install uv (see https://docs.astral.sh/uv/getting-started/installation/)
# copy the checkout, e.g. from your machine: git ls-files -z | rsync -a --from0 --files-from=- ./ <thor-host>:kcoral/
cd ~/kcoral
uv sync --locked --group gpu --python 3.12
# TVM compiles kernels with NVRTC and misses JetPack's CCCL headers without this.
export TVM_CUDA_NVRTC_EXTRA_OPTS=-I/usr/local/cuda/include/cccl
nohup .venv/bin/kcoral server --device gpu --gpus 0 --host 127.0.0.1 --port 8000 \
  > server.log 2>&1 &
```

Then on your machine, keep a tunnel open and point the example at it:

```bash
ssh -N -L 8000:127.0.0.1:8000 <thor-host>
export KCORAL_URL=http://127.0.0.1:8000
```

### 3. Optional: stable clocks

Jetson Thor scales its GPU and memory clocks with load and throttles when hot.
`bench.py` copes with the default governor. For the most repeatable
numbers, run the board in its maximum power mode with clocks pinned (this
needs root on Thor):

```bash
sudo nvpmodel -m 0        # MAXN power mode
sudo jetson_clocks        # pin clocks at maximum; `sudo jetson_clocks --restore` undoes it
```

## Run the initial kernel

```bash
uv run python bench.py initial_kernel.py
```

This takes a few minutes, because the initial kernel is slow by design. Every run
prints a per-shape table (median ms, spread across trials, TF/s, speedup
over the initial kernel, ratio to cuBLAS), the aggregate totals and the
clock/temperature line, and saves the full report to `results/`. Other useful
forms:

```bash
uv run python bench.py work/new.py --check-only --shapes 0   # quick correctness check
uv run python bench.py work/best.py work/new.py             # A/B in one request
uv run python bench.py work/new.py --emit-source            # also save generated CUDA C
```

## Let a coding agent optimize the kernel

`PROMPT.md` holds the agent's instructions; the agent works in `work/`. With
Claude Code, start `claude` in this directory with `KCORAL_URL` exported and
set the goal. `/goal` keeps the agent working until the stop rule in
`PROMPT.md` is met:

```text
/goal Follow PROMPT.md in this directory exactly. The goal is met when the stop rule in its
section 8 is satisfied (three consecutive different ideas without a confirmed 3% improvement,
or 40 rounds, or 4 hours), the final verification `uv run python bench.py initial_kernel.py
work/best.py` has been run and its output shown with ALL PASSED, work/RESULTS.md is written,
and the FINAL status line has been printed.
```

For an unattended run, pass the same goal headless:

```bash
claude -p "/goal Follow PROMPT.md in this directory exactly. ..." \
  --permission-mode auto --permission-prompts none
```

Other agents with a similar goal loop, such as Codex CLI's `/goal`, can be
given the same text. You can adjust the budget in the goal text.

## Verify the result yourself

Everything the agent claims can be re-measured independently:

```bash
uv run python bench.py initial_kernel.py work/best.py
```

- **Untouched harness.** `git status .` should show no changes: the
  benchmark files, the initial kernel and `PROMPT.md` must be as committed. The
  agent's `work/` and `results/` output is git-ignored.
- **Correctness.** Every row must read `PASS` and the run must end with
  `ALL PASSED` (the exit code is non-zero otherwise).
- **Speedup.** The aggregate line gives `speedup_vs_initial` and
  `ratio_vs_cublas` for `best`. Run the command two or three times.
  Differences smaller than the `spread` column and the warnings are noise.
- **Measurement conditions.** Check the telemetry line. A GPU clock below its
  maximum or a stability warning means the numbers are suspect, so rerun.
- **What actually ran.** `results/<run>.json` lists, per shape, the GPU kernel
  names each implementation launched. A candidate that calls cuBLAS is failed
  automatically.
- **How it got there.** Read `work/LOG.md` for the round-by-round history and
  `work/RESULTS.md` for the agent's own summary.

## Results from our validation run

One unattended run on 2026-09-24 used the `/goal` above with this setup:

| Setting | Value |
|---|---|
| Agent | Claude Code 2.1.282, headless (`claude -p`) |
| Model | Claude Opus 5.5 (1M context), high effort |
| Server | existing Thor KCoral server, JetPack 7 |
| Clocks | default governor, not pinned |

The agent stopped under the plateau rule after 10 rounds and about 45
minutes. Its path, from `work/LOG.md`:

| Round | Idea | Total ms (A/B run) |
|---|---|---|
| 0 | initial kernel: shared-memory tiled kernel on CUDA cores | 738.5 |
| 1 | TMA + `tcgen05` into TMEM, 128×256 tile, no overlap | 14.75 |
| 3 | 4-stage warp-specialized pipeline + L2-grouped tile order per shape | 5.58 |
| 6 | persistent 2-CTA clusters with `cta_group=2` MMAs, double-buffered TMEM, overlapped epilogue | 5.04 |
| 7 | 256×512 cluster tiles on the DRAM-bound shapes | 4.60 |

Rounds 2, 4, 5 and 8–10 tried other ideas without a confirmed gain; the log
records why. We then re-measured the final kernel independently three times
with `bench.py initial_kernel.py work/best.py`:

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| correctness | ALL PASSED | ALL PASSED | ALL PASSED |
| total ms, best / cuBLAS | 4.67 / 5.01 | 4.84 / 5.00 | 5.03 / 5.01 |
| speedup vs initial kernel | 158× | 152× | 147× |
| ratio vs cuBLAS | 1.07× | 1.03× | 1.00× |

Per shape, the kernel is about 1.2× faster than cuBLAS on the long-K
`m2048-n4096-k11008` projection and within a few percent of it on the square
shapes. On `m2048-n11008-k4096` its timing is bimodal: `bench.py` flags a
10–24% spread across trials there. That is the kind of lead a further run could
pick up. Your numbers will differ with clocks, temperature, the model and
effort level you run, and the agent's choices. The speedup over the initial
kernel should land well above 100×.

## Troubleshooting

- **`cannot reach KCoral server`**: the server or your connection to it (for
  example the SSH tunnel) is down, or `KCORAL_URL` points elsewhere.
  `uv run python bench.py --health` should answer.
- **Every kernel fails with `cannot open source file "cuda/std/cstdint"`**:
  TVM 0.26 compiles with NVRTC and looks for the CCCL headers under
  `targets/aarch64-linux`, but JetPack installs the toolkit as
  `targets/sbsa-linux`. Start the server with
  `TVM_CUDA_NVRTC_EXTRA_OPTS=-I/usr/local/cuda/include/cccl` (the setup script
  does this).
- **A request is slow to start**: another request holds the GPU. KCoral runs
  one GPU request at a time and queues the rest.
- **The server log warns `bubblewrap could not start`**: some hosts forbid the
  unprivileged namespaces KCoral's sandbox uses. The server then runs without
  filesystem isolation. That is fine on a machine you control; pass
  `--sandbox none` to silence the warning.
- **A kernel hangs until the request times out**: usually an mbarrier phase or
  expected-transaction byte count is wrong. The request fails with a timeout
  and the next request gets a fresh worker.
- **Large spread or clock warnings**: the board is throttling or other work is
  running on it. Let it cool down, pin clocks (see above) and rerun.
