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
| `bench.py` | Benchmark harness: checks and times kernels plus cuBLAS in one KCoral request (it uploads itself as the server-side code) |
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
shapes), and the correctness rule and timing protocol are implemented in the
server-side part of `bench.py`.

## Setup

You need:

- [uv](https://docs.astral.sh/uv/getting-started/installation/) on your machine.
- This KCoral checkout.
- One of the following:
  - access to a running Thor KCoral server;
  - SSH access to a Jetson Thor (JetPack 7, CUDA 13 driver) with
    [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, to
    start one yourself.

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

```bash
./launch_server.sh <thor-host>                 # e.g. user@thor.local
ssh -f -N -L 8000:127.0.0.1:8000 <thor-host>   # tunnel to the server, in the background
export KCORAL_URL=http://127.0.0.1:8000
```

`launch_server.sh` sets up and starts a KCoral server on Thor for you. The
server listens only on the host's `127.0.0.1`, hence the tunnel. See the script
for its options (`--port`, `--dir`, `--stop`).

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

- **Correctness.** The run must end with `ALL PASSED` (the exit code is
  non-zero otherwise), and `git status .` must show the example's files
  unchanged; the agent's `work/` and `results/` are git-ignored.
- **Speed.** The aggregate line gives `speedup_vs_initial` and
  `ratio_vs_cublas`. Rerun two or three times: differences within the
  `spread` column are noise, and a stability warning means the numbers are
  suspect.
- **Details.** `results/<run>.json` lists the GPU kernels each implementation
  launched (a candidate that calls cuBLAS fails automatically); `work/LOG.md`
  and `work/RESULTS.md` hold the agent's history and summary.

## Results from our validation run

One unattended run on 2026-09-28 used the `/goal` above with this setup:

| Setting | Value |
|---|---|
| Agent | Claude Code 2.1.283, headless (`claude -p`) |
| Model | Claude Opus 5.5 (1M context), high effort |
| Server | fresh Thor KCoral server from `launch_server.sh`, JetPack 7 |
| Clocks | default governor, not pinned |

The agent stopped under the plateau rule after 17 rounds and about 66
minutes. The rounds that improved its best kernel, from `work/LOG.md`:

| Round | Idea | Total ms (A/B run) |
|---|---|---|
| 0 | initial kernel: shared-memory tiled kernel on CUDA cores | 738.1 |
| 1 | TMA + `tcgen05` into TMEM, 128×256 tile, no overlap | 16.69 |
| 3 | L2-grouped tile order with a per-shape group size, on a 4-stage warp-specialized pipeline | 8.38 |
| 4 | 16-byte vectorized epilogue stores | 5.72 |
| 6 | persistent 2-CTA clusters with `cta_group=2` MMAs and double-buffered TMEM | 5.03 |
| 7 | 256×512 cluster tiles on the large and long-K shapes | 4.73 |
| 11 | K-direction serpentine walk over consecutive tiles | 4.34 |
| 14 | per-tile widths and L2-capacity-aware grouping on `m2048-n11008-k4096` | 4.22 |

The other rounds tried ideas such as a TMA-store epilogue, L2 cache hints and
explicit prefetching without a confirmed gain; the log records why. Two calls
were the agent's own: it did not count rounds 8–10 as three different ideas
(round 10 re-applied round 8 per shape), so it kept going past round 13; and it
promoted round 14 although both confirmation runs warned about spread, which
came from the previous best's bimodal timing on `m2048-n11008-k4096`.

Its final verification, `bench.py initial_kernel.py work/best.py`, passed
without warnings:

| | Total ms | vs initial kernel | vs cuBLAS |
|---|---|---|---|
| initial kernel | 738.2 | 1× | 0.007× |
| cuBLAS | 5.00 | 148× | 1× |
| agent's kernel | **4.21** | **175×** | **1.19×** |

The kernel beats cuBLAS on every shape, by 1.04–1.07× on three of them and by
1.42× on the long-K `m2048-n4096-k11008`. Re-running the same command afterwards
on the same server reproduced it within 1.18–1.20× of cuBLAS (4.23–4.24 ms) in
four warning-free runs. Your numbers will differ with clocks, temperature, the
model and effort level you run, and the agent's choices. The speedup over the
initial kernel should land well above 100×.

## Troubleshooting

- **`cannot reach KCoral server`**: the server or your connection to it (for
  example the SSH tunnel) is down, or `KCORAL_URL` points elsewhere.
  `curl -fsS "$KCORAL_URL/health"` should answer.
- **Every kernel fails with `cannot open source file "cuda/std/cstdint"`**: TVM
  0.26's NVRTC misses JetPack's CCCL headers. Start the server with
  `TVM_CUDA_NVRTC_EXTRA_OPTS=-I/usr/local/cuda/include/cccl`, as
  `launch_server.sh` does.
- **Large spread or clock warnings**: the board is throttling or other work is
  running on it. Let it cool down and rerun.
