# AccRL CUDA turn corpus

This directory contains every CUDA kernel that AccRL could extract from every
assistant turn of the selected architecture cohort. The checked-in corpus is the
B200 cohort; every manifest row says `source_arch: "b200"` explicitly. Kernels
are not filtered by evaluation outcome: compile errors, incorrect results,
runtime errors, timeouts, and successful turns are all retained.

## Extraction

The corpus is generated with AccRL's
`fib_runtime/mini_swe_agent_docker/plots/analyze_kernel_per_turn.py` utility.
The wrapper classifies `compute_100a`/`sm_100a` as B200 and
`compute_90a`/`sm_90a` as H100, with experiment tags as a fallback, then filters
on the explicitly selected architecture before calling AccRL's
`extract_turns_from_trajectory()` function:

```bash
uv run python scripts/extract_accrl_cuda_kernels.py \
  --accrl-root /home/yixind/dev/AccRL \
  --eval-root /home/yixind/AccRL-exps/eval_runs \
  --source-arch b200 \
  --output ./cuda_kernels \
  --force
```

AccRL skips an assistant turn only when it cannot extract a CUDA/CPP code block
from that response. No correctness or performance status is used as a filter.

## Layout and provenance

Each workload has one top-level subdirectory. Paths preserve the evaluation run,
experiment, and assistant-turn index:

```text
cuda_kernels/
  <workload>/
    <evaluation-run>/
      <experiment>/
        kernel_t<turn>.cu
```

For example:

```text
cuda_kernels/gemm_n7168_k5120/glm52-b200-gemm/exp_000/kernel_t0.cu
```

[`manifest.jsonl`](manifest.jsonl) records the workload, run, experiment, turn,
source architecture, relative source trajectory, prompt tag, trajectory exit
status, and SHA-256 for all extracted files.

Run-level correctness and architecture tables are copied into
[`turn_correctness_arch/`](turn_correctness_arch/). Files are keyed by evaluation
run to avoid duplicating a run-wide CSV under every workload. Its
`manifest.jsonl` records source paths and SHA-256 digests; `missing.txt` lists
represented runs for which AccRL has no `figures/turn_correctness_arch.csv`.

## Inventory

| Workload | Turns | Unique SHA-256 contents |
|---|---:|---:|
| `gemm_n7168_k5120` | 652 | 652 |
| `mha_bwd_d128` | 528 | 524 |
| `mha_bwd_d128_causal` | 570 | 567 |
| `mha_with_lse_d128` | 826 | 689 |
| `mha_with_lse_d128_causal` | 627 | 624 |
| `mha_with_lse_h48_d128` | 159 | 159 |
| **Total** | **3,362** | **3,215 per-workload unique entries** |

The extraction covers 435 trajectories. Another 74 selected Blackwell
trajectories contained no valid CUDA turn to collect.

## Stress driver

Start benchmark-server on the B200, then run a conservative one-minute test
with one kernel from each workload:

```bash
uv run python scripts/stress_gpu.py \
  --url http://127.0.0.1:8000 \
  --duration-seconds 60 \
  --concurrency 8
```

Use the entire content-deduplicated turn corpus for a longer run:

```bash
uv run python scripts/stress_gpu.py \
  --source-arch b200 \
  --kernels-per-workload 0 \
  --duration-seconds 3600 \
  --concurrency 16 \
  --rate 8
```

Pass `--keep-duplicates` to schedule all 3,362 turns. Because this corpus
intentionally contains failed kernels, pass `--allow-errors` when failures are
expected as part of the stress workload. `--prewarm` attempts to compile every
selected kernel before measurement and is therefore unsuitable for this full,
unfiltered corpus unless that cost is intentional.

The driver synthesizes deterministic tensors with the known workload shapes. It
tests compilation, execution, scheduling, backpressure, and timing; it does not
currently compare outputs against numerical references.
