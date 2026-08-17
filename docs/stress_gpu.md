# AccRL GPU corpus stress test

`scripts/stress_gpu.py` exercises benchmark-server with the checked-in AccRL
CUDA and Triton corpora on H100 or B200. Every kernel has a `source_arch` of
`h100` or `b200`; every result also records the server's `target_arch` and
`target_sm`. Native and cross-architecture results are therefore never merged
silently.

List both source architectures without contacting a server:

```bash
uv run python scripts/stress_gpu.py --list
```

Run a bounded native-architecture smoke test. With no `--source-arch`, the
driver selects the architecture matching the server (`sm_90a` -> `h100`,
`sm_100a` -> `b200`):

```bash
uv run python scripts/stress_gpu.py \
  --url http://127.0.0.1:8000 \
  --requests 20 \
  --concurrency 8 \
  --warmup 1 \
  --repeat 5
```

Use every content-distinct B200 kernel, or request an explicitly labelled
cross-architecture Triton run:

```bash
uv run python scripts/stress_gpu.py --source-arch b200 --kernels-per-workload 0 --requests 6000
uv run python scripts/stress_gpu.py --language triton --source-arch h100 --requests 100
uv run python scripts/stress_gpu.py --language cuda --source-arch b200 \
  --workload gemm_n7168_k5120
```

Repeat `--language`, `--source-arch`, or `--workload` to select several values.
Byte-identical sources are deduplicated within each
language/source-architecture/workload cohort by default; `--keep-duplicates`
retains every corpus file. Bounded selections prefer CUDA turns marked
`Correct` by AccRL and Triton sources recorded under `success/`. An unlimited
selection includes successful, failed, and intermediate attempts.

CUDA source is uploaded as `language="cuda"` and passed through
`builtin.compile_cuda`. Triton files contain their own destination-passing
`run(...)` wrapper, so they are uploaded as Python modules and benchmarked
directly. Each request record includes `language`, `source_arch`, `target_arch`,
and `target_sm`; the summary groups requests by the same dimensions.

The checked-in CUDA corpus currently contains the B200 cohort only. AccRL-exps
also has more than 60,000 H100 CUDA turns, so the extractor does not import that
much larger cohort unless `--source-arch h100` is explicitly supplied. The
Triton corpus contains both architectures. On an H100, the default run therefore
uses the available H100 Triton cohort; explicitly selecting absent H100 CUDA
reports an error instead of relabelling B200 source.

JSONL output goes to `stress-results/` by default. Use `--allow-errors` when the
goal is to finish a corpus sweep that intentionally includes unsuccessful model
turns. Targets other than H100 and B200 require `--allow-unsupported-target` and
an explicit `--source-arch`.

## Refreshing the corpora

The CUDA extractor recognizes `NVCC_GENCODE` first and prompt tags second. It
defaults to B200 to keep accidental refreshes bounded; repeat `--source-arch` to
request both cohorts deliberately:

```bash
uv run python scripts/extract_accrl_cuda_kernels.py --source-arch b200 --force
uv run python scripts/extract_accrl_cuda_kernels.py \
  --source-arch b200 --source-arch h100 --output /large/cuda-kernels
```

The Triton extractor searches all nested run directories under
`/home/yixind/AccRL-exps/eval_runs`, rather than relying on run names. It retains
every assistant-turn Python candidate, every `success/kernel_v*.py`, and any
smoke-run workspace `kernel.py`. `TRITON_GPU_ARCH` and prompt tags are normalized
to `h100` or `b200` in the manifest:

```bash
uv run python scripts/extract_accrl_triton_kernels.py --force
```

The current Triton corpus contains 3,296 files from 30 runs and 305
trajectories: 2,412 turn candidates, 880 successful versions, and four workspace
versions. Its 783 B200-authored and 2,513 H100-authored files remain separate in
selection and reporting even when both are run against one target.

Any future change to corpus discovery or program construction should keep the
architecture-aware regression tests passing:

```bash
uv run pytest -q \
  tests/test_stress_gpu.py \
  tests/test_extract_accrl_cuda_kernels.py \
  tests/test_extract_accrl_triton_kernels.py
```
