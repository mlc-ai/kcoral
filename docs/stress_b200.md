# B200 AccRL corpus stress test

`scripts/stress_b200.py` exercises benchmark-server with the checked-in AccRL
CUDA and Triton corpora. It selects one distinct source per language and
workload by default and interleaves the two languages deterministically. A
workload that has no corpus for one language is still exercised by the other.

List the default selection without contacting a server:

```bash
uv run python scripts/stress_b200.py --list
```

Run a bounded mixed-language smoke test against a B200 server:

```bash
uv run python scripts/stress_b200.py \
  --url http://127.0.0.1:8000 \
  --requests 20 \
  --concurrency 8 \
  --warmup 1 \
  --repeat 5
```

Use every content-distinct kernel, or select one language explicitly:

```bash
uv run python scripts/stress_b200.py --kernels-per-workload 0 --requests 6000
uv run python scripts/stress_b200.py --language triton --requests 100
uv run python scripts/stress_b200.py --language cuda --workload gemm_n7168_k5120
```

Repeat `--language` or `--workload` to select several values. Byte-identical
sources are deduplicated within a language/workload by default;
`--keep-duplicates` retains every corpus file. Bounded Triton selections prefer
sources recorded under `success/`, while an unlimited selection includes
successful, failed, and intermediate attempts.

CUDA source is uploaded as `language="cuda"` and passed through
`builtin.compile_cuda`. Triton files contain their own destination-passing
`run(...)` wrapper, so they are uploaded as Python modules and benchmarked
directly. Each request record and the final summary include the kernel language.

The driver rejects a non-`sm_100a` target unless `--allow-non-b200` is passed.
JSONL output goes to `stress-results/` by default. Use `--allow-errors` when the
goal is to finish a corpus sweep that intentionally includes unsuccessful model
turns.

## Refreshing the corpora

The CUDA corpus is generated from Blackwell trajectories:

```bash
uv run python scripts/extract_accrl_blackwell_kernels.py --force
```

The Triton extractor searches all nested run directories under
`/home/yixind/AccRL-exps/eval_runs`, rather than relying on run names. It retains
every assistant-turn Python candidate, every `success/kernel_v*.py`, and any
smoke-run workspace `kernel.py`. The manifest records source kind, prompt tag,
source architecture, exit/evaluation status, and SHA-256:

```bash
uv run python scripts/extract_accrl_triton_kernels.py --force
```

The current Triton corpus contains 3,296 files from 30 runs and 305
trajectories: 2,412 turn candidates, 880 successful versions, and four workspace
versions. They cover five supported workloads and both Hopper- and
Blackwell-authored sources; all are run against the B200 target.

Any future change to corpus discovery or program construction should keep the
mixed-language regression tests passing:

```bash
uv run pytest -q \
  tests/test_stress_b200.py \
  tests/test_extract_accrl_blackwell_kernels.py \
  tests/test_extract_accrl_triton_kernels.py
```
