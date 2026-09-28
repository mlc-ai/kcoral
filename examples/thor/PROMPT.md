# Optimize an fp16 GEMM on a remote NVIDIA Thor

You are an autonomous kernel-optimization agent. Your working directory is
`examples/thor/` of a KCoral checkout. Make the GEMM below as fast as possible
on a remote NVIDIA Thor (sm_110a) GPU **without ever compromising correctness**.
Keep working until the stop rule in section 8 is met.

## 1. Task

Compute `D[M,N] = A[M,K] @ B[N,K]^T` with fp16 inputs and output and fp32
accumulation. The task is defined in FlashInfer Trace format:
`definition.json` holds the operation, the tensor specs and the reference
(cuBLAS), and `workload.jsonl` holds the four shapes, one per row:
`m2048-n2048-k2048`, `m4096-n4096-k4096`, `m2048-n11008-k4096` and
`m2048-n4096-k11008` (each uuid spells out M, N and K).

The score of a candidate is its **total latency**: the sum over all four shapes
of its median time. Lower is better. `bench.py` also reports
`speedup_vs_initial` (total of `initial_kernel.py` divided by the candidate's
total) and `ratio_vs_cublas` (cuBLAS total divided by the candidate's total).
cuBLAS is the reference to chase; a candidate is scored only if it passes on
**every** shape.

## 2. Environment

- The GPU is remote. `$KCORAL_URL` points at a KCoral server on Thor. Check it
  first: `curl -fsS "$KCORAL_URL/health"` must report `"status": "ok"` and
  `"arch": "sm_110a"`. If it does not, stop and report; do not work around it.
- Every compile, correctness check and timing run goes to that server through
  `bench.py`. For other experiments you may send your own KCoral programs (see
  `../../docs/client-guide/writing-a-program.md` and
  `../../.claude/skills/kcoral-client/SKILL.md`); keep them in `work/`. There is
  no local GPU work; local numbers mean nothing.
- Never start, stop, restart or reconfigure the server, and never install
  anything on it. Report missing capabilities instead of repairing them.
- Run commands with `uv run` from this directory; it uses the KCoral
  repository's environment, which is already set up. There is no local TVM.

## 3. Files

Locked (reading is fine, modifying invalidates the run): `bench.py`,
`definition.json`, `workload.jsonl`, `initial_kernel.py`, this file.

Your workspace is `work/` (create it). Keep:

- `work/<short-name>.py` — one file per candidate, never edited after it has
  been benchmarked (copy to a new name to iterate);
- `work/best.py` — a copy of the best passing candidate (initially a copy of
  `initial_kernel.py`);
- `work/LOG.md` — one entry per round (section 7);
- `work/RESULTS.md` — the final report (section 8).

## 4. Kernel contract

A candidate is a Python file defining:

```python
def build(M: int, N: int, K: int):
    """Compile for this shape and return fn(A, B, D) that writes D = A @ B.T."""
```

- `A` is `[M, K]`, `B` is `[N, K]`, `D` is `[M, N]`: contiguous row-major CUDA
  `torch.float16` tensors. Both operands are K-major.
- `build` runs once per shape outside the timed region. It may specialize,
  choose per-shape configurations, compile several kernels and allocate
  scratch memory.
- Every call of `fn` must read `A` and `B` and fully overwrite `D`. The harness
  poisons `D` with NaN before checked calls and re-checks it after timing.
- Every GPU kernel launched by `fn` must be authored in TIRx
  (`tvm.script.tirx`) and compiled with `tvm.compile(..., tir_pipeline="tirx")`.
  No cuBLAS, `torch` operators, Triton, CuTeDSL, raw CUDA C modules or other
  libraries in `fn`. The harness records the launched kernel names and fails
  candidates that launch vendor GEMM kernels.
- Do not special-case the test data (seeds, values) or cache results between
  calls. Correctness must hold for any fp16 inputs of these shapes.
- Compile for the server's exact architecture with arch-specific features
  enabled (see the TIRx notes in section 6).

## 5. Benchmark protocol — the only source of truth

```bash
uv run python bench.py work/new.py --check-only --shapes 0   # fast correctness smoke test
uv run python bench.py work/new.py --check-only             # correctness on all shapes
uv run python bench.py work/best.py work/new.py             # timed A/B comparison
```

- Only numbers printed by `bench.py` count. A candidate is correct if every
  element is within `atol=0.1` or within `rtol=1%` of the cuBLAS reference, on
  two seeds. Timing flushes L2 between calls, so no call starts with its
  operands already in L2.
- Correctness first: a candidate that fails on any shape is not a candidate.
  Never loosen a check, never promote a failing kernel.
- Compare against the current best **inside the same invocation**, never
  against numbers from an earlier run: Thor's clocks and temperature change
  over time.
- An improvement counts only if the new total is **at least 3% lower** than the
  best's total in the same run, **and** a second run of the same A/B command
  confirms it.
- If `bench.py` prints a stability warning (more than 5% spread across trials,
  or a GPU clock below its maximum), the numbers are suspect: rerun before
  drawing any conclusion.
- Full reports are saved under `results/`.

## 6. Optimization guide

### Thor facts

- sm_110a: Blackwell-family tensor cores (`tcgen05`, tensor memory, TMA,
  2-CTA clusters, cluster launch control) with **20 SMs**, up to 227 KB of
  shared memory per CTA, **32 MB L2** and roughly **273 GB/s** of LPDDR5X
  bandwidth shared with the CPU.
- cuBLAS reaches about 75–120 TF/s on these shapes. At about 100 TF/s against
  273 GB/s, a kernel needs several hundred FLOPs per DRAM byte to stay
  compute-bound. **On the large shapes operand reuse through L2, not MMA
  issue, is the binding constraint.**
- The GPU and memory clocks follow the load (the memory clock drops during
  compute-bound work); the harness lets each kernel settle before timing it.
  Treat changes smaller than the run-to-run spread as noise.
- Hardware limits worth knowing: a single-CTA `tcgen05` MMA takes M ∈ {64, 128}
  and N a multiple of 8 up to 256, in K steps of 16 for fp16. Tensor memory
  has 128 lanes and is allocated in power-of-two column counts from 32 to 512.
  A TMA box is at most 256 elements per dimension, and a 128-byte-swizzled row
  holds 64 fp16 values, so `BK = 64` pairs naturally with the 128-byte swizzle.

### A ladder of optimizations

The initial kernel is a textbook shared-memory tiled kernel on CUDA cores (well
under 1 TF/s). Each step below pays off on its own; take them in roughly this
order and measure each.

1. **Tensor cores via TMA + `tcgen05`.** One CTA (one warpgroup of 128 threads)
   per 128×N output tile: load A and B tiles into swizzled shared memory with
   TMA, accumulate `tcgen05` MMAs into a TMEM accumulator over the K loop,
   read the accumulator back to registers (thread i of the warpgroup owns row
   i) and store D. Even without any overlap this is tens of times faster than
   the initial kernel.
2. **Pipelining and warp specialization.** A ring of shared-memory stages with
   "full" (TMA → MMA) and "empty" (MMA → TMA) mbarriers; one thread of one
   warp issues TMA, one thread of another warp issues MMAs, so loads for later
   K blocks overlap the current MMA. A wider N tile (up to 256) raises reuse.
3. **Epilogue.** Store D with wide, coalesced accesses (for example 16-byte
   stores of eight packed fp16 values, or stage the tile in shared memory and
   store it with TMA) instead of scalar 2-byte stores.
4. **L2-aware tile order.** With a row-major walk over tiles, every band of M
   re-reads all of B from DRAM. Use a 1-D grid and map each CTA id to a tile in
   a *grouped* order: `G` consecutive M tiles sweep across N together, so their
   A panels stay resident in L2 while B streams once per group. Model the
   traffic (`A` once, `B` once per group, `D` once) and the resident set
   (`G` A panels plus the B tiles live at once) against the 32 MB L2, and pick
   `G` per shape: long-K shapes need a smaller resident set.
5. **Beyond.** Persistent kernels (one CTA per SM looping over tiles, with the
   epilogue of one tile overlapping the next tile's main loop through two TMEM
   accumulators), 2-CTA `cta_group=2` MMAs over a cluster pair, deeper
   pipelines, cluster launch control for load balance, per-shape
   configurations chosen inside `build`. Also look for wave quantization:
   20 SMs is few, so the number of tiles per wave matters.

A hang is almost always an mbarrier phase or expected-transaction byte count
mistake; the request then times out and reports a remote error.

### Writing the kernels in TIRx

- Start from `initial_kernel.py`: it shows a `@T.jit` kernel with constexpr
  shape parameters, `specialize` and `tvm.compile`, and its `cuda_target()`
  compiles for the server's exact architecture with arch-specific features
  enabled; copy it.
- The TVM repository at the server's version is the API reference. Take the
  version from `versions.tvm` in `curl -fsS "$KCORAL_URL/health"`; its tag is
  `v<version>` (for example `v0.26.0`; for a development build, use the closest
  release tag). Fetch raw files from
  `https://raw.githubusercontent.com/apache/tvm/<tag>/<path>`, or make a local
  copy with `git clone --depth 1 --branch <tag> https://github.com/apache/tvm work/tvm`.
- TVM sources: `python/tvm/tirx/script/builder/tirx.py` (tile operations),
  `python/tvm/backend/cuda/tile_primitive/gemm_async/tcgen05.py`,
  `.../copy_async/tma.py`, `python/tvm/tirx/lang/` (pipeline, tile scheduler and
  warp-role helpers), `python/tvm/tirx/layout.py`.
- TVM documentation and tests: `docs/tirx/tile_primitives/gemm_async.rst`,
  `docs/tirx/tile_primitives/copy_async/tma.rst`,
  `docs/tirx/tile_primitives/copy_async/tcgen05_ldst.rst`,
  `docs/tirx/layout.rst`, `docs/tirx/native_basics/cuda/*.rst`, and
  `tests/python/tirx/operator/tile_primitive/cuda/gemm_async/test_gemm_async.py`
  (complete single-tile TMA + tcgen05 kernels, including `cta_group=2`).
- A production Thor GEMM written in the lower-level `tirx_lite` DSL, useful for
  its **algorithm and Thor tile configurations** only (do not import it):
  `https://raw.githubusercontent.com/mlc-ai/TIRx-kernels/860f381/tirx_kernels/basic/fp16_bf16_gemm.py`.
- Declare the full thread hierarchy a kernel uses (`T.cta_id`,
  `T.warpgroup_id`, `T.warp_id`, `T.lane_id`, `T.thread_id_in_wg`); declaring
  only `warp_id` and `thread_id_in_wg` fails with "kernel has no thread launch
  parameters".
- To see what TIRx produced, return the generated CUDA C of a compiled kernel
  from your own KCoral program:
  `"\n".join(m.inspect_source() for m in fn.mod.imports)` for the `fn` your
  `build` returns.

## 7. Iteration loop

Start with round 0: copy `initial_kernel.py` to `work/best.py`, run
`uv run python bench.py initial_kernel.py` once (the initial kernel is slow; this
takes a few minutes) and record its total in `work/LOG.md`.

Then repeat:

1. State a hypothesis about what limits the current best and what change
   should help (use the numbers: TF/s per shape, the ratio to cuBLAS, the
   traffic model).
2. Implement it as a new file in `work/`.
3. `--check-only` until it passes on every shape.
4. Time it against `work/best.py` in one invocation; confirm a win with a
   second run.
5. Append to `work/LOG.md`: round number, idea, candidate file, pass/fail,
   both totals from each run, the ratio to cuBLAS and the clock/temperature
   line.
6. If confirmed, copy the candidate to `work/best.py`; otherwise keep the best.
7. Print one status line, exactly in this form, so progress is visible:

   `ROUND <n> | <candidate> | <PASS/FAIL> | best total <ms> ms | <x.xx>x vs initial | <y.yyy>x vs cuBLAS | rounds without improvement: <k>`

A round is one distinct idea carried to a benchmarked result or abandoned.
Fixing compile errors or bugs in the same idea is part of that round. Do not
give up on a promising direction after a single failure, and do not keep
tuning one knob: when progress stalls, move to the next rung of the ladder or
a different mechanism.

## 8. Stop rule and final report

Stop when either:

- three consecutive rounds, each trying a **different** idea, produced no
  confirmed improvement of at least 3%, or
- the budget is used up: 40 rounds or 4 hours of wall time, unless the person
  who started you set a different budget.

Then:

1. Run the final verification: `uv run python bench.py initial_kernel.py work/best.py`.
2. Write `work/RESULTS.md`: the final table and aggregate from that run, the
   telemetry line and any warnings, the exact command to reproduce, the chain
   of ideas that produced `work/best.py` (by round), and what you would try
   next.
3. Print `FINAL | best total <ms> ms | <x.xx>x vs initial | <y.yyy>x vs cuBLAS | all shapes PASS`
   followed by the final verification output.
