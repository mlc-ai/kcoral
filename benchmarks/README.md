# KCoral benchmarks

Latency and throughput of a KCoral server for a minimal job (`c = a + b`,
PyTorch, float32), compared with running the same job without KCoral. Use them
to track KCoral's per-request overhead across commits and machines.

- `latency.py`, `throughput.py`: the benchmarks; each run writes one JSON results file
- `summary.py`: merges results files into `tables.md` and figures
- `run.sh`: starts a local server per configuration, runs the benchmarks, summarizes

Each script's `--help` lists its options.

## Run

Requires an NVIDIA GPU, [uv](https://docs.astral.sh/uv/) and `curl`. From the
repository root:

```bash
benchmarks/run.sh --gpu-id 1                                         # everything on GPU 1
benchmarks/run.sh --gpu-id 1 --benchmark latency --force-cache-miss  # then add cache misses
LATENCY_ARGS="--trials 5" THROUGHPUT_ARGS="--duration 5" benchmarks/run.sh --gpu-id 1  # quick check
```

Results go to `benchmarks/results/NAME/` (default NAME: the short commit), with
server logs and `env.txt` (GPU, driver, CPU, OS, commit, sandbox mode).

To compare commits or machines, summarize several results directories together:

```bash
git checkout main && benchmarks/run.sh --gpu-id 1 --name main
git checkout my-branch && benchmarks/run.sh --gpu-id 1 --name my-branch
uv run --group server --with matplotlib==3.10.7 python benchmarks/summary.py \
  benchmarks/results/main benchmarks/results/my-branch -o benchmarks/results/compare
```

## Request breakdown

`latency.py` splits each KCoral request's wall time into four parts:

- **client: build program + hash:** the client's time before `execute`.
- **server: worker handoff:** passing the program and its inputs to the worker
  process and the result back, plus workspace setup (`elapsed_ms - lease_held_ms`).
- **server: GPU lease held:** `lease_held_ms`: running the program plus
  per-request cleanup. With fresh workers this includes waiting for the previous
  worker process on the GPU to exit.
- **HTTP + request handling:** the rest: HTTP transfer, the server's handling
  outside the worker and the client's response decoding.
