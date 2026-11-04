# KCoral overhead profile

Measures what running a job through a KCoral server costs compared with running
it directly, using a minimal kernel (`c = a + b`, PyTorch, float32). The questions,
results and conclusions are in [REPORT.md](REPORT.md).

## Files

| File | Report section | What it does |
|---|---|---|
| `benchmark/overhead.py` | Q1–Q3 | Local vs KCoral for one tensor size: end-to-end time with cached and new inputs, and a request breakdown |
| `benchmark/throughput.py` | Q4, Discussion: fresh workers | Requests per second for 1–16 concurrent clients |
| `benchmark/pipe_cost.py` | Discussion: worker handoff | Moving tensors through a bare `multiprocessing` pipe, as the server does with its worker |
| `benchmark/reuse_state.py` | Discussion: reused workers | What one reused worker carries from one request to the next |
| `benchmark/common.py` | — | Shared by the benchmarks: the job, one KCoral request, summaries |
| `report.py` | Q1–Q4, Discussion: fresh workers, worker handoff | Figures (`figures/`, one function per section) and tables (`out/tables.md`) from `out/*.json` and the server event logs |
| `run.sh` | all | Full reproduction: environment, servers, every experiment, figures |

## Reproduce

Requires an NVIDIA GPU, [uv](https://docs.astral.sh/uv/) and `curl`. From the repository root:

```bash
examples/profile/run.sh -g 1                        # all steps on GPU 1 (~35 min)
examples/profile/run.sh -g 1 throughput figures     # only some steps
```

Steps: `overhead` (Q1–Q3, ~20 min), `throughput` (Q4, ~15 min), `discussion` (~1 min)
and `figures`. `run.sh` installs the locked environment (`uv sync --group server`)
and starts a local server for each configuration with server defaults, except
`--max-requests-per-worker` (1 = fresh, 0 = reused) and a private disk cache.
It stops each server when its step finishes.

Results are JSON files in `out/` (not committed): raw samples plus min / p5 /
median / p95 / max / mean summaries. `out/env.txt` records the GPU, driver, CPU,
OS, commit and each server's sandbox mode. `report.py` regenerates `figures/`
and `out/tables.md` from them.
