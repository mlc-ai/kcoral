"""Q1-Q3: the vector_add job run locally vs. through KCoral, for one tensor size.

Each mode runs in its own loop: --warmup untimed runs, then --trials timed runs.

  local          numpy -> GPU, a + b, GPU -> numpy in this process (warm CUDA context)
  kcoral_cached  the job as a KCoral request whose inputs the server has cached
  kcoral_new     the same request with new tensor contents (CACHE_MISS, then upload)

Before each KCoral request the script waits until every worker is ready, so the
results exclude waiting for worker replacement. Raw samples go to --out (JSON).
"""

import argparse
import json
import sys
import time

import numpy as np
from common import (
    SOURCE,
    environment,
    kcoral_request,
    make_inputs,
    summary,
    wait_all_workers_ready,
)

from kcoral import Client


def make_local_job():
    """The job in this process. Returns a function that runs it once and returns wall ms."""
    import torch

    namespace = {}
    exec(SOURCE, namespace)
    vector_add = namespace["vector_add"]

    def run(a, b):
        t0 = time.perf_counter()
        c = vector_add(torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()).cpu().numpy()
        wall = (time.perf_counter() - t0) * 1e3
        assert np.allclose(c, a + b)
        return wall

    return run


def run_modes(client, n, trials, warmup):
    """Run each mode in its own loop. Returns the first request and per-mode samples."""
    capacity = client.health()["load"]["request_capacity"]
    a, b = make_inputs(n, seed=0)

    def kcoral_job(a, b):
        wait_all_workers_ready(client, capacity)
        return kcoral_request(client, a, b)

    # The first request after server start uploads a and b, so later cached requests hit.
    first_wall, first_breakdown = kcoral_job(a, b)
    local_job = make_local_job()
    seeds = iter(range(1, warmup + trials + 1))  # a new input pair for every kcoral_new run
    modes = {
        "local": lambda: (local_job(a, b), None),
        "kcoral_cached": lambda: kcoral_job(a, b),
        "kcoral_new": lambda: kcoral_job(*make_inputs(n, seed=next(seeds))),
    }
    wall, breakdown = {}, {}
    for mode, run in modes.items():
        for _ in range(warmup):
            run()
        samples = [run() for _ in range(trials)]
        wall[mode] = [ms for ms, _ in samples]
        if mode != "local":
            breakdown[mode] = [parts for _, parts in samples]
        print(f"{mode}: done", file=sys.stderr)
    return {"wall_ms": first_wall, **first_breakdown}, wall, breakdown


def print_summary(report):
    print(f"{report['env']['bytes_per_tensor'] / 2**20:.3g} MiB per tensor, {report['env']['gpu']}")
    print(f"first request after server start: {report['first_request']['wall_ms']:.1f} ms")
    for mode, s in report["e2e_ms"].items():
        print(f"{mode:14s} ms: min {s['min']:9.2f}  median {s['median']:9.2f}  max {s['max']:9.2f}")
    for mode, fields in report["breakdown_ms"].items():
        medians = ", ".join(f"{k}={s['median']:.2f}" for k, s in fields.items())
        print(f"{mode} breakdown medians (ms): {medians}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=1 << 20, help="elements per float32 vector")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with Client(args.url) as client:
        env = environment(client.health())
        first, wall, breakdown = run_modes(client, args.n, args.trials, args.warmup)

    report = {
        "env": {**env, "n": args.n, "bytes_per_tensor": args.n * 4},
        "first_request": first,
        "e2e_ms": {mode: summary(v) for mode, v in wall.items()},
        "breakdown_ms": {
            mode: {k: summary([s[k] for s in v]) for k in v[0]} for mode, v in breakdown.items()
        },
        "raw": {"e2e_ms": wall, "breakdown_ms": breakdown},
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print_summary(report)


if __name__ == "__main__":
    main()
