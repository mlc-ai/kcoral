"""Latency of the vector_add job per tensor size, as a KCoral request (cache hit
by default) or without KCoral. Each KCoral request waits until every worker is
ready, so the results exclude worker replacement.
"""

import argparse
import sys
import time

import numpy as np
from common import (
    PARTS,
    SOURCE,
    add_common_args,
    environment,
    kcoral_request,
    make_inputs,
    summary,
    wait_all_workers_ready,
    write_results,
)

from kcoral import Client


def local_job():
    """The job in this process: (a, b) -> (wall ms, None)."""
    import torch

    namespace = {}
    exec(SOURCE, namespace)
    vector_add = namespace["vector_add"]

    def run(a, b):
        t0 = time.perf_counter()
        c = vector_add(torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()).cpu().numpy()
        wall = (time.perf_counter() - t0) * 1e3
        assert np.allclose(c, a + b)
        return wall, None

    return run


def kcoral_job(client):
    """The job as a KCoral request: (a, b) -> (wall ms, timing)."""
    capacity = client.health()["load"]["request_capacity"]

    def run(a, b):
        wait_all_workers_ready(client, capacity)
        return kcoral_request(client, a, b)

    return run


def measure(job, cache_miss, n, trials, warmup):
    """One record: warmup untimed and trials timed runs, n elements per tensor."""
    a, b = make_inputs(n, seed=0)
    if not cache_miss:
        job(a, b)  # with KCoral, uploads the inputs so later requests hit the cache

    def once():
        return job(*make_inputs(n)) if cache_miss else job(a, b)

    for _ in range(warmup):
        once()
    samples = [once() for _ in range(trials)]
    wall = [ms for ms, _ in samples]
    print(f"n={n}: median {np.median(wall):.2f} ms", file=sys.stderr)
    record = {"n": n, "e2e_ms": summary(wall)}
    raw = {"e2e_ms": wall}
    timing = [t for _, t in samples]
    if timing[0] is not None:
        record["timing_ms"] = {k: summary([t[k] for t in timing]) for k, _ in PARTS}
        raw["timing_ms"] = timing
    return {**record, "raw": raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="unused with --no-kcoral")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--force-cache-miss",
        action="store_true",
        help="send new inputs with every request, so the client uploads them after CACHE_MISS",
    )
    mode.add_argument(
        "--no-kcoral",
        action="store_true",
        help="run the job in this process, without KCoral (needs the GPU: CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--sizes",
        default="1024,1048576,16777216",
        help="float32 elements per tensor, comma-separated",
    )
    parser.add_argument("--trials", type=int, default=50, help="timed runs per size")
    parser.add_argument("--warmup", type=int, default=5, help="untimed runs per size")
    add_common_args(parser)
    args = parser.parse_args()
    sizes = [int(n) for n in args.sizes.split(",")]

    def run_all(job):
        return [measure(job, args.force_cache_miss, n, args.trials, args.warmup) for n in sizes]

    if args.no_kcoral:
        import torch

        env = {**environment(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
        records = run_all(local_job())
    else:
        with Client(args.url) as client:
            env = environment(client.health())
            records = run_all(kcoral_job(client))
    params = {
        "kcoral": not args.no_kcoral,
        "cache": None if args.no_kcoral else "miss" if args.force_cache_miss else "hit",
        "trials": args.trials,
        "warmup": args.warmup,
    }
    write_results(args, "latency", env, params, records)


if __name__ == "__main__":
    main()
