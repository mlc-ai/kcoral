"""Compare a vector_add job run locally against the same job run through KCoral.

Measured (each repeated --trials times, interleaved so drift affects all modes alike):

  e2e local         in-process: numpy -> GPU, a + b, GPU -> numpy (warm CUDA context)
  e2e local_proc    the same job in a fresh `python` process (import torch + CUDA init)
  e2e kcoral_cached client.execute with tensors already in the server blob cache
  e2e kcoral_new    client.execute with new tensor contents (CACHE_MISS, then upload)
  kernel local/kcoral   GPU time of a + b from kcoral.builtins.benchmark (CUPTI)

Before each KCoral request the script waits until every worker is ready, so
results exclude waiting for worker replacement. Raw samples go to --out (JSON).
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

from kcoral import Client, Program

# Shared by the local and remote runs, so both execute identical code.
SOURCE = """
import torch

def vector_add(a, b):
    return a + b

def kernel_latency(n):
    from kcoral.builtins import benchmark
    a = torch.rand(n, device="cuda")
    b = torch.rand(n, device="cuda")
    return benchmark(vector_add, a, b, {"warmup": 100, "repeat": 1000})
"""

# The local_proc job: a new interpreter that runs the same steps as `local`.
PROC_JOB = """
import sys, numpy as np, torch
ns = {}
exec(sys.argv[1], ns)
n = int(sys.argv[2])
a = np.random.default_rng(0).random(n, dtype=np.float32)
b = np.random.default_rng(1).random(n, dtype=np.float32)
c = ns["vector_add"](torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()).cpu().numpy()
assert np.allclose(c, a + b)
"""


def summary(xs):
    q = statistics.quantiles(xs, n=20, method="inclusive")  # q[0] = p5, q[18] = p95
    return {
        "n": len(xs),
        "min": min(xs),
        "p5": q[0],
        "median": statistics.median(xs),
        "p95": q[18],
        "max": max(xs),
        "mean": statistics.mean(xs),
    }


def wait_all_workers_ready(client, capacity):
    while client.health()["load"]["request_capacity"] < capacity:
        time.sleep(0.02)


def kcoral_job(client, a, b):
    """One vector_add request. Returns (wall ms, timing breakdown in ms)."""
    t0 = time.perf_counter()
    program = Program()
    module = program.upload(kind="module", source=SOURCE)
    fn = program.get_function(module=module, name="vector_add")
    out = program.run(
        fn=fn, args=[program.upload(kind="tensor", value=a), program.upload(kind="tensor", value=b)]
    )
    program.return_(key="c", value=out)
    build = (time.perf_counter() - t0) * 1e3  # includes hashing the tensor blobs
    result = client.execute(program, timeout_seconds=60)
    wall = (time.perf_counter() - t0) * 1e3
    if not result.completed:
        raise SystemExit(f"KCoral request failed: {result.error}")
    assert np.allclose(result.results["c"], a + b)
    breakdown = {"client_build_ms": build}
    breakdown.update(
        {
            k: getattr(result, k)
            for k in ("queue_ms", "elapsed_ms", "lease_wait_ms", "lease_held_ms")
        }
    )
    return wall, breakdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=1 << 20, help="elements per float32 vector")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--kernel-trials", type=int, default=30)
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()

    import torch

    ns = {}
    exec(SOURCE, ns)
    vector_add, kernel_latency = ns["vector_add"], ns["kernel_latency"]
    rng = np.random.default_rng(0)
    a, b = rng.random(args.n, dtype=np.float32), rng.random(args.n, dtype=np.float32)

    def local_job():
        t0 = time.perf_counter()
        c = vector_add(torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()).cpu().numpy()
        wall = (time.perf_counter() - t0) * 1e3
        assert np.allclose(c, a + b)
        return wall

    def local_proc_job():
        t0 = time.perf_counter()
        subprocess.run([sys.executable, "-c", PROC_JOB, SOURCE, str(args.n)], check=True)
        return (time.perf_counter() - t0) * 1e3

    e2e = {m: [] for m in ("local", "local_proc", "kcoral_cached", "kcoral_new")}
    breakdown = {m: [] for m in ("kcoral_cached", "kcoral_new")}
    kernel = {"local": [], "kcoral": []}

    with Client(args.url) as client:
        health = client.health()
        capacity = health["load"]["request_capacity"]
        # The first request of the run uploads `a` and `b`; later cached-input trials hit it.
        wait_all_workers_ready(client, capacity)
        first_wall, first_breakdown = kcoral_job(client, a, b)
        local_job()  # warm the local CUDA context and caching allocator

        for i in range(args.trials):
            e2e["local"].append(local_job())
            e2e["local_proc"].append(local_proc_job())
            wait_all_workers_ready(client, capacity)
            wall, s = kcoral_job(client, a, b)
            e2e["kcoral_cached"].append(wall)
            breakdown["kcoral_cached"].append(s)
            fresh_a = np.random.default_rng(1000 + i).random(args.n, dtype=np.float32)
            fresh_b = np.random.default_rng(2000 + i).random(args.n, dtype=np.float32)
            wait_all_workers_ready(client, capacity)
            wall, s = kcoral_job(client, fresh_a, fresh_b)
            e2e["kcoral_new"].append(wall)
            breakdown["kcoral_new"].append(s)
            print(f"e2e trial {i + 1}/{args.trials}", file=sys.stderr, end="\r")

        program = Program()
        module = program.upload(kind="module", source=SOURCE)
        fn = program.get_function(module=module, name="kernel_latency")
        program.return_(key="r", value=program.run(fn=fn, args=[args.n]))
        for i in range(args.kernel_trials):
            kernel["local"].append(kernel_latency(args.n)["latency_ms_median"] * 1e3)
            wait_all_workers_ready(client, capacity)
            result = client.execute(program, timeout_seconds=60)
            if not result.completed:
                raise SystemExit(f"KCoral request failed: {result.error}")
            kernel["kcoral"].append(result.results["r"]["latency_ms_median"] * 1e3)
            print(f"kernel trial {i + 1}/{args.kernel_trials}", file=sys.stderr, end="\r")
    print(file=sys.stderr)

    report = {
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "server_target": health["target"],
            "server_versions": health["versions"],
            "server_workers": capacity,
            "n": args.n,
            "bytes_per_tensor": args.n * 4,
        },
        "first_request": {"wall_ms": first_wall, **first_breakdown},
        "e2e_ms": {m: summary(v) for m, v in e2e.items()},
        "breakdown_ms": {
            m: {k: summary([s[k] for s in v]) for k in v[0]} for m, v in breakdown.items()
        },
        "kernel_us": {m: summary(v) for m, v in kernel.items()},
        "raw": {"e2e_ms": e2e, "breakdown_ms": breakdown, "kernel_us": kernel},
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    print(f"n={args.n} ({args.n * 4 / 2**20:.3g} MiB/tensor), {report['env']['gpu']}")
    print(f"first request after server start: {first_wall:.1f} ms")
    row = "{:<22}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}"
    print(row.format("", "min", "p5", "median", "p95", "max", "mean"))
    for section, unit in (("e2e_ms", "ms"), ("kernel_us", "us")):
        for mode, s in report[section].items():
            print(
                row.format(
                    f"{section.split('_')[0]} {mode} ({unit})",
                    *(f"{s[k]:.3f}" for k in ("min", "p5", "median", "p95", "max", "mean")),
                )
            )
    for mode, fields in report["breakdown_ms"].items():
        print(
            f"{mode} breakdown medians (ms): "
            + ", ".join(f"{k}={s['median']:.2f}" for k, s in fields.items())
        )


if __name__ == "__main__":
    main()
