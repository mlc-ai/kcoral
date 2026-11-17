"""Shared by the benchmarks: the job, one KCoral request for it, and summaries."""

import platform
import statistics
import time

import numpy as np

from kcoral import Program

# The job. Local runs and KCoral requests execute this same source.
SOURCE = """
def vector_add(a, b):
    return a + b
"""


def make_inputs(n, seed):
    """Two float32 vectors of n elements; a new seed gives new tensor contents."""
    rng = np.random.default_rng(seed)
    return rng.random(n, dtype=np.float32), rng.random(n, dtype=np.float32)


def kcoral_request(client, a, b):
    """Run the job as one KCoral request. Returns (wall ms, timing breakdown in ms)."""
    t0 = time.perf_counter()
    program = Program()
    module = program.upload(kind="module", source=SOURCE)
    fn = program.get_function(module=module, name="vector_add")
    inputs = [program.upload(kind="tensor", value=a), program.upload(kind="tensor", value=b)]
    program.return_(key="c", value=program.run(fn=fn, args=inputs))
    build = (time.perf_counter() - t0) * 1e3  # includes hashing the tensor blobs
    result = client.execute(program, timeout_seconds=60)
    wall = (time.perf_counter() - t0) * 1e3
    if not result.completed:
        raise SystemExit(f"KCoral request failed: {result.error}")
    assert np.allclose(result.results["c"], a + b)
    breakdown = {"client_build_ms": build}
    for key in ("queue_ms", "elapsed_ms", "lease_wait_ms", "lease_held_ms"):
        breakdown[key] = getattr(result, key)
    return wall, breakdown


def wait_all_workers_ready(client, capacity):
    """Block until no worker is being replaced, so the next request gets a ready worker."""
    while client.health()["load"]["request_capacity"] < capacity:
        time.sleep(0.02)


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


def environment(health):
    """The client's GPU and software, plus what the server reports about itself."""
    import torch

    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "server_target": health["target"],
        "server_versions": health["versions"],
        "server_workers": health["load"]["request_capacity"],
    }
