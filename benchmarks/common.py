"""Shared by the benchmarks: the job, one KCoral request for it, the results format."""

import json
import platform
import statistics
import time

import numpy as np

import kcoral
from kcoral import Program

# The job. Local runs and KCoral requests execute this same source.
SOURCE = """
def vector_add(a, b):
    return a + b
"""

SERVER_FIELDS = ("queue_ms", "elapsed_ms", "lease_wait_ms", "lease_held_ms")

# The parts of a request's wall time, in request order: (timing key, label).
PARTS = (
    ("client_build_ms", "client: build program + hash"),
    ("worker_handoff_ms", "server: worker handoff"),
    ("lease_held_ms", "server: GPU lease held"),
    ("rest_ms", "HTTP + request handling"),
)


def make_inputs(n, seed=None):
    """Two float32 vectors of n elements. Without a seed, the contents are new
    every call, so the server has never cached them."""
    rng = np.random.default_rng(seed)
    return rng.random(n, dtype=np.float32), rng.random(n, dtype=np.float32)


def kcoral_request(client, a, b):
    """Run the job as one KCoral request. Returns (wall ms, {SERVER_FIELDS and PARTS: ms})."""
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
    timing = {k: getattr(result, k) for k in SERVER_FIELDS}
    timing.update(
        client_build_ms=build,
        worker_handoff_ms=result.elapsed_ms - result.lease_held_ms,
        rest_ms=wall - build - result.elapsed_ms,
    )
    return wall, timing


def wait_all_workers_ready(client, capacity):
    """Block until no worker is being replaced, so the next request gets a ready worker."""
    while client.health()["load"]["request_capacity"] < capacity:
        time.sleep(0.02)


def summary(xs):
    q = statistics.quantiles(xs, n=20, method="inclusive")  # q[0] = p5, q[18] = p95
    return {
        "n": len(xs),
        "p5": q[0],
        "median": statistics.median(xs),
        "p95": q[18],
    }


def environment(health=None):
    """The client's software, plus what the server reports about itself."""
    env = {"kcoral": kcoral.__version__, "python": platform.python_version()}
    if health is not None:
        env.update(
            server_target=health["target"],
            server_versions=health["versions"],
            server_workers=health["load"]["request_capacity"],
        )
    return env


def add_common_args(parser):
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="label recorded with the results, e.g. worker_type=reused. Repeatable.",
    )
    parser.add_argument("--out", required=True, help="JSON results file")


def write_results(args, benchmark, env, params, records):
    tags = dict(tag.split("=", 1) for tag in args.tag)
    with open(args.out, "w") as f:
        json.dump(
            {
                "benchmark": benchmark,
                "tags": tags,
                "env": env,
                "params": params,
                "records": records,
            },
            f,
            indent=1,
        )
