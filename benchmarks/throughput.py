"""Requests per second one server serves, by number of concurrent clients. Each
client sends the 4 KiB vector_add job (cache hit) back to back; only requests
that start after the warmup count.
"""

import argparse
import threading
import time

from common import (
    SERVER_FIELDS,
    add_common_args,
    environment,
    kcoral_request,
    make_inputs,
    summary,
    wait_all_workers_ready,
    write_results,
)

from kcoral import Client


def run_level(url, concurrency, duration, warmup, a, b):
    start = time.perf_counter()
    deadline = start + warmup + duration
    samples, lock = [], threading.Lock()

    def client_loop():
        with Client(url) as client:
            while (sent := time.perf_counter()) < deadline:
                wall, timing = kcoral_request(client, a, b)
                with lock:
                    samples.append({"start_s": sent - start, "wall_ms": wall, **timing})

    threads = [threading.Thread(target=client_loop) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    measured = [s for s in samples if s["start_s"] >= warmup]
    return {
        "concurrency": concurrency,
        "requests_per_s": len(measured) / duration,
        **{k: summary([s[k] for s in measured]) for k in ("wall_ms", *SERVER_FIELDS)},
        "raw": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--concurrency", default="1,2,4,8,16", help="client counts, comma-separated"
    )
    parser.add_argument("--duration", type=float, default=30.0, help="measured seconds per level")
    parser.add_argument("--warmup", type=float, default=5.0, help="unmeasured seconds per level")
    add_common_args(parser)
    args = parser.parse_args()

    a, b = make_inputs(1024, seed=0)
    levels = []
    with Client(args.url) as client:
        health = client.health()
        capacity = health["load"]["request_capacity"]
        wait_all_workers_ready(client, capacity)
        # Upload the inputs once, so every measured request hits the cache.
        kcoral_request(client, a, b)
        for concurrency in map(int, args.concurrency.split(",")):
            wait_all_workers_ready(client, capacity)
            level = run_level(args.url, concurrency, args.duration, args.warmup, a, b)
            levels.append(level)
            print(
                f"C={concurrency:2d}: {level['requests_per_s']:6.2f} req/s, "
                f"wall median {level['wall_ms']['median']:7.1f} ms, "
                f"queue median {level['queue_ms']['median']:7.1f} ms, "
                f"lease_wait median {level['lease_wait_ms']['median']:6.1f} ms"
            )
    params = {"duration_s": args.duration, "warmup_s": args.warmup}
    write_results(args, "throughput", environment(health), params, levels)


if __name__ == "__main__":
    main()
