"""Q4: requests per second one GPU serves, by concurrency.

Each of C client threads sends the 4 KiB vector_add job with cached inputs (the
same request as overhead.py) back to back: --warmup seconds, then --duration
seconds measured. Only requests that start inside the measured window count.
Before each concurrency level the script waits until every worker is ready.
Raw samples go to --out (JSON).
"""

import argparse
import json
import threading
import time

from common import kcoral_request, make_inputs, summary, wait_all_workers_ready

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
        "wall_ms": summary([s["wall_ms"] for s in measured]),
        "queue_ms": summary([s["queue_ms"] for s in measured]),
        "lease_wait_ms": summary([s["lease_wait_ms"] for s in measured]),
        "lease_held_ms": summary([s["lease_held_ms"] for s in measured]),
        "raw": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", default="1,2,4,8,16")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--out", required=True)
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
    with open(args.out, "w") as f:
        json.dump(
            {
                "server": {"workers": capacity, "target": health["target"]},
                "duration_s": args.duration,
                "warmup_s": args.warmup,
                "levels": levels,
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
