"""Discussion: what moving tensors between the server and a worker process costs.

The server passes a program to its worker, input tensor bytes included, over a
multiprocessing pipe (spawn context, as in kcoral.runtime.worker), and the
worker sends the result back the same way. This times that pipe alone, with no
KCoral or GPU, for the Q1/Q2 shapes: two inputs in, one output of the same size out.

  inputs      send the two inputs, receive a small acknowledgement
  round_trip  send the two inputs, receive one output of the same size
"""

import argparse
import json
import multiprocessing as mp
import os
import time

from common import summary


def child(conn):
    while (message := conn.recv()) is not None:
        inputs, reply = message
        conn.send(inputs[0] if reply else b"ok")


def run(conn, inputs, reply):
    t0 = time.perf_counter()
    conn.send((inputs, reply))
    conn.recv()
    return (time.perf_counter() - t0) * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1024,1048576,16777216", help="float32 elements")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=child, args=(child_conn,))
    proc.start()
    report = {}
    for n in map(int, args.sizes.split(",")):
        inputs = [os.urandom(n * 4), os.urandom(n * 4)]
        run(conn, inputs, reply=True)  # warm up
        report[n] = {
            f"{mode}_ms": summary([run(conn, inputs, reply) for _ in range(args.trials)])
            for mode, reply in (("inputs", False), ("round_trip", True))
        }
        print(
            f"{n * 4 / 2**20:8.3f} MiB per tensor: inputs "
            f"{report[n]['inputs_ms']['median']:7.2f} ms, round trip "
            f"{report[n]['round_trip_ms']['median']:7.2f} ms (medians)"
        )
    conn.send(None)
    proc.join()
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
