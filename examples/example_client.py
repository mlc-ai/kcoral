"""Send one benchmark request to a running server using the bundled client.

    python examples/example_client.py [PORT]      # default 8000

It uploads an input tensor, a TIRx kernel, and a torch reference, compiles the
kernel, runs it on the uploaded tensor, asserts the result matches the
reference (``assert_close`` fails the request if they differ), and benchmarks
it, then prints the reply. Content keys and the ``CACHE_MISS`` retry are
handled by the client.
"""

import array
import sys

from benchmark_server.client import Client, ref, run, upload_function, upload_tensor_bytes

KERNEL = """from __future__ import annotations
from tvm.script import tirx as T

@T.jit
def main(A: T.Buffer((N,), "float32"), B: T.Buffer((N,), "float32"), *, N: T.constexpr):
    T.device_entry()
    i = T.cta_id([N])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0
"""
REF = "def main(a):\n    return a + 1.0\n"


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "8000"
    raw = array.array("f", (float(i) for i in range(256))).tobytes()
    with Client(f"http://127.0.0.1:{port}") as client:
        outcome = client.execute(
            [
                upload_function("kernel", KERNEL),
                upload_function("reffn", REF),
                upload_tensor_bytes("a", "float32", [256], raw),
                run("out", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
                run("mod", "builtin.compile_tirx", [ref("kernel"), {"N": 256}]),
                run("krun", ref("mod"), [ref("a"), ref("out")]),
                run("refv", ref("reffn"), [ref("a")]),
                run("chk", "builtin.assert_close", [ref("out"), ref("refv")]),
                run(
                    "perf",
                    "builtin.benchmark",
                    [ref("mod"), ref("a"), ref("out"), {"warmup": 10, "repeat": 50}],
                ),
            ],
            timeout_seconds=120,
        )
    print("status:", outcome.status, " request:", outcome.request_id)
    for result in outcome.results:
        line = f"  {result.id:>6}  {result.status}"
        if result.value is not None:
            line += f"  {result.value}"
        if result.error is not None:
            line += f"  {result.error}"
        print(line)


if __name__ == "__main__":
    main()
