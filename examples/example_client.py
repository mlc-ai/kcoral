"""Submit a complete TIRx benchmark program with the high-level client."""

from __future__ import annotations

import numpy as np

from benchmark_server import Client, Program

KERNEL_SOURCE = r"""
from __future__ import annotations
from tvm.script import tirx as T


@T.jit
def main(
    A: T.Buffer((N,), "float32"),
    B: T.Buffer((N,), "float32"),
    *,
    N: T.constexpr,
):
    T.device_entry()
    i = T.cta_id([N])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0
"""


def build_program() -> Program:
    values = np.arange(256, dtype=np.float32)

    program = Program()
    kernel = program.upload(id="kernel", kind="module", source=KERNEL_SOURCE)
    input_tensor = program.upload(id="input", kind="tensor", value=values)
    output = program.run(
        id="output",
        fn="builtin.empty",
        args=[{"shape": [len(values)], "dtype": "float32"}],
    )
    compiled = program.run(
        id="compiled",
        fn="builtin.compile_tirx",
        args=[kernel, {"N": len(values)}],
    )
    program.run(id="invoke", fn=compiled, args=[input_tensor, output])
    timing = program.run(
        id="timing",
        fn="builtin.benchmark",
        args=[compiled, input_tensor, output, {"warmup": 10, "repeat": 50}],
    )
    program.return_(key="timing", value=timing)
    program.return_(key="output", value=output)
    return program


def main() -> None:
    with Client("http://localhost:8000") as client:
        result = client.execute(build_program(), timeout_seconds=120)

    print(f"status={result.status} elapsed_ms={result.elapsed_ms:.3f}")
    print(result.results["timing"])
    print(result.results["output"])


if __name__ == "__main__":
    main()
