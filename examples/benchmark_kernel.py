"""Compile a TIRx kernel, check its output, and measure GPU activity."""

import os

import numpy as np

from kcoral import Client, Program

KERNEL = r"""
from __future__ import annotations
from tvm.script import tirx as T

@T.jit
def main(A: T.Buffer((N,), "float32"), B: T.Buffer((N,), "float32"), *, N: T.constexpr):
    T.device_entry()
    i = T.cta_id([N])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0
"""
REFERENCE = "def main(a):\n    return a + 1.0\n"


def build_program() -> Program:
    program = Program()
    kernel_module = program.upload(id="kernel_module", kind="module", source=KERNEL)
    kernel = program.get_function(id="kernel", module=kernel_module, name="main")
    reference_module = program.upload(id="reference_module", kind="module", source=REFERENCE)
    reference = program.get_function(id="reference", module=reference_module, name="main")

    src = program.upload(id="src", kind="tensor", value=np.arange(256, dtype=np.float32))
    dst = program.run(id="dst", fn="builtin.empty", args=[{"shape": [256], "dtype": "float32"}])

    compiled = program.run(id="compiled", fn="builtin.compile_tirx", args=[kernel, {"N": 256}])
    program.run(id="invoke", fn=compiled, args=[src, dst])

    expected = program.run(id="expected", fn=reference, args=[src])
    check = program.run(id="check", fn="builtin.assert_close", args=[dst, expected])
    timing = program.run(id="timing", fn="builtin.benchmark", args=[compiled, src, dst])

    program.return_(key="check", value=check)
    program.return_(key="timing", value=timing)
    return program


def main() -> None:
    with Client(os.environ.get("KCORAL_URL", "http://localhost:8000")) as client:
        result = client.execute(build_program(), timeout_seconds=120)
    if not result.completed:
        raise SystemExit(f"Benchmark failed: {result.error}")
    print(result.results["check"])
    print(result.results["timing"])


if __name__ == "__main__":
    main()
