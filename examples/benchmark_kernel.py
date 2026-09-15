"""Compile a TIRx kernel, check its output, and measure GPU activity."""

import os

import numpy as np

from kcoral import Client, Program

SOURCE = r"""
from __future__ import annotations

import torch
from tvm.script import tirx as T
from kcoral.builtins import compile_tirx, benchmark

@T.jit
def add_one(A: T.Buffer((N,), "float32"), B: T.Buffer((N,), "float32"), *, N: T.constexpr):
    T.device_entry()
    i = T.cta_id([N])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0


def evaluate(src):
    dst = torch.empty_like(src)
    compiled = compile_tirx(add_one, {"N": src.numel()})
    compiled(src, dst)
    torch.testing.assert_close(dst, src + 1.0, rtol=1e-2, atol=1e-3)
    return {"check": {"passed": True}, "timing": benchmark(compiled, src, dst)}
"""


def build_program() -> Program:
    program = Program()
    module = program.upload(id="module", kind="module", source=SOURCE)
    evaluate = program.get_function(id="evaluate", module=module, name="evaluate")
    values = np.arange(256, dtype=np.float32)
    src = program.upload(id="src", kind="tensor", value=values)
    report = program.run(id="report", fn=evaluate, args=[src])
    program.return_(key="report", value=report)
    return program


def main() -> None:
    with Client(os.environ.get("KCORAL_URL", "http://localhost:8000")) as client:
        result = client.execute(build_program(), timeout_seconds=120)
    if not result.completed:
        raise SystemExit(f"Benchmark failed: {result.error}")
    print(result.results["report"]["check"])
    print(result.results["report"]["timing"])


if __name__ == "__main__":
    main()
