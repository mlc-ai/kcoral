"""Compile on a remote server's CPU, then check and time the kernel on its GPU."""

import os

import numpy as np

from kcoral import Client, Program

SOURCE = r"""
from __future__ import annotations

from pathlib import Path

import torch
import tvm
import tvm_ffi
from tvm.script import tirx as T
from kcoral.builtins import benchmark

@T.jit
def add_one(A: T.Buffer((N,), "float32"), B: T.Buffer((N,), "float32"), *, N: T.constexpr):
    T.device_entry()
    i = T.cta_id([N])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0


def compile_kernel(arch):
    # An explicit architecture avoids querying the GPU during compilation.
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    module = tvm.IRModule({"add_one": add_one.specialize(N=256)})
    with target:
        compiled = tvm.compile(module, target=target, tir_pipeline="tirx")
    compiled.export_library("add_one.so")
    return "add_one.so"


def evaluate(library, src):
    module = tvm_ffi.load_module(Path(library).resolve())
    compiled = module["add_one"]
    dst = torch.empty_like(src)
    compiled(src, dst)
    torch.testing.assert_close(dst, src + 1.0, rtol=1e-2, atol=1e-3)
    return {"check": {"passed": True}, "timing": benchmark(compiled, src, dst)}
"""


def build_program(arch: str) -> Program:
    program = Program()
    module = program.upload(kind="module", source=SOURCE)
    compile_kernel = program.get_function(module=module, name="compile_kernel", cpu_only=True)
    evaluate = program.get_function(module=module, name="evaluate")
    library = program.run(fn=compile_kernel, args=[arch])
    values = np.arange(256, dtype=np.float32)
    src = program.upload(kind="tensor", value=values)
    report = program.run(fn=evaluate, args=[library, src])
    program.return_(key="report", value=report)
    return program


def main() -> None:
    with Client(os.environ.get("KCORAL_URL", "http://127.0.0.1:8000")) as client:
        result = client.execute(build_program(client.target()["arch"]), timeout_seconds=120)
    if not result.completed:
        raise SystemExit(f"Benchmark failed: {result.error}")
    print(result.results["report"]["check"])
    print(result.results["report"]["timing"])


if __name__ == "__main__":
    main()
