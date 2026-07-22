"""GPU integration tests for GPURuntime + the tvm/TIRX builtins.

Skipped unless BENCH_GPU_TEST=1 and a GPU with a TIRX-enabled tvm are available.
tvm may be pip-installed (no extra env needed) or built from source (put its
Python tree on PYTHONPATH and point TVM_LIBRARY_PATH at the built lib dir). With
a source build::

    BENCH_GPU_TEST=1 PYTHONPATH=<tvm>/python:src TVM_LIBRARY_PATH=<tvm>/build/lib \\
        python -m pytest tests/test_gpu_runtime.py -q
"""

import os

import pytest

from benchmark_server.engine import execute
from benchmark_server.keys import canonical_bytes
from benchmark_server.schemas import Program, Run, Upload

pytestmark = pytest.mark.skipif(
    os.environ.get("BENCH_GPU_TEST") != "1",
    reason="GPU integration test; set BENCH_GPU_TEST=1 with the TIRX env to run",
)

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


def _runtime():
    from benchmark_server.gpu_runtime import GPURuntime

    return GPURuntime()


def _up(id, kind, inline):
    return Upload(id=id, kind=kind, key="sha256:unused", inline=inline)


def _run(program):
    program.upload_bytes = {
        i.id: canonical_bytes(i.kind, i.inline)
        for i in program.instructions
        if i.op == "upload"
    }
    return execute(program, _runtime())


def _by_id(results, id):
    return next(r for r in results if r.id == id)


def test_compile_run_correctness_and_benchmark():
    res = _run(Program(instructions=[
        _up("kernel", "function", {"source": KERNEL}),
        _up("reffn", "function", {"source": REF}),
        Run("x", "builtin.randn", [{"shape": [256], "dtype": "float32", "seed": 0}]),
        Run("out", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
        Run("mod", "builtin.compile_tirx", [{"$ref": "kernel"}, {"N": 256}]),
        Run("_run", {"$ref": "mod"}, [{"$ref": "x"}, {"$ref": "out"}]),
        Run("ref", {"$ref": "reffn"}, [{"$ref": "x"}]),
        Run("chk", "builtin.check_close", [{"$ref": "out"}, {"$ref": "ref"}]),
        Run("perf", "builtin.benchmark",
            [{"$ref": "mod"}, {"$ref": "x"}, {"$ref": "out"}, {"warmup": 5, "repeat": 20}]),
    ]))
    assert [r.status for r in res] == ["OK"] * 9
    assert _by_id(res, "x").value == {"handle": "x"}  # tensor -> handle, not transmitted
    chk = _by_id(res, "chk").value
    assert chk["passed"] and chk["max_abs_err"] == 0.0
    perf = _by_id(res, "perf").value
    assert perf["latency_ms"] > 0 and perf["repeat"] == 20


def test_python_syntax_error_is_parse():
    res = _run(Program(instructions=[_up("k", "function", {"source": "def bad(:\n pass\n"})]))
    assert res[0].status == "FAILED" and res[0].error["kind"] == "parse"


def test_tirx_error_is_parse():
    bad = KERNEL.replace("B[i] = A[i] + 1.0", "B[i] = A[i] + undefined_symbol")
    res = _run(Program(instructions=[
        _up("k", "function", {"source": bad}),
        Run("m", "builtin.compile_tirx", [{"$ref": "k"}, {"N": 16}]),
    ]))
    assert res[0].status == "OK"
    assert res[1].status == "FAILED" and res[1].error["kind"] == "parse"


def test_compile_on_non_kernel_is_compile_error():
    res = _run(Program(instructions=[
        Run("x", "builtin.randn", [{"shape": [4], "dtype": "float32"}]),
        Run("m", "builtin.compile_tirx", [{"$ref": "x"}]),
    ]))
    assert res[1].status == "FAILED" and res[1].error["kind"] == "compile"


def _correctness_program(kernel_src):
    return Program(instructions=[
        _up("kernel", "function", {"source": kernel_src}),
        _up("reffn", "function", {"source": REF}),  # reference is A + 1.0
        Run("x", "builtin.randn", [{"shape": [256], "dtype": "float32", "seed": 0}]),
        Run("out", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
        Run("mod", "builtin.compile_tirx", [{"$ref": "kernel"}, {"N": 256}]),
        Run("_run", {"$ref": "mod"}, [{"$ref": "x"}, {"$ref": "out"}]),
        Run("ref", {"$ref": "reffn"}, [{"$ref": "x"}]),
        Run("chk", "builtin.assert_close", [{"$ref": "out"}, {"$ref": "ref"}]),
        Run("perf", "builtin.benchmark", [{"$ref": "mod"}, {"$ref": "x"}, {"$ref": "out"}]),
    ])


def test_assert_close_passes_when_correct():
    res = _run(_correctness_program(KERNEL))  # kernel is A + 1.0, matches the reference
    assert [r.status for r in res] == ["OK"] * 9


def test_assert_close_fails_correctness_and_skips_rest():
    wrong = KERNEL.replace("A[i] + 1.0", "A[i] + 2.0")  # kernel disagrees with the reference
    res = _run(_correctness_program(wrong))
    chk = _by_id(res, "chk")
    assert chk.status == "FAILED" and chk.error["kind"] == "correctness"
    assert _by_id(res, "perf").status == "SKIPPED"  # benchmark does not run


def test_upload_tensor_and_run_kernel():
    import base64

    import numpy as np

    a = np.arange(256, dtype=np.float32)
    tensor = {"dtype": "float32", "shape": [256],
              "data_b64": base64.b64encode(a.tobytes()).decode()}
    res = _run(Program(instructions=[
        _up("kernel", "function", {"source": KERNEL}),
        _up("reffn", "function", {"source": REF}),
        _up("a", "tensor", tensor),  # client-provided input tensor
        Run("out", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
        Run("mod", "builtin.compile_tirx", [{"$ref": "kernel"}, {"N": 256}]),
        Run("_run", {"$ref": "mod"}, [{"$ref": "a"}, {"$ref": "out"}]),
        Run("ref", {"$ref": "reffn"}, [{"$ref": "a"}]),
        Run("chk", "builtin.check_close", [{"$ref": "out"}, {"$ref": "ref"}]),
    ]))
    assert [r.status for r in res] == ["OK"] * 8
    chk = _by_id(res, "chk").value
    assert chk["passed"] and chk["max_abs_err"] == 0.0  # kernel ran on the uploaded tensor
