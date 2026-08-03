"""GPU integration tests for the runtime and TIRx builtins."""

import os

import pytest

from benchmark_server.engine import execute
from benchmark_server.keys import compute_blob_hash
from benchmark_server.schemas import Program, Ref, Return, Run, Upload

pytestmark = pytest.mark.skipif(
    os.environ.get("BENCH_GPU_TEST") != "1",
    reason="GPU integration test requires BENCH_GPU_TEST=1",
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

PRIM_KERNEL = """from tvm.script import tirx as T

@T.prim_func
def main(A: T.Buffer((256,), "float32"), B: T.Buffer((256,), "float32")):
    T.device_entry()
    i = T.cta_id([256])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0
"""


def ref(handle):
    return Ref(handle)


def runtime():
    from benchmark_server.gpu_runtime import GPURuntime

    return GPURuntime()


def decode_structural(encoded):
    value_type = encoded["type"]
    if value_type == "object":
        return {key: decode_structural(value) for key, value in encoded["value"].items()}
    if value_type == "array":
        return [decode_structural(value) for value in encoded["value"]]
    if value_type == "null":
        return None
    return encoded["value"]


def test_compile_correctness_and_benchmark():
    program = Program(
        [
            Upload("kernel", "module", source=KERNEL),
            Upload("reference", "module", source=REF),
            Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32", "seed": 0}]),
            Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
            Run("compiled", "builtin.compile_tirx", [ref("kernel"), {"N": 256}]),
            Run("invoke", ref("compiled"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            Run("check", "builtin.check_close", [ref("output"), ref("expected")]),
            Run(
                "timing",
                "builtin.benchmark",
                [ref("compiled"), ref("input"), ref("output"), {"warmup": 5, "repeat": 20}],
            ),
            Return("check", ref("check")),
            Return("timing", ref("timing")),
        ]
    )
    outcome = execute(program, runtime())
    assert outcome.status == "COMPLETED"
    check = decode_structural(outcome.results["check"])
    timing = decode_structural(outcome.results["timing"])
    assert check["passed"] and check["max_abs_err"] == 0
    assert timing["latency_ms_median"] > 0 and timing["repeat"] == 20


def test_benchmark_budget_counts_and_no_flush():
    program = Program(
        [
            Upload("kernel", "module", source=KERNEL),
            Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32", "seed": 0}]),
            Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
            Run("compiled", "builtin.compile_tirx", [ref("kernel"), {"N": 256}]),
            Run(
                "timing",
                "builtin.benchmark",
                [
                    ref("compiled"),
                    ref("input"),
                    ref("output"),
                    {"warmup_ms": 5, "repeat_ms": 20, "flush_l2": False},
                ],
            ),
            Return("timing", ref("timing")),
        ]
    )
    outcome = execute(program, runtime())
    assert outcome.status == "COMPLETED"
    timing = decode_structural(outcome.results["timing"])
    assert timing["latency_ms_median"] > 0 and timing["flush_l2"] is False
    assert timing["warmup"] >= 1 and timing["repeat"] > 10


def test_python_syntax_error_is_parse_failure():
    outcome = execute(
        Program([Upload("kernel", "module", source="def bad(:\n    pass\n")]), runtime()
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "parse"
    assert outcome.error["instruction_index"] == 0


def test_tirx_error_is_parse_failure():
    bad_kernel = KERNEL.replace("B[i] = A[i] + 1.0", "B[i] = A[i] + undefined_symbol")
    outcome = execute(
        Program(
            [
                Upload("kernel", "module", source=bad_kernel),
                Run("compiled", "builtin.compile_tirx", [ref("kernel"), {"N": 16}]),
            ]
        ),
        runtime(),
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "parse"
    assert outcome.error["instruction_index"] == 1


def test_compile_on_non_kernel_is_compile_failure():
    outcome = execute(
        Program(
            [
                Run("input", "builtin.randn", [{"shape": [4], "dtype": "float32"}]),
                Run("compiled", "builtin.compile_tirx", [ref("input")]),
            ]
        ),
        runtime(),
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "compile"
    assert outcome.error["instruction_index"] == 1


def test_prim_func_kernel_compiles_directly():
    outcome = execute(
        Program(
            [
                Upload("kernel", "module", source=PRIM_KERNEL),
                Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32"}]),
                Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
                Run("compiled", "builtin.compile_tirx", [ref("kernel")]),
                Run("invoke", ref("compiled"), [ref("input"), ref("output")]),
            ]
        ),
        runtime(),
    )
    assert outcome.status == "COMPLETED"


def test_prim_func_with_bindings_is_compile_failure():
    outcome = execute(
        Program(
            [
                Upload("kernel", "module", source=PRIM_KERNEL),
                Run("compiled", "builtin.compile_tirx", [ref("kernel"), {"N": 256}]),
            ]
        ),
        runtime(),
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "compile"
    assert outcome.error["instruction_index"] == 1


def test_bad_binding_name_is_compile_failure():
    outcome = execute(
        Program(
            [
                Upload("kernel", "module", source=KERNEL),
                Run("compiled", "builtin.compile_tirx", [ref("kernel"), {"WRONG": 1}]),
            ]
        ),
        runtime(),
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "compile"
    assert outcome.error["instruction_index"] == 1


def test_assert_close_failure_stops_without_results():
    wrong = KERNEL.replace("A[i] + 1.0", "A[i] + 2.0")
    program = Program(
        [
            Upload("kernel", "module", source=wrong),
            Upload("reference", "module", source=REF),
            Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32"}]),
            Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
            Run("compiled", "builtin.compile_tirx", [ref("kernel"), {"N": 256}]),
            Run("invoke", ref("compiled"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            Run("check", "builtin.assert_close", [ref("output"), ref("expected")]),
            Return("output", ref("output")),
        ]
    )
    outcome = execute(program, runtime())
    assert outcome.status == "FAILED" and outcome.results == {}
    assert outcome.error["kind"] == "correctness" and outcome.error["instruction_index"] == 7


def test_uploaded_and_returned_tensor_bytes():
    import numpy as np

    array = np.arange(16, dtype=np.float32)
    raw = array.tobytes()
    digest = compute_blob_hash(raw)
    program = Program(
        [
            Upload("tensor", "tensor", blob=digest, dtype="float32", shape=[16]),
            Return("tensor", ref("tensor")),
        ],
        blob_bytes={digest: raw},
    )
    outcome = execute(program, runtime())
    assert outcome.status == "COMPLETED"
    np.testing.assert_array_equal(
        np.frombuffer(outcome.binary_parts["return:0"], dtype=np.float32), array
    )
