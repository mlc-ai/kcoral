import os

import pytest
from fastapi.testclient import TestClient

from benchmark_server.app import create_app
from benchmark_server.config import ServerConfig
from benchmark_server.gpu_runtime import gpu_runtime_factory
from benchmark_server.keys import compute_key
from benchmark_server.testing import fake_runtime_factory

STUB_FN = "def main(a):\n    return a\n"  # a callable the fake materializes; body irrelevant


def make_client():
    return TestClient(create_app(runtime_factory=fake_runtime_factory))


def _fn_upload(inline: bool):
    src = {"source": STUB_FN}
    up = {"id": "fn", "op": "upload", "kind": "function", "key": compute_key("function", src)}
    if inline:
        up["inline"] = src
    return up


def bench_program(inline_upload: bool = True):
    # An upload (so CACHE_MISS has a key to report) followed by a few runs that
    # thread a handle through the engine. The fake's builtins fake only the work.
    return {
        "instructions": [
            _fn_upload(inline_upload),
            {"id": "x", "op": "run", "fn": "builtin.opaque", "args": []},
            {"id": "y", "op": "run", "fn": {"$ref": "fn"}, "args": [{"$ref": "x"}]},
            {"id": "r", "op": "run", "fn": "builtin.structural", "args": [{"$ref": "y"}]},
        ]
    }


def test_health():
    with make_client() as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_full_program_completed():
    with make_client() as c:
        r = c.post("/benchmark", json=bench_program())
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "COMPLETED"
        last = data["results"][-1]
        assert last["status"] == "OK" and last["value"] == {"ok": True}
        assert data["results"][0] == {"id": "fn", "op": "upload", "status": "OK"}


def test_failed_instruction_reports_failed_status():
    prog = {
        "instructions": [
            {"id": "a", "op": "run", "fn": "builtin.opaque", "args": []},
            {"id": "b", "op": "run", "fn": "builtin.nope", "args": []},  # unknown -> FAILED
            {"id": "c", "op": "run", "fn": "builtin.opaque", "args": []},  # -> SKIPPED
        ]
    }
    with make_client() as c:
        r = c.post("/benchmark", json=prog)
    assert r.status_code == 200  # 200 = "we ran your program"
    data = r.json()
    assert data["status"] == "FAILED"  # body status reflects the instruction outcome
    assert [x["status"] for x in data["results"]] == ["OK", "FAILED", "SKIPPED"]


def test_cache_miss_then_retry_with_inline():
    with make_client() as c:
        miss = c.post("/benchmark", json=bench_program(inline_upload=False)).json()
        assert miss["status"] == "CACHE_MISS"
        assert compute_key("function", {"source": STUB_FN}) in miss["missing_keys"]
        done = c.post("/benchmark", json=bench_program(inline_upload=True)).json()
        assert done["status"] == "COMPLETED"


def test_cache_hit_lets_key_only_upload_run():
    with make_client() as c:
        assert c.post("/benchmark", json=bench_program(True)).json()["status"] == "COMPLETED"
        # key now cached: a key-only program resolves from the cache
        assert c.post("/benchmark", json=bench_program(False)).json()["status"] == "COMPLETED"


def test_key_mismatch_is_400():
    with make_client() as c:
        prog = {
            "instructions": [
                {
                    "id": "k",
                    "op": "upload",
                    "kind": "function",
                    "key": "sha256:wrong",
                    "inline": {"source": STUB_FN},
                }
            ]
        }
        assert c.post("/benchmark", json=prog).status_code == 400


def test_bad_json_is_400():
    with make_client() as c:
        r = c.post("/benchmark", content=b"not json", headers={"content-type": "application/json"})
        assert r.status_code == 400


def test_unknown_op_is_400():
    with make_client() as c:
        r = c.post("/benchmark", json={"instructions": [{"id": "a", "op": "frob"}]})
        assert r.status_code == 400


def test_worker_crash_is_500():
    with make_client() as c:
        prog = {"instructions": [{"id": "boom", "op": "run", "fn": "builtin.crash", "args": []}]}
        r = c.post("/benchmark", json=prog)
        assert r.status_code == 500 and r.json()["error"]["kind"] == "engine"


def test_timeout_is_504():
    with make_client() as c:
        prog = {
            "instructions": [{"id": "s", "op": "run", "fn": "builtin.sleep", "args": [3.0]}],
            "options": {"timeout_seconds": 0.5},
        }
        r = c.post("/benchmark", json=prog)
        assert r.status_code == 504 and r.json()["error"]["kind"] == "timeout"


# --- real kernel over the full stack (HTTP -> spawned worker -> GPURuntime) ---
# Skipped unless BENCH_GPU_TEST=1 and a GPU with a TIRX-enabled tvm are available.

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


def _gpu_id() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(cvd) if cvd.isdigit() else 0


def _upload(id, source):
    return {
        "id": id,
        "op": "upload",
        "kind": "function",
        "key": compute_key("function", {"source": source}),
        "inline": {"source": source},
    }


def _tensor_upload(id, arr):
    import base64

    inline = {
        "dtype": "float32",
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes()).decode(),
    }
    return {
        "id": id,
        "op": "upload",
        "kind": "tensor",
        "key": compute_key("tensor", inline),
        "inline": inline,
    }


@pytest.mark.skipif(
    os.environ.get("BENCH_GPU_TEST") != "1",
    reason="real-kernel e2e; set BENCH_GPU_TEST=1 with the TIRX env to run",
)
def test_real_kernel_end_to_end():
    import numpy as np

    body = {
        "instructions": [
            _upload("kernel", KERNEL),
            _upload("reffn", REF),
            _tensor_upload("x", np.arange(256, dtype=np.float32)),  # client-provided input tensor
            {
                "id": "out",
                "op": "run",
                "fn": "builtin.empty",
                "args": [{"shape": [256], "dtype": "float32"}],
            },
            {
                "id": "mod",
                "op": "run",
                "fn": "builtin.compile_tirx",
                "args": [{"$ref": "kernel"}, {"N": 256}],
            },
            {
                "id": "_run",
                "op": "run",
                "fn": {"$ref": "mod"},
                "args": [{"$ref": "x"}, {"$ref": "out"}],
            },
            {"id": "ref", "op": "run", "fn": {"$ref": "reffn"}, "args": [{"$ref": "x"}]},
            {
                "id": "chk",
                "op": "run",
                "fn": "builtin.check_close",
                "args": [{"$ref": "out"}, {"$ref": "ref"}],
            },
            {
                "id": "perf",
                "op": "run",
                "fn": "builtin.benchmark",
                "args": [
                    {"$ref": "mod"},
                    {"$ref": "x"},
                    {"$ref": "out"},
                    {"warmup": 5, "repeat": 20},
                ],
            },
        ],
        "options": {"timeout_seconds": 120},
    }

    app = create_app(ServerConfig(gpus=[_gpu_id()]), runtime_factory=gpu_runtime_factory)
    with TestClient(app) as c:
        data = c.post("/benchmark", json=body).json()

    assert data["status"] == "COMPLETED"
    assert [r["status"] for r in data["results"]] == ["OK"] * 9
    results = {r["id"]: r for r in data["results"]}
    assert results["chk"]["value"]["passed"] and results["chk"]["value"]["max_abs_err"] == 0.0
    assert results["perf"]["value"]["latency_ms"] > 0 and results["perf"]["value"]["repeat"] == 20
