import json
import linecache
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from kcoral import _bench_worker as worker
from kcoral import bench_cli

SOURCES = {
    "flashinfer_bench_evolve/__init__.py": "",
    "flashinfer_bench_evolve/tasks/__init__.py": "",
    "flashinfer_bench_evolve/tasks/test/__init__.py": "",
    "flashinfer_bench_evolve/benchmark_common.py": "",
    "flashinfer_bench_evolve/tasks/test/definition.json": json.dumps(
        {"reference": "def run(x): return x + 1"}
    ),
    "flashinfer_bench_evolve/tasks/test/benchmark.py": """
from dataclasses import dataclass
from flashinfer_bench_evolve.benchmark_common import load_task_reference
@dataclass
class Config:
    warmup: int = 1
def default_config(): return Config()
def tirx_run(value): return value
def tirx_prepare(lowered, value): return lowered.setup(value)
def run_suite(config, *, workloads, candidate_fn=None, candidate_prepare_fn=None):
    value = workloads[0]['value']
    reference = load_task_reference('test')(value)
    actual = reference if candidate_fn is None else candidate_fn(candidate_prepare_fn(value))
    return [{'passed': actual == reference, 'value': actual, 'warmup': config.warmup}]
""",
}


@pytest.mark.parametrize(
    "candidate", [None, b"def setup(x): return x + 1", b"def setup(x): return x - 1"]
)
def test_uploaded_benchmark_runs_candidate_and_restores_imports(monkeypatch, candidate):
    monkeypatch.setattr(worker, "SOURCES", SOURCES)
    original_finders = list(sys.meta_path)
    original_cache = dict(linecache.cache)
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("flashinfer_bench_evolve")
    }
    worker.main("init", "test", {"warmup": 3}, [{"value": 41}], [], candidate)
    rows = worker.main("run", 0)
    assert rows == [
        {
            "passed": candidate != b"def setup(x): return x - 1",
            "value": 40 if candidate == b"def setup(x): return x - 1" else 42,
            "warmup": 3,
        }
    ]
    assert sys.meta_path == original_finders
    assert {
        key: value
        for key, value in linecache.cache.items()
        if key.startswith("<bundle:") or key == "<lowered.py>"
    } == {
        key: value
        for key, value in original_cache.items()
        if key.startswith("<bundle:") or key == "<lowered.py>"
    }
    assert {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("flashinfer_bench_evolve")
    } == original_modules


def test_failed_benchmark_initialization_cleans_up(monkeypatch):
    monkeypatch.setattr(worker, "SOURCES", SOURCES)
    finders = list(sys.meta_path)
    with pytest.raises(RuntimeError, match="compile failed"):
        worker.main(
            "init", "test", {}, [{"value": 41}], [], b"raise RuntimeError('compile failed')"
        )
    assert sys.meta_path == finders
    assert worker._STATE == {}


def test_bundle_preserves_runtime_annotations_and_nonfinite_results():
    namespace = {}
    worker._exec("def f(x: int): pass", "<annotation-test>", namespace)
    assert namespace["f"].__annotations__["x"] is int
    assert worker._sanitize([float("nan"), float("inf"), -float("inf")]) == [
        "NaN",
        "Infinity",
        "-Infinity",
    ]
    worker._cleanup()


def test_summary_restores_nonfinite_numeric_columns_only():
    rows = bench_cli.decode_rows([{"speedup": "NaN", "max_abs": "Infinity", "id": "NaN"}])
    assert math.isnan(rows[0]["speedup"])
    assert rows[0]["max_abs"] == float("inf")
    assert rows[0]["id"] == "NaN"


def test_benchmark_request_is_self_contained():
    entry = {"id": 1}
    tensor = np.arange(4, dtype=np.float32)
    adapter = SimpleNamespace(blobs=lambda rows: ([{"path": "x", "tensor_key": "y"}], [tensor]))
    source = "def main(*args): pass"
    program = bench_cli.build_request(adapter, source, "test", {"warmup": 2}, entry, b"candidate")
    instructions = program._instructions
    assert instructions[0]["source"] == source
    assert {item.get("kind") for item in instructions} >= {"module", "bytes", "tensor"}
    init = next(item for item in instructions if item.get("id") == "init")
    assert init["args"][:5] == [
        "init",
        "test",
        {"warmup": 2},
        [entry],
        [{"path": "x", "tensor_key": "y"}],
    ]
    assert instructions[-1]["key"] == "row"
    assert program._blobs


@pytest.mark.parametrize("passed", [True, False])
@pytest.mark.parametrize("connection", [[], ["--host", "gpu.example", "--port", "9000"]])
def test_bench_command_sends_each_workload_and_summarizes(
    monkeypatch, tmp_path, capsys, passed, connection
):
    monkeypatch.setenv("KCORAL_URL", "http://server")
    expected_url = "http://gpu.example:9000" if connection else "http://server"
    common = SimpleNamespace(summarize=lambda rows, label: summaries.append((rows, label)))
    monkeypatch.setitem(sys.modules, "flashinfer_bench_evolve.benchmark_common", common)
    summaries, requests = [], []

    def plan(task, directory, version, **kwargs):
        assert (task, directory, version) == ("test", tmp_path / "kda/decode", "v0")
        assert kwargs == {"warmup": 4, "repeat": 50, "shape_mode": "all"}
        return {"warmup": 4}, [{"value": 1}, {"value": 2}], b"def setup(x): return x + 1"

    adapter = SimpleNamespace(
        workload_key=lambda key: key,
        PACKAGED={"kda/decode": ("test", 3, 50, "all")},
        KERNEL_EVOLUTION_ROOT=tmp_path,
        plan=plan,
        harness_sources=lambda task: SOURCES,
        blobs=lambda rows: ([], []),
    )
    monkeypatch.setattr(bench_cli, "load_adapter", lambda repo: adapter)

    def execute(args, program):
        assert args.url == expected_url
        requests.append(program)
        instructions = program._instructions
        assert len(next(item for item in instructions if item.get("id") == "init")["args"][3]) == 1
        return SimpleNamespace(
            completed=True,
            results={"init": {"worker": "test"}, "row": [{"passed": passed}]},
            __getitem__=None,
        )

    class Result:
        completed = True

        def __init__(self, data):
            self.results = data.results

        def __getitem__(self, key):
            return self.results[key]

    monkeypatch.setattr(bench_cli, "execute", lambda args, program: Result(execute(args, program)))
    assert bench_cli.main(["kda/decode", "v0", "--warmup", "4", *connection]) == int(not passed)
    assert len(requests) == 2
    assert summaries == [([{"passed": passed}, {"passed": passed}], "")]
    captured = capsys.readouterr()
    assert f"benchmark server: {expected_url}" in captured.out
    assert captured.err == (
        f"kcoral: warning: --host/--port override KCORAL_URL; using {expected_url}\n"
        if connection
        else ""
    )


def test_adapter_discovery_from_nested_workload(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    directory = root / "kernel-evolution/kda/decode"
    directory.mkdir(parents=True)
    (root / "kernel-evolution/bench_adapter.py").write_text("VALUE = 42\n")
    monkeypatch.chdir(directory)
    assert bench_cli.load_adapter(None).VALUE == 42
    assert bench_cli.load_adapter(root).VALUE == 42
    with pytest.raises(FileNotFoundError, match=r"bench_adapter\.py"):
        bench_cli.load_adapter(tmp_path / "missing")
