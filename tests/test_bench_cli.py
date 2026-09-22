import json
import linecache
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kcoral import Client, bench_cli, tool_cli
from kcoral import _bench_worker as worker
from kcoral._tool_inputs import pack_inputs
from kcoral._tool_worker import collect_files, unpack_inputs
from kcoral.app import create_app
from kcoral.config import ServerConfig
from kcoral.testing import fake_runtime_factory

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
    original = [{"speedup": "NaN", "max_abs": "Infinity", "id": "NaN"}]
    rows = bench_cli.decode_rows(original)
    assert math.isnan(rows[0]["speedup"])
    assert rows[0]["max_abs"] == float("inf")
    assert rows[0]["id"] == "NaN"
    assert original == [{"speedup": "NaN", "max_abs": "Infinity", "id": "NaN"}]


def test_benchmark_request_is_self_contained():
    entry = {"id": 1}
    tensor = np.arange(4, dtype=np.float32)
    adapter = SimpleNamespace(blobs=lambda rows: ([{"path": "x", "tensor_key": "y"}], [tensor]))
    source = "def main(*args): pass"
    program = bench_cli.build_request(
        adapter,
        source,
        "test",
        {"warmup": 2},
        entry,
        b"candidate",
        inputs=pack_inputs([]),
        environment={"MODE": "debug"},
        fetch=[],
    )
    instructions = program._instructions
    assert instructions[0]["source"] == source
    assert {item.get("kind") for item in instructions} >= {"module", "bytes", "tensor"}
    run = next(item for item in instructions if item.get("id") == "run")
    assert run["args"][:4] == [
        "test",
        {"warmup": 2},
        entry,
        [{"path": "x", "tensor_key": "y"}],
    ]
    assert run["args"][6] == {"MODE": "debug"}
    assert instructions[-1]["key"] == "outcome"
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
        assert next(item for item in instructions if item.get("id") == "run")["args"][2] in [
            {"value": 1},
            {"value": 2},
        ]
        return SimpleNamespace(
            completed=True,
            results={
                "outcome": {
                    "worker": {"worker": "test"},
                    "rows": [{"passed": passed}],
                    "missing": [],
                    "error": None,
                }
            },
            __getitem__=None,
        )

    class Result:
        completed = True

        def __init__(self, data):
            self.results = data.results

        def __getitem__(self, key):
            return self.results[key]

    monkeypatch.setattr(bench_cli, "execute", lambda args, program: Result(execute(args, program)))
    assert bench_cli.main([*connection, "--", "kda/decode", "v0", "--warmup", "4"]) == int(
        not passed
    )
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


def test_bench_separates_shared_and_native_options(monkeypatch):
    monkeypatch.delenv("KCORAL_URL", raising=False)
    monkeypatch.setenv("LOCAL_VALUE", "copied")
    args = bench_cli.parse_args(
        [
            "--host",
            "gpu.example",
            "--port",
            "9000",
            "--send",
            "experiment",
            "--env",
            "MODE=debug",
            "-e",
            "LOCAL_VALUE",
            "--fetch",
            "results",
            "--out",
            "artifacts",
            "--",
            "kda/decode",
            "v0",
            "--repo",
            "checkout",
            "--warmup",
            "0",
            "--repeat",
            "2",
        ]
    )
    assert args.url == "http://gpu.example:9000"
    assert args.env == {"MODE": "debug", "LOCAL_VALUE": "copied"}
    assert args.send == [Path("experiment")] and args.fetch == ["results"]
    assert args.out == Path("artifacts") and args.repo == Path("checkout")
    assert (args.workload, args.version, args.warmup, args.repeat) == ("kda/decode", "v0", 0, 2)
    assert (
        bench_cli.parse_args(["--port", "9000", "--out", "artifacts", "--", "kda/decode"]).version
        == "baseline"
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["kda/decode", "v0"],
        ["--repo", "checkout", "--", "kda/decode"],
        ["--", "kda/decode", "--host", "gpu.example"],
        ["--fetch", "results", "--", "kda/decode"],
        ["--env", "CUDA_VISIBLE_DEVICES=0", "--", "kda/decode"],
        ["--", "kda/decode", "--warmup", "-1"],
        ["--", "kda/decode", "--repeat", "0"],
    ],
)
def test_bench_rejects_invalid_parameter_placement(monkeypatch, argv):
    monkeypatch.setenv("KCORAL_URL", "http://server")
    with pytest.raises(SystemExit) as exc:
        bench_cli.parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", [["--help"], ["--", "--help"]])
def test_bench_help_does_not_need_a_server(monkeypatch, capsys, argv):
    monkeypatch.delenv("KCORAL_URL", raising=False)
    with pytest.raises(SystemExit) as exc:
        bench_cli.main(argv)
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--repo" in output and "--warmup" in output
    if argv == ["--help"]:
        assert "--send" in output and "--env" in output


@pytest.mark.parametrize("fail", [False, True])
def test_benchmark_files_and_environment_are_request_local(monkeypatch, tmp_path, fail):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(worker, "SOURCES", SOURCES)
    monkeypatch.setenv("MODE", "original")
    inputs = tmp_path / "experiment"
    inputs.mkdir()
    (inputs / "__init__.py").write_text("")
    (inputs / "provided_helper.py").write_text("VALUE = 'uploaded'\n")
    candidate = (
        "import os\nfrom pathlib import Path\nfrom experiment import provided_helper\n"
        "assert os.environ['MODE'] == 'remote'\n"
        "assert Path(os.environ['KCORAL_DIR']) == Path.cwd()\n"
        "Path('results').mkdir()\n"
        "Path('results/value.txt').write_text(provided_helper.VALUE)\n"
        "os.environ['BENCH_TEMP_VALUE'] = 'temporary'\n"
        + ("raise RuntimeError('candidate failed')\n" if fail else "def setup(x): return x + 1\n")
    ).encode()
    environment, search_path = dict(os.environ), list(sys.path)
    outcome = worker.execute(
        "test",
        {},
        {"value": 41},
        [],
        candidate,
        pack_inputs([inputs]),
        {"MODE": "remote"},
        ["results", "missing"],
        unpack_inputs,
        collect_files,
    )
    assert (tmp_path / "outputs/results/value.txt").read_text() == "uploaded"
    assert outcome["missing"] == ["missing"]
    if fail:
        assert "candidate failed" in outcome["error"] and outcome["rows"] == []
    else:
        assert outcome["error"] is None and outcome["rows"][0]["passed"] is True
    assert Path.cwd() == tmp_path
    assert dict(os.environ) == environment and sys.path == search_path
    assert "experiment.provided_helper" not in sys.modules
    assert "experiment" not in sys.modules
    assert worker._STATE == {}


@pytest.fixture
def benchmark_remote(monkeypatch, tmp_path):
    monkeypatch.delenv("MODE", raising=False)
    config = ServerConfig(
        device="cpu", sandbox="none", max_requests_per_worker=0, disk_cache_dir=tmp_path / "cache"
    )
    with TestClient(create_app(config, runtime_factory=fake_runtime_factory)) as server:
        client = Client("http://testserver")
        client.close()
        client._http = server
        monkeypatch.setattr(tool_cli, "Client", lambda url: nullcontext(client))
        monkeypatch.setenv("KCORAL_URL", "http://testserver")
        yield server


@pytest.mark.parametrize("mode", ["passed", "incorrect", "error", "later-error", "missing"])
def test_bench_roundtrip_files_and_summary(benchmark_remote, monkeypatch, tmp_path, mode, capsys):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "factor.txt").write_text("1")
    candidate = (
        "import os\nfrom pathlib import Path\n"
        "assert os.environ['MODE'] == 'remote'\n"
        "assert Path(os.environ['KCORAL_DIR']) == Path.cwd()\n"
        "def setup(x):\n"
        "    Path('results').mkdir()\n"
        "    Path('results/value.txt').write_text(str(x))\n"
        + (
            "    if x == 2: raise RuntimeError('candidate failed')\n"
            if mode == "later-error"
            else ""
        )
        + (
            "    raise RuntimeError('candidate failed')\n"
            if mode == "error"
            else f"    return x {'-' if mode == 'incorrect' else '+'} "
            "int(Path('experiment/factor.txt').read_text())\n"
        )
    ).encode()
    summaries = []
    monkeypatch.setitem(
        sys.modules,
        "flashinfer_bench_evolve.benchmark_common",
        SimpleNamespace(summarize=lambda rows, label: summaries.append(rows)),
    )
    adapter = SimpleNamespace(
        workload_key=lambda key: key,
        PACKAGED={"test": ("test", 1, 1, "all")},
        KERNEL_EVOLUTION_ROOT=tmp_path,
        plan=lambda *a, **kw: ({}, [{"value": 1}, {"value": 2}], candidate),
        harness_sources=lambda task: SOURCES,
        blobs=lambda rows: ([], []),
    )
    monkeypatch.setattr(bench_cli, "load_adapter", lambda repo: adapter)
    out = tmp_path / "artifacts"
    selected = "missing" if mode == "missing" else "results"
    code = bench_cli.main(
        [
            "--send",
            str(experiment),
            "--env",
            "MODE=remote",
            "--fetch",
            selected,
            "--out",
            str(out),
            "--",
            "test",
            "v0",
        ]
    )
    assert code == int(mode != "passed")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["completed"] is (mode not in {"error", "later-error"})
    assert summary["passed"] is (mode == "passed")
    count = 1 if mode == "error" else 2
    assert len(summary["workloads"]) == count
    for index in range(1, count + 1):
        directory = out / "workloads" / f"{index:04d}"
        result = json.loads((directory / "result.json").read_text())
        assert result["index"] == index
        if mode == "missing":
            assert result["missing"] == ["missing"]
        else:
            assert (directory / "files/results/value.txt").read_text() == str(index)
    if mode in {"error", "later-error"}:
        assert "candidate failed" in summary["error"]
        assert len(summary["results"]) == int(mode == "later-error")
    else:
        assert len(summaries[0]) == 2
    capsys.readouterr()
    assert tool_cli.main("python", ["--", "-c", "import os; print('MODE' in os.environ)"]) == 0
    assert capsys.readouterr().out == "False\n"


def test_bench_out_without_fetch_keeps_nonfinite_json(benchmark_remote, monkeypatch, tmp_path):
    sources = dict(SOURCES)
    path = "flashinfer_bench_evolve/tasks/test/benchmark.py"
    sources[path] = sources[path].replace("'warmup': config.warmup", "'speedup': float('inf')")
    adapter = SimpleNamespace(
        workload_key=lambda key: key,
        PACKAGED={"test": ("test", 1, 1, "all")},
        KERNEL_EVOLUTION_ROOT=tmp_path,
        plan=lambda *a, **kw: ({}, [{"value": 1}], None),
        harness_sources=lambda task: sources,
        blobs=lambda rows: ([], []),
    )
    summaries = []
    monkeypatch.setitem(
        sys.modules,
        "flashinfer_bench_evolve.benchmark_common",
        SimpleNamespace(summarize=lambda rows, label: summaries.append(rows)),
    )
    monkeypatch.setattr(bench_cli, "load_adapter", lambda repo: adapter)
    out = tmp_path / "baseline"
    assert bench_cli.main(["--out", str(out), "--", "test"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["passed"] is True and summary["version"] == "baseline"
    assert summary["results"][0]["speedup"] == "Infinity"
    assert summaries[0][0]["speedup"] == float("inf")
    assert (out / "workloads/0001/result.json").is_file()
    assert not (out / "workloads/0001/files").exists()


def test_existing_bench_output_does_not_contact_server(monkeypatch, tmp_path):
    monkeypatch.setenv("KCORAL_URL", "http://unused")
    monkeypatch.setattr(
        bench_cli, "load_adapter", lambda *a: pytest.fail("must not prepare benchmark")
    )
    assert bench_cli.main(["--out", str(tmp_path), "--", "test"]) == 1
