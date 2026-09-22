"""Standalone benchmark definitions need no external benchmark checkout."""

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kcoral import Client, bench_cli, builtins, tool_cli
from kcoral import _bench_worker as worker
from kcoral._tool_inputs import pack_inputs
from kcoral._tool_worker import collect_files, unpack_inputs
from kcoral.app import create_app
from kcoral.config import ServerConfig
from kcoral.testing import FakeRuntime

# Comparison double: timing uses a deterministic KCoral benchmark substitute.
FAKE_TORCH = """
import math
from contextlib import nullcontext
from types import SimpleNamespace
__version__ = 'test'
class Tensor: pass
cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda: 'test GPU',
                       synchronize=lambda: None)
no_grad = nullcontext
def assert_close(actual, expected, **kwargs):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected: assert_close(actual[key], expected[key], **kwargs)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected): assert_close(a, b, **kwargs)
    else:
        assert math.isclose(actual, expected, abs_tol=kwargs['atol'], rel_tol=kwargs['rtol'])
testing = SimpleNamespace(assert_close=assert_close)
"""
BENCH = """
from .helper import increment

def make_inputs(case): return [case['value']]
def reference(x): return increment(x)
"""
CANDIDATE = """
import os
from pathlib import Path

def run(x):
    assert os.environ['MODE'] == 'remote'
    assert Path(os.environ['KCORAL_DIR']) == Path.cwd()
    Path('results').mkdir(exist_ok=True)
    Path('results/value.txt').write_text(str(x))
    return x + int(Path('experiment/factor.txt').read_text())
"""


def fake_benchmark(call, config):
    for _ in range(config["warmup"] + config["repeat"]):
        call()
    return {
        "latency_ms_median": 0.5,
        "latency_ms_mean": 0.5,
        "latency_ms_min": 0.5,
        "latency_ms_max": 0.5,
        "activities_stable": True,
        **config,
    }


def benchmark_runtime_factory():
    builtins.benchmark = fake_benchmark
    return FakeRuntime()


@pytest.fixture
def benchmark(tmp_path):
    directory = tmp_path / "benchmarks/test"
    directory.mkdir(parents=True)
    (directory / "bench.json").write_text(json.dumps({"cases": [{"value": 1}, {"value": 2}]}))
    (directory / "bench.py").write_text(BENCH)
    (directory / "helper.py").write_text("def increment(x): return x + 1\n")
    (directory / "v0.py").write_text("def run(x): return x + 1\n")
    (directory / "torch.py").write_text(FAKE_TORCH)
    return directory


@pytest.fixture
def remote(monkeypatch, tmp_path):
    monkeypatch.delenv("MODE", raising=False)
    config = ServerConfig(
        device="cpu", sandbox="none", max_requests_per_worker=0, disk_cache_dir=tmp_path / "cache"
    )
    with TestClient(create_app(config, runtime_factory=benchmark_runtime_factory)) as server:
        client = Client("http://testserver")
        client.close()
        client._http = server
        monkeypatch.setattr(tool_cli, "Client", lambda url: nullcontext(client))
        monkeypatch.setenv("KCORAL_URL", "http://testserver")
        yield server


def test_load_definition_without_importing_user_code(benchmark, monkeypatch):
    (benchmark / "bench.py").write_text("raise RuntimeError('must only execute on worker')")
    monkeypatch.chdir(benchmark.parent)
    directory, cases, candidate, config = bench_cli.load_benchmark("test", "v0", None, 0, 2)
    assert directory == benchmark and candidate == "v0.py"
    assert cases == [{"value": 1}, {"value": 2}]
    assert config == {"warmup": 0, "repeat": 2, "atol": 1e-5, "rtol": 1e-5}
    assert bench_cli.load_benchmark("test", "v0.py", benchmark.parent, None, None)[2] == "v0.py"
    assert bench_cli.load_benchmark(str(benchmark), "baseline", None, None, None)[2] is None


@pytest.mark.parametrize(
    "settings",
    [
        [],
        {},
        {"cases": []},
        {"cases": [1]},
        {"cases": [{"value": float("nan")}]},
        {"cases": [{}], "warmup": -1},
        {"cases": [{}], "warmup": True},
        {"cases": [{}], "repeat": 0},
        {"cases": [{}], "repeat": 1.5},
        {"cases": [{}], "atol": -1},
        {"cases": [{}], "rtol": float("inf")},
        {"cases": [{}], "atol": True},
        {"cases": [{}], "unknown": 1},
    ],
)
def test_invalid_manifest(benchmark, settings):
    (benchmark / "bench.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        bench_cli.load_benchmark(str(benchmark), "baseline", None, None, None)


@pytest.mark.parametrize("version", ["../v0", ".", "..", "dir/v0", "dir\\v0"])
def test_candidate_must_be_in_benchmark_directory(benchmark, version):
    with pytest.raises(ValueError, match="VERSION"):
        bench_cli.load_benchmark(str(benchmark), version, None, None, None)


def test_missing_definition_or_candidate(benchmark):
    with pytest.raises(FileNotFoundError, match="candidate not found"):
        bench_cli.load_benchmark(str(benchmark), "absent", None, None, None)
    (benchmark / "bench.py").unlink()
    with pytest.raises(FileNotFoundError, match="definition not found"):
        bench_cli.load_benchmark(str(benchmark), "baseline", None, None, None)


def test_benchmark_request_is_self_contained(benchmark):
    program = bench_cli.build_request(
        "test",
        "v0.py",
        {"value": 1},
        {"warmup": 0, "repeat": 2},
        inputs=pack_inputs([benchmark]),
        environment={"MODE": "debug"},
        fetch=["results"],
    )
    run = next(item for item in program._instructions if item.get("id") == "run")
    assert run["args"][:4] == ["test", "v0.py", {"value": 1}, {"warmup": 0, "repeat": 2}]
    assert run["args"][5:7] == [{"MODE": "debug"}, ["results"]]
    assert [item["key"] for item in program._instructions if item["op"] == "return"] == [
        "outcome",
        "artifacts",
    ]
    assert program._blobs


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


@pytest.mark.parametrize(
    "argv",
    [
        ["kda/decode", "v0"],
        ["--repo", "checkout", "--", "test"],
        ["--", "test", "--host", "gpu.example"],
        ["--fetch", "results", "--", "test"],
        ["--env", "CUDA_VISIBLE_DEVICES=0", "--", "test"],
        ["--", "test", "--warmup", "-1"],
        ["--", "test", "--repeat", "0"],
    ],
)
def test_invalid_arguments(monkeypatch, argv):
    monkeypatch.setenv("KCORAL_URL", "http://server")
    with pytest.raises(SystemExit) as exc:
        bench_cli.parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", [["--help"], ["--", "--help"]])
def test_help_needs_no_server_or_gpu_libraries(monkeypatch, capsys, argv):
    monkeypatch.delenv("KCORAL_URL", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit) as exc:
        bench_cli.main(argv)
    assert exc.value.code == 0
    assert "--repo" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["passed", "incorrect", "error", "later-error", "missing"])
def test_remote_cases_files_and_summary(remote, benchmark, tmp_path, capsys, mode):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "factor.txt").write_text("1")
    source = CANDIDATE
    if mode == "incorrect":
        source = source.replace("return x +", "return x -")
    elif mode == "error":
        source = source.replace(
            "    return x +", "    raise RuntimeError('candidate failed')\n    return x +"
        )
    elif mode == "later-error":
        source = source.replace(
            "    return x +",
            "    if x == 2: raise RuntimeError('candidate failed')\n    return x +",
        )
    (benchmark / "v0.py").write_text(source)
    out = tmp_path / "artifacts"
    selected = "missing" if mode == "missing" else "results"
    code = bench_cli.main(
        [
            "--send",
            str(experiment),
            "-e",
            "MODE=remote",
            "--fetch",
            selected,
            "--out",
            str(out),
            "--",
            "test",
            "v0",
            "--repo",
            str(benchmark.parent),
            "--warmup",
            "0",
            "--repeat",
            "2",
        ]
    )
    assert code == int(mode != "passed")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["completed"] is (mode not in {"error", "later-error"})
    assert summary["passed"] is (mode == "passed")
    assert summary["config"]["warmup"] == 0 and summary["config"]["repeat"] == 2
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
        assert len(summary["results"]) == 2
    if mode == "passed":
        assert all(row["speedup"] == 1.0 for row in summary["results"])
    capsys.readouterr()
    assert tool_cli.main("python", ["--", "-c", "import os; print('MODE' in os.environ)"]) == 0
    assert capsys.readouterr().out == "False\n"


@pytest.mark.parametrize("candidate", [False, True])
def test_baseline_and_prepared_candidate(remote, benchmark, tmp_path, candidate, capsys):
    (benchmark / "v0.py").write_text("def prepare(x): return lambda: x + 1\n")
    out = tmp_path / "artifacts"
    version = "v0" if candidate else "baseline"
    assert bench_cli.main(["--out", str(out), "--", str(benchmark), version]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["passed"] is True
    assert summary["results"][0]["baseline_ms"] == 0.5
    assert summary["results"][0]["kernel_ms"] == 0.5
    assert not (out / "workloads/0001/files").exists()
    assert "2/2 cases passed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "definition,candidate,message",
    [
        (BENCH + "\ndef baseline(x): return x - 1\n", "def run(x): return x + 1", "baseline"),
        (BENCH, "def prepare(x): return 42", "prepare() must return"),
        (BENCH, "VALUE = 1", "must define callable run"),
        (
            BENCH.replace("return [case", "return {0: case").replace("['value']]", "['value']}"),
            "def run(x): return x + 1",
            "must return a tuple or list",
        ),
        (
            BENCH.replace("return increment(x)", "return None"),
            "def run(x): return x + 1",
            "not None",
        ),
    ],
)
def test_invalid_contracts_and_baseline_checks(
    remote, benchmark, tmp_path, definition, candidate, message
):
    (benchmark / "bench.py").write_text(definition)
    (benchmark / "v0.py").write_text(candidate)
    out = tmp_path / "artifacts"
    assert bench_cli.main(["--out", str(out), "--", str(benchmark), "v0"]) == 1
    summary = (out / "summary.json").read_text()
    assert message in summary


def test_post_timing_check_detects_stateful_candidate(remote, benchmark, tmp_path):
    (benchmark / "v0.py").write_text("""
def prepare(x):
    calls = 0
    def run():
        nonlocal calls
        calls += 1
        return x + (1 if calls <= 2 else 2)
    return run
""")
    out = tmp_path / "artifacts"
    assert bench_cli.main(["--out", str(out), "--", str(benchmark), "v0"]) == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["completed"] is True
    assert summary["results"][0]["message"].startswith("candidate after timing")


@pytest.mark.parametrize("fail", [False, True])
def test_worker_restores_environment_and_imports(monkeypatch, benchmark, tmp_path, fail):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("MODE", "original")
    monkeypatch.setattr(builtins, "benchmark", fake_benchmark)
    extra = "import os\nos.environ['MODE'] = 'changed'\n"
    if fail:
        extra += "raise RuntimeError('load failed')\n"
    (benchmark / "bench.py").write_text(extra + BENCH)
    environment, search_path = dict(os.environ), list(sys.path)
    modules = dict(sys.modules)
    outcome = worker.execute(
        "test",
        "v0.py",
        {"value": 1},
        {"warmup": 0, "repeat": 2, "atol": 1e-5, "rtol": 1e-5},
        pack_inputs([benchmark]),
        {"MODE": "remote"},
        [],
        unpack_inputs,
        collect_files,
    )
    assert ("load failed" in outcome["error"]) if fail else outcome["rows"][0]["passed"]
    assert Path.cwd() == workspace
    assert dict(os.environ) == environment and sys.path == search_path
    assert set(sys.modules) == set(modules)
    assert all(sys.modules[name] is module for name, module in modules.items())


def test_measure_uses_builtin_cupti_and_preserves_zero_warmup(monkeypatch):
    calls = []
    monkeypatch.setattr(builtins, "benchmark", fake_benchmark)
    result = worker._measure(lambda: calls.append(True), 0, 3)
    assert result["latency_ms_median"] == 0.5
    assert result["warmup"] == 0 and result["repeat"] == 3 and result["flush_l2"] is True
    assert len(calls) == 3


def test_existing_output_does_not_prepare_or_contact_server(monkeypatch, tmp_path):
    monkeypatch.setenv("KCORAL_URL", "http://unused")
    monkeypatch.setattr(bench_cli, "load_benchmark", lambda *a: pytest.fail("must not prepare"))
    assert bench_cli.main(["--out", str(tmp_path), "--", "test"]) == 1
