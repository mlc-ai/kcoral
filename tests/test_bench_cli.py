"""Standalone definitions, correctness, reports, and CUPTI delegation."""

import json
from contextlib import nullcontext

import pytest
from fastapi.testclient import TestClient

from kcoral import Client, builtins
from kcoral import client as client_module
from kcoral.app import create_app
from kcoral.config import ServerConfig
from kcoral.testing import FakeRuntime
from kcoral.tools import bench, cli

# Scalar comparison double; no CPU or GPU performance is measured in these tests.
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
    assert math.isclose(actual, expected, abs_tol=kwargs['atol'], rel_tol=kwargs['rtol'])
testing = SimpleNamespace(assert_close=assert_close)
"""


def fake_benchmark(call, config):
    config = {**config, "warmup": max(config["warmup"], 1)}
    for _ in range(config["warmup"] + config["repeat"]):
        call()
    return {"latency_ms_median": 0.5, "activities_stable": True, **config}


def runtime_factory():
    builtins.benchmark = fake_benchmark
    return FakeRuntime()


@pytest.fixture
def benchmark(tmp_path):
    directory = tmp_path / "benchmark"
    directory.mkdir()
    (directory / "bench.json").write_text(json.dumps({"cases": [{"value": 1}, {"value": 2}]}))
    (directory / "bench.py").write_text(
        "from .helper import increment\n"
        "def make_inputs(case): return [case['value']]\n"
        "def reference(x): return increment(x)\n"
    )
    (directory / "helper.py").write_text("def increment(x): return x + 1\n")
    (directory / "v0.py").write_text("def run(x): return x + 1\n")
    (directory / "torch.py").write_text(FAKE_TORCH)
    return directory


@pytest.fixture
def remote(monkeypatch, tmp_path):
    monkeypatch.delenv("MODE", raising=False)
    config = ServerConfig(
        device="cpu",
        sandbox="none",
        max_requests_per_worker=0,
        log_console=False,
        disk_cache_dir=tmp_path / "cache",
    )
    with TestClient(create_app(config, runtime_factory=runtime_factory)) as server:
        client = Client("http://testserver")
        client.close()
        client._http = server
        monkeypatch.setattr(client_module, "Client", lambda url: nullcontext(client))
        monkeypatch.setenv("KCORAL_URL", "http://testserver")
        yield


def test_definition_loading_and_validation(benchmark):
    (benchmark / "bench.py").write_text("raise RuntimeError('worker only')")
    directory, cases, candidate, config = bench.load_benchmark(str(benchmark), "v0", None, 0, 2)
    assert directory == benchmark and candidate == "v0.py"
    assert cases == [{"value": 1}, {"value": 2}]
    assert config["warmup"] == 0 and config["repeat"] == 2
    with pytest.raises(ValueError, match="VERSION"):
        bench.load_benchmark(str(benchmark), "../v0", None, None, None)
    (benchmark / "bench.json").write_text('{"cases": []}')
    with pytest.raises(ValueError, match="cases"):
        bench.load_benchmark(str(benchmark), "v0", None, None, None)


def test_shared_and_benchmark_options(monkeypatch):
    monkeypatch.delenv("KCORAL_URL", raising=False)
    args = bench.parse_args(
        "--host gpu.example --port 9000 --output-limit-bytes 524288 --out artifacts "
        "-- kda/decode v0 --repo checkout --warmup 0 --repeat 2".split()
    )
    assert args.url == "http://gpu.example:9000" and args.output_limit_bytes == 524288
    assert (args.workload, args.version, args.warmup, args.repeat) == ("kda/decode", "v0", 0, 2)
    assert str(args.repo) == "checkout" and str(args.out) == "artifacts"
    with pytest.raises(SystemExit) as error:
        bench.parse_args(["--", "--help"])
    assert error.value.code == 0


@pytest.mark.parametrize("mode", ["passed", "incorrect", "later-error"])
def test_remote_cases_and_partial_results(remote, benchmark, tmp_path, capsys, mode):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "factor.txt").write_text("1")
    source = """
import os
from pathlib import Path

def run(x):
    assert os.environ['MODE'] == 'remote'
    assert Path(os.environ['KCORAL_DIR']) == Path.cwd()
    Path('results').mkdir(exist_ok=True)
    Path('results/value.txt').write_text(str(x))
    return x + int(Path('experiment/factor.txt').read_text())
"""
    if mode == "incorrect":
        source = source.replace("return x +", "return x -")
    elif mode == "later-error":
        source = source.replace(
            "    return x +",
            "    if x == 2: raise RuntimeError('candidate failed')\n    return x +",
        )
    (benchmark / "v0.py").write_text(source)
    out = tmp_path / "artifacts"
    code = cli.run_main(
        [
            "bench",
            "--send",
            str(experiment),
            "-e",
            "MODE=remote",
            "--fetch",
            "results",
            "--out",
            str(out),
            "--",
            "benchmark",
            "v0",
            "--repo",
            str(tmp_path),
            "--warmup",
            "0",
            "--repeat",
            "2",
        ]
    )
    summary = json.loads((out / "summary.json").read_text())
    assert code == int(mode != "passed")
    assert summary["passed"] is (mode == "passed")
    assert summary["completed"] is (mode != "later-error")
    assert len(summary["workloads"]) == 2
    for index in (1, 2):
        assert (out / f"workloads/{index:04d}/files/results/value.txt").read_text() == str(index)
    if mode == "later-error":
        assert "candidate failed" in summary["error"] and len(summary["results"]) == 1
    elif mode == "passed":
        for row in summary["results"]:
            assert row["speedup"] == 1
            assert row["kernel_timing"]["warmup"] == 1 and row["kernel_timing"]["repeat"] == 2
    capsys.readouterr()
    assert cli.main("python", ["--", "-c", "import os; print('MODE' in os.environ)"]) == 0
    assert capsys.readouterr().out == "False\n"


@pytest.mark.parametrize("version", ["baseline", "v0"])
def test_baseline_and_prepared_candidate(remote, benchmark, tmp_path, version):
    (benchmark / "v0.py").write_text("def prepare(x): return lambda: x + 1\n")
    out = tmp_path / "artifacts"
    assert bench.main(["--out", str(out), "--", str(benchmark), version]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["passed"] and summary["results"][0]["kernel_ms"] == 0.5
    assert not (out / "workloads/0001/files").exists()


def test_post_timing_correctness(remote, benchmark, tmp_path):
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
    assert bench.main(["--out", str(out), "--", str(benchmark), "v0"]) == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["results"][0]["message"].startswith("candidate after timing")


def test_cupti_delegation(monkeypatch):
    calls = []
    monkeypatch.setattr(builtins, "benchmark", fake_benchmark)
    result = bench._measure(lambda: calls.append(True), 0, 3)
    assert len(calls) == 4
    assert result["warmup"] == 1 and result["repeat"] == 3 and result["flush_l2"] is True
