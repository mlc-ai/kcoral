import io
import json
import os
import subprocess
import sys
import tarfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from kcoral import Client
from kcoral import tool_cli as cli
from kcoral._tool_inputs import pack_inputs
from kcoral._tool_worker import unpack_inputs
from kcoral.app import create_app
from kcoral.config import ServerConfig
from kcoral.testing import fake_runtime_factory


@pytest.mark.parametrize("tool", cli.COMMANDS)
@pytest.mark.parametrize("explicit_argv", [False, True])
def test_run_command_preserves_tool_arguments(monkeypatch, tool, explicit_argv):
    from kcoral.__main__ import main

    arguments = (
        ["kda/decode", "v0"]
        if tool == "bench"
        else ["--send", "experiment", "--", "--flag", "--", "script.py", "a b"]
    )
    calls = []
    monkeypatch.setattr(cli, "main", lambda name, argv: calls.append((name, argv)) or 7)
    monkeypatch.setattr(sys, "argv", ["kcoral", "run", tool, *arguments])
    with pytest.raises(SystemExit) as exc:
        main(["run", tool, *arguments]) if explicit_argv else main()
    assert exc.value.code == 7
    assert calls == [(tool, arguments)]


@pytest.mark.parametrize("argv", [[], ["unknown"]])
def test_run_requires_a_known_tool(argv):
    with pytest.raises(SystemExit) as exc:
        cli.run_main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv,usage",
    [
        (["run", "--help"], "kcoral run"),
        (["run", "ncu", "--help"], "kcoral run ncu"),
        (["run", "bench", "--help"], "kcoral run bench"),
    ],
)
def test_run_command_help(monkeypatch, argv, usage, capsys):
    from kcoral.__main__ import main

    monkeypatch.setattr(sys, "argv", ["kcoral", *argv])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert usage in capsys.readouterr().out


@pytest.fixture
def remote(monkeypatch, tmp_path):
    config = ServerConfig(
        device="cpu",
        sandbox="none",
        max_requests_per_worker=0,
        disk_cache_dir=tmp_path / "cache",
    )
    with TestClient(create_app(config, runtime_factory=fake_runtime_factory)) as server:
        client = Client("http://testserver")
        client.close()
        client._http = server
        monkeypatch.setattr(cli, "Client", lambda url: nullcontext(client))
        monkeypatch.setenv("KCORAL_URL", "http://testserver")
        yield server


def test_python_roundtrip_and_exit_code(remote, tmp_path, capsys):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "value.txt").write_text("uploaded")
    (experiment / "check.py").write_text(
        "import os, sys\nfrom pathlib import Path\n"
        "assert Path(os.environ['KCORAL_DIR']) == Path.cwd()\n"
        "print(Path('value.txt').read_text(), os.environ['TEST_VALUE'], sys.argv[1:])\n"
        "print('remote stderr', file=sys.stderr)\nsys.exit(7)\n"
    )
    code = cli.main(
        "python",
        ["--send", str(experiment), "-e", "TEST_VALUE=a b", "--", "check.py", "--url", "literal"],
    )
    output = capsys.readouterr()
    assert code == 7
    assert "uploaded a b ['--url', 'literal']" in output.out
    assert "remote stderr" in output.err


def test_shell_fetches_binary_and_empty_directory_on_failure(remote, tmp_path):
    script = tmp_path / "setup.sh"
    script.write_text("mkdir -p result/empty\nprintf '\\000\\377' > result/data.bin\nexit 9\n")
    out = tmp_path / "artifacts"
    assert (
        cli.main(
            "shell",
            [
                "--send",
                str(script),
                "--fetch",
                "result",
                "--out",
                str(out),
                "--",
                "bash",
                "setup.sh",
            ],
        )
        == 9
    )
    assert (out / "result/data.bin").read_bytes() == b"\x00\xff"
    assert (out / "result/empty").is_dir()


def test_executable_inputs_and_missing_artifacts(remote, tmp_path, capsys):
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\necho executable\n")
    script.chmod(0o700)
    assert cli.main("shell", ["--send", str(script), "--", "./run.sh"]) == 0
    assert "executable" in capsys.readouterr().out
    assert (
        cli.main(
            "python", ["--fetch", "missing", "--out", str(tmp_path / "out"), "--", "-c", "pass"]
        )
        == 1
    )
    assert "missing artifacts: missing" in capsys.readouterr().err


@pytest.fixture
def tool_binaries(tmp_path):
    source = (
        f"#!{sys.executable}\n"
        + """
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
print(json.dumps({"tool": name, "args": args, "ncu_profile": os.environ.get("NCU_PROFILE")}))
if name == "ncu":
    Path(args[args.index("--export") + 1]).write_bytes(b"NCU report\\x00")
elif name == "run-iket":
    root = Path(args[args.index("--output-dir") + 1])
    root.mkdir()  # The real profiler requires a new output directory.
    (root / "trace.json").write_text('{"traceEvents": []}')
sys.exit(int(os.environ.get("TOOL_EXIT", "0")))
"""
    )
    for name in ("ncu", "run-iket", "compute-sanitizer"):
        path = tmp_path / name
        path.write_text(source)
        path.chmod(0o700)
    return ["-e", f"PATH={tmp_path}{os.pathsep}{os.environ['PATH']}"]


@pytest.mark.parametrize("options", [[], ["--tool", "racecheck"]])
def test_compute_sanitizer_native_arguments(remote, tool_binaries, capsys, options):
    assert (
        cli.main("compute-sanitizer", [*tool_binaries, "--", *options, "python", "check.py"]) == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["args"] == [*options, "python", "check.py"]


@pytest.mark.parametrize("tool", ["ncu", "run-iket"])
def test_profilers_return_reports_even_when_application_fails(
    remote, tool_binaries, tmp_path, capsys, tool
):
    out = tmp_path / "artifacts" / tool
    options = (
        ["--set", "basic", "--launch-count", "1"]
        if tool == "ncu"
        else ["profile", "--postprocess", "json"]
    )
    assert (
        cli.main(
            tool,
            [
                *tool_binaries,
                "-e",
                "TOOL_EXIT=3",
                "--out",
                str(out),
                "--",
                *options,
                "--",
                "python",
                "capture.py",
                "a b",
            ],
        )
        == 3
    )
    output = json.loads(capsys.readouterr().out)
    assert output["args"][-3:] == ["python", "capture.py", "a b"]
    if tool == "ncu":
        assert output["ncu_profile"] == "1"
        assert output["args"][:2] == ["--config-file", "0"]
        assert (out / "capture.ncu-rep").read_bytes() == b"NCU report\x00"
    else:
        assert output["args"][2:7] == ["profile", "--postprocess", "json", "--", "python"]
        assert json.loads((out / "trace.json").read_text()) == {"traceEvents": []}


def test_no_report_and_missing_tool_are_failures(remote, tmp_path, capsys):
    assert cli.main("shell", ["--", "no-such-kcoral-test-executable"]) == 1
    assert "not installed in the remote server environment" in capsys.readouterr().err
    path = tmp_path / "ncu"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o700)
    assert (
        cli.main(
            "ncu",
            [
                "-e",
                f"PATH={tmp_path}",
                "--out",
                str(tmp_path / "out"),
                "--",
                "--",
                "python",
                "capture.py",
            ],
        )
        == 1
    )
    assert "missing artifacts: capture.ncu-rep" in capsys.readouterr().err


def test_truncated_output_signal_and_remote_exception(remote, capsys):
    assert cli.main("python", ["--output-limit-bytes", "8", "--", "-c", "print('x' * 100)"]) == 0
    assert "remote output was truncated" in capsys.readouterr().err
    assert (
        cli.main("python", ["--", "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"])
        == 143
    )
    assert cli.main("python", ["--", "-c", "raise RuntimeError('bad kernel')"]) == 1
    assert "bad kernel" in capsys.readouterr().err


def test_output_collision_does_not_execute(monkeypatch, tmp_path):
    monkeypatch.setenv("KCORAL_URL", "http://unused")
    monkeypatch.setattr(cli, "execute", lambda *args: pytest.fail("must not contact server"))
    assert cli.main("ncu", ["--out", str(tmp_path), "--", "--", "python", "capture.py"]) == 1


def test_return_rejects_symlinks(remote, tmp_path, capsys):
    code = "from pathlib import Path; Path('link').symlink_to('/etc/passwd')"
    assert (
        cli.main("python", ["--fetch", "link", "--out", str(tmp_path / "out"), "--", "-c", code])
        == 1
    )
    assert "symlink" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "tool,args",
    [
        ("python", ["--", "-i", "check.py"]),
        ("python", ["--", "-"]),
        ("python", ["--", "-c"]),
        ("python", ["--send", "x"]),
        ("python", ["-e", "CUDA_VISIBLE_DEVICES=1", "--", "check.py"]),
        ("python", ["-e", "KCORAL_DIR=/tmp", "--", "check.py"]),
        ("python", ["-e", "BAD-NAME=x", "--", "check.py"]),
        ("python", ["--timeout", "0", "--", "check.py"]),
        ("python", ["--fetch", "../x", "--out", "out", "--", "check.py"]),
        ("ncu", ["--out", "out", "--", "--export=x", "--", "python", "capture.py"]),
        ("ncu", ["--out", "out", "--", "-ifile", "--", "python", "capture.py"]),
        ("ncu", ["--out", "out", "--", "--mode=attach", "--", "python", "capture.py"]),
        (
            "run-iket",
            ["--out", "out", "--", "--output-dir=x", "profile", "--", "python", "capture.py"],
        ),
        ("run-iket", ["--out", "out", "--", "profile"]),
    ],
)
def test_invalid_arguments(monkeypatch, tool, args):
    monkeypatch.setenv("KCORAL_URL", "http://unused")
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(tool, args)
    assert exc.value.code == 2


def test_url_and_environment_defaults(monkeypatch):
    monkeypatch.delenv("KCORAL_URL", raising=False)
    with pytest.raises(SystemExit):
        cli.parse_args("python", ["--", "check.py"])
    monkeypatch.setenv("LOCAL_VALUE", "a b")
    args, forwarded = cli.parse_args(
        "python",
        [
            "--url",
            "http://server",
            "-e",
            "LOCAL_VALUE",
            "--",
            "-Wignore",
            "-X",
            "dev",
            "-m",
            "module",
            "-i",
        ],
    )
    assert args.url == "http://server"
    assert args.env == {"LOCAL_VALUE": "a b"}
    assert forwarded[-1] == "-i"


@pytest.mark.parametrize("environment_url", [None, "https://environment.example:9443/prefix"])
@pytest.mark.parametrize(
    "options,expected_url",
    [
        (["--host", "gpu.example", "--port", "9000"], "http://gpu.example:9000"),
        (["--host", "gpu.example"], "http://gpu.example:8000"),
        (["--port", "9000"], "http://127.0.0.1:9000"),
        (["--host", "127.0.0.2", "--port", "65535"], "http://127.0.0.2:65535"),
        (["--host", "::1", "--port", "9000"], "http://[::1]:9000"),
        (["--host", "[::1]", "--port", "1"], "http://[::1]:1"),
    ],
)
def test_explicit_address_overrides_environment(
    monkeypatch, capsys, environment_url, options, expected_url
):
    if environment_url is None:
        monkeypatch.delenv("KCORAL_URL", raising=False)
    else:
        monkeypatch.setenv("KCORAL_URL", environment_url)
    args, forwarded = cli.parse_args("python", [*options, "--", "check.py"])
    assert args.url == expected_url
    assert forwarded == ["check.py"]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"kcoral: warning: --host/--port override KCORAL_URL; using {expected_url}\n"
        if environment_url
        else ""
    )


@pytest.mark.parametrize("explicit_url", [None, "https://explicit.example/prefix"])
def test_url_selection_preserves_native_host_and_port(monkeypatch, capsys, explicit_url):
    monkeypatch.setenv("KCORAL_URL", "https://environment.example/prefix")
    options = ["--url", explicit_url] if explicit_url else []
    native_args = ["check.py", "--host", "native-host", "--port", "invalid-for-kcoral"]
    args, forwarded = cli.parse_args("python", [*options, "--", *native_args])
    assert args.url == (explicit_url or "https://environment.example/prefix")
    assert forwarded == native_args
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "tool,native_args",
    [
        ("python", ["check.py"]),
        ("shell", ["bash", "setup.sh"]),
        ("compute-sanitizer", ["python", "check.py"]),
        ("ncu", ["--set", "basic", "--", "python", "capture.py"]),
        ("run-iket", ["profile", "--", "python", "capture.py"]),
    ],
)
def test_all_tools_accept_connection_flags(monkeypatch, tool, native_args):
    monkeypatch.delenv("KCORAL_URL", raising=False)
    options = ["--host", "gpu.example", "--port", "9000"]
    if tool in {"ncu", "run-iket"}:
        options += ["--out", "artifacts"]
    args, forwarded = cli.parse_args(tool, [*options, "--", *native_args])
    assert args.url == "http://gpu.example:9000"
    assert forwarded == native_args


@pytest.mark.parametrize(
    "options,message",
    [
        (["--port", "0"], "--port must be between 1 and 65535"),
        (["--port", "65536"], "--port must be between 1 and 65535"),
        (["--port", "-1"], "--port must be between 1 and 65535"),
        (["--port", "abc"], "invalid int value"),
        *[
            (["--host", host], "--host must be a hostname or IP address")
            for host in (
                "",
                "http://gpu",
                "gpu:9000",
                "gpu/path",
                "user@gpu",
                "a b",
                "[gpu]",
                "::x",
            )
        ],
        (["--url", "http://gpu", "--host", "other"], "--url cannot be combined"),
        (["--url", "http://gpu", "--port", "9000"], "--url cannot be combined"),
    ],
)
def test_invalid_connection_flags(monkeypatch, capsys, options, message):
    monkeypatch.setenv("KCORAL_URL", "http://environment.example")
    with pytest.raises(SystemExit) as exc:
        cli.parse_args("python", [*options, "--", "check.py"])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_input_snapshots_are_stable_and_reject_conflicts(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    file = source / "run.sh"
    file.write_text("#!/bin/sh\n")
    file.chmod(0o700)
    archive = pack_inputs([source])
    os.utime(file, (1, 1))
    assert pack_inputs([source]) == archive
    out = tmp_path / "out"
    out.mkdir()
    unpack_inputs(archive, out)
    assert (out / "run.sh").stat().st_mode & 0o100
    with pytest.raises(ValueError, match="conflicting"):
        pack_inputs([source, file])
    (source / "link").symlink_to(file)
    with pytest.raises(ValueError, match="symlink"):
        pack_inputs([source])


@pytest.mark.parametrize(
    "name,kind",
    [("../escape", tarfile.REGTYPE), ("/escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)],
)
def test_archive_extraction_rejects_unsafe_members(tmp_path, name, kind):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        archive.addfile(member)
    with pytest.raises(ValueError):
        unpack_inputs(buffer.getvalue(), tmp_path)


def test_help_and_tool_imports_need_no_server_extra():
    code = """
import sys
for name in ('fastapi', 'uvicorn', 'grpc', 'torch', 'tvm', 'tvm_ffi'):
    sys.modules[name] = None
from kcoral.__main__ import main
sys.argv = ['kcoral', 'run', 'ncu', '--help']
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "python")},
    )
    assert result.returncode == 0, result.stderr
    assert "--send" in result.stdout and "--out" in result.stdout


def test_bench_command_roundtrip(remote, monkeypatch, tmp_path, capsys):
    from test_bench_cli import SOURCES

    from kcoral import bench_cli

    evolution = tmp_path / "kernel-evolution"
    evolution.mkdir()
    candidate = evolution / "kda/decode/v0/lowered.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("def setup(x): return x + 1\n")
    (evolution / "bench_adapter.py").write_text(
        "from pathlib import Path\n"
        "KERNEL_EVOLUTION_ROOT = Path(__file__).parent\n"
        "PACKAGED = {'kda/decode': ('test', 1, 1, 'all')}\n"
        "def workload_key(key): return key\n"
        "def plan(task, path, version, **kwargs):\n"
        "    candidate = (path / version / 'lowered.py').read_bytes()\n"
        "    return {'warmup': kwargs['warmup']}, [{'value': 41}, {'value': 42}], candidate\n"
        "def blobs(rows): return [], []\n"
        f"def harness_sources(task): return {SOURCES!r}\n"
    )
    summaries = []
    monkeypatch.setitem(
        sys.modules,
        "flashinfer_bench_evolve.benchmark_common",
        SimpleNamespace(summarize=lambda rows, label: summaries.append(rows)),
    )
    monkeypatch.chdir(tmp_path)
    assert bench_cli.main(["kda/decode", "v0"]) == 0
    assert summaries == [
        [
            {"passed": True, "value": 42, "warmup": 1},
            {"passed": True, "value": 43, "warmup": 1},
        ]
    ]
    assert "workload 2/2" in capsys.readouterr().out
