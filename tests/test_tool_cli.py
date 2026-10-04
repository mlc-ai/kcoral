"""Shared options and remote execution for the built-in commands."""

import io
import json
import os
import subprocess
import sys
import tarfile
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from support.runtime import fake_runtime_factory

from kcoral import Client
from kcoral import client as client_module
from kcoral.config import ServerConfig
from kcoral.server.app import create_app
from kcoral.tools import cli
from kcoral.tools.cli import pack_inputs
from kcoral.tools.runner import unpack_inputs


def isolated_tool_runtime_factory(python_executable):
    # The runner searches beside Python before PATH. Keep that directory under
    # test control so installed profilers cannot shadow the fake executables.
    sys.executable = str(python_executable)
    return fake_runtime_factory()


@pytest.fixture
def remote(monkeypatch, tmp_path):
    python_executable = tmp_path / "python"
    python_executable.symlink_to(sys.executable)
    runtime_factory = partial(isolated_tool_runtime_factory, python_executable)
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


def test_shared_options_and_native_argument_boundary(monkeypatch, capsys):
    monkeypatch.setenv("KCORAL_URL", "http://environment")
    monkeypatch.setenv("LOCAL_VALUE", "copied")
    args, forwarded = cli.get_tool("python").parse_args(
        "--host gpu.example --port 9000 -e LOCAL_VALUE -e MODE=debug "
        "--send experiment --output-limit-bytes 524288 -- check.py --host native".split()
    )
    assert args.url == "http://gpu.example:9000"
    assert args.env == {"LOCAL_VALUE": "copied", "MODE": "debug"}
    assert args.send == [Path("experiment")] and args.output_limit_bytes == 524288
    assert forwarded == ["check.py", "--host", "native"]
    warning = capsys.readouterr().err
    assert "override KCORAL_URL" in warning and args.url in warning


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("python", "-- -i check.py"),
        ("python", "--output-limit-bytes 0 -- check.py"),
        ("python", "-e CUDA_VISIBLE_DEVICES=0 -- check.py"),
        ("ncu", "--out reports -- --export=custom -- python check.py"),
        ("run-iket", "--out reports -- profile"),
    ],
)
def test_invalid_tool_options(monkeypatch, tool, arguments):
    monkeypatch.setenv("KCORAL_URL", "http://unused")
    with pytest.raises(SystemExit) as error:
        cli.get_tool(tool).parse_args(arguments.split())
    assert error.value.code == 2


def test_python_upload_output_and_exit_status(remote, tmp_path, capsys):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "value.txt").write_text("uploaded")
    (experiment / "check.py").write_text(
        "import os, sys\nfrom pathlib import Path\n"
        "assert Path(os.environ['KCORAL_DIR']) == Path.cwd()\n"
        "print(Path('experiment/value.txt').read_text(), os.environ['MODE'], sys.argv[1:])\n"
        "print('remote stderr', file=sys.stderr)\nsys.exit(7)\n"
    )
    assert (
        cli.run_main(
            [
                "python",
                "--send",
                str(experiment),
                "-e",
                "MODE=debug",
                "--",
                "experiment/check.py",
                "--url",
                "literal",
            ]
        )
        == 7
    )
    output = capsys.readouterr()
    assert output.out == "uploaded debug ['--url', 'literal']\n"
    assert output.err == "remote stderr\n"


def test_shell_returns_files_on_failure(remote, tmp_path):
    directory = tmp_path / "experiment"
    directory.mkdir()
    script = directory / "setup.sh"
    script.write_text(
        "#!/bin/sh\nmkdir -p result/empty\nprintf '\\000\\377' > result/data.bin\nexit 9\n"
    )
    script.chmod(0o700)
    out = tmp_path / "artifacts"
    assert (
        cli.main(
            "shell",
            [
                "--send",
                str(directory),
                "--fetch",
                "result",
                "--out",
                str(out),
                "--",
                "./experiment/setup.sh",
            ],
        )
        == 9
    )
    assert (out / "result/data.bin").read_bytes() == b"\x00\xff"
    assert (out / "result/empty").is_dir()


@pytest.mark.parametrize("tool", ["compute-sanitizer", "ncu", "run-iket"])
def test_native_tools_and_profiler_reports(remote, tmp_path, capsys, tool):
    executable = tmp_path / tool
    executable.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
print(json.dumps(args))
if Path(sys.argv[0]).name == 'ncu':
    assert os.environ['NCU_PROFILE'] == '1'
    Path(args[args.index('--export') + 1]).write_bytes(b'report')
elif Path(sys.argv[0]).name == 'run-iket':
    out = Path(args[args.index('--output-dir') + 1])
    out.mkdir()  # IKET requires a directory that does not already exist.
    (out / 'trace.json').write_text('{}')
sys.exit(3)
"""
    )
    executable.chmod(0o700)
    out = tmp_path / "artifacts"
    options = ["-e", f"PATH={tmp_path}"]
    native = {
        "compute-sanitizer": ["--tool", "racecheck"],
        "ncu": ["--set", "basic", "--"],
        "run-iket": ["profile", "--"],
    }[tool]
    if tool != "compute-sanitizer":
        options += ["--out", str(out)]
    assert cli.main(tool, [*options, "--", *native, "python", "check.py", "a b"]) == 3
    arguments = json.loads(capsys.readouterr().out)
    assert arguments[-3:] == ["python", "check.py", "a b"]
    if tool == "compute-sanitizer":
        assert arguments[:2] == ["--tool", "racecheck"]
    elif tool == "ncu":
        assert arguments[:2] == ["--config-file", "0"]
        assert (out / "capture.ncu-rep").read_bytes() == b"report"
    else:
        assert (out / "trace.json").read_text() == "{}"


def test_capture_and_failure_paths(remote, tmp_path, capsys):
    assert (
        cli.main("python", ["--output-limit-bytes", "524288", "--", "-c", "print('x' * 600000)"])
        == 0
    )
    output = capsys.readouterr()
    assert len(output.out) == 524288 and "truncated" in output.err
    assert (
        cli.main("python", ["--", "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"])
        == 143
    )
    assert cli.main("shell", ["--", "no-such-kcoral-executable"]) == 1
    assert (
        cli.main(
            "python", ["--fetch", "missing", "--out", str(tmp_path / "out"), "--", "-c", "pass"]
        )
        == 1
    )
    assert "missing artifacts" in capsys.readouterr().err


def test_upload_names_permissions_and_path_validation(tmp_path):
    directory = tmp_path / "experiment"
    directory.mkdir()
    script = directory / "run.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o700)
    snapshot = pack_inputs([directory, script])
    os.utime(script, (1, 1))
    assert pack_inputs([directory, script]) == snapshot
    out = tmp_path / "out"
    out.mkdir()
    unpack_inputs(snapshot, out)
    assert (out / "experiment/run.sh").read_text() == (out / "run.sh").read_text()
    assert (out / "experiment/run.sh").stat().st_mode & 0o100
    with pytest.raises(ValueError, match="conflicting"):
        pack_inputs([directory, directory])
    (directory / "link").symlink_to(script)
    with pytest.raises(ValueError, match="symlink"):
        pack_inputs([directory])
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.addfile(tarfile.TarInfo("../escape"))
    with pytest.raises(ValueError):
        unpack_inputs(buffer.getvalue(), out)


def test_all_tool_help_needs_only_client_dependencies():
    code = """
import sys
for name in ('fastapi', 'uvicorn', 'grpc', 'torch', 'tvm', 'tvm_ffi'):
    sys.modules[name] = None
from kcoral.tools import COMMANDS
from kcoral.tools.cli import run_main
for tool in COMMANDS:
    try:
        run_main([tool, '--help'])
    except SystemExit as error:
        assert error.code == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "python")},
    )
    assert result.returncode == 0, result.stderr
    assert "--output-limit-bytes" in result.stdout
