"""Slurm launch, transport and cancellation behavior without a cluster or GPU."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kcoral.__main__ import main
from kcoral.server import slurm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("KCORAL_", "SLURM_")):
            monkeypatch.delenv(name)


def test_command_allocates_one_node_and_forwards_server_options():
    args = slurm.build_parser().parse_args(
        [
            "--router",
            "http://router:9000",
            "--account",
            "team",
            "--partition",
            "gpu",
            "--qos",
            "normal",
            "--constraint",
            "h100",
            "--gpus",
            "2",
            "--cpus-per-task",
            "4",
            "--mem",
            "64G",
            "--time",
            "01:00:00",
            "--python",
            "/opt/env with spaces/bin/python",
            "--node-id",
            "node-a",
            "--",
            "--workers-per-gpu",
            "2",
            "--port=8123",
            "--log-dir",
            "logs with spaces",
        ]
    )
    command = slurm.allocation_command(args, "kcoral-test")
    assert command[:2] == ["srun", "--job-name=kcoral-test"]
    for option in (
        "--nodes=1",
        "--ntasks=1",
        "--gpus-per-node=2",
        "--gpu-bind=none",
        "--account=team",
        "--partition=gpu",
        "--qos=normal",
        "--constraint=h100",
        "--mem=64G",
        "--time=01:00:00",
        "--cpus-per-task=4",
    ):
        assert option in command
    boundary = command.index("/opt/env with spaces/bin/python")
    assert command[boundary + 1 : boundary + 4] == ["-m", "kcoral.server.slurm", "2"]
    assert command[-5:] == [
        "--workers-per-gpu",
        "2",
        "--port=8123",
        "--log-dir",
        "logs with spaces",
    ]
    assert "--node-id=node-a" in command


def test_pyxis_options_are_opt_in():
    parser = slurm.build_parser()
    plain = parser.parse_args(["--router", "http://router:9000"])
    assert not any("container" in argument for argument in slurm.allocation_command(plain, "job"))
    args = parser.parse_args(
        [
            "--router",
            "http://router:9000",
            "--container-image",
            "/images/worker.sqsh",
            "--container-mount",
            "/shared/cache:/cache",
            "--container-mount",
            "/shared/logs:/logs",
            "--container-workdir",
            "/workspace",
        ]
    )
    command = slurm.allocation_command(args, "job")
    assert "--container-image=/images/worker.sqsh" in command
    assert "--container-mounts=/shared/cache:/cache,/shared/logs:/logs" in command
    assert "--container-workdir=/workspace" in command
    assert "--no-container-mount-home" in command


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--gpus=0"],
        ["--cpus-per-task=-1"],
        ["--container-mount=/a:/b"],
        ["--container-workdir=/a"],
        ["--ssh=-bad"],
        ["--", "--gpus=0"],
        ["--", "--device=cpu"],
        ["--", "--node-token=secret"],
        ["--", "--node-id=another"],
        ["--", "--router=http://other"],
        ["--", "--port=70000"],
        ["--", "--workers-per-gpu=0"],
    ],
)
def test_invalid_arguments_do_not_allocate(monkeypatch, arguments):
    monkeypatch.setattr(slurm, "run_allocation", lambda *args, **kwargs: pytest.fail("allocated"))
    with pytest.raises(SystemExit):
        slurm.slurm_main((["--router=http://router:9000"] if arguments else []) + arguments)


def test_rejects_nested_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    with pytest.raises(SystemExit, match="2"):
        slurm.slurm_main(["--router=http://router:9000"])


def test_root_dispatch_environment_defaults_and_unique_job_names(monkeypatch):
    calls = []
    monkeypatch.setenv("KCORAL_ROUTER_ENDPOINT", "http://router:9000")
    monkeypatch.setattr(
        slurm, "run_allocation", lambda command, **kwargs: calls.append((command, kwargs)) or 7
    )
    for _ in range(2):
        with pytest.raises(SystemExit) as raised:
            main(["slurm"])
        assert raised.value.code == 7
    assert calls[0][1]["job_name"] != calls[1][1]["job_name"]
    for command, options in calls:
        assert "--router=http://router:9000" in command
        assert "--node-id=" + options["job_name"] in command


def test_ssh_stdin_preserves_arguments_and_keeps_token_out_of_argv(tmp_path):
    arguments = ["spaces and 'quotes'", "line1\nline2", "$(touch should-not-exist)", "`id`"]
    token = "secret with 'quotes'\nand $expansion"
    command = [
        sys.executable,
        "-c",
        "import json, os, sys; print(json.dumps([sys.argv[1:], os.environ['KCORAL_NODE_TOKEN']]))",
        *arguments,
    ]
    remote = slurm._ssh_command("user@login")
    assert remote[-2:] == ["user@login", "sh -s"]
    assert token not in " ".join(remote)
    result = subprocess.run(
        ["sh", "-s"],
        input=slurm._remote_script(command, token=token),
        text=True,
        capture_output=True,
        check=True,
        cwd=tmp_path,
    )
    assert json.loads(result.stdout) == [arguments, token]
    assert not (tmp_path / "should-not-exist").exists()


@pytest.mark.parametrize("variable", ["SLURM_JOB_ID", "SLURM_JOBID"])
def test_remote_rejects_nested_allocation(monkeypatch, variable):
    monkeypatch.setenv(variable, "42")
    result = subprocess.run(
        ["sh", "-s"],
        input=slurm._remote_script(["exit", "99"], allocation=True),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "outside an existing Slurm allocation" in result.stderr


def test_node_resolves_gpus_and_selects_a_local_port(monkeypatch):
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(slurm, "visible_device_ids", lambda: [5, 2])
    monkeypatch.setattr(slurm.cli, "server_main", calls.append)
    slurm.node_main(["2", "--router=http://router:9000", "--node-id=job"])
    config = slurm.cli.config_from_args(slurm.cli.server_parser().parse_args(calls[0]))
    assert config.gpus == [5, 2]
    assert config.node_id == "job"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert 0 < int(calls[0][-1].split("=")[1]) <= 65535


def test_node_preserves_explicit_port_and_rejects_allocation_mismatch(monkeypatch):
    calls = []
    monkeypatch.setattr(slurm, "visible_device_ids", lambda: [3])
    monkeypatch.setattr(slurm.cli, "server_main", calls.append)
    with pytest.raises(SystemExit, match="requested 2 GPUs but CUDA can see 1"):
        slurm.node_main(["2", "--router=http://router:9000", "--node-id=job"])
    assert not calls
    slurm.node_main(["1", "--router=http://router:9000", "--node-id=job", "--port=8123"])
    assert calls[0][-1] == "--port=8123"


@pytest.fixture
def fake_cluster(tmp_path, monkeypatch):
    source = (ROOT / "tests/support/slurm.py").read_text()
    for name in ("srun", "scancel", "ssh"):
        executable = tmp_path / name
        executable.write_text(f"#!{sys.executable}\n" + source)
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("SLURM_TEST_STATE", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "python"))
    (tmp_path / "unrelated-job").write_text("leave running")
    return tmp_path


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("phase", ["pending", "running", "hangup", "exit"])
def test_allocation_lifecycle_cancels_only_its_job(fake_cluster, monkeypatch, remote, phase):
    monkeypatch.setenv("SLURM_TEST_PHASE", phase)
    monkeypatch.setenv("KCORAL_NODE_TOKEN", "node-secret")
    arguments = [sys.executable, "-m", "kcoral", "slurm", "--router=http://router:9000"]
    if remote:
        arguments += ["--ssh=login"]
    process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        started = fake_cluster / "started.json"
        while not started.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.02)
        assert started.exists()
        if phase != "exit":
            process.send_signal(
                {"pending": signal.SIGINT, "running": signal.SIGTERM, "hangup": signal.SIGHUP}[
                    phase
                ]
            )
        stdout, stderr = process.communicate(timeout=15)
        assert (
            process.returncode == {"pending": 130, "running": 143, "hangup": 129, "exit": 7}[phase]
        ), stderr
        allocation = json.loads(started.read_text())
        canceled = json.loads((fake_cluster / "canceled.json").read_text())
        assert canceled == ["--name=" + allocation["job_name"]]
        assert not (fake_cluster / allocation["job_name"]).exists()
        assert (fake_cluster / "unrelated-job").exists()
        assert allocation["token"] == "node-secret"
        assert "node-secret" not in stdout + stderr
        assert "node-secret" not in " ".join(allocation["arguments"])
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_transport_failure_preserves_status_and_reports_cleanup_failure(
    fake_cluster, monkeypatch, capsys
):
    monkeypatch.setenv("SLURM_TEST_SSH_FAILURE", "1")
    assert slurm.slurm_main(["--router=http://router:9000", "--ssh=login"]) == 255
    assert "could not cancel" in capsys.readouterr().err
    assert not (fake_cluster / "started.json").exists()


def test_missing_srun_restores_signal_handlers(monkeypatch, capsys):
    before = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setenv("PATH", "")
    with pytest.raises(SystemExit) as raised:
        slurm.slurm_main(["--router=http://router:9000"])
    assert raised.value.code == 1
    assert "cannot launch Slurm" in capsys.readouterr().err
    assert before == {signum: signal.getsignal(signum) for signum in before}


def test_ssh_disconnect_after_allocation_still_cancels_job(fake_cluster, monkeypatch):
    monkeypatch.setenv("SLURM_TEST_PHASE", "running")
    monkeypatch.setenv("SLURM_TEST_DISCONNECT", "1")
    result = subprocess.run(
        [sys.executable, "-m", "kcoral", "slurm", "--router=http://router:9000", "--ssh=login"],
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 255, result.stderr
    allocation = json.loads((fake_cluster / "started.json").read_text())
    assert not (fake_cluster / allocation["job_name"]).exists()
    assert (fake_cluster / "canceled.json").exists()
    assert (fake_cluster / "unrelated-job").exists()


def test_cancel_runs_even_when_local_process_cleanup_fails(monkeypatch):
    calls = []

    def fail(process):
        raise OSError("process cleanup failed")

    monkeypatch.setattr(slurm, "_stop_process", fail)
    monkeypatch.setattr(slurm, "_cancel", lambda host, job_name: calls.append((host, job_name)))
    with pytest.raises(OSError, match="cleanup failed"):
        slurm.run_allocation([sys.executable, "-c", "pass"], host=None, job_name="owned-job")
    assert calls == [(None, "owned-job")]
