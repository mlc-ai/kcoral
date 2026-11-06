"""Keep a routed KCoral node running in one foreground Slurm allocation."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import socket
import subprocess
import sys
import uuid

from kcoral.server import cli
from kcoral.support.cuda import visible_device_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kcoral slurm",
        allow_abbrev=False,
        description="Allocate one Slurm GPU node and connect it to an existing Router.",
        epilog="Pass server options after --. Ctrl+C cancels even a pending allocation.",
    )
    parser.add_argument("--router", default=os.environ.get("KCORAL_ROUTER_ENDPOINT"))
    parser.add_argument(
        "--ssh", metavar="HOST", help="run Slurm commands through an SSH login host"
    )
    parser.add_argument("--node-id", help="Router node identifier; defaults to a unique job name")
    parser.add_argument("--account")
    parser.add_argument("--partition")
    parser.add_argument("--qos")
    parser.add_argument("--constraint")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs to allocate (default: 1)")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", default="32G", help="memory per node (default: 32G)")
    parser.add_argument(
        "--time", default="04:00:00", help="allocation time limit (default: 04:00:00)"
    )
    parser.add_argument("--python", default="python3", help="Python executable on the compute node")
    parser.add_argument("--container-image", help="Pyxis container image or Enroot .sqsh path")
    parser.add_argument("--container-mount", action="append", default=[], metavar="SOURCE:TARGET")
    parser.add_argument("--container-workdir", help="working directory inside the container")
    parser.add_argument("server_args", nargs=argparse.REMAINDER, help="server options after --")
    return parser


def _server_arguments(args: argparse.Namespace, job_name: str) -> list[str]:
    extra = args.server_args[1:] if args.server_args[:1] == ["--"] else args.server_args
    owned = {"--router", "--node-id", "--node-token", "--device", "--gpus"}
    for argument in extra:
        if argument.split("=", 1)[0] in owned:
            raise ValueError(f"{argument.split('=', 1)[0]} is managed by the Slurm launcher")
    arguments = [
        f"--router={args.router}",
        f"--node-id={args.node_id or job_name}",
        "--device=gpu",
        "--host=127.0.0.1",
        *extra,
    ]
    cli.config_from_args(cli.server_parser().parse_args([*arguments, "--gpus=0"]))
    return arguments


def allocation_command(args: argparse.Namespace, job_name: str) -> list[str]:
    command = [
        "srun",
        f"--job-name={job_name}",
        "--nodes=1",
        "--ntasks=1",
        f"--gpus-per-node={args.gpus}",
        "--gpu-bind=none",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem={args.mem}",
        f"--time={args.time}",
        "--export=ALL",
        "--kill-on-bad-exit=1",
    ]
    for name in ("account", "partition", "qos", "constraint"):
        value = getattr(args, name)
        if value:
            command.append(f"--{name}={value}")
    if args.container_image:
        command.extend([f"--container-image={args.container_image}", "--no-container-mount-home"])
        if args.container_mount:
            command.append("--container-mounts=" + ",".join(args.container_mount))
        if args.container_workdir:
            command.append(f"--container-workdir={args.container_workdir}")
    command.extend(
        [
            "--",
            args.python,
            "-m",
            "kcoral.server.slurm",
            str(args.gpus),
            *_server_arguments(args, job_name),
        ]
    )
    return command


def _ssh_command(host: str) -> list[str]:
    return [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "--",
        host,
        "sh -s",
    ]


def _remote_script(
    command: list[str], *, token: str | None = None, allocation: bool = False
) -> str:
    script = ""
    if allocation:
        script += (
            'if [ -n "${SLURM_JOB_ID:-${SLURM_JOBID:-}}" ]; then\n'
            "  echo 'run this launcher outside an existing Slurm allocation' >&2\n"
            "  exit 2\n"
            "fi\n"
        )
    if token is not None:
        script += "export KCORAL_NODE_TOKEN=" + shlex.quote(token) + "\n"
    return script + "exec " + shlex.join(command) + "\n"


def _cancel(host: str | None, job_name: str) -> None:
    command = ["scancel", f"--name={job_name}"]
    try:
        result = subprocess.run(
            _ssh_command(host) if host else command,
            input=_remote_script(command) if host else "",
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"exit status {result.returncode}")
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        location = f" on {host}" if host else ""
        print(
            f"warning: could not cancel {job_name}: {exc}; run scancel --name={job_name}{location}",
            file=sys.stderr,
        )


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except ProcessLookupError:
        process.wait()
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_allocation(command: list[str], *, host: str | None, job_name: str) -> int:
    interrupted = 0

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = signum

    previous = {
        signum: signal.signal(signum, stop)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    process = None
    try:
        process = subprocess.Popen(
            _ssh_command(host) if host else command,
            stdin=subprocess.PIPE if host else subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        if host:
            try:
                with process.stdin:
                    process.stdin.write(
                        _remote_script(
                            command, token=os.environ.get("KCORAL_NODE_TOKEN"), allocation=True
                        )
                    )
            except BrokenPipeError:
                pass
        while not interrupted:
            try:
                status = process.wait(timeout=0.25)
                return status if status >= 0 else 128 - status
            except subprocess.TimeoutExpired:
                pass
        return 128 + interrupted
    finally:
        try:
            if process is not None:
                try:
                    _stop_process(process)
                finally:
                    _cancel(host, job_name)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def slurm_main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.router:
        parser.error("--router or KCORAL_ROUTER_ENDPOINT is required")
    if args.gpus < 1 or args.cpus_per_task < 1:
        parser.error("--gpus and --cpus-per-task must be positive")
    if (args.container_mount or args.container_workdir) and not args.container_image:
        parser.error("container mounts and workdir require --container-image")
    if args.ssh is not None and (not args.ssh.strip() or args.ssh.startswith("-")):
        parser.error("--ssh must name a login host")
    if not args.ssh and (os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")):
        parser.error("run this launcher on a login host, outside an existing Slurm allocation")
    job_name = "kcoral-" + uuid.uuid4().hex
    try:
        command = allocation_command(args, job_name)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Slurm job {job_name}: waiting for an allocation; Ctrl+C cancels it", file=sys.stderr)
    try:
        return run_allocation(command, host=args.ssh, job_name=job_name)
    except OSError as exc:
        parser.exit(1, f"cannot launch Slurm: {exc}\n")


def node_main(argv: list[str]) -> None:
    """Resolve the allocation's GPUs on the compute node, then start its supervisor."""
    expected_gpus = int(argv[0])
    arguments = argv[1:]
    parser = cli.server_parser()
    parser.set_defaults(port=0)
    args = parser.parse_args(arguments)
    try:
        devices = visible_device_ids()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if len(devices) != expected_gpus:
        raise SystemExit(f"requested {expected_gpus} GPUs but CUDA can see {len(devices)}")
    port = args.port
    if not port:
        family, kind, protocol, _, address = socket.getaddrinfo(
            args.host, 0, type=socket.SOCK_STREAM
        )[0]
        with socket.socket(family, kind, protocol) as listener:
            listener.bind(address)
            port = listener.getsockname()[1]
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    cli.server_main([*arguments, "--gpus=" + ",".join(map(str, devices)), f"--port={port}"])


if __name__ == "__main__":
    node_main(sys.argv[1:])
