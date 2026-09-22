"""Uploaded command runner; uses only the Python standard library on the worker."""

import io
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath


def relative_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"expected a relative file path: {name!r}")
    return path


def unpack_inputs(archive, workdir):
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar:
            path = relative_path(member.name)
            if not member.isfile():
                raise ValueError(f"input is not a regular file: {member.name!r}")
            destination = workdir.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            destination.chmod(0o700 if member.mode & 0o111 else 0o600)


def collect_files(workdir, reports, paths):
    missing = []
    for name in paths:
        source = workdir.joinpath(*relative_path(name).parts)
        # Check every ancestor before traversing or reading the selection.
        current = workdir
        for part in source.relative_to(workdir).parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"artifact is a symlink: {current}")
        if not source.exists():
            missing.append(name)
            continue
        entries = [source, *sorted(source.rglob("*"))] if source.is_dir() else [source]
        for entry in entries:
            mode = entry.lstat().st_mode
            destination = reports / entry.relative_to(workdir)
            if stat.S_ISDIR(mode):
                destination.mkdir(parents=True, exist_ok=True)
            elif stat.S_ISREG(mode):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(entry, destination)
            else:
                raise ValueError(f"artifact is not a regular file or directory: {entry}")
    return missing


def run(archive, tool_run, arguments, overrides, fetch):
    workdir = Path("inputs").absolute()
    reports = Path("outputs").absolute()
    workdir.mkdir()
    unpack_inputs(archive, workdir)
    env = {**os.environ, **overrides, "KCORAL_DIR": str(workdir)}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")

    def execute(command, *, create_reports=True):
        if create_reports:
            reports.mkdir(exist_ok=True)
        target = command[0]
        if "/" in target and not os.path.isabs(target):
            target = str(workdir / target)
        executable = shutil.which(target, path=env["PATH"])
        if executable is None:
            raise RuntimeError(f"{command[0]} is not installed in the remote server environment")
        # Stay in the worker's process group so its timeout also kills subprocesses.
        return subprocess.run(
            [executable, *command[1:]],
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
        ).returncode

    outcome = tool_run(arguments, env, reports, execute)
    reports.mkdir(exist_ok=True)
    outcome["missing"].extend(collect_files(workdir, reports, fetch))
    return outcome
