"""Bubblewrap filesystem isolation for trusted, optionally reused workers.

Each process has one private writable /work. The parent clears its contents
between requests; it never mounts the directory containing other workspaces.
This does not turn arbitrary code in a shared interpreter into mutually
untrusted principals.
"""

from __future__ import annotations

import gc
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType

WORKSPACE = "/work"
PRIVATE = ".kcoral"
_active = False
_DIRECTORIES = (
    "tmp",
    "home",
    "cache",
    "cuda",
    "triton",
    "extensions",
    "libraries",
    "output",
    "shm",
)


def active() -> bool:
    return _active


def activate() -> None:
    """Called before runtime imports in the sandbox child."""
    global _active
    _active = True
    tempfile.tempdir = f"{WORKSPACE}/{PRIVATE}/tmp"


def _environment() -> dict[str, str]:
    private = f"{WORKSPACE}/{PRIVATE}"
    return {
        "HOME": f"{private}/home",
        "TMPDIR": f"{private}/tmp",
        "TMP": f"{private}/tmp",
        "TEMP": f"{private}/tmp",
        "XDG_CACHE_HOME": f"{private}/cache",
        "CUDA_CACHE_PATH": f"{private}/cuda",
        "TRITON_CACHE_DIR": f"{private}/triton",
        "TORCH_EXTENSIONS_DIR": f"{private}/extensions",
        "TVM_FFI_CACHE_DIR": f"{private}/cache/tvm-ffi",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


class Sandbox:
    """Parent-owned storage and launch policy for a single worker generation."""

    def __init__(self, readonly_paths: tuple[Path, ...] = ()) -> None:
        if sys.platform != "linux":
            raise ValueError("bubblewrap isolation requires Linux")
        self.executable = shutil.which("bwrap")
        if self.executable is None:
            raise ValueError("bubblewrap isolation requires bwrap on PATH; install bubblewrap")
        self.readonly_paths = tuple(Path(p).absolute() for p in readonly_paths)
        for path in self.readonly_paths:
            if not path.exists():
                raise ValueError(f"sandbox read-only path does not exist: {path}")
        self._directory = tempfile.TemporaryDirectory(prefix="kcoral-sandbox-")
        self.root = Path(self._directory.name)
        self.workspace = self.root / "work"
        self.workspace.mkdir(mode=0o700)
        self.prepare()

    def prepare(self) -> None:
        """Empty the existing bind mount without replacing its root inode."""
        for entry in self.workspace.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                entry.unlink()
            else:
                shutil.rmtree(entry)
        for name in _DIRECTORIES:
            (self.workspace / PRIVATE / name).mkdir(parents=True, mode=0o700)

    def close(self) -> None:
        self._directory.cleanup()

    def command(self, gpu_id: int | None) -> list[str]:
        command = [
            self.executable,
            "--unshare-all",
            "--unshare-user",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--disable-userns",
        ]
        package_path = Path(__file__).absolute().parent.parent
        paths = [
            Path(p)
            for p in (
                "/usr",
                "/bin",
                "/sbin",
                "/lib",
                "/lib64",
                "/sys",
                "/etc/ld.so.cache",
                "/etc/ld.so.conf",
                "/etc/ld.so.conf.d",
                "/etc/alternatives",
                "/etc/localtime",
            )
            if Path(p).exists()
        ]
        paths += [Path(sys.prefix), Path(sys.base_prefix), package_path]
        paths += list(self.readonly_paths)
        seen: set[Path] = set()
        for path in paths:
            path = path.absolute()
            resolved = path.resolve()
            if path in seen:
                continue
            # A runtime mount must never reveal the parent of private workspaces.
            if resolved == Path("/") or self.root.resolve().is_relative_to(resolved):
                raise ValueError(f"read-only runtime path contains sandbox workspaces: {path}")
            seen.add(path)
            command += ["--ro-bind", str(resolved), str(path)]
        command += ["--bind", str(self.workspace), WORKSPACE, "--proc", "/proc", "--dir", "/dev"]
        for name in ("null", "zero", "random", "urandom", "full"):
            command += ["--dev-bind", f"/dev/{name}", f"/dev/{name}"]
        if gpu_id is not None:
            devices = [
                Path(f"/dev/nvidia{gpu_id}"),
                Path("/dev/nvidiactl"),
                Path("/dev/nvidia-uvm"),
                Path("/dev/nvidia-uvm-tools"),
            ]
            # Bind device nodes individually: a writable host directory here
            # would also permit ordinary files outside the workspace.
            devices.extend(Path("/dev/nvidia-caps").glob("nvidia-cap*"))
            for path in devices:
                if path.exists():
                    command += ["--dev-bind", str(path), str(path)]
        for target, link in (
            (f"{WORKSPACE}/{PRIVATE}/tmp", "/tmp"),
            (f"{WORKSPACE}/{PRIVATE}/tmp", "/var/tmp"),
            (f"{WORKSPACE}/{PRIVATE}/shm", "/dev/shm"),
            ("/proc/self/fd", "/dev/fd"),
            ("/proc/self/fd/0", "/dev/stdin"),
            ("/proc/self/fd/1", "/dev/stdout"),
            ("/proc/self/fd/2", "/dev/stderr"),
        ):
            command += ["--symlink", target, link]
        python_paths = [str(package_path), *(str(p) for p in self.readonly_paths if p.is_dir())]
        environment = {
            **_environment(),
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONNOUSERSITE": "1",
        }
        for key, value in environment.items():
            command += ["--setenv", key, value]
        # NVIDIA's driver can fail CUDA initialization with error 304 when the
        # private proc mount is read-only. It contains kernel interfaces, not
        # ordinary writable files; bubblewrap still masks sensitive proc nodes.
        if gpu_id is None:
            command += ["--remount-ro", "/proc"]
        command += [
            "--chdir",
            WORKSPACE,
            "--remount-ro",
            "/",
            "--",
            sys.executable,
            "-m",
            "kcoral.sandbox_worker",
        ]
        return command


class SandboxProcess:
    """The small multiprocessing.Process surface used by Worker.

    The duplex control socket is passed as stdin, which bubblewrap preserves.
    The child duplicates it before replacing stdin with /dev/null. No host file
    or directory descriptors are inherited. A PID namespace also contains
    descendants that create new process groups.
    """

    def __init__(self, sandbox: Sandbox, child, gpu_id: int | None) -> None:
        self._output = bytearray()
        self._started = threading.Event()
        self._start_error: BaseException | None = None
        command = sandbox.command(gpu_id)
        # Linux parent-death signals follow the *creating thread*. Pool
        # replacement threads are short-lived, so launch from a thread that
        # stays alive for the whole child lifetime.
        self._reader = threading.Thread(
            target=self._launch, args=(command, child.fileno()), daemon=True
        )
        self._reader.start()
        self._started.wait()
        if self._start_error is not None:
            raise self._start_error
        self.pid = self._process.pid

    def _launch(self, command: list[str], control_fd: int) -> None:
        try:
            self._process = subprocess.Popen(
                command,
                stdin=control_fd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        except BaseException as exc:
            self._start_error = exc
        finally:
            self._started.set()
        if self._start_error is None:
            self._drain()
            self._process.wait()

    def _drain(self) -> None:
        assert self._process.stdout is not None
        with self._process.stdout as stream:
            while data := stream.read1(4096):
                self._output.extend(data)
                del self._output[:-8192]

    def output_tail(self) -> str:
        return bytes(self._output).decode("utf-8", errors="replace")

    @property
    def exitcode(self) -> int | None:
        return self._process.poll()

    def is_alive(self) -> bool:
        return self._process.poll() is None

    def join(self, timeout: float | None = None) -> None:
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return
        self._reader.join(timeout=1)

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()


class RequestState:
    """Restore workspace imports and detect resources unsafe to carry forward.

    This is lifecycle hygiene for trusted code, not an adversarial interpreter
    reset. Any detected surviving resource retires the process instead of making
    the next request's files available to it.
    """

    def __init__(self) -> None:
        self.path = sys.path[:]
        self.threads = set(threading.enumerate())
        self.native_threads = set(os.listdir("/proc/self/task"))

    def finish(self) -> None:
        sys.path[:] = self.path
        for name, module in list(sys.modules.items()):
            if not isinstance(module, ModuleType):
                continue
            # Do not invoke module-level __getattr__: torch.classes, for
            # example, synthesizes objects for names such as __path__.
            path = vars(module).get("__file__")
            locations = vars(module).get("__path__", ())
            if (isinstance(path, str) and path.startswith(WORKSPACE + "/")) or any(
                isinstance(p, str) and p.startswith(WORKSPACE + "/") for p in locations
            ):
                sys.modules.pop(name, None)
        for path in list(sys.path_importer_cache):
            if path.startswith(WORKSPACE + "/"):
                sys.path_importer_cache.pop(path, None)
        importlib.invalidate_caches()
        gc.collect()
        if set(threading.enumerate()) - self.threads:
            raise RuntimeError("sandbox request left background threads running")
        if threads := set(os.listdir("/proc/self/task")) - self.native_threads:
            names = [Path(f"/proc/self/task/{tid}/comm").read_text().strip() for tid in threads]
            raise RuntimeError(f"sandbox request left native threads running: {names}")
        # /proc is private to this worker. PID 1 is bubblewrap's reaper.
        own_pid = os.getpid()
        if any(
            p.name.isdigit() and int(p.name) not in (1, own_pid) for p in Path("/proc").iterdir()
        ):
            raise RuntimeError("sandbox request left child processes running")
        for entry in Path("/proc/self/fd").iterdir():
            try:
                target = os.readlink(entry)
            except FileNotFoundError:
                continue
            if target == WORKSPACE or target.startswith(WORKSPACE + "/"):
                raise RuntimeError(f"sandbox request left workspace file open: {target}")
        for line in Path("/proc/self/maps").read_text().splitlines():
            if WORKSPACE + "/" in line:
                raise RuntimeError(f"sandbox request left workspace file mapped: {line}")
