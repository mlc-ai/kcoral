"""Authenticated Modal server for ``kcoral server --backend modal``."""

from __future__ import annotations

import threading
import time

import modal

from kcoral import __version__

app = modal.App("kcoral-modal-worker")
image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("bubblewrap")
    .uv_sync(extras=["server", "modal"], frozen=True, uv_version="0.11.30")
    .add_local_dir("python", "/opt/kcoral/python", copy=True)
    .add_local_file("pyproject.toml", "/opt/kcoral/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/opt/kcoral/uv.lock", copy=True)
    .add_local_file("setup.py", "/opt/kcoral/setup.py", copy=True)
    .add_local_file("README.md", "/opt/kcoral/README.md", copy=True)
    .uv_pip_install(
        "/opt/kcoral",
        extra_options="--no-deps",
        uv_version="0.11.30",
        env={"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_KCORAL": __version__},
    )
    .workdir("/opt/kcoral")
)


@app.server(
    image=image,
    gpu="B200",
    cpu=4,
    memory=16384,
    port=8000,
    unauthenticated=False,
    min_containers=1,
    max_containers=1,
    max_concurrency=2,
    startup_timeout=300,
    exit_grace_period=960,
)
class ModalWorker:
    @modal.enter()
    def start(self) -> None:
        import uvicorn

        from kcoral.config import ServerConfig
        from kcoral.server.app import ShutdownServer, create_app

        app = create_app(
            ServerConfig(
                gpus=[0],
                workers_per_gpu=1,
                max_requests_per_worker=1,
                worker_wait_timeout_seconds=0,
                cache_capacity_bytes=0,
                disk_cache_capacity_mbytes=0,
                sandbox_required=True,
            )
        )
        self._server = ShutdownServer(uvicorn.Config(app, host="0.0.0.0", port=8000), app)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 300
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                self._server.should_exit = True
                raise RuntimeError("KCoral Modal worker failed to start")
            time.sleep(0.05)

        def watch_pool():
            while self._thread.is_alive():
                if app.state.pool.closing:
                    # Stop listening so Modal replaces the container.
                    self._server.should_exit = True
                    return
                time.sleep(0.1)

        threading.Thread(target=watch_pool, daemon=True).start()

    @modal.exit()
    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(30)
        if self._thread.is_alive():
            raise RuntimeError("KCoral Modal worker did not finish draining")
