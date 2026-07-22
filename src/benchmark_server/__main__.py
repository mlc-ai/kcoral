"""Run the benchmark server: ``python -m benchmark_server``.

Launch it where a TIRX-enabled tvm is importable — either pip-installed
(``pip install apache-tvm``) or a from-source build (put its Python tree on
``PYTHONPATH`` and point ``TVM_LIBRARY_PATH`` at the built library directory). The
front-end process itself touches no GPU; each worker process imports torch/tvm
and owns one GPU.

    BENCH_GPUS=1,2,3 python -m benchmark_server

Environment:
  BENCH_GPUS   comma-separated physical GPU ids the workers pin (default "0")
  BENCH_HOST   bind host (default 127.0.0.1)
  BENCH_PORT   bind port (default 8000)
"""

from __future__ import annotations

import os

from .app import create_app
from .config import ServerConfig
from .gpu_runtime import gpu_runtime_factory


def _gpus() -> list[int]:
    raw = os.environ.get("BENCH_GPUS", "0")
    return [int(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    import uvicorn

    app = create_app(ServerConfig(gpus=_gpus()), runtime_factory=gpu_runtime_factory)
    uvicorn.run(
        app,
        host=os.environ.get("BENCH_HOST", "127.0.0.1"),
        port=int(os.environ.get("BENCH_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
