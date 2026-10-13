import os
import subprocess
import sys
from pathlib import Path

import pytest

PYTHON_DIR = Path(__file__).resolve().parent.parent / "python"

# Everything the `server`, `gpu` and `compiler` installs add. A client install
# has none of them, so importing the client must not reach for any.
SERVER_ONLY = (
    "fastapi",
    "starlette",
    "pydantic",
    "uvicorn",
    "torch",
    "tvm",
    "tvm_ffi",
    "grpc",
    "google.protobuf",
)

CHECK_CLIENT_IS_THIN = f"""
import sys

from kcoral import Client, Program, Register

reached = sorted(name for name in {SERVER_ONLY!r} if name in sys.modules)
print(",".join(reached))
"""


def run_isolated(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(PYTHON_DIR)},
        capture_output=True,
        text=True,
    )


def test_client_import_pulls_no_server_dependency():
    """`pip install kcoral` installs the client alone, so a fresh interpreter
    importing it must load nothing that only the extras provide."""
    result = run_isolated(CHECK_CLIENT_IS_THIN)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_create_app_still_reachable():
    """Deferring it must not remove it from the package's surface."""
    import kcoral

    assert kcoral.create_app is not None
    assert "create_app" in kcoral.__all__


def test_unknown_attribute_still_raises():
    import kcoral

    with pytest.raises(AttributeError):
        kcoral.no_such_symbol


def test_cli_reports_a_missing_server_extra():
    """Without the extra the front-end import fails; the message has to name the
    fix instead of surfacing a bare ModuleNotFoundError."""
    code = """
import sys

sys.modules["uvicorn"] = None  # force the ImportError a client-only install gives

from kcoral.__main__ import main

sys.argv = ["kcoral", "server"]
main()
"""
    result = run_isolated(code)
    assert result.returncode != 0
    assert "kcoral[server]" in result.stderr


def _extras():
    tomllib = pytest.importorskip("tomllib")
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    return tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]


def _resolve(extras, name):
    """Requirement names an extra installs, following kcoral[...] self-references."""
    names = set()
    for requirement in extras[name]:
        base = requirement.split(";")[0].strip()
        if base.startswith("kcoral["):
            for inner in base[len("kcoral[") : -1].split(","):
                names |= _resolve(extras, inner.strip())
            continue
        names.add(base.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip().lower())
    return names


@pytest.mark.parametrize("extra", ["rocm", "server-common"])
def test_rocm_extras_never_install_torch_or_nvidia_packages(extra):
    """pip cannot pick a ROCm torch wheel from an extra; any torch here would
    replace the ROCm build already installed."""
    names = _resolve(_extras(), extra)
    assert "torch" not in names
    assert not {name for name in names if name.startswith(("nvidia", "cupti", "flashinfer"))}


def test_server_extra_still_installs_the_cuda_server():
    names = _resolve(_extras(), "server")
    assert {"torch", "fastapi", "apache-tvm-ffi", "cupti-python"} <= names
    assert names == _resolve(_extras(), "cuda")
