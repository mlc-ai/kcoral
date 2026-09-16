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

sys.argv = ["kcoral"]
main()
"""
    result = run_isolated(code)
    assert result.returncode != 0
    assert "kcoral[server]" in result.stderr
