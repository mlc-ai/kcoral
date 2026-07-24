"""Multi-file package uploads over the full stack (HTTP -> spawned worker)."""

import os

import pytest
from fastapi.testclient import TestClient

from benchmark_server.app import create_app
from benchmark_server.client import upload_package
from benchmark_server.config import ServerConfig
from benchmark_server.keys import compute_key
from benchmark_server.testing import fake_runtime_factory


def make_client():
    return TestClient(create_app(runtime_factory=fake_runtime_factory))


def program(package, args):
    return {
        "instructions": [
            package,
            {"id": "y", "op": "run", "fn": {"$ref": package["id"]}, "args": args},
        ]
    }


def test_package_entry_imports_sibling_module():
    files = {
        "pkg/__init__.py": "",
        "pkg/main.py": "from pkg.helper import triple\ndef main(x):\n    return triple(x)\n",
        "pkg/helper.py": "def triple(x):\n    return 3 * x\n",
    }
    package = upload_package("pkg", files, "pkg/main.py:main")
    with make_client() as c:
        data = c.post("/benchmark", json=program(package, [14])).json()
    assert data["status"] == "COMPLETED"
    assert data["results"][1]["value"] == 42


def test_package_reads_bundled_data_file():
    files = {
        "reader.py": (
            "from pathlib import Path\n"
            "def main():\n"
            "    return (Path(__file__).parent / 'data.txt').read_text()\n"
        ),
        "data.txt": "hello from the package",
    }
    package = upload_package("pkg", files, "reader.py:main")
    with make_client() as c:
        data = c.post("/benchmark", json=program(package, [])).json()
    assert data["status"] == "COMPLETED"
    assert data["results"][1]["value"] == "hello from the package"


def test_same_module_name_is_isolated_across_requests():
    # The worker persists between requests; a second package reusing the same
    # module path must not see the first one's module.
    def versioned(value):
        files = {"mod.py": f"VALUE = {value}\ndef main():\n    return VALUE\n"}
        return upload_package("pkg", files, "mod.py:main")

    with make_client() as c:
        first = c.post("/benchmark", json=program(versioned(1), [])).json()
        second = c.post("/benchmark", json=program(versioned(2), [])).json()
    assert first["results"][1]["value"] == 1
    assert second["results"][1]["value"] == 2


def test_traversal_path_is_rejected():
    files = {"../evil.py": "def main():\n    return 0\n"}
    package = upload_package("pkg", files, "../evil.py:main")
    with make_client() as c:
        data = c.post("/benchmark", json=program(package, [])).json()
    assert data["status"] == "FAILED"
    result = data["results"][0]
    assert result["status"] == "FAILED" and result["error"]["kind"] == "parse"
    assert "unsafe" in result["error"]["message"]


def test_entry_must_be_among_files():
    files = {"mod.py": "def main():\n    return 0\n"}
    package = upload_package("pkg", files, "other.py:main")
    with make_client() as c:
        data = c.post("/benchmark", json=program(package, [])).json()
    assert data["results"][0]["status"] == "FAILED"
    assert data["results"][0]["error"]["kind"] == "parse"


def test_broken_import_fails_and_cleans_up():
    files = {"mod.py": "import does_not_exist_anywhere\ndef main():\n    return 0\n"}
    package = upload_package("pkg", files, "mod.py:main")
    with make_client() as c:
        data = c.post("/benchmark", json=program(package, [])).json()
        assert data["results"][0]["error"]["kind"] == "parse"
        # the worker stays healthy for the next request
        retry = upload_package("ok", {"mod.py": "def main():\n    return 7\n"}, "mod.py:main")
        data = c.post("/benchmark", json=program(retry, [])).json()
    assert data["status"] == "COMPLETED" and data["results"][1]["value"] == 7


def test_package_key_is_order_independent():
    files_a = {"a.py": "def main():\n    return 1\n", "b.py": "x = 2\n"}
    files_b = {"b.py": "x = 2\n", "a.py": "def main():\n    return 1\n"}
    key_a = compute_key("package", {"files": files_a, "entry": "a.py:main"})
    key_b = compute_key("package", {"files": files_b, "entry": "a.py:main"})
    assert key_a == key_b


# --- real tensors through a package on the GPU --------------------------------


@pytest.mark.skipif(
    os.environ.get("BENCH_GPU_TEST") != "1",
    reason="package-on-GPU e2e; set BENCH_GPU_TEST=1 with a GPU to run",
)
def test_package_runs_on_gpu_tensors():
    import numpy

    from benchmark_server.client import upload_tensor
    from benchmark_server.gpu_runtime import gpu_runtime_factory

    gpu_id_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    gpu_id = int(gpu_id_raw) if gpu_id_raw.isdigit() else 0
    files = {
        "kernels/__init__.py": "",
        "kernels/scale.py": (
            "from kernels.util import factor\ndef main(a):\n    return a * factor()\n"
        ),
        "kernels/util.py": "def factor():\n    return 4\n",
    }
    x = numpy.arange(32, dtype=numpy.float32)
    body = {
        "instructions": [
            upload_package("pkg", files, "kernels/scale.py:main"),
            upload_tensor("x", x),
            upload_tensor("expected", x * 4),
            {"id": "y", "op": "run", "fn": {"$ref": "pkg"}, "args": [{"$ref": "x"}]},
            {
                "id": "chk",
                "op": "run",
                "fn": "builtin.check_close",
                "args": [{"$ref": "y"}, {"$ref": "expected"}],
            },
        ],
        "options": {"timeout_seconds": 120},
    }
    app = create_app(ServerConfig(gpus=[gpu_id]), runtime_factory=gpu_runtime_factory)
    with TestClient(app) as c:
        data = c.post("/benchmark", json=body).json()
    assert data["status"] == "COMPLETED"
    results = {r["id"]: r for r in data["results"]}
    assert results["chk"]["value"]["passed"] and results["chk"]["value"]["max_abs_err"] == 0.0
