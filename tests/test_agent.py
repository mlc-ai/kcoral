"""The shell tool call an agent gets over MCP, sharing the CLI's program builder."""

import asyncio
import json

import pytest
from test_tool_cli import remote  # noqa: F401

from kcoral import agent
from kcoral.tools.cli import pack_inputs
from kcoral.tools.runner import unpack_inputs


def test_pack_inputs_uploads_under_a_given_name(tmp_path):
    source = tmp_path / "local.txt"
    source.write_text("x")
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "inner.py").write_text("y")
    archive = pack_inputs([("solution/kernel.py", source), ("dataset", folder), source])
    unpack_inputs(archive, tmp_path / "remote")
    assert (tmp_path / "remote/solution/kernel.py").read_text() == "x"
    assert (tmp_path / "remote/dataset/inner.py").read_text() == "y"
    assert (tmp_path / "remote/local.txt").read_text() == "x"


def test_send_shell_uploads_runs_and_fetches(remote, tmp_path):  # noqa: F811
    kernel = tmp_path / "kernel.py"
    kernel.write_text("print('kernel')")
    data = tmp_path / "data"
    data.mkdir()
    (data / "w.jsonl").write_text("{}")
    out = tmp_path / "artifacts"
    report = agent.send_shell(
        "http://testserver",
        "cat solution/kernel.py\nls dataset\necho $MODE\nmkdir bench_results\n"
        "echo row > bench_results/r.jsonl\necho 'remote stderr' >&2\nexit 3\n",
        {"solution/kernel.py": str(kernel), "dataset": str(data)},
        env={"MODE": "fast"},
        fetch=["bench_results", "benchmark.csv"],
        out_dir=str(out),
    )
    assert report["status"] == "COMPLETED" and report["returncode"] == 3
    assert report["stdout"] == "print('kernel')w.jsonl\nfast\n"
    assert report["stderr"] == "remote stderr\n"
    assert report["missing"] == ["benchmark.csv"]
    assert report["artifacts_dir"] == str(out)
    assert (out / "bench_results/r.jsonl").read_text() == "row\n"
    assert report["error"] is None and report["request_id"]
    json.dumps(report)


def test_send_shell_reports_remote_failure_without_raising(remote, tmp_path):  # noqa: F811
    report = agent.send_shell("http://testserver", "echo ok", {}, timeout_seconds=1)
    assert report["status"] == "COMPLETED" and report["returncode"] == 0
    # A symlinked artifact fails collection after the script ran: no outcome, an error.
    report = agent.send_shell(
        "http://testserver",
        "ln -s /etc/passwd leak",
        {},
        fetch=["leak"],
        out_dir=str(tmp_path / "out"),
    )
    assert report["status"] == "FAILED" and report["returncode"] is None
    assert "symlink" in report["error"]["message"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"env": {"CUDA_VISIBLE_DEVICES": "1"}},
        {"env": {"bad name": "1"}},
        {"fetch": ["outputs"]},
        {"uploaded_files": {"../escape": "."}},
    ],
)
def test_send_shell_rejects_invalid_arguments(tmp_path, kwargs):
    arguments = {"uploaded_files": {}, **kwargs}
    with pytest.raises(ValueError):
        agent.send_shell("http://unused", "true", **arguments)


def test_send_python_runs_a_function_with_uploads_and_args(remote, tmp_path):  # noqa: F811
    data = tmp_path / "data"
    data.mkdir()
    (data / "weights.txt").write_text("3")
    report = agent.send_python(
        "http://testserver",
        "from pathlib import Path\n"
        "def main(x, label):\n"
        "    print('computing', label)\n"
        "    return {'label': label, 'sum': x + int(Path('inputs/weights.txt').read_text())}\n",
        "main",
        {"inputs": str(data)},
        args=[4, "demo"],
    )
    assert report["status"] == "COMPLETED"
    assert report["value"] == {"label": "demo", "sum": 7}
    assert report["stdout"] == "computing demo\n" and report["error"] is None
    json.dumps(report)


def test_send_python_reports_failures_without_raising(remote, tmp_path):  # noqa: F811
    report = agent.send_python(
        "http://testserver", "def main():\n    raise RuntimeError('boom')\n", "main", {}
    )
    assert report["status"] == "FAILED" and report["value"] is None
    assert "boom" in report["error"]["message"]
    report = agent.send_python("http://testserver", "def main():\n    return b'raw'\n", "main", {})
    assert report["status"] == "ERROR" and report["value"] is None
    assert report["error"]["kind"] == "unsupported_value" and "bytes" in report["error"]["message"]
    json.dumps(report)


@pytest.mark.parametrize("name", ["", "not an identifier", "1abc"])
def test_send_python_rejects_invalid_entry_function(name):
    with pytest.raises(ValueError):
        agent.send_python("http://unused", "def main(): pass", name, {})


def test_send_python_rejects_unsafe_uploads(tmp_path):
    source = tmp_path / "file.txt"
    source.write_text("x")
    link = tmp_path / "link.txt"
    link.symlink_to(source)
    for uploads in ({"../escape": str(source)}, {"/abs": str(source)}, {"ok.txt": str(link)}):
        with pytest.raises(ValueError):
            agent.send_python("http://unused", "def main(): pass", "main", uploads)


def test_mcp_server_exposes_and_calls_the_tools(remote, tmp_path):  # noqa: F811
    pytest.importorskip("mcp")
    server = agent.build_server()
    tools = asyncio.run(server.list_tools())
    assert [tool.name for tool in tools] == ["kcoral_send_shell", "kcoral_send_python"]
    assert set(tools[0].input_schema["required"]) == {"endpoint", "shell_script", "uploaded_files"}
    assert set(tools[1].input_schema["required"]) == {
        "endpoint",
        "python_script",
        "entry_function",
        "uploaded_files",
    }
    assert "nothing persists between calls" in tools[0].description.lower()
    result = asyncio.run(
        server.call_tool(
            "kcoral_send_python",
            {
                "endpoint": "http://testserver",
                "python_script": "def main(n):\n    return [n, n * 2]\n",
                "entry_function": "main",
                "uploaded_files": {},
                "args": [21],
            },
        )
    )
    assert json.loads(result.content[0].text)["value"] == [21, 42]
    kernel = tmp_path / "kernel.py"
    kernel.write_text("x")
    arguments = {
        "endpoint": "http://testserver",
        "shell_script": "cat solution/kernel.py",
        "uploaded_files": {"solution/kernel.py": str(kernel)},
    }
    result = asyncio.run(server.call_tool("kcoral_send_shell", arguments))
    report = json.loads(result.content[0].text)
    assert report["returncode"] == 0 and report["stdout"] == "x"
