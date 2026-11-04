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
    out = tmp_path / "runs" / "artifacts"
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


def test_send_shell_keeps_outcome_when_artifact_save_fails(remote, tmp_path, monkeypatch):  # noqa: F811
    from kcoral.artifacts import ReturnedFolder

    def fail_save(self, destination):
        raise OSError("disk full")

    monkeypatch.setattr(ReturnedFolder, "save", fail_save)
    report = agent.send_shell(
        "http://testserver",
        "echo finished; echo diagnostic >&2; echo row > report.txt; exit 3",
        {},
        fetch=["report.txt", "missing.txt"],
        out_dir=str(tmp_path / "runs" / "out"),
    )
    assert report["status"] == "ERROR" and report["returncode"] == 3
    assert report["stdout"] == "finished\n" and report["stderr"] == "diagnostic\n"
    assert report["missing"] == ["missing.txt"] and report["artifacts_dir"] is None
    assert report["error"]["kind"] == "save_failed" and "disk full" in report["error"]["message"]
    assert report["result_errors"]["artifacts"] == report["error"]
    assert report["request_id"]
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


MODULE = """
from pathlib import Path
def main(n, text):
    Path("report").mkdir()
    Path("report/out.txt").write_text(text)
    print("ran", n)
    return {"n": n, "doubled": n * 2}
def raw():
    return b"raw"
def boom():
    raise RuntimeError("boom")
"""


def _program(*extra, data=None):
    return [
        {"op": "upload", "kind": "module", "id": "mod", "source": MODULE},
        {"op": "get_function", "id": "fn", "module": {"$ref": "mod"}, "name": "main"},
        {"op": "run", "id": "out", "fn": {"$ref": "fn"}, "args": [4, "hello"]},
        {"op": "return", "key": "value", "value": {"$ref": "out"}},
        *extra,
    ]


def test_send_program_builds_the_same_instructions_as_the_builder():
    from kcoral import Program, Register

    built = Program()
    for item in _program():
        agent._translate(built, item, None)
    expected = Program()
    module = expected.upload(id="mod", kind="module", source=MODULE)
    fn = expected.get_function(id="fn", module=module, name="main")
    out = expected.run(id="out", fn=fn, args=[4, "hello"])
    expected.return_(key="value", value=out)
    assert built.instructions == expected.instructions
    with pytest.raises(ValueError):
        agent._translate(built, {"op": "run", "fn": {"$ref": "nope"}}, None)
    assert isinstance(agent._register({"$ref": "x"}), Register)


def test_send_program_runs_uploads_and_saves_artifacts(remote, tmp_path):  # noqa: F811
    source = tmp_path / "w.bin"
    source.write_bytes(b"\x00\x01")
    folder = tmp_path / "assets"
    folder.mkdir()
    (folder / "a.txt").write_text("a")
    out = tmp_path / "out"
    program = [
        {"op": "upload", "kind": "folder", "local": str(folder), "path": "assets"},
        {"op": "upload", "kind": "file", "id": "w", "local": str(source), "path": "inputs/w.bin"},
        *_program({"op": "return", "key": "report", "kind": "folder", "path": "report"}),
    ]
    report = agent.send_program("http://testserver", program, out_dir=str(out))
    assert report["status"] == "COMPLETED" and report["error"] is None
    assert report["results"] == {"value": {"n": 4, "doubled": 8}, "report": str(out / "report")}
    assert report["result_errors"] == {} and report["stdout"] == "ran 4\n"
    assert (out / "report/out.txt").read_text() == "hello"
    json.dumps(report)


@pytest.mark.parametrize(
    "index,instruction",
    [
        (0, {"op": "return", "key": "x", "kind": "folder", "path": "report"}),  # no out_dir
        (0, {"op": "upload", "kind": "file", "id": "w", "blob": "ab" * 32, "path": "w"}),
        (0, {"op": "teleport"}),
        (0, {"op": "upload", "kind": "hologram", "id": "x"}),
        (0, {"op": "get_function", "id": "f", "module": {"$ref": "missing"}, "name": "m"}),
        (0, {"op": "run", "id": "r", "fn": {"$ref": "fn"}, "extra": 1}),
    ],
)
def test_send_program_reports_invalid_instructions_with_their_index(index, instruction):
    program = [instruction]
    if index:
        program = [
            {"op": "upload", "kind": "module", "id": "m", "source": "x = 1"}
        ] * index + program
    report = agent.send_program("http://unused", program)
    assert report["status"] == "ERROR"
    assert report["error"]["kind"] == "invalid_program"
    assert report["error"]["instruction_index"] == index
    assert report["error"]["instruction_op"] == instruction["op"]


def test_send_program_indexes_after_a_folder_upload_and_checks_keys(tmp_path):
    folder = tmp_path / "assets"
    folder.mkdir()
    for name in ("a", "b", "c"):
        (folder / name).write_text(name)
    link = tmp_path / "link"
    link.symlink_to(folder / "a")
    out = tmp_path / "out"
    for bad_key in ("../x", "a/b", "/abs", "."):
        program = [
            {"op": "upload", "kind": "folder", "local": str(folder), "path": "assets"},
            *_program({"op": "return", "key": bad_key, "kind": "folder", "path": "report"}),
        ]
        report = agent.send_program("http://unused", program, out_dir=str(out))
        assert report["error"]["kind"] == "invalid_program"
        assert report["error"]["instruction_index"] == 5, bad_key
    program = [
        {"op": "upload", "kind": "folder", "local": str(folder), "path": "assets"},
        {"op": "upload", "kind": "file", "id": "w", "local": str(link), "path": "w"},
    ]
    report = agent.send_program("http://unused", program)
    assert report["error"]["instruction_index"] == 1 and "symlink" in report["error"]["message"]
    assert not out.exists()


def test_send_program_reports_parser_errors_without_an_index():
    program = _program()
    program[2]["args"] = [{"$ref": "ghost"}]
    report = agent.send_program("http://unused", program)
    assert report["status"] == "ERROR" and report["error"]["kind"] == "invalid_program"
    assert report["error"]["instruction_index"] is None
    assert "ghost" in report["error"]["message"]


@pytest.mark.parametrize(
    "index,changes,message",
    [
        *((2, {"args": args}, "args") for args in ({}, "", 0, False, None)),
        (3, {"path": "report.txt"}, "path"),
        (3, {"kind": "file", "path": "report.txt"}, "value"),
        (3, {"kind": "folder", "path": "report"}, "value"),
        (3, {"kind": None}, "kind"),
    ],
)
def test_mcp_rejects_malformed_program_without_submission(
    tmp_path, monkeypatch, index, changes, message
):
    pytest.importorskip("mcp")
    monkeypatch.setattr(
        agent, "_execute", lambda *args: pytest.fail("invalid program was submitted")
    )
    program = _program()
    program[index].update(changes)
    out = tmp_path / "out"
    result = asyncio.run(
        agent.build_server().call_tool(
            "kcoral_send_program",
            {"endpoint": "http://unused", "program": program, "out_dir": str(out)},
        )
    )
    report = json.loads(result.content[0].text)
    assert report["status"] == "ERROR" and report["error"]["kind"] == "invalid_program"
    assert report["error"]["instruction_index"] == index
    assert message in report["error"]["message"]
    assert not out.exists()


def test_send_program_out_dir_handling(remote, tmp_path, monkeypatch):  # noqa: F811
    from kcoral import artifacts

    existing = tmp_path / "existing"
    existing.mkdir()
    report = agent.send_program("http://testserver", _program(), out_dir=str(existing))
    assert report["status"] == "ERROR" and report["error"]["kind"] == "out_dir"
    assert "already exists" in report["error"]["message"]

    def fail(self, destination):
        raise OSError("disk full")

    monkeypatch.setattr(artifacts.ReturnedFolder, "save", fail)
    out = tmp_path / "nested" / "out"
    program = _program({"op": "return", "key": "report", "kind": "folder", "path": "report"})
    report = agent.send_program("http://testserver", program, out_dir=str(out))
    assert out.is_dir()
    assert report["results"] == {"value": {"n": 4, "doubled": 8}}
    assert report["result_errors"]["report"]["kind"] == "save_failed"
    assert "disk full" in report["result_errors"]["report"]["message"]
    assert report["status"] == "ERROR" and report["error"]["kind"] == "save_failed"


def test_send_program_keeps_partial_results_and_flags_unsupported_values(remote):  # noqa: F811
    program = _program(
        {"op": "get_function", "id": "boom", "module": {"$ref": "mod"}, "name": "boom"},
        {"op": "run", "id": "crash", "fn": {"$ref": "boom"}},
        {"op": "return", "key": "never", "value": {"$ref": "crash"}},
    )
    report = agent.send_program("http://testserver", program)
    assert report["status"] == "FAILED" and "boom" in report["error"]["message"]
    assert report["results"] == {"value": {"n": 4, "doubled": 8}} and report["result_errors"] == {}
    program = _program(
        {"op": "get_function", "id": "rawfn", "module": {"$ref": "mod"}, "name": "raw"},
        {"op": "run", "id": "rawout", "fn": {"$ref": "rawfn"}},
        {"op": "return", "key": "raw", "value": {"$ref": "rawout"}},
    )
    report = agent.send_program("http://testserver", program)
    assert report["results"] == {"value": {"n": 4, "doubled": 8}}
    assert report["result_errors"]["raw"]["kind"] == "unsupported_value"
    assert report["status"] == "ERROR" and report["error"]["kind"] == "unsupported_value"
    json.dumps(report)


def test_mcp_reports_local_validation_errors(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setattr(agent, "_execute", lambda *args: pytest.fail("invalid call was submitted"))
    server = agent.build_server()
    missing_upload = {"uploaded_files": {"kernel.py": str(tmp_path / "missing.py")}}
    for tool, invalid, message in [
        ("shell", missing_upload, "input must exist"),
        ("python", missing_upload, "input must exist"),
        ("shell", {"env": {"CUDA_VISIBLE_DEVICES": "1"}}, "managed environment variable"),
        ("python", {"entry_function": "not an identifier"}, "Python identifier"),
    ]:
        arguments = {"endpoint": "http://unused", "uploaded_files": {}}
        if tool == "shell":
            arguments["shell_script"] = "true"
        else:
            arguments.update(python_script="def main(): pass", entry_function="main")
        result = asyncio.run(server.call_tool(f"kcoral_send_{tool}", arguments | invalid))
        report = json.loads(result.content[0].text)
        assert report["status"] == "ERROR" and report["error"]["kind"] == "invalid_arguments"
        assert message in report["error"]["message"], (tool, report)


@pytest.mark.parametrize("existing_parent", [True, False])
def test_mcp_reports_unusable_shell_output_before_submission(
    tmp_path, monkeypatch, existing_parent
):
    pytest.importorskip("mcp")
    monkeypatch.setattr(
        agent, "_execute", lambda *args: pytest.fail("invalid output was submitted")
    )
    occupied = tmp_path / "occupied"
    occupied.write_text("keep")
    out = occupied / "out" if existing_parent else occupied
    result = asyncio.run(
        agent.build_server().call_tool(
            "kcoral_send_shell",
            {
                "endpoint": "http://unused",
                "shell_script": "true",
                "uploaded_files": {},
                "fetch": ["report.txt"],
                "out_dir": str(out),
            },
        )
    )
    report = json.loads(result.content[0].text)
    assert report["status"] == "ERROR" and report["error"]["kind"] == "invalid_arguments"
    assert str(occupied) in report["error"]["message"]
    assert occupied.read_text() == "keep"


def test_mcp_server_exposes_and_calls_the_tools(remote, tmp_path):  # noqa: F811
    pytest.importorskip("mcp")
    server = agent.build_server()
    tools = asyncio.run(server.list_tools())
    assert [tool.name for tool in tools] == [
        "kcoral_send_shell",
        "kcoral_send_python",
        "kcoral_send_program",
    ]
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
    result = asyncio.run(
        server.call_tool(
            "kcoral_send_program", {"endpoint": "http://testserver", "program": _program()}
        )
    )
    assert json.loads(result.content[0].text)["results"] == {"value": {"n": 4, "doubled": 8}}
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
