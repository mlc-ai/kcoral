"""Call the KCoral MCP tools from Python, the way a coding agent does.

The script starts ``kcoral mcp`` as a subprocess and talks to it over standard
input and output, exactly as an MCP-enabled agent would. It lists the tools,
then uses each one against a running KCoral server: ``kcoral_run_python``
probes the worker, ``kcoral_run_shell`` uploads ``check.py``, runs it, and
fetches its report, and ``kcoral_run_program`` returns both a value and a
folder. Install KCoral first: ``pip install kcoral``.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import Client, StdioServerParameters

HERE = Path(__file__).resolve().parent

PROBE = """
import torch

def probe(n):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.arange(n, dtype=torch.float32, device=device) + 1
    name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    return {"device": name, "sum": float(x.sum())}
"""

SUMMARIZE = """
import json
from pathlib import Path

def summarize(values):
    Path("report").mkdir()
    Path("report/summary.json").write_text(json.dumps({"count": len(values), "max": max(values)}))
    return sum(values) / len(values)
"""


def report_of(result) -> dict:
    """Every tool returns one JSON report, as structured content and as text."""
    return result.structured_content or json.loads(result.content[0].text)


class ToolFailed(Exception):
    """A tool call that did not complete, or a script that failed inside one."""


def completed(report: dict, step: str) -> dict:
    if report["status"] != "COMPLETED":
        raise ToolFailed(f"{step} {report['status']}: {report['error']}")
    return report


async def main(endpoint: str, out_dir: Path) -> int:
    # The same command an agent's "mcpServers" entry runs: kcoral mcp.
    server = StdioServerParameters(command=sys.executable, args=["-m", "kcoral", "mcp"])
    async with Client(server) as client:
        try:
            await call_each_tool(client, endpoint, out_dir)
        except ToolFailed as failure:
            print(failure, file=sys.stderr)
            return 1
    return 0


async def call_each_tool(client: Client, endpoint: str, out_dir: Path) -> None:
    tools = await client.list_tools()
    print("tools:", ", ".join(tool.name for tool in tools.tools))

    # Call one function of a Python script and get its JSON value back.
    arguments = {
        "endpoint": endpoint,
        "python_script": PROBE,
        "entry_function": "probe",
        "uploaded_files": {},
        "args": [4],
    }
    result = await client.call_tool("kcoral_run_python", arguments)
    print("python:", completed(report_of(result), "python")["value"])

    # Run a shell script beside an uploaded file and fetch what it writes.
    arguments = {
        "endpoint": endpoint,
        "shell_script": "mkdir results && python experiment/check.py > results/check.json",
        "uploaded_files": {"experiment/check.py": str(HERE / "check.py")},
        "fetch": ["results/check.json"],
        "out_dir": str(out_dir / "shell"),
    }
    result = await client.call_tool("kcoral_run_shell", arguments)
    report = completed(report_of(result), "shell")
    if report["returncode"] != 0:
        raise ToolFailed(f"check.py exited with {report['returncode']}:\n{report['stderr']}")
    print("shell:", (out_dir / "shell/results/check.json").read_text().strip())

    # Submit a program: several instructions, returning a value and a folder.
    arguments = {
        "endpoint": endpoint,
        "program": [
            {"op": "upload", "kind": "module", "id": "mod", "source": SUMMARIZE},
            {"op": "get_function", "id": "fn", "module": {"$ref": "mod"}, "name": "summarize"},
            {"op": "run", "id": "mean", "fn": {"$ref": "fn"}, "args": [[1, 2, 3, 6]]},
            {"op": "return", "key": "mean", "value": {"$ref": "mean"}},
            {"op": "return", "key": "report", "kind": "folder", "path": "report"},
        ],
        "out_dir": str(out_dir / "program"),
    }
    result = await client.call_tool("kcoral_run_program", arguments)
    print("program:", completed(report_of(result), "program")["results"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("KCORAL_URL", "http://127.0.0.1:8000"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(tempfile.mkdtemp(prefix="kcoral-mcp-")),
        help="Fetched files go under here; each tool call needs a new directory of its own.",
    )
    args = parser.parse_args()
    # Saving refuses to follow symlinks, and macOS places temporary files under one (/var).
    out_dir = args.out_dir.resolve()
    status = asyncio.run(main(args.endpoint, out_dir))
    print("fetched files are under", out_dir)
    sys.exit(status)
