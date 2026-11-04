# Tool Calls

Coding agents can use KCoral over MCP to run shell scripts, call Python
functions, or submit a program to a remote worker. The MCP server runs beside
the agent; execution happens on an existing KCoral server or Router.

## Setup

Install MCP support in the agent's environment:

```bash
python -m pip install 'kcoral[client]'
```

Configure the agent to start `kcoral mcp` over standard input and output. For
clients that use an `mcpServers` configuration, the entry is:

```json
{
  "mcpServers": {
    "kcoral": {"command": "kcoral", "args": ["mcp"]}
  }
}
```

Use the full path to `kcoral` if it is not on the agent's `PATH`. Supply the
remote server URL as `endpoint` in each tool call. The worker needs the
libraries and executables your experiment uses.

## Choose a tool

| Tool | Use it to |
| --- | --- |
| `kcoral_send_shell` | Run a shell script, benchmark, or profiler and retrieve files. |
| `kcoral_send_python` | Call a function from Python source and return a JSON-compatible value. |
| `kcoral_send_program` | Submit an instruction list for a request that needs several modules, tensor uploads, or file returns. |

Every call starts in a fresh working directory. Upload everything it needs
each time; files and Python state do not persist between calls.
`uploaded_files` maps remote relative paths to files or directories on the
machine running the MCP server. Directories are uploaded recursively under
the specified remote path. Symlinks are not supported.

## Run a shell script

Call `kcoral_send_shell` with these arguments to upload a local `experiment`
directory, run its script, and retrieve a report:

```json
{
  "endpoint": "http://127.0.0.1:8000",
  "shell_script": "python experiment/check.py",
  "uploaded_files": {"experiment": "./experiment"},
  "fetch": ["results/report.json"],
  "out_dir": "./runs/001",
  "timeout_seconds": 300
}
```

The script runs with `bash -c` from the remote working directory. In this
example, `check.py` must write `results/report.json`; the returned file is
saved as `runs/001/results/report.json`. Use `fetch` and `out_dir` together,
and choose a new output directory for each call.

An optional `env` object sets subprocess environment variables. Use it, or an
explicit interpreter in the script, when a benchmark needs its own runtime.
`CUDA_VISIBLE_DEVICES` and `KCORAL_DIR` are managed by KCoral.

## Call a Python function

Call `kcoral_send_python` with source, a function name, and positional arguments:

```json
{
  "endpoint": "http://127.0.0.1:8000",
  "python_script": "def main(n):\n    return {'value': n + 1}\n",
  "entry_function": "main",
  "uploaded_files": {},
  "args": [41]
}
```

The result's `value` is `{"value": 42}`. The function runs in the KCoral
worker's Python environment. Return JSON-compatible values; convert tensors
with `.tolist()` or `float()`. Uploaded source runs as a module without
`__file__`; refer to uploaded files by their relative paths.

## Submit a program

`kcoral_send_program` takes `endpoint` and a `program` list using the
[protocol instructions](protocol.md): `upload`, `get_function`, `run`, and
`return`. Reference earlier instruction IDs with `{"$ref": "id"}`.

For uploads, provide module `source`, or a `local` file path instead of a
wire-protocol `blob`. A folder upload uses `kind: "folder"`, `local`, and
its remote `path`. File and folder returns require a new `out_dir` and are
saved under `out_dir/<return key>`. Returned values and saved paths appear
in `results`; individual return failures appear in `result_errors`.

## Check the result

All tools return `status`, captured output, and an `error` on failure.
`COMPLETED` means the program finished, not that a benchmark passed. For
shell calls, check `returncode`, `missing`, and `result_errors`, then inspect
the benchmark's correctness results. Output can be truncated; retrieve files
when you need complete reports.

`timeout_seconds` defaults to 300 and is capped by the server. A timeout or
transport failure may prevent results from returning. For detailed file and
execution behavior, see [Builtin CLI Tools](builtin-cli-tools.md).
