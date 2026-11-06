"""Executable substitutes for exercising allocation ownership and transport."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

state = Path(os.environ["SLURM_TEST_STATE"])
command = Path(sys.argv[0]).name
arguments = sys.argv[1:]

if command == "ssh":
    if os.environ.get("SLURM_TEST_SSH_FAILURE"):
        raise SystemExit(255)
    if os.environ.get("SLURM_TEST_DISCONNECT"):
        script = sys.stdin.read()
        if "exec srun " in script:
            process = subprocess.Popen(
                ["sh", "-s"], stdin=subprocess.PIPE, text=True, start_new_session=True
            )
            with process.stdin:
                process.stdin.write(script)
            deadline = time.monotonic() + 5
            while not (state / "started.json").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            raise SystemExit(255)
        raise SystemExit(subprocess.run(["sh", "-s"], input=script, text=True).returncode)
    os.execvp("sh", ["sh", "-s"])
elif command == "srun":
    job_name = next(
        argument.split("=", 1)[1] for argument in arguments if argument.startswith("--job-name=")
    )
    (state / job_name).write_text(str(os.getpid()))
    (state / "started.json").write_text(
        json.dumps(
            {
                "job_name": job_name,
                "arguments": arguments,
                "token": os.environ.get("KCORAL_NODE_TOKEN"),
            }
        )
    )
    if os.environ["SLURM_TEST_PHASE"] == "exit":
        raise SystemExit(7)
    while True:
        signal.pause()
elif command == "scancel":
    job_name = arguments[0].split("=", 1)[1]
    if (state / job_name).exists():
        try:
            os.kill(int((state / job_name).read_text()), signal.SIGTERM)
        except ProcessLookupError:
            pass
    (state / job_name).unlink(missing_ok=True)
    (state / "canceled.json").write_text(json.dumps(arguments))
