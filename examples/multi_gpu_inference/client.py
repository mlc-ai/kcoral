"""Upload and run an unchanged whole-model script on a reserved GPU set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kcoral import Client, Program

LAUNCHER = """
def launch(gpu_count, use_torchrun, steps):
    import subprocess
    import sys

    command = [sys.executable]
    if use_torchrun:
        command += [
            "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc-per-node={gpu_count}", "--max-restarts=0",
        ]
    command += ["infer.py", "--steps", str(steps)]
    subprocess.run(command, cwd="job", check=True)
    return "job/result.json"
"""


def build_program(gpu_count: int, *, torchrun: bool = True, steps: int = 3) -> Program:
    program = Program()
    program.upload_file(blob=Path(__file__).with_name("infer.py").read_bytes(), path="job/infer.py")
    module = program.upload(id="launcher", kind="module", source=LAUNCHER)
    launch = program.get_function(id="launch", module=module, name="launch")
    path = program.run(id="inference", fn=launch, args=[gpu_count, torchrun, steps])
    program.return_file(key="report", path=path)
    return program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--single-process", action="store_true")
    args = parser.parse_args()
    with Client(args.url) as client:
        result = client.execute(
            build_program(args.gpus, torchrun=not args.single_process),
            gpu_count=args.gpus,
            timeout_seconds=120,
        )
    if not result.completed:
        raise RuntimeError(f"{result.error}\n{result.stdout}\n{result.stderr}")
    print(f"Allocated physical GPUs: {result.gpu_ids}")
    print(json.loads(result["report"].read_bytes()))


if __name__ == "__main__":
    main()
