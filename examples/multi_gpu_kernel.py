"""Run a library multi-GPU kernel in ONE process, locally or through KCoral.

Local:  CUDA_VISIBLE_DEVICES=0,1 python examples/multi_gpu_kernel.py --local
Remote: python examples/multi_gpu_kernel.py --url http://localhost:8000 --gpus 2
Server: kcoral --gpus 0,1 --workers-per-gpu 1

This one file contains the kernel call, correctness checks, and client Program.
PyTorch's single-process NCCL interface launches precompiled all-reduce kernels
that sum the input tensors across GPUs and place the sum on every GPU. The
execution machine needs 2-8 GPUs and PyTorch built with NCCL support. No separate
compiler, process group, or torchrun launcher is needed by the script.
"""

import argparse
import json

SCRIPT = """
import json
import os
from pathlib import Path
import torch
from torch.cuda import nccl

count = torch.cuda.device_count()
if not 2 <= count <= 8:
    raise RuntimeError("this example needs 2-8 visible GPUs")
elements, iterations = 65539, 3
base = (torch.arange(elements, dtype=torch.float32) % 257) / 1024
inputs = [torch.empty(elements, device=f"cuda:{i}") for i in range(count)]
outputs = [torch.empty_like(x) for x in inputs]
if not nccl.is_available(inputs):
    raise RuntimeError("PyTorch must be built with NCCL support")

for step in range(iterations):
    for rank, x in enumerate(inputs):
        x.copy_(base).add_(rank + 1 + 8 * step)

    nccl.all_reduce(inputs, outputs)  # The multi-GPU library kernel call.

    for device in range(count):
        torch.cuda.synchronize(device)
    expected = count * base + count * (count + 1) / 2 + count * 8 * step
    for output in outputs:
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

report = {
    "ok": True, "kernel": "nccl_all_reduce", "gpu_count": count,
    "gpu_processes": 1, "pids": [os.getpid()],
    "elements": elements, "iterations": iterations,
    "checked_elements": count * elements * iterations,
}
Path("result.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
"""


def build_program():
    from kcoral import Program

    program = Program()
    program.upload(id="kernel", kind="module", source=SCRIPT)
    program.return_file(key="report", path="result.json")
    return program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--gpus", type=int, choices=range(2, 9), default=2)
    parser.add_argument("--local", action="store_true", help="use all visible local GPUs")
    args = parser.parse_args()
    if args.local:
        exec(SCRIPT, {})
        return

    from kcoral import Client

    with Client(args.url) as client:
        result = client.execute(build_program(), gpu_count=args.gpus, timeout_seconds=120)
    if not result.completed:
        raise RuntimeError(f"{result.error}\n{result.stdout}\n{result.stderr}")
    print(f"Allocated physical GPUs: {result.gpu_ids}")
    print(json.loads(result["report"].read_bytes()))


if __name__ == "__main__":
    main()
