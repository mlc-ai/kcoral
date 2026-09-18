"""Run a library multi-GPU kernel with ONE process PER GPU.

Local:  CUDA_VISIBLE_DEVICES=0,1 python examples/multi_gpu_kernel_multiprocess.py --local
Remote: python examples/multi_gpu_kernel_multiprocess.py --url http://localhost:8000 --gpus 2
Server: kcoral --gpus 0,1 --workers-per-gpu 1

This single file contains the kernel call, process setup, checks, and client
Program. Each worker selects its GPU, joins a communication group, and invokes
NCCL all-reduce through torch.distributed. A CPU controller starts and joins the
workers; no external launcher is needed. Requires PyTorch with NCCL and 2-8 GPUs.
This demonstrates the per-GPU process model. SGLang's custom kernels using
inter-process GPU memory sharing are outside this example.
"""

import argparse
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path


def worker(rank, count, rendezvous):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=count,
        device_id=torch.device("cuda", rank),
        timeout=timedelta(seconds=90),
    )
    try:
        elements, iterations = 65539, 3
        base = (torch.arange(elements, dtype=torch.float32) % 257) / 1024
        value = torch.empty(elements, device="cuda")
        for step in range(iterations):
            value.copy_(base).add_(rank + 1 + 8 * step)
            dist.all_reduce(value)  # Every worker calls the library kernel.
            torch.cuda.synchronize()
            expected = count * base + count * (count + 1) / 2 + count * 8 * step
            torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)

        info = {"rank": rank, "pid": os.getpid(), "device": torch.cuda.current_device()}
        print(json.dumps(info), flush=True)
        ranks = [None] * count
        dist.all_gather_object(ranks, info)
        if rank == 0:
            report = {
                "ok": True,
                "kernel": "nccl_all_reduce",
                "gpu_count": count,
                "gpu_processes": count,
                "pids": [entry["pid"] for entry in ranks],
                "ranks": ranks,
                "elements": elements,
                "iterations": iterations,
                "checked_elements": count * elements * iterations,
            }
            Path("result.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


def run_local():
    import torch
    import torch.multiprocessing as mp

    count = torch.cuda.device_count()
    if not 2 <= count <= 8:
        raise RuntimeError("this example needs 2-8 visible GPUs")
    with tempfile.TemporaryDirectory(prefix="nccl-rendezvous-") as directory:
        rendezvous = (Path(directory) / "store").as_uri()
        # A private file coordinates startup without a shared, fixed TCP port.
        mp.spawn(worker, args=(count, rendezvous), nprocs=count, join=True)


def build_program():
    from kcoral import Program

    program = Program()
    # spawn needs an importable worker and a real __main__ entry point.
    program.upload_file(path="kernel.py", blob=Path(__file__).read_bytes())
    program.upload(
        id="launch",
        kind="module",
        source="""
import subprocess
import sys
subprocess.run([sys.executable, "kernel.py", "--local"], check=True)
""",
    )
    program.return_file(key="report", path="result.json")
    return program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--gpus", type=int, choices=range(2, 9), default=2)
    parser.add_argument("--local", action="store_true", help="use all visible local GPUs")
    args = parser.parse_args()
    if args.local:
        run_local()
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
