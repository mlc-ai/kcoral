"""Discussion: what a reused worker carries from one request to the next.

Run against a server with one reused worker (--workers-per-gpu 1
--max-requests-per-worker 0), so every request lands on the same process. The
script inspects the worker, changes state in one request, inspects again, then
triggers a CUDA device-side assert and inspects the worker that serves next.
"""

import argparse
import json
import time

from kcoral import Client, Program

SOURCE = """
import ctypes, os, torch

def inspect():
    torch.cuda.synchronize()
    return {
        "pid": os.getpid(),
        "gpu_free_mib": torch.cuda.mem_get_info()[0] // 2**20,
        "torch_allocated_mib": torch.cuda.memory_allocated() // 2**20,
        "tensor_on_torch_module": hasattr(torch, "_probe_tensor"),
        "tensor_add_patched": getattr(torch.Tensor.__add__, "_probe", False),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "env_var": os.environ.get("PROBE_ENV"),
        "rand": torch.rand(1).item(),
    }

def change_state():
    torch._probe_tensor = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
    original = torch.Tensor.__add__
    def patched(a, b):
        return original(a, b)
    patched._probe = True
    torch.Tensor.__add__ = patched
    torch.backends.cudnn.benchmark = True
    os.environ["PROBE_ENV"] = "set"
    cuda = ctypes.CDLL("libcuda.so.1")
    pointer = ctypes.c_uint64()
    assert cuda.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(512 * 2**20)) == 0
    return inspect()  # the raw 512 MiB allocation is never freed

def fault():
    x = torch.zeros(4, device="cuda")
    x[torch.tensor([1 << 30], device="cuda")]  # out-of-range index: device-side assert
    torch.cuda.synchronize()
"""

STEPS = ("inspect", "change_state", "inspect", "fault", "inspect")


def call(client, name):
    program = Program()
    module = program.upload(kind="module", source=SOURCE)
    program.return_(key="r", value=program.run(fn=program.get_function(module=module, name=name)))
    result = client.execute(program, timeout_seconds=60)
    return {
        "step": name,
        "status": result.status,
        "result": result.results.get("r"),
        "error": None
        if result.completed
        else {"kind": result.error["kind"], "message": result.error["message"][:200]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    steps = []
    with Client(args.url) as client:
        for name in STEPS:
            if steps and steps[-1]["step"] == "fault":
                while client.health()["load"]["request_capacity"] < 1:
                    time.sleep(0.05)  # the faulted worker is being replaced
            steps.append(call(client, name))
            print(json.dumps(steps[-1]))
    with open(args.out, "w") as f:
        json.dump(steps, f, indent=1)


if __name__ == "__main__":
    main()
