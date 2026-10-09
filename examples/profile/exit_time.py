"""Discussion: why a fresh worker adds ~300 ms - how long a warmed worker process takes to exit.

Each child builds the worker runtime (kcoral.runtime.gpu.GPURuntime: torch, TVM,
CuTe DSL, CUDA context), runs one small kernel, prints "ready", then ends in one of
these ways. The parent times from "ready" until the process is gone. No server needed.

  normal       return normally (what a retiring worker does today)
  sigterm      killed by SIGTERM (what retirement did before #59)
  os_exit      os._exit(0): skips Python finalization
  ctx_destroy  destroy the CUDA context explicitly (timed in the child), then return
"""

import argparse
import json
import signal
import subprocess
import sys
import time

from overhead import summary

CHILD = """
import ctypes, os, sys, time
from kcoral.runtime.gpu import GPURuntime
runtime = GPURuntime()
import torch
(torch.ones(1 << 20, device="cuda") + 1).sum().item()
torch.cuda.synchronize()
mode = sys.argv[1]
if mode == "ctx_destroy":
    cuda = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    cuda.cuCtxGetDevice(ctypes.byref(device))
    t = time.perf_counter()
    cuda.cuDevicePrimaryCtxReset_v2(device)
    print(f"ctx_destroy_ms {(time.perf_counter() - t) * 1e3}", flush=True)
print("ready", flush=True)
if mode == "sigterm":
    time.sleep(60)
if mode == "os_exit":
    os._exit(0)
"""

MODES = ("normal", "sigterm", "os_exit", "ctx_destroy")


def one(mode):
    proc = subprocess.Popen([sys.executable, "-c", CHILD, mode], stdout=subprocess.PIPE, text=True)
    ctx_ms = None
    for line in proc.stdout:
        if line.startswith("ctx_destroy_ms"):
            ctx_ms = float(line.split()[1])
        if line.strip() == "ready":
            break
    t = time.perf_counter()
    if mode == "sigterm":
        proc.send_signal(signal.SIGTERM)
    proc.wait()
    return (time.perf_counter() - t) * 1e3, ctx_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = {}
    for mode in MODES:
        runs = [one(mode) for _ in range(args.trials)]
        report[mode] = {"exit_ms": summary([r[0] for r in runs])}
        if mode == "ctx_destroy":
            report[mode]["ctx_destroy_ms"] = summary([r[1] for r in runs])
        print(f"{mode:12s} exit median {report[mode]['exit_ms']['median']:6.1f} ms")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
