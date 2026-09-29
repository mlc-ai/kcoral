"""Compile on a CPU server, then execute the returned library on a GPU server.

Start two instances of the same server before running this example:

    kcoral server --device cpu --num-workers 8 --port 8000
    kcoral server --device gpu --gpus 0 --workers-per-gpu 8 --port 8001

Both requests use the regular ``POST /execute`` instruction protocol. The first
returns shared-object bytes; the second uploads those bytes as ``kind="library"``.
"""

from __future__ import annotations

import os

from kcoral import Client, Program

N = 1 << 20

CUDA_SOURCE = r"""
__global__ void add_one_kernel(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + 1.0f;
}

void add_one(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  int n = static_cast<int>(x.numel());
  add_one_kernel<<<(n + 255) / 256, 256>>>(static_cast<const float*>(x.data_ptr()),
                                           static_cast<float*>(y.data_ptr()), n);
}
"""

REFERENCE = "def main(a):\n    return a + 1.0\n"


# These functions and imports are uploaded to the server.
OPERATIONS = r"""
from kcoral.builtins import benchmark


def empty(spec):
    import torch

    return torch.empty(spec["shape"], dtype=getattr(torch, spec["dtype"]), device="cuda")


def randn(spec):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(spec.get("seed", 0))
    return torch.randn(
        spec["shape"], dtype=getattr(torch, spec["dtype"]), device="cuda", generator=generator
    )


def assert_close(actual, expected):
    import torch

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-2, atol=1e-3)
    return {"ok": True}


def compile_cuda_binary(source, cfg):
    import os
    from pathlib import Path

    import tvm_ffi.cpp

    arch = cfg["arch"].removeprefix("sm_")
    suffix = "a" if arch.endswith("a") else ""
    digits = arch.removesuffix("a")
    key = "TVM_FFI_CUDA_ARCH_LIST"
    previous = os.environ.get(key)
    os.environ[key] = f"{int(digits[:-1])}.{digits[-1]}{suffix}"
    try:
        path = tvm_ffi.cpp.build_inline(
            name=f"example_{source.name}",
            cuda_sources=source.source,
            functions=source.name,
            backend="cuda",
            extra_cuda_cflags=cfg.get("extra_cuda_cflags"),
        )
        return Path(path).read_bytes()
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
"""


def compile_program(arch: str) -> Program:
    program = Program()
    operations = program.upload(id="operations", kind="module", source=OPERATIONS)
    compile_cuda_binary = program.get_function(
        id="compile_cuda_binary", module=operations, name="compile_cuda_binary", cpu_only=True
    )
    module = program.upload(
        id="source",
        kind="module",
        language="cuda",
        source=CUDA_SOURCE,
    )
    source = program.get_function(id="add_one_source", module=module, name="add_one")
    library = program.run(
        id="library",
        fn=compile_cuda_binary,
        args=[source, {"arch": arch, "extra_cuda_cflags": ["-O3"]}],
    )
    program.return_(key="library", value=library)
    return program


def benchmark_program(library: bytes) -> Program:
    program = Program()
    operations = program.upload(id="operations", kind="module", source=OPERATIONS)
    empty = program.get_function(id="empty", module=operations, name="empty")
    randn = program.get_function(id="randn", module=operations, name="randn")
    assert_close = program.get_function(id="assert_close", module=operations, name="assert_close")
    benchmark = program.get_function(id="benchmark", module=operations, name="benchmark")
    module = program.upload(id="kernel_module", kind="library", value=library)
    kernel = program.get_function(id="kernel", module=module, name="add_one")
    reference_module = program.upload(id="reference_module", kind="module", source=REFERENCE)
    reference = program.get_function(id="reference", module=reference_module, name="main")
    src = program.run(
        id="src",
        fn=randn,
        args=[{"shape": [N], "dtype": "float32", "seed": 0}],
    )
    dst = program.run(id="dst", fn=empty, args=[{"shape": [N], "dtype": "float32"}])
    program.run(id="invoke", fn=kernel, args=[src, dst])
    expected = program.run(id="expected", fn=reference, args=[src])
    check = program.run(id="check", fn=assert_close, args=[dst, expected])
    timing = program.run(
        id="timing",
        fn=benchmark,
        args=[kernel, src, dst, {"warmup_ms": 25, "repeat_ms": 100}],
    )
    program.return_(key="check", value=check)
    program.return_(key="timing", value=timing)
    return program


def main() -> None:
    cpu_url = os.environ.get("KCORAL_CPU_URL", "http://localhost:8000")
    gpu_url = os.environ.get("KCORAL_GPU_URL", "http://localhost:8001")
    with Client(cpu_url) as cpu_client, Client(gpu_url) as gpu_client:
        arch = gpu_client.target()["arch"]
        compiled = cpu_client.execute(compile_program(arch), timeout_seconds=120)
        if not compiled.completed:
            raise SystemExit(f"compile failed: {compiled.error}")
        library = compiled.results["library"]
        if not isinstance(library, bytes):
            raise SystemExit("compile server returned a non-binary library")

        result = gpu_client.execute(benchmark_program(library), timeout_seconds=120)
        if not result.completed:
            raise SystemExit(f"benchmark failed: {result.error}")

    timing = result.results["timing"]
    print(f"compiled {len(library) / 1024:.0f} KiB for {arch}")
    print(f"CPU request held a GPU lease for {compiled.lease_held_ms:.0f} ms")
    print(
        f"GPU request held its lease for {result.lease_held_ms:.0f} ms; "
        f"kernel median {timing['latency_ms_median'] * 1e3:.1f} us"
    )


if __name__ == "__main__":
    main()
