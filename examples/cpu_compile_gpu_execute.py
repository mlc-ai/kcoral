"""Compile on a CPU server, then execute the returned library on a GPU server.

Start two instances of the same server before running this example:

    benchmark-server --device cpu --num-workers 8 --port 8000
    benchmark-server --device gpu --gpus 0 --workers-per-gpu 8 --port 8001

Both requests use the regular ``POST /execute`` instruction protocol. The first
returns shared-object bytes; the second uploads those bytes as ``kind="library"``.
"""

from __future__ import annotations

import os

from benchmark_server import Client, Program

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


def compile_program(arch: str) -> Program:
    program = Program()
    source = program.upload(
        id="source",
        kind="module",
        language="cuda",
        source=CUDA_SOURCE,
        entry="add_one",
    )
    library = program.run(
        id="library",
        fn="builtin.compile_cuda_binary",
        args=[source, {"arch": arch, "extra_cuda_cflags": ["-O3"]}],
    )
    program.return_(key="library", value=library)
    return program


def benchmark_program(library: bytes) -> Program:
    program = Program()
    kernel = program.upload(id="kernel", kind="library", value=library, entry="add_one")
    reference = program.upload(id="reference", kind="module", source=REFERENCE)
    src = program.run(
        id="src",
        fn="builtin.randn",
        args=[{"shape": [N], "dtype": "float32", "seed": 0}],
    )
    dst = program.run(id="dst", fn="builtin.empty", args=[{"shape": [N], "dtype": "float32"}])
    program.run(id="invoke", fn=kernel, args=[src, dst])
    expected = program.run(id="expected", fn=reference, args=[src])
    check = program.run(id="check", fn="builtin.assert_close", args=[dst, expected])
    timing = program.run(
        id="timing",
        fn="builtin.benchmark",
        args=[kernel, src, dst, {"warmup_ms": 25, "repeat_ms": 100}],
    )
    program.return_(key="check", value=check)
    program.return_(key="timing", value=timing)
    return program


def main() -> None:
    cpu_url = os.environ.get("BENCH_CPU_URL", "http://localhost:8000")
    gpu_url = os.environ.get("BENCH_GPU_URL", "http://localhost:8001")
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
