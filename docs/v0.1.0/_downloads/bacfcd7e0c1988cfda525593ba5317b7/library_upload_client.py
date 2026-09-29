"""Build a kernel locally, upload the library, check its output, and time it.

The client owns the compiler flags and builds for the architecture reported by
GET /health. Allocation, comparison and CUPTI measurement run on the server.
The client needs a CUDA toolchain.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

from kcoral import Client, Program

N = 1 << 20

# The export macro is what makes the object loadable: it emits the
# `__tvm_ffi_add_one` symbol the server looks up by function name.
SOURCE = r"""
#include <tvm/ffi/container/tensor.h>

__global__ void add_one_kernel(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + 1.0f;
}

void add_one(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  int n = static_cast<int>(x.numel());
  add_one_kernel<<<(n + 255) / 256, 256>>>(static_cast<const float*>(x.data_ptr()),
                                           static_cast<float*>(y.data_ptr()), n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, add_one);
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
"""


def build_library(arch: str, directory: str) -> bytes:
    """Compile SOURCE for `arch` — the server's, not this machine's."""
    import tvm_ffi.cpp

    source_path = os.path.join(directory, "add_one.cu")
    pathlib.Path(source_path).write_text(SOURCE)
    # Whatever the local toolchain accepts belongs here; this freedom is the
    # reason to upload a library rather than let the server build one.
    library = tvm_ffi.cpp.build(
        "add_one",
        cuda_files=source_path,
        extra_cuda_cflags=[
            f"-gencode=arch=compute_{arch.removeprefix('sm_')},code={arch}",
            "-O3",
        ],
        build_directory=directory,
        output=os.path.join(directory, "add_one.so"),
    )
    return pathlib.Path(library).read_bytes()


def build_program(library: bytes) -> Program:
    program = Program()
    operations = program.upload(id="operations", kind="module", source=OPERATIONS)
    empty = program.get_function(id="empty", module=operations, name="empty")
    randn = program.get_function(id="randn", module=operations, name="randn")
    assert_close = program.get_function(id="assert_close", module=operations, name="assert_close")
    benchmark = program.get_function(id="benchmark", module=operations, name="benchmark")
    # No compile instruction follows: get_function binds the precompiled callable.
    module = program.upload(id="kernel_module", kind="library", value=library)
    kernel = program.get_function(id="kernel", module=module, name="add_one")
    reference_module = program.upload(id="reference_module", kind="module", source=REFERENCE)
    reference = program.get_function(id="reference", module=reference_module, name="main")

    src = program.run(id="src", fn=randn, args=[{"shape": [N], "dtype": "float32", "seed": 0}])
    dst = program.run(id="dst", fn=empty, args=[{"shape": [N], "dtype": "float32"}])
    program.run(id="invoke", fn=kernel, args=[src, dst])

    # Compared on the server against a plain-Python reference, so the output
    # tensor never travels; assert_close stops the program before timing a
    # kernel that is wrong.
    expected = program.run(id="expected", fn=reference, args=[src])
    check = program.run(id="check", fn=assert_close, args=[dst, expected])
    timing = program.run(id="timing", fn=benchmark, args=[kernel, src, dst])
    program.return_(key="check", value=check)
    program.return_(key="timing", value=timing)
    return program


def main() -> None:
    with Client(os.environ.get("KCORAL_URL", "http://localhost:8000")) as client:
        arch = client.target()["arch"]
        with tempfile.TemporaryDirectory() as directory:
            library = build_library(arch, directory)
        print(f"built {len(library) / 1024:.0f} KiB for {arch}")

        result = client.execute(build_program(library), timeout_seconds=120)
        if result.status != "COMPLETED":
            raise SystemExit(f"{result.status} — {result.error}")
        print(
            f"{result.elapsed_ms:.0f} ms total, "
            f"{result.lease_held_ms:.0f} ms on the GPU, "
            f"kernel {result.results['timing']['latency_ms_median'] * 1e3:.1f} us"
        )


if __name__ == "__main__":
    main()
