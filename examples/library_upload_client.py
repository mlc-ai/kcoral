"""Build a kernel on the client and upload the shared object.

A prebuilt library is the path for a build the server's compile builtins cannot
express — several translation units, a code generator, flags beyond the
``extra_cuda_cflags`` that ``compile_cuda`` exposes. The client builds for the
arch ``GET /health`` reports, uploads the bytes once, and every later request
costs only their hash.

This needs a CUDA toolchain here; `remote_compile_client.py` needs none.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

from kcoral import Client, Program

N = 1 << 20

# The export macro is what makes the object loadable: it emits the
# `__tvm_ffi_add_one` symbol the server looks up by `entry`. A server-side
# compile needs neither it nor the include — `compile_cuda` supplies both.
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
    # No compile instruction follows: the handle is already the callable.
    kernel = program.upload(id="kernel", kind="library", value=library, entry="add_one")
    reference = program.upload(id="reference", kind="module", source=REFERENCE)

    src = program.run(
        id="src", fn="builtin.randn", args=[{"shape": [N], "dtype": "float32", "seed": 0}]
    )
    dst = program.run(id="dst", fn="builtin.empty", args=[{"shape": [N], "dtype": "float32"}])
    program.run(id="invoke", fn=kernel, args=[src, dst])

    # Compared on the server against a plain-Python reference, so the output
    # tensor never travels; assert_close stops the program before timing a
    # kernel that is wrong.
    expected = program.run(id="expected", fn=reference, args=[src])
    check = program.run(id="check", fn="builtin.assert_close", args=[dst, expected])
    timing = program.run(id="timing", fn="builtin.benchmark", args=[kernel, src, dst])
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
