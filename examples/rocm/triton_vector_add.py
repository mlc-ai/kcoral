"""Run and benchmark a Triton vector add on any GPU server, CUDA or ROCm.

Prints the server's target and the kernel's latency; checks the sum on the client.
Run with KCORAL_URL pointing at the server.
"""

import os

import numpy as np

from kcoral import Client, Program

N = 1 << 24

SOURCE = """
import torch
import triton
import triton.language as tl

from kcoral.builtins import benchmark


@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    total = tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask)
    tl.store(out + offsets, total, mask=mask)


def main(x, y):
    out = torch.empty_like(x)
    n = x.numel()

    def launch():
        add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)

    launch()
    timing = benchmark(launch)
    return {"out": out, "timing": timing}
"""


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N, dtype=np.float32)
    y = rng.standard_normal(N, dtype=np.float32)
    program = Program()
    module = program.upload(kind="module", source=SOURCE)
    fn = program.get_function(module=module, name="main")
    result_ref = program.run(
        fn=fn,
        args=[program.upload(kind="tensor", value=x), program.upload(kind="tensor", value=y)],
    )
    program.return_(key="result", value=result_ref)
    with Client(os.environ.get("KCORAL_URL", "http://127.0.0.1:8000")) as client:
        print(f"target: {client.target()}")
        result = client.execute(program, timeout_seconds=120)
    if not result.completed:
        raise SystemExit(f"Program failed: {result.error}")
    output = result.results["result"]
    np.testing.assert_allclose(output["out"], x + y, rtol=1e-6)
    timing = output["timing"]
    print(
        f"timing={timing.get('timing')} median={timing['latency_ms_median']:.4f} ms "
        f"min={timing['latency_ms_min']:.4f} ms repeat={timing['repeat']}"
    )


if __name__ == "__main__":
    main()
