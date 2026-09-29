"""Upload a tensor, add one on the GPU, and return the result."""

import os

import numpy as np

from kcoral import Client, Program

SOURCE = """
def add_one(x):
    if not x.is_cuda:
        raise RuntimeError("This program requires a GPU tensor")
    return x + 1
"""


def build_program() -> Program:
    program = Program()
    module = program.upload(id="module", kind="module", source=SOURCE)
    add_one = program.get_function(id="add_one", module=module, name="add_one")
    x = program.upload(id="x", kind="tensor", value=np.arange(4, dtype=np.float32))
    y = program.run(id="y", fn=add_one, args=[x])
    program.return_(key="output", value=y)
    return program


def main() -> None:
    with Client(os.environ.get("KCORAL_URL", "http://localhost:8000")) as client:
        result = client.execute(build_program(), timeout_seconds=30)
    if not result.completed:
        raise SystemExit(f"Program failed: {result.error}")
    np.testing.assert_array_equal(result.results["output"], np.arange(1, 5, dtype=np.float32))
    print(result.status)
    print(result.results["output"])


if __name__ == "__main__":
    main()
