"""Upload a function and tensor, run on a GPU, and receive a NumPy array."""

import argparse
import os

import numpy as np

from kcoral import Client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("KCORAL_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args()

    with Client(args.endpoint) as client:

        @client.function(timeout=30)
        def add_one(x):
            if not x.is_cuda:
                raise RuntimeError("This example requires a GPU server")
            return x + 1

        values = np.arange(4, dtype=np.float32)
        output = add_one.remote(values)
        np.testing.assert_array_equal(output, values + 1)
        print(output)


if __name__ == "__main__":
    main()
