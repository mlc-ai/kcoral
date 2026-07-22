# Benchmark Server

A v2 implementation of the synchronous remote GPU execution protocol in this
repository. It includes the HTTP server, content-addressed blob cache, isolated
per-request Python runtimes, GPU-slot scheduling, structured logs, and a
synchronous Python SDK.

## Install and run

```bash
python -m pip install -e .
benchmark-server \
  --devices 0,1 \
  --cache-capacity-bytes 10737418240
```

For protocol tests on a machine without GPUs, `--devices 0` is sufficient. The
server exposes the configured device to an executing request through
`CUDA_VISIBLE_DEVICES`; uploaded programs still need a working GPU runtime if
they actually use CUDA.

## Docker development

Build the development image and start the server:

```bash
docker compose -f docker/compose.yaml up --build
```

After the image has been built, source changes are available through the bind
mount and do not require another build:

```bash
docker compose -f docker/compose.yaml up
```

Dependencies are locked in `uv.lock`. After changing dependencies in
`pyproject.toml`, refresh the lockfile and rebuild the image:

```bash
uv lock
docker compose -f docker/compose.yaml up --build
```

Run the test suite in a temporary container:

```bash
docker compose -f docker/compose.yaml run --rm benchmark-server \
  pytest -q /workspace/benchmark-server/tests/test_server.py
```

The server uses GPU `0` and host port `8000` by default. Override them when
needed:

```bash
BENCHMARK_DEVICES=0,1 BENCHMARK_PORT=8001 \
  docker compose -f docker/compose.yaml up
```

## Python client

```python
from benchmark_server import Client

with Client("http://127.0.0.1:8000") as client:
    result = client.execute({"main.py": "def main():\n    return 42\n"})
    print(result.value)
```

## Instruction execution

The same `/execute` endpoint can run request-local instruction programs. Uploaded
Python modules may register TVM FFI global functions, calls may read and write
integer-indexed registers, and return instructions expose named results:

```python
import hashlib

from benchmark_server import Client

source = """
import tvm_ffi

@tvm_ffi.register_global_func("example.add")
def add(left, right):
    return left + right
"""
module_hash = hashlib.sha256(source.encode()).hexdigest()
instructions = [
    {"op": "upload_module", "blob": module_hash},
    {"op": "call", "dst": 0, "func": "example.add", "args": [2, 3]},
    {"op": "return", "reg": 0, "key": "sum"},
]

with Client("http://127.0.0.1:8000") as client:
    result = client.execute_instructions(
        instructions,
        {module_hash: source},
    )
    print(result.value["sum"])
```

`upload_tensor` supports `cpu` and the worker-local `cuda:0` device. Tensor
upload requires PyTorch in the server runtime.

`random_tensor` creates a deterministic uniformly distributed floating-point
tensor directly in the worker from `shape`, `dtype`, `seed`, and `device`
fields, then stores it in `dst`.

## Documentation

- [English API reference](api-reference.md)
- [Chinese API reference](api-reference-zh.md)
