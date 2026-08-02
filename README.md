# benchmark-server

`benchmark-server` executes GPU benchmark programs over HTTP. A program uploads
modules and tensors, runs registered functions, and explicitly returns selected
values.

The wire format and validation rules are defined in
[`docs/protocol.md`](docs/protocol.md).

## Install

Python 3.10 or newer is required.

```bash
pip install .
```

The CUDA runtime also requires PyTorch and TVM FFI supplied by its environment.
The checked-in lockfile can create the development environments:

```bash
uv sync --extra test
uv sync --group gpu --extra test
```

## Run the server

```bash
benchmark-server --host 0.0.0.0 --port 8000
```

Useful options include:

```text
--gpus 0,1                  CUDA devices exposed to workers
--worker-wait-timeout-seconds 30  Queue wait before HTTP 503 responses
--default-timeout-seconds 300     Default execution timeout
--output-limit-bytes 1048576      Request-level stdout/stderr capture limit
--max-request-bytes 268435456     Maximum request body size
```

Check readiness with `GET /health`. Submit programs with `POST /execute` using
`multipart/form-data`.

## Python client

The client builds the protocol JSON and binary tensor parts. Tensor uploads can
use NumPy arrays, PyTorch tensors, objects implementing the DLPack protocol, or
raw bytes accompanied by `dtype` and `shape`.

```python
from benchmark_server import Client, Program

source = r"""
def main(a, b):
    return a + b
"""

program = Program()
module = program.upload(id="module", kind="module", source=source)
result = program.run(id="sum", fn=module, args=[20, 22])
program.return_(key="answer", value=result)

with Client("http://localhost:8000") as client:
    response = client.execute(program)

print(response.status)
print(response.results["answer"])
```

`Program.upload()` and `Program.run()` return a `Register`, which can be passed
to later instructions. `Program.return_()` adds an explicit result; run values
that are not returned do not appear in the response.

The client starts with a cache-only request. When the server reports a
`CACHE_MISS`, it retries once with only the requested blobs. Returned tensor
values are decoded as CPU `tvm_ffi.Tensor` objects.

## Protocol summary

A request contains one `program` JSON part and zero or more binary parts named
`blob:<sha256>`. The SHA-256 digest is calculated over the raw tensor bytes.
Programs contain three instruction types:

- `upload`: register a module source string or a tensor blob.
- `run`: call a registered function with recursively encoded arguments.
- `return`: expose a previously computed value under a result key.

Successful execution returns HTTP 200 with status `COMPLETED`. Instruction
failures also return HTTP 200, with status `FAILED` and structured error
details. Missing cached blobs return HTTP 200 with status `CACHE_MISS`. Request,
capacity, timeout, and server errors use the HTTP status codes documented in the
protocol.

Responses are JSON unless a returned value contains bytes or tensors. Binary
responses use `multipart/form-data`: the `result` part contains JSON metadata,
and `return:<index>` parts contain raw bytes in depth-first traversal order.

## Development

```bash
pip install '.[test]'
pytest -q
ruff check src tests
ruff format --check src tests
```

GPU integration tests are opt-in:

```bash
BENCH_GPU_TEST=1 pytest -q
```
