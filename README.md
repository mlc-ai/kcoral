# benchmark-server

`benchmark-server` executes GPU benchmark programs over HTTP. A program uploads
modules and tensors, runs registered functions, and explicitly returns selected
values.

[`docs/client_guide.md`](docs/client_guide.md) is the guide to writing one, and
covers where to compile, which language to use, how to measure, and how failures
arrive. [`docs/protocol.md`](docs/protocol.md) defines the wire format
and validation rules.

## Install

Python 3.10 or newer is required.

```bash
pip install .
```

This gives the Python client and the HTTP front-end, neither of which touches a
GPU. Running programs needs a worker environment holding more, and how much
depends on which builtins the programs use:

| To run | The worker environment needs |
|---|---|
| any program | PyTorch and TVM FFI |
| `compile_tirx` | TVM as well |
| `compile_cuda` | `nvcc`, a host C++ compiler, and `ninja` as well |
| `compile_cutedsl`, or a CuTeDSL library upload | `nvidia-cutlass-dsl` as well |
| `compile_triton` | `triton`, which the CUDA PyTorch wheels already carry |
| `benchmark` | `cupti-python` as well |

A builtin whose requirement is absent answers `unavailable` and the rest of the
server is unaffected, so a partial environment is a usable deployment.

Build whichever of the two environments below matches the work.

### Front-end, engine, and client

The lockfile builds this one, CPU-only, with no GPU or compiler needed:

```bash
uv sync
```

### Running GPU programs

The `gpu` group adds PyTorch, TVM, TVM FFI, CuTeDSL, and the CUPTI Python
bindings, which together cover every builtin and every kind of library upload:

```bash
uv sync --group gpu
```

`nvcc` and a host C++ compiler still come from the system; everything else is a
wheel, and no environment variables are needed.

## Run the server

```bash
benchmark-server --host 0.0.0.0 --port 8000
```

Useful options include:

```text
--gpus 0,1                        CUDA devices exposed to workers
--workers-per-gpu 8               Workers sharing each GPU
--worker-wait-timeout-seconds 30  Queue wait before HTTP 503 responses
--default-timeout-seconds 300     Default execution timeout
--output-limit-bytes 1048576      Request-level stdout/stderr capture limit
--max-request-bytes 268435456     Maximum request body size
```

Check readiness with `GET /health`, which also reports the `target` an uploaded
library must be built for and the `versions` the worker runs. Submit programs with `POST /execute` using
`multipart/form-data`.

Several workers share each GPU, so one can compile while another measures on the
GPU it is not using; they take turns through a per-GPU lease and never run on it
at once. Raising `--workers-per-gpu` keeps the GPUs busier at the cost of dividing
their memory among more concurrent benchmarks.

## Python client

The client builds the protocol JSON and binary parts. Byte uploads preserve
files and other binary data unchanged. Tensor uploads can use NumPy arrays,
PyTorch tensors, objects implementing the DLPack protocol, or raw bytes
accompanied by `dtype` and `shape`.

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

A kernel built elsewhere can be uploaded instead of source, which is the path
when the build is customized beyond what a `compile_*` builtin expresses:

```python
arch = client.target()["arch"]                # "sm_100a" — build the object for this
so_bytes = pathlib.Path("add_one.so").read_bytes()

program = Program()
kernel = program.upload(id="kernel", kind="library", value=so_bytes, entry="add_one")
program.run(id="invoke", fn=kernel, args=[x, y])   # no compile instruction
```

The server builds nothing here; it loads the shared object and calls `entry`. The
protocol document states what a library must export, and
[`examples/library_upload_client.py`](examples/library_upload_client.py) builds
one end to end.

`Program.upload()` and `Program.run()` return a `Register`, which can be passed
to later instructions. `Program.return_()` adds an explicit result; run values
that are not returned do not appear in the response.

The client starts with a cache-only request. When the server reports a
`CACHE_MISS`, it retries with only the requested blobs. If concurrent cache
changes cause another miss, one final request carries every local blob. Returned
tensor values are decoded as CPU `numpy.ndarray` objects.

`bfloat16` and `float8` arrays reach torch through a byte reinterpretation:

```python
torch.from_numpy(value.view(np.uint8)).view(torch.bfloat16)
```

### FlashInfer Trace client

The distribution includes a separate `flashinfer_bench` Python package containing
the self-contained FlashInfer Trace data model and all benchmark orchestration.
It loads a `TraceSet`, generates one general server program per solution and
workload pair, submits programs concurrently, and returns `Trace` evaluation
objects. The `benchmark_server` package receives no FlashInfer-specific API or
scheduler.

```bash
pip install benchmark-server
python examples/flashinfer_client.py Example-FlashInfer-Trace
```

The bundled package has no dependency on the external `flashinfer-bench` project.
Every tensor uses the existing content-addressed upload flow: a request first
sends only the SHA-256 content key, then sends only missing blobs after
`CACHE_MISS`. Inputs are prepared once per workload, so solution requests reuse
cached data. See the [FlashInfer client guide](docs/flashinfer.md) for the
supported scope and extension points.

## Protocol summary

A request contains one `program` JSON part and zero or more binary parts named
`blob:<sha256>`. The SHA-256 digest is calculated over the raw bytes.
Programs contain three instruction types:

- `upload`: register module source, raw bytes, a tensor, or a library.
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

`--group test` adds pytest to either environment from [Install](#install):

```bash
uv sync --group test             # or --group test --group gpu
pytest -q
ruff check python tests
ruff format --check python tests
```

The GPU integration tests are opt-in, and need the GPU environment:

```bash
BENCH_GPU_TEST=1 pytest -q
```
