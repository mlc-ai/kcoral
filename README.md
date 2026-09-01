# KCoral

KCoral executes GPU benchmark programs over HTTP. The same server
can also run without a GPU as a CUDA compilation service. A program uploads
modules and tensors, runs registered functions, and explicitly returns selected
values.

[`docs/client_guide.md`](docs/client_guide.md) is the guide to writing one, and
covers where to compile, which language to use, how to measure, and how failures
arrive. [`docs/protocol.md`](docs/protocol.md) defines the wire format
and validation rules.

## Install

Python 3.10 or newer is required.

### The client

```bash
pip install .
```

This gives the Python client on its own, which is all that sending programs to
a running server needs. It does not depend on a GPU, so it can install on a
machine with no GPU, no CUDA and no compiler.

### The server

The `server` extra adds the HTTP front-end, FastAPI and uvicorn, which the
client never imports:

```bash
pip install '.[server]'
```

Neither the client nor the front-end touches a GPU. Running programs needs a
worker environment holding more, and how much depends on what the programs use:

| To run | The worker environment needs |
|---|---|
| a CPU compilation server | TVM FFI, `nvcc`, `ninja`, and a host C++ compiler |
| any GPU program | PyTorch and TVM FFI |
| `compile_tirx` | TVM as well |
| `compile_cuda` | `nvcc`, a host C++ compiler, and `ninja` as well |
| `compile_cutedsl`, or a CuTeDSL library upload | `nvidia-cutlass-dsl` as well |
| `compile_triton` | `triton`, which the CUDA PyTorch wheels already carry |
| `benchmark` | `cupti-python` as well |
| a program importing FlashInfer | `flashinfer-python` 0.6.17 or newer as well |

A builtin whose requirement is absent answers `unavailable` and the rest of the
server is unaffected, so a partial environment is a usable deployment.

Build whichever of the environments below matches the work.

### Front-end, engine, and client

The lockfile builds this one, CPU-only, with no GPU or compiler needed:

```bash
uv sync
```

`uv sync --no-default-groups` narrows it to the client's dependencies alone.

### Running GPU programs

The `gpu` group adds PyTorch, TVM, TVM FFI, CuTeDSL, and the CUPTI Python
bindings, which together cover every builtin and every kind of library upload,
plus other dependencies for programs whose reference calls it:

```bash
uv sync --group gpu
```

`nvcc` and a host C++ compiler still come from the system; everything else is a
wheel, and no environment variables are needed.

### Running CPU compilation workers

The `compiler` group adds TVM FFI and `ninja` without installing PyTorch or
other GPU runtimes. The CUDA toolkit and a host C++ compiler still come from the
system:

```bash
uv sync --group compiler
```

## Run the server

```bash
kcoral --host 127.0.0.1 --port 8000
```

The server binds to `127.0.0.1` by default. Use `--host` or `KCORAL_SERVER_HOST` to
select another address.

To compile on a machine without a GPU and execute on a separate GPU machine,
run two instances of this same command:

```bash
kcoral --device cpu --num-workers 16 --host 0.0.0.0 --port 8000
kcoral --device gpu --gpus 0 --workers-per-gpu 8 --host 0.0.0.0 --port 8001
```

Useful options include:

```text
--device cpu|gpu                  Worker type (default: gpu)
--num-workers 16                  CPU workers in CPU mode
--gpus 0,1                        CUDA devices exposed to workers
--workers-per-gpu 8               Workers sharing each GPU
--max-requests-per-worker 1       Fresh process/context per request; 0 reuses workers
--worker-wait-timeout-seconds 30  Queue wait before HTTP 503 responses
--default-timeout-seconds 300     Default execution timeout
--output-limit-bytes 1048576      Request-level stdout/stderr capture limit
--max-request-bytes 268435456     Maximum request body size
--log-dir logs                    Event log directory; empty disables logging
--no-log-console                  Stop mirroring events to stderr
--no-log-programs                 Stop keeping each request's program JSON
```

Check readiness with `GET /health`, which also reports the `target` an uploaded
library must be built for and the `versions` the worker runs. A CPU server
reports `gpu_count: 0`; the client reads the target from the GPU server. Submit
programs with `POST /execute` using `multipart/form-data`.

Several workers share each GPU, so one can compile while another measures on the
GPU it is not using; they take turns through a per-GPU lease and never run on it
at once. Raising `--workers-per-gpu` keeps the GPUs busier at the cost of dividing
their memory among more concurrent benchmarks.

Workers serve one request by default, then the pool replaces them before making
the slot idle again, so an out-of-bounds or race-sensitive kernel cannot make a
later request depend on its process history. `--max-requests-per-worker 0` reuses
workers instead: reset and poison detection still run, but undefined CUDA
behaviour is no longer contained, and replacement costs enough on short requests
that throughput numbers should record the setting.

## Logs

Each run appends to `<log-dir>/runs/<timestamp>/events.jsonl`, one JSON object
per line, mirrored to stderr and never capped or rotated. Only the front-end
writes it, so one file holds a request's whole history: `request_received`,
`request_accepted` with the shape of the workload, `request_routed` with the
worker that took it, and `request_finished`. One that never comes back stops
after the record naming its worker.

`request_finished` carries a `finish_reason`: why the request ended when it did.

| `finish_reason` | what happened | worker |
|---|---|---|
| `completed` | the program ran to the end | keeps serving |
| `program_failed` | the program raised; `error_kind` and `instruction_index` say where | keeps serving |
| `request_limit` | `--max-requests-per-worker` reached, after answering | replaced |
| `poisoned_context` | cleanup after the program failed, after answering | replaced |
| `timeout` | no answer inside `timeout_seconds` | killed, replaced |
| `crashed` | the process exited mid-request; `exitcode` says how | killed, replaced |
| `no_worker` | saturated, nothing ran | untouched |
| `rejected`, `cache_miss`, `server_error` | never reached a worker | untouched |

A failing program is the client's kernel and stays `INFO`; only what the server
itself did wrong reaches `ERROR`.

```bash
jq -c 'select(.level == "ERROR")' logs/runs/*/events.jsonl
jq -c 'select(.request_id == "<id>")' logs/runs/*/events.jsonl   # one request end to end
jq -c 'select(.worker_id == "gpu0/w3")' logs/runs/*/events.jsonl # one worker's history
```

`worker_id` names a seat on a GPU and outlives the processes that fill it, which
`generation` and `pid` identify; the `worker_*` records cover what happens
between requests. A killed worker's output reaches `request_finished` as
`output_tail`.

Each program's JSON is kept beside the log as `programs/<request-id>.json`, so a
failed request names the kernel that failed. Blob uploads travel by hash, so this
costs a few KB per request; `--no-log-programs` turns it off.

## Python client

The client builds the protocol JSON and binary parts. Byte uploads preserve
files and other binary data unchanged. Tensor uploads can use NumPy arrays,
PyTorch tensors, objects implementing the DLPack protocol, or raw bytes
accompanied by `dtype` and `shape`.

```python
from kcoral import Client, Program

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

[`examples/cpu_compile_gpu_execute.py`](examples/cpu_compile_gpu_execute.py)
shows the two-server form: the client reads the GPU target, sends a normal
`/execute` request to a CPU server to return the shared-object bytes, then sends
another normal `/execute` request to upload and run that library on the GPU
server. No additional endpoint or instruction format is involved.

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
KCORAL_GPU_TEST=1 pytest -q
```

The CPU compilation integration test needs the compiler group and a CUDA
toolchain, but no GPU. It runs whenever `nvcc`, `ninja` and a host C++ compiler
are present, and skips itself otherwise:

```bash
uv run --group test --group compiler pytest -q
```
