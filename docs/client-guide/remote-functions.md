# Remote Python Functions

Decorate a self-contained Python function and call `.remote()` to execute it
on a KCoral server. Each invocation uploads the function and its arguments,
runs it, and returns the selected result using the existing program protocol.
The client machine needs no GPU (graphics processing unit) libraries.

Save the function in a Python file so its source is available:

```python
import kcoral

@kcoral.function(endpoint="http://localhost:8000", timeout=30)
def gpu_sum(n: int):
    import torch

    x = torch.arange(n, dtype=torch.float32, device="cuda")
    return x.sum().item()

if __name__ == "__main__":
    print(gpu_sum.remote(4))  # 6.0
```

Start a [GPU server](../server-guide/launch-the-server.md) with PyTorch installed
before running this script. PyTorch is a tensor computation library; imports
inside the decorated function execute on the server during `.remote()`.
Calling `gpu_sum(4)` without `.remote()` instead runs the original function
locally, requiring the same dependencies and hardware on the client.

## Configure the endpoint in Python

`endpoint` is the server base URL and can include a reverse proxy's path prefix.
A reverse proxy forwards requests to the KCoral service. Set `execute_path` if
that proxy exposes execution under a different route:

```python
@kcoral.function(
    endpoint="https://compute.example.org/lab",
    execute_path="/jobs/run",
    timeout=60,
)
def double(value):
    return value * 2
```

This posts to `https://compute.example.org/lab/jobs/run`. A leading slash in
`execute_path` does not remove the base URL's prefix. The default path is
`/execute`. These options configure the client; they do not change server routes.
Paths cannot contain a scheme, host, query, or fragment.

If `endpoint` is omitted, each invocation reads `KCORAL_URL`, falling back to
`http://localhost:8000`. An explicit endpoint takes precedence over the environment.
The standalone decorator opens and closes a client for each invocation.

To reuse connections or provide request headers, bind functions to a `Client`:

```python
from kcoral import Client

with Client(
    "https://compute.example.org/lab",
    execute_path="/jobs/run",
    health_path="/status",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
) as client:

    @client.function(timeout=30)
    def double(value):
        return value * 2

    print(double.remote(21))
```

The client owns these connections; keep it open until the calls finish.
Its custom paths also apply to ordinary `Client.execute()` and `Client.health()`.

## Arguments and results

Positional arguments, keyword arguments, defaults, `*args`, and `**kwargs` follow
the original Python signature. Defaults are bound on the client and sent as
explicit arguments. Supported values are:

- JSON values: null (`None`), booleans, finite numbers, strings, lists, and
  dictionaries with string keys. JSON is the structured text format used by the
  execution protocol. Tuples and arbitrary Python objects are not accepted.
- Bytes, byte arrays, and memory views, uploaded as bytes.
- NumPy arrays and tensor objects implementing DLPack, a tensor interchange
  protocol also provided by PyTorch. They are uploaded as tensors.

Pass bytes and tensors as whole arguments, not nested inside lists or dictionaries.
For example, `evaluate.remote(inputs, reference=expected)` can upload two tensors.
JSON objects resembling protocol references, such as `{"$ref": "name"}`, remain data.
Uploads and JSON containers are snapshotted when a program is built.

Results follow the ordinary [client decoding rules](writing-a-program.md#submit-and-read-results):
tensor results arrive as NumPy arrays in client memory, bytes as Python bytes,
and structured results as ordinary Python values. No pickle serialization is used.

Here is a complete example with a local NumPy input, uploaded to the GPU:

```{literalinclude} ../../examples/remote_function.py
:language: python
```

Run `python examples/remote_function.py --endpoint http://localhost:8000`.
For a proxy, also pass `--execute-path /jobs/run` as appropriate.
You can {download}`download remote_function.py <../../examples/remote_function.py>`.

## Errors and execution metadata

`.remote()` returns the function value on success and raises
`RemoteExecutionError` on an instruction failure. The exception's `result`
contains the full `ProgramResult`, including `request_id`, `error`, `stdout`
and `stderr`; `error["traceback"]` retains the remote traceback. HTTP request
errors and connection failures retain the ordinary client exception types.

Use `.execute()` to receive the full outcome on both success and failure:

```python
result = double.execute(21)
if result.completed:
    print(result.results["output"])
else:
    print(result.error)
print(result.request_id, result.stdout, result.stderr)
```

`timeout` maps to the program's `timeout_seconds`; `output_limit_bytes` limits
captured output per stream. Both are subject to server limits. See
[time limits](../server-guide/launch-the-server.md#time-and-size-limits) for
queue and execution timing. Functions declared with `cpu_only=True` must touch
no GPU; their execution releases the GPU reservation under the existing protocol.

## Source and execution boundaries

Only the decorated function's source is uploaded. Import dependencies **inside
the function** and install them on the server. External helper functions,
module globals, and captured variables are not copied; their use is rejected
when decorating. Pass values as arguments and define helpers inside the function.
Additional decorators, asynchronous functions, generators, and functions without
available source are unsupported. Annotations are omitted from uploaded signatures.

Each call is a separate program. Remote tensors, mutations and function state do
not survive into the next call. Put allocation, kernel execution, correctness
checks and measurement into one remote function when they need the same data.

For explicit instruction control, keep using `Program`. A decorated function's
`.build_program(*args, **kwargs)` returns an ordinary program selecting `output`,
without executing it. You can inspect its instructions or submit it with a
different client. When doing so, pass execution limits to `Client.execute()`;
the decorator's limits apply only through its `.remote()` and `.execute()` methods.
