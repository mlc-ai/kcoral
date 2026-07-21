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

## Python client

```python
from benchmark_server import Client

with Client("http://127.0.0.1:8000") as client:
    result = client.execute({"main.py": "def main():\n    return 42\n"})
    print(result.value)
```

## Documentation

- [English API reference](api-reference.md)
- [Chinese API reference](api-reference-zh.md)
