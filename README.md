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
docker compose up --build
```

After the image has been built, source changes are available through the bind
mount and do not require another build:

```bash
docker compose up
```

Run the test suite in a temporary container:

```bash
docker compose run --rm benchmark-server \
  pytest -q /workspace/benchmark-server/tests/test_server.py
```

The server uses GPU `0` and host port `8000` by default. Override them when
needed:

```bash
BENCHMARK_DEVICES=0,1 BENCHMARK_PORT=8001 docker compose up
```

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
