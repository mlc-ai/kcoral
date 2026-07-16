# Benchmark Server

Design documents for a synchronous remote GPU execution service. A client
uploads a `main.py` script together with artifact files; the server runs
`main()` in an isolated per-request working directory on one GPU and returns
the function's return value.

## Documents

- [api-reference.md](api-reference.md): full API reference (in Chinese),
  covering the `POST /execute` protocol, the `main()` entry contract, the
  recursive return-value encoding, error types, the execution model, and the
  file cache design.
