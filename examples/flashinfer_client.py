"""Run a FlashInfer Trace dataset through benchmark-server.

Usage: ``python examples/flashinfer_client.py TRACE_SET [SERVER_URL]``

The bundled Trace API does not depend on the separate ``flashinfer-bench``
package.
"""

from __future__ import annotations

import sys

from flashinfer_bench import BenchmarkConfig, FlashInferBenchmark, TraceSet


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: flashinfer_client.py TRACE_SET [SERVER_URL]")
    trace_set = TraceSet.from_path(sys.argv[1])
    server_url = sys.argv[2] if len(sys.argv) == 3 else "http://127.0.0.1:8000"
    config = BenchmarkConfig()

    with FlashInferBenchmark(trace_set, server_url, config) as benchmark:
        result = benchmark.run_all(dump_traces=True, resume=True)

    summary = result.summary()
    print(f"evaluated={summary.total} passed={summary.passed} failed={summary.failed}")


if __name__ == "__main__":
    main()
