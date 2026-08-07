"""Run a FlashInfer Trace dataset through benchmark-server.

Usage: ``python examples/flashinfer_client.py TRACE_SET DEFINITION SOLUTION [SERVER_URL]``

The bundled Trace API does not depend on the separate ``flashinfer-bench``
package.
"""

from __future__ import annotations

import sys

from flashinfer_bench import BenchmarkConfig, FlashInferBenchmark, TraceSet


def main() -> None:
    if len(sys.argv) not in (4, 5):
        raise SystemExit("usage: flashinfer_client.py TRACE_SET DEFINITION SOLUTION [SERVER_URL]")
    trace_set = TraceSet.from_path(sys.argv[1])
    definition_name = sys.argv[2]
    solution_name = sys.argv[3]
    server_url = sys.argv[4] if len(sys.argv) == 5 else "http://127.0.0.1:8000"
    try:
        definition = trace_set.definitions[definition_name]
    except KeyError:
        raise SystemExit(f"definition not found: {definition_name}") from None
    solution = trace_set.get_solution(solution_name)
    if solution is None:
        raise SystemExit(f"solution not found: {solution_name}")
    workloads = [item.workload for item in trace_set.workloads.get(definition_name, [])]
    config = BenchmarkConfig()

    with FlashInferBenchmark(server_url, config) as benchmark:
        traces = benchmark.run(
            definition,
            solution,
            workloads,
            trace_set_root=trace_set.root,
        )

    trace_set.add_traces(traces)
    passed = sum(trace.evaluation.status.value == "PASSED" for trace in traces)
    print(f"evaluated={len(traces)} passed={passed} failed={len(traces) - passed}")


if __name__ == "__main__":
    main()
