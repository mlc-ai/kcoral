"""Command-line entry point for remote FlashInfer Trace evaluation."""

from __future__ import annotations

import argparse

from .client import FlashInferTraceClient
from .dataset import TraceSet


def main() -> None:
    """Evaluate one solution against every workload for a definition."""

    parser = argparse.ArgumentParser(prog="python -m flashinfer_trace")
    parser.add_argument("trace_set")
    parser.add_argument("definition")
    parser.add_argument("solution")
    parser.add_argument("server_url", nargs="?", default="http://127.0.0.1:8000")
    arguments = parser.parse_args()

    trace_set = TraceSet.from_path(arguments.trace_set)
    if arguments.definition not in trace_set.definitions:
        parser.error(f"unknown definition: {arguments.definition!r}")
    solution = trace_set.get_solution(arguments.solution)
    if solution is None:
        parser.error(f"unknown solution: {arguments.solution!r}")
    definition = trace_set.definitions[arguments.definition]
    workload_traces = trace_set.workloads.get(definition.name, [])

    with FlashInferTraceClient(arguments.server_url) as client:
        traces = client.evaluate_many(
            definition,
            solution,
            workload_traces,
            resource_root=trace_set.root,
        )
    trace_set.add_traces(traces)

    passed = sum(trace.is_successful() for trace in traces)
    print(f"evaluated={len(traces)} passed={passed} failed={len(traces) - passed}")


if __name__ == "__main__":
    main()
