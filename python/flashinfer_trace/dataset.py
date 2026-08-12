"""Directory-backed storage for FlashInfer Trace datasets."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import Definition, Solution, Trace, TraceSetSummary


@dataclass
class TraceSet:
    """An in-memory or directory-backed FlashInfer Trace dataset."""

    root: Path | None = None
    definitions: dict[str, Definition] = field(default_factory=dict)
    solutions: dict[str, list[Solution]] = field(default_factory=dict)
    workloads: dict[str, list[Trace]] = field(default_factory=dict)
    traces: dict[str, list[Trace]] = field(default_factory=dict)
    _solutions_by_name: dict[str, Solution] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.root is not None:
            self.root = Path(self.root)
        for solutions in self.solutions.values():
            for solution in solutions:
                if solution.name in self._solutions_by_name:
                    raise ValueError(f"duplicate solution name: {solution.name}")
                self._solutions_by_name[solution.name] = solution

    @property
    def definitions_path(self) -> Path:
        """Return the definitions directory."""

        return self._root_path("definitions")

    @property
    def solutions_path(self) -> Path:
        """Return the solutions directory."""

        return self._root_path("solutions")

    @property
    def workloads_path(self) -> Path:
        """Return the workload declarations directory."""

        return self._root_path("workloads")

    @property
    def traces_path(self) -> Path:
        """Return the evaluation traces directory."""

        return self._root_path("traces")

    def _root_path(self, name: str) -> Path:
        if self.root is None:
            raise ValueError("TraceSet root is not set")
        return self.root / name

    @classmethod
    def from_path(cls, path: str | Path | None = None) -> TraceSet:
        """Load definitions, solutions, workloads, and traces from a directory."""

        if path is None:
            path = os.environ.get(
                "FIB_DATASET_PATH",
                str(Path.home() / ".cache" / "flashinfer_bench" / "dataset"),
            )
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        trace_set = cls(root=root)

        for file_path in sorted(trace_set.definitions_path.rglob("*.json")):
            definition = Definition.model_validate_json(file_path.read_text(encoding="utf-8"))
            if definition.name in trace_set.definitions:
                raise ValueError(f"duplicate definition name: {definition.name}")
            trace_set.definitions[definition.name] = definition

        for file_path in sorted(trace_set.solutions_path.rglob("*.json")):
            solution = Solution.model_validate_json(file_path.read_text(encoding="utf-8"))
            if solution.name in trace_set._solutions_by_name:
                raise ValueError(f"duplicate solution name: {solution.name}")
            trace_set.solutions.setdefault(solution.definition, []).append(solution)
            trace_set._solutions_by_name[solution.name] = solution

        for file_path in sorted(trace_set.workloads_path.rglob("*.jsonl")):
            for value in _load_json_lines(file_path):
                trace = Trace.model_validate(value)
                if not trace.is_workload_trace():
                    raise ValueError(f"workload file contains an evaluation trace: {file_path}")
                trace_set.workloads.setdefault(trace.definition, []).append(trace)

        for file_path in sorted(trace_set.traces_path.rglob("*.jsonl")):
            for value in _load_json_lines(file_path):
                trace = Trace.model_validate(value)
                if trace.is_workload_trace():
                    raise ValueError(f"trace file contains a workload-only trace: {file_path}")
                trace_set.traces.setdefault(trace.definition, []).append(trace)
        return trace_set

    def get_solution(self, name: str) -> Solution | None:
        """Find a solution by its globally unique name."""

        return self._solutions_by_name.get(name)

    def add_traces(self, traces: list[Trace]) -> None:
        """Validate and append evaluation traces as one in-memory batch."""

        validated_traces = [Trace.model_validate(trace) for trace in traces]
        output_files: dict[Path, list[Trace]] = defaultdict(list)

        for trace in validated_traces:
            if trace.is_workload_trace():
                raise ValueError("add_traces does not accept workload-only traces")
            definition = self.definitions.get(trace.definition)
            if definition is None:
                raise ValueError(f"unknown definition: {trace.definition}")
            solution = self._solutions_by_name.get(trace.solution or "")
            if solution is None:
                raise ValueError(f"unknown solution: {trace.solution}")
            if solution.definition != definition.name:
                raise ValueError(
                    f"solution {solution.name!r} targets definition "
                    f"{solution.definition!r}, not {definition.name!r}"
                )
            if self.root is not None:
                for label, segment in (
                    ("solution author", solution.author),
                    ("operation type", definition.op_type),
                    ("definition name", definition.name),
                ):
                    _validate_path_segment(label, segment)
                output_path = self.traces_path / solution.author / definition.op_type
                output_files[output_path / f"{definition.name}.jsonl"].append(trace)

        for path, values in output_files.items():
            _append_json_lines(path, values)
        for trace in validated_traces:
            self.traces.setdefault(trace.definition, []).append(trace)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the complete dataset."""

        return {
            "definitions": {
                name: definition.model_dump(mode="json")
                for name, definition in self.definitions.items()
            },
            "solutions": {
                name: [solution.model_dump(mode="json") for solution in solutions]
                for name, solutions in self.solutions.items()
            },
            "workloads": {
                name: [trace.model_dump(mode="json") for trace in traces]
                for name, traces in self.workloads.items()
            },
            "traces": {
                name: [trace.model_dump(mode="json") for trace in traces]
                for name, traces in self.traces.items()
            },
        }

    def summary(self) -> TraceSetSummary:
        """Count passed and failed evaluation traces."""

        traces = [trace for values in self.traces.values() for trace in values]
        passed = sum(trace.is_successful() for trace in traces)
        return TraceSetSummary(total=len(traces), passed=passed, failed=len(traces) - passed)


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            values.append(value)
    return values


def _append_json_lines(path: Path, values: list[Trace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    appended = "".join(value.model_dump_json() + "\n" for value in values).encode()
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        existing = path.read_bytes() if path.exists() else b""
        if existing and not existing.endswith(b"\n"):
            existing += b"\n"
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as temporary_file:
                temporary_file.write(existing)
                temporary_file.write(appended)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _validate_path_segment(label: str, value: str) -> None:
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a single path segment: {value!r}")


__all__ = ["TraceSet"]
