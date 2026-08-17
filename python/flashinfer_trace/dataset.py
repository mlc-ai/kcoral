"""Directory-backed storage for FlashInfer Trace datasets."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .schema import Definition, Solution, Trace


class TraceSet:
    """A loaded FlashInfer Trace directory."""

    def __init__(self, path: str | Path) -> None:
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"trace set directory does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"trace set path is not a directory: {root}")

        definitions: dict[str, Definition] = {}
        solutions: dict[str, Solution] = {}
        workloads: defaultdict[str, list[Trace]] = defaultdict(list)
        traces: defaultdict[str, list[Trace]] = defaultdict(list)

        for file_path in sorted((root / "definitions").rglob("*.json")):
            definition = Definition.model_validate_json(file_path.read_text(encoding="utf-8"))
            if definition.name in definitions:
                raise ValueError(f"duplicate definition name: {definition.name}")
            definitions[definition.name] = definition

        for file_path in sorted((root / "solutions").rglob("*.json")):
            solution = Solution.model_validate_json(file_path.read_text(encoding="utf-8"))
            if solution.name in solutions:
                raise ValueError(f"duplicate solution name: {solution.name}")
            solutions[solution.name] = solution

        for file_path in sorted((root / "workloads").rglob("*.jsonl")):
            for value in _load_json_lines(file_path):
                trace = Trace.model_validate(value)
                if not trace.is_workload_trace():
                    raise ValueError(f"workload file contains an evaluation trace: {file_path}")
                workloads[trace.definition].append(trace)

        for file_path in sorted((root / "traces").rglob("*.jsonl")):
            for value in _load_json_lines(file_path):
                trace = Trace.model_validate(value)
                if trace.is_workload_trace():
                    raise ValueError(f"trace file contains a workload-only trace: {file_path}")
                traces[trace.definition].append(trace)

        self._path = root
        self._definitions = MappingProxyType(definitions)
        self._solutions = MappingProxyType(solutions)
        self._workloads = MappingProxyType(
            {name: tuple(values) for name, values in workloads.items()}
        )
        self._trace_values = {name: tuple(values) for name, values in traces.items()}
        self._traces = MappingProxyType(self._trace_values)

    @property
    def path(self) -> Path:
        """Return the dataset directory."""

        return self._path

    @property
    def definitions(self) -> Mapping[str, Definition]:
        """Return definitions indexed by name."""

        return self._definitions

    @property
    def solutions(self) -> Mapping[str, Solution]:
        """Return solutions indexed by their globally unique names."""

        return self._solutions

    @property
    def workloads(self) -> Mapping[str, tuple[Trace, ...]]:
        """Return workload declarations grouped by definition name."""

        return self._workloads

    @property
    def traces(self) -> Mapping[str, tuple[Trace, ...]]:
        """Return evaluation traces grouped by definition name."""

        return self._traces

    def append(self, traces: Sequence[Trace]) -> None:
        """Validate and persist one batch of evaluation traces."""

        validated_traces = [Trace.model_validate(trace) for trace in traces]
        output_files: dict[Path, list[Trace]] = defaultdict(list)
        traces_by_definition: dict[str, list[Trace]] = defaultdict(list)

        for trace in validated_traces:
            if trace.is_workload_trace():
                raise ValueError("append does not accept workload-only traces")
            definition = self.definitions.get(trace.definition)
            if definition is None:
                raise ValueError(f"unknown definition: {trace.definition}")
            solution = self.solutions.get(trace.solution or "")
            if solution is None:
                raise ValueError(f"unknown solution: {trace.solution}")
            if solution.definition != definition.name:
                raise ValueError(
                    f"solution {solution.name!r} targets definition "
                    f"{solution.definition!r}, not {definition.name!r}"
                )
            for label, segment in (
                ("solution author", solution.author),
                ("operation type", definition.op_type),
                ("definition name", definition.name),
            ):
                _validate_path_segment(label, segment)
            output_path = self.path / "traces" / solution.author / definition.op_type
            output_files[output_path / f"{definition.name}.jsonl"].append(trace)
            traces_by_definition[trace.definition].append(trace)

        for path, values in output_files.items():
            _append_json_lines(path, values)
        for definition_name, values in traces_by_definition.items():
            self._trace_values[definition_name] = (
                *self._trace_values.get(definition_name, ()),
                *values,
            )


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
