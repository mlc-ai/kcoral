"""Tests for FlashInfer Trace schemas and directory storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashinfer_trace import (
    AxisConst,
    AxisVar,
    BenchmarkConfig,
    BuildSpec,
    Correctness,
    Definition,
    Environment,
    EvalConfig,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    SafetensorsInput,
    ScalarInput,
    Solution,
    SourceFile,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
)


def _definition() -> Definition:
    return Definition(
        name="vector_add",
        op_type="elementwise",
        axes={
            "batch": AxisVar(description="Batch size"),
            "width": AxisConst(value=4),
        },
        inputs={
            "left": TensorSpec(shape=["batch", "width"], dtype="float32"),
            "scale": TensorSpec(shape=None, dtype="float32"),
            "stored": TensorSpec(shape=["batch", "width"], dtype="float32"),
        },
        outputs={
            "output": TensorSpec(shape=["batch", "width"], dtype="float32"),
        },
        reference="def run(left, scale, stored):\n    return left + scale + stored\n",
        constraints=["batch > 0"],
    )


def _solution(definition: Definition) -> Solution:
    return Solution(
        name="python_vector_add",
        definition=definition.name,
        author="tests",
        spec=BuildSpec(
            language="python",
            target_hardware=["cuda"],
            entry_point="solution.py::run",
            destination_passing_style=False,
        ),
        sources=[
            SourceFile(
                path="solution.py",
                content="def run(left, scale, stored):\n    return left + scale + stored\n",
            )
        ],
    )


def _workload() -> Workload:
    return Workload(
        axes={"batch": 2},
        inputs={
            "left": RandomInput(),
            "scale": ScalarInput(value=2.0),
            "stored": SafetensorsInput(path="inputs.safetensors", tensor_key="stored"),
        },
        uuid="workload-1",
    )


def _evaluation(status: EvaluationStatus) -> Evaluation:
    if status == EvaluationStatus.PASSED:
        return Evaluation(
            status=status,
            environment=Environment(hardware="test-gpu", libs={"cuda": "13.0"}),
            timestamp="2026-08-12T00:00:00Z",
            correctness=Correctness(max_relative_error=0.01, max_absolute_error=0.02),
            performance=Performance(
                latency_ms=1.0,
                reference_latency_ms=2.0,
                speedup_factor=2.0,
            ),
        )
    return Evaluation(
        status=status,
        environment=Environment(hardware="test-gpu"),
        timestamp="2026-08-12T00:00:01Z",
        log="execution timed out",
    )


def _write_dataset_files(
    root: Path,
    definition: Definition,
    solution: Solution,
    workload: Workload,
) -> None:
    definition_path = root / "definitions" / "elementwise" / "vector_add.json"
    solution_path = root / "solutions" / "tests" / "elementwise" / "python_vector_add.json"
    workload_path = root / "workloads" / "elementwise" / "vector_add.jsonl"
    definition_path.parent.mkdir(parents=True)
    solution_path.parent.mkdir(parents=True)
    workload_path.parent.mkdir(parents=True)

    definition_data = definition.model_dump(mode="json")
    definition_data["future_field"] = {"ignored": True}
    definition_path.write_text(json.dumps(definition_data), encoding="utf-8")
    solution_path.write_text(solution.model_dump_json(), encoding="utf-8")
    workload_trace = Trace(definition=definition.name, workload=workload)
    workload_path.write_text(f"\n{workload_trace.model_dump_json()}\n", encoding="utf-8")


def test_models_validate_trace_json_and_resolve_configuration() -> None:
    definition_data = _definition().model_dump(mode="json")
    definition_data["unknown_field"] = "future-compatible"
    definition = Definition.model_validate(definition_data)

    assert definition.get_input_shapes({"batch": 2}) == [(2, 4), None, (2, 4)]
    assert definition.get_output_shapes({"batch": 2}) == [(2, 4)]
    assert definition.input_dtypes == ["float32", "float32", "float32"]
    assert "unknown_field" not in definition.model_dump()

    configuration = BenchmarkConfig(
        iterations=7,
        op_type_config={"elementwise": EvalConfig(warmup_runs=2, extra={"operation": True})},
        definition_config={
            "vector_add": EvalConfig(
                iterations=5,
                num_trials=4,
                extra={"definition": True},
            )
        },
    )
    resolved = configuration.resolve_eval_config(definition)

    assert resolved.warmup_runs == 2
    assert resolved.iterations == 7
    assert resolved.num_trials == 4
    assert resolved.extra == {"operation": True, "definition": True}

    invalid_axis = definition_data | {
        "axes": {"batch": {"type": "var"}, "width": {"type": "const", "value": True}}
    }
    with pytest.raises(ValueError):
        Definition.model_validate(invalid_axis)

    asynchronous_reference = definition_data | {
        "reference": "async def run(left, scale, stored):\n    return left + scale + stored\n"
    }
    with pytest.raises(ValueError, match="synchronous"):
        Definition.model_validate(asynchronous_reference)


def test_directory_load_append_and_reload_use_real_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        TraceSet(tmp_path / "missing")

    definition = _definition()
    solution = _solution(definition)
    workload = _workload()
    _write_dataset_files(tmp_path, definition, solution, workload)

    trace_set = TraceSet(tmp_path)

    assert trace_set.definitions[definition.name] == definition
    assert trace_set.solutions[solution.name] == solution
    assert trace_set.workloads[definition.name][0].workload == workload
    assert trace_set.traces == {}
    with pytest.raises(TypeError):
        trace_set.definitions["other"] = definition  # type: ignore[index]
    assert isinstance(trace_set.workloads[definition.name], tuple)

    passed_trace = Trace(
        definition=definition.name,
        workload=workload,
        solution=solution.name,
        evaluation=_evaluation(EvaluationStatus.PASSED),
    )
    invalid_trace = passed_trace.model_copy(update={"solution": "missing-solution"})
    trace_path = (
        tmp_path / "traces" / solution.author / definition.op_type / f"{definition.name}.jsonl"
    )

    with pytest.raises(ValueError, match="unknown solution"):
        trace_set.append([passed_trace, invalid_trace])

    assert trace_set.traces == {}
    assert not trace_path.exists()

    unsafe_solution = solution.model_copy(
        update={"name": "unsafe-solution", "author": "../outside"}
    )
    unsafe_root = tmp_path / "unsafe"
    _write_dataset_files(unsafe_root, definition, unsafe_solution, workload)
    unsafe_trace_set = TraceSet(unsafe_root)
    unsafe_trace = passed_trace.model_copy(update={"solution": unsafe_solution.name})
    with pytest.raises(ValueError, match="single path segment"):
        unsafe_trace_set.append([unsafe_trace])
    assert unsafe_trace_set.traces == {}

    timeout_trace = Trace(
        definition=definition.name,
        workload=workload.model_copy(update={"uuid": "workload-2"}),
        solution=solution.name,
        evaluation=_evaluation(EvaluationStatus.TIMEOUT),
    )
    trace_set.append([passed_trace, timeout_trace])

    persisted_records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(persisted_records) == 2
    assert persisted_records[1]["evaluation"]["correctness"] is None
    assert persisted_records[1]["evaluation"]["performance"] is None

    reloaded = TraceSet(tmp_path)

    assert [trace.workload.uuid for trace in reloaded.traces[definition.name]] == [
        "workload-1",
        "workload-2",
    ]
    assert reloaded.traces[definition.name][1].evaluation is not None
    assert reloaded.traces[definition.name][1].evaluation.performance is None
