"""Tests for the self-contained FlashInfer Trace client adapter."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
import uvicorn

import flashinfer_bench as flashinfer
from benchmark_server.app import create_app
from benchmark_server.client import Client, ProgramResult
from benchmark_server.config import ServerConfig
from benchmark_server.gpu_runtime import gpu_runtime_factory
from benchmark_server.schemas import parse_program


def test_flashinfer_logic_lives_in_dedicated_package():
    assert flashinfer.FlashInferBenchmark.__module__.startswith("flashinfer_bench.")
    assert flashinfer.TraceSet.__module__.startswith("flashinfer_bench.")
    assert importlib.util.find_spec("benchmark_server.flashinfer") is None
    assert importlib.util.find_spec("benchmark_server.flashinfer_trace") is None


@dataclass(frozen=True)
class _Source:
    path: str
    content: str


@dataclass(frozen=True)
class _Spec:
    language: str = "python"
    entry_point: str = "solution.py::run"
    destination_passing_style: bool = False
    binding: str | None = None
    dependencies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Solution:
    name: str = "solution"
    spec: _Spec = field(default_factory=_Spec)
    sources: list[_Source] = field(
        default_factory=lambda: [_Source("solution.py", "def run(a):\n    return a + 1\n")]
    )

    def get_entry_path(self) -> Path:
        return Path(self.spec.entry_point.split("::", 1)[0])

    def get_entry_symbol(self) -> str:
        return self.spec.entry_point.split("::", 1)[1]


class _Definition:
    name = "add_one"
    reference = "def run(a):\n    return a + 1\n"
    inputs: ClassVar[dict[str, Any]] = {"a": SimpleNamespace(dtype="float32")}
    outputs: ClassVar[dict[str, Any]] = {"out": SimpleNamespace(dtype="float32")}

    @staticmethod
    def get_output_shapes(_axes):
        return [(4,)]


@dataclass(frozen=True)
class _Workload:
    uuid: str = "workload-1"
    axes: dict[str, int] = field(default_factory=dict)


def _eval_config(**overrides):
    values = {
        "rtol": 1e-2,
        "atol": 1e-3,
        "warmup_runs": 2,
        "iterations": 5,
        "num_trials": 1,
        "profile_baseline": True,
        "required_matched_ratio": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _job(*, solution=None, trials=None, eval_config=None):
    return flashinfer.BenchmarkJob(
        definition=_Definition(),
        solution=solution or _Solution(),
        workload=_Workload(),
        trials=trials or [[np.arange(4, dtype=np.float32)]],
        eval_config=eval_config or _eval_config(),
        timeout_seconds=30,
    )


def _completed(built, *, overrides=None):
    results: dict[str, Any] = {}
    for key in built.metadata_keys:
        results[key] = {
            "shape_matches": True,
            "dtype_matches": True,
            "actual_shape": [4],
            "expected_shape": [4],
            "actual_dtype": "float32",
            "expected_dtype": "float32",
        }
    for key in built.correctness_keys:
        results[key] = {
            "passed": True,
            "nonfinite": None,
            "max_abs_err": 0.25,
            "max_rel_err": 0.125,
            "rtol": 0.01,
            "atol": 0.001,
        }
    for key in built.solution_timing_keys:
        results[key] = {"latency_ms_median": 2.0}
    for key in built.reference_timing_keys:
        results[key] = {"latency_ms_median": 4.0}
    results.update(overrides or {})
    return ProgramResult(
        status="COMPLETED",
        request_id="request-1",
        queue_ms=1.0,
        elapsed_ms=20.0,
        results=results,
        stdout="",
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
    )


def test_program_builder_keeps_benchmark_logic_client_side_and_uploads_tensor_blobs():
    value = np.arange(4, dtype=np.float32)
    built = flashinfer.FlashInferProgramBuilder().build(
        _job(trials=[[value], [value]], eval_config=_eval_config(num_trials=2))
    )

    # Parsing the emitted wire program proves every handle dependency is ordered.
    parse_program({"instructions": built.program.instructions})
    tensor_uploads = [
        instruction
        for instruction in built.program.instructions
        if instruction["op"] == "upload" and instruction["kind"] == "tensor"
    ]
    assert len(tensor_uploads) == 2
    assert tensor_uploads[0]["blob"] == tensor_uploads[1]["blob"]
    assert set(built.program._blobs) == {tensor_uploads[0]["blob"]}
    assert built.metadata_keys == ("metadata_0_0", "metadata_1_0")
    assert built.correctness_keys == ("correctness_0_0", "correctness_1_0")
    assert built.solution_timing_keys == ("solution_timing_0", "solution_timing_1")

    operations = [
        (item["op"], item.get("id"), item.get("key")) for item in built.program.instructions
    ]
    assert operations.index(("run", "metadata_gate", None)) > operations.index(
        ("return", None, "metadata_1_0")
    )
    assert operations.index(("run", "correctness_gate", None)) > operations.index(
        ("return", None, "correctness_1_0")
    )
    assert operations.index(("run", "solution_timing_0", None)) > operations.index(
        ("run", "correctness_gate", None)
    )


def test_program_builder_supports_destination_passing_and_optional_reference_timing():
    solution = _Solution(spec=_Spec(destination_passing_style=True))
    built = flashinfer.FlashInferProgramBuilder().build(
        _job(solution=solution, eval_config=_eval_config(profile_baseline=False))
    )
    instructions = built.program.instructions
    invocation = next(item for item in instructions if item.get("id") == "solution_result_0")
    assert invocation["args"][-1] == {"$ref": "solution_output_0_0"}
    assert built.reference_timing_keys == ()


def test_default_solution_adapter_rejects_unsupported_builds_explicitly():
    builder = flashinfer.FlashInferProgramBuilder()
    multi_source = _Solution(
        sources=[
            _Source("solution.py", "def run(a): return a"),
            _Source("helper.py", "VALUE = 1"),
        ]
    )
    with pytest.raises(flashinfer.UnsupportedFlashInferFeature, match="one source file"):
        builder.build(_job(solution=multi_source))

    cpp = _Solution(spec=_Spec(language="cpp"))
    with pytest.raises(flashinfer.UnsupportedFlashInferFeature, match="no SolutionAdapter"):
        builder.build(_job(solution=cpp))


def test_input_provider_is_deterministic_and_reads_safetensors_without_extra_package(tmp_path):
    stored = np.arange(4, dtype=np.float32)
    header = json.dumps(
        {
            "stored": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [0, stored.nbytes],
            }
        }
    ).encode()
    header += b" " * (-len(header) % 8)
    tensor_path = tmp_path / "inputs.safetensors"
    tensor_path.write_bytes(struct.pack("<Q", len(header)) + header + stored.tobytes())
    definition = flashinfer.Definition(
        name="inputs",
        op_type="test",
        axes={"N": flashinfer.AxisConst(value=4)},
        inputs={
            "random": flashinfer.TensorSpec(shape=["N"], dtype="float32"),
            "scalar": flashinfer.TensorSpec(shape=None, dtype="int32"),
            "stored": flashinfer.TensorSpec(shape=["N"], dtype="float32"),
        },
        outputs={"out": flashinfer.TensorSpec(shape=["N"], dtype="float32")},
        reference="def run(random, scalar, stored):\n    return random + scalar + stored\n",
    )
    workload = flashinfer.Workload(
        axes={},
        inputs={
            "random": flashinfer.RandomInput(),
            "scalar": flashinfer.ScalarInput(value=2),
            "stored": flashinfer.SafetensorsInput(path="inputs.safetensors", tensor_key="stored"),
        },
        uuid="deterministic-inputs",
    )
    provider = flashinfer.DefaultWorkloadInputProvider()

    first = provider.prepare(definition, workload, num_trials=2, trace_set_root=tmp_path)
    second = provider.prepare(definition, workload, num_trials=2, trace_set_root=tmp_path)

    assert np.array_equal(first[0][0], second[0][0])
    assert not np.array_equal(first[0][0], first[1][0])
    assert first[0][1] == 2
    assert np.array_equal(first[0][2], stored)


def test_result_mapper_produces_correctness_speed_and_failure_statuses():
    built = flashinfer.FlashInferProgramBuilder().build(_job())
    mapper = flashinfer.FlashInferResultMapper()

    passed = mapper.map(_completed(built), built)
    assert passed.status == "PASSED"
    assert passed.correctness["max_absolute_error"] == 0.25
    assert passed.performance == {
        "latency_ms": 2.0,
        "reference_latency_ms": 4.0,
        "speedup_factor": 2.0,
    }

    bad_shape = _completed(
        built, overrides={built.metadata_keys[0]: {"shape_matches": False, "dtype_matches": True}}
    )
    assert mapper.map(bad_shape, built).status == "INCORRECT_SHAPE"

    bad_dtype = _completed(
        built, overrides={built.metadata_keys[0]: {"shape_matches": True, "dtype_matches": False}}
    )
    assert mapper.map(bad_dtype, built).status == "INCORRECT_DTYPE"

    bad_number = _completed(
        built,
        overrides={
            built.correctness_keys[0]: {
                "passed": False,
                "nonfinite": "inf",
                "max_abs_err": 0.0,
                "max_rel_err": 0.0,
            }
        },
    )
    numerical = mapper.map(bad_number, built)
    assert numerical.status == "INCORRECT_NUMERICAL"
    assert numerical.correctness["max_absolute_error"] == float("inf")


def test_result_mapper_classifies_server_compile_failure():
    built = flashinfer.FlashInferProgramBuilder().build(_job())
    outcome = ProgramResult(
        status="FAILED",
        request_id="request-2",
        queue_ms=0.0,
        elapsed_ms=1.0,
        results={},
        stdout="",
        stderr="compiler diagnostic",
        stdout_truncated=False,
        stderr_truncated=False,
        error={"kind": "compile", "message": "bad kernel"},
    )
    data = flashinfer.FlashInferResultMapper().map(outcome, built)
    assert data.status == "COMPILE_ERROR"
    assert "compiler diagnostic" in data.log


class _InputProvider:
    def __init__(self):
        self.calls = 0

    def prepare(self, *_args, **_kwargs):
        self.calls += 1
        return [[np.arange(4, dtype=np.float32)]]


class _RecordingClient:
    def __init__(self):
        self.programs = []
        self.threads = set()
        self._lock = threading.Lock()

    def health(self):
        return {"gpu_count": 2}

    def execute(self, program, **_kwargs):
        with self._lock:
            self.programs.append(program)
            self.threads.add(threading.current_thread().name)
        returns = [item["key"] for item in program.instructions if item["op"] == "return"]
        results = {}
        for key in returns:
            if key.startswith("metadata"):
                results[key] = {"shape_matches": True, "dtype_matches": True}
            elif key.startswith("correctness"):
                results[key] = {
                    "passed": True,
                    "nonfinite": None,
                    "max_abs_err": 0.0,
                    "max_rel_err": 0.0,
                }
            elif key.startswith("solution_timing"):
                results[key] = {"latency_ms_median": 1.0}
            elif key.startswith("reference_timing"):
                results[key] = {"latency_ms_median": 2.0}
        return ProgramResult(
            status="COMPLETED",
            request_id="recorded",
            queue_ms=0.0,
            elapsed_ms=1.0,
            results=results,
            stdout="",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
        )


class _Config:
    timeout_seconds = 30

    @staticmethod
    def resolve_eval_config(_definition):
        return _eval_config()


def _bundled_trace_set(tmp_path, solution_names=("one", "two")):
    definition = flashinfer.Definition(
        name="add_one",
        op_type="elementwise",
        axes={"N": flashinfer.AxisConst(value=4)},
        inputs={"a": flashinfer.TensorSpec(shape=["N"], dtype="float32")},
        outputs={"out": flashinfer.TensorSpec(shape=["N"], dtype="float32")},
        reference="def run(a):\n    return a + 1\n",
    )
    solutions = [
        flashinfer.Solution(
            name=name,
            definition=definition.name,
            author="test",
            spec=flashinfer.BuildSpec(
                language="python",
                target_hardware=["cuda"],
                entry_point="solution.py::run",
                destination_passing_style=False,
            ),
            sources=[flashinfer.SourceFile(path="solution.py", content=definition.reference)],
        )
        for name in solution_names
    ]
    workload = flashinfer.Workload(
        axes={}, inputs={"a": flashinfer.RandomInput()}, uuid="workload-1"
    )
    trace_set = flashinfer.TraceSet(
        root=tmp_path,
        definitions={definition.name: definition},
        solutions={definition.name: solutions},
        workloads={
            definition.name: [flashinfer.Trace(definition=definition.name, workload=workload)]
        },
        traces={},
    )
    return definition, solutions, workload, trace_set


def test_high_level_client_runs_one_definition_solution_pair(tmp_path):
    definition, solutions, workload, _trace_set = _bundled_trace_set(tmp_path)
    workloads = [workload, workload.model_copy(update={"uuid": "workload-2"})]
    inputs = _InputProvider()
    client = _RecordingClient()
    benchmark = flashinfer.FlashInferBenchmark(
        client,
        _Config(),
        input_provider=inputs,
    )

    traces = benchmark.run(
        definition,
        solutions[0],
        workloads,
        trace_set_root=tmp_path,
    )

    assert not hasattr(benchmark, "run_all")
    assert inputs.calls == 2
    assert len(client.programs) == 2
    assert [trace.workload.uuid for trace in traces] == ["workload-1", "workload-2"]
    assert [trace.solution for trace in traces] == ["one", "one"]


def test_pair_calls_reuse_tensor_keys_and_leave_persistence_to_caller(tmp_path):
    definition, solutions, workload, trace_set = _bundled_trace_set(tmp_path)
    client = _RecordingClient()
    benchmark = flashinfer.FlashInferBenchmark(
        client,
        _Config(),
        input_provider=_InputProvider(),
        max_workers=1,
    )

    traces = []
    for solution in solutions:
        traces.extend(
            benchmark.run(definition, solution, [workload], trace_set_root=trace_set.root)
        )

    assert len(client.programs) == 2
    input_blobs = []
    for program in client.programs:
        upload = next(
            item
            for item in program.instructions
            if item["op"] == "upload" and item["kind"] == "tensor"
        )
        input_blobs.append(upload["blob"])
        assert upload["blob"] in program._blobs
    assert input_blobs[0] == input_blobs[1]
    assert [trace.solution for trace in traces] == ["one", "two"]
    assert trace_set.traces == {}

    trace_set.add_traces(traces)

    assert [trace.solution for trace in trace_set.traces["add_one"]] == ["one", "two"]
    persisted = tmp_path / "traces" / "test" / "elementwise" / "add_one.jsonl"
    assert persisted.exists()
    assert '"solution":"one"' in persisted.read_text(encoding="utf-8")
    assert '"solution":"two"' in persisted.read_text(encoding="utf-8")


def test_high_level_client_rejects_a_solution_for_another_definition(tmp_path):
    definition, solutions, workload, _trace_set = _bundled_trace_set(tmp_path)
    other = definition.model_copy(update={"name": "other"})
    benchmark = flashinfer.FlashInferBenchmark(_RecordingClient(), _Config())

    with pytest.raises(ValueError, match="targets definition"):
        benchmark.run(other, solutions[0], [workload], trace_set_root=tmp_path)


def test_bundled_trace_objects_round_trip_without_external_package(tmp_path):
    definition = flashinfer.Definition(
        name="remote_add_one",
        op_type="elementwise",
        axes={"N": flashinfer.AxisConst(value=4)},
        inputs={"a": flashinfer.TensorSpec(shape=["N"], dtype="float32")},
        outputs={"out": flashinfer.TensorSpec(shape=["N"], dtype="float32")},
        reference="def run(a):\n    return a + 1\n",
    )
    solution = flashinfer.Solution(
        name="python_add_one",
        definition=definition.name,
        author="test",
        spec=flashinfer.BuildSpec(
            language="python",
            target_hardware=["cuda"],
            entry_point="solution.py::run",
            destination_passing_style=False,
        ),
        sources=[flashinfer.SourceFile(path="solution.py", content=definition.reference)],
    )
    workload = flashinfer.Workload(
        axes={}, inputs={"a": flashinfer.RandomInput()}, uuid="public-interface-workload"
    )
    trace_set = flashinfer.TraceSet(
        root=tmp_path,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={
            definition.name: [flashinfer.Trace(definition=definition.name, workload=workload)]
        },
        traces={},
    )
    config = flashinfer.BenchmarkConfig(
        warmup_runs=1,
        iterations=1,
        num_trials=1,
        profile_baseline=True,
    )

    traces = flashinfer.FlashInferBenchmark(
        _RecordingClient(),
        config,
        input_provider=_InputProvider(),
        max_workers=1,
    ).run(definition, solution, [workload], trace_set_root=trace_set.root)

    trace = traces[0]
    assert isinstance(trace, flashinfer.Trace)
    assert trace.solution == solution.name
    assert trace.evaluation.status == flashinfer.EvaluationStatus.PASSED
    assert trace.evaluation.performance.speedup_factor == 2.0


def test_trace_set_loads_established_layout_and_round_trips_json(tmp_path):
    definition, solutions, workload, _trace_set = _bundled_trace_set(tmp_path)
    definition_path = tmp_path / "definitions" / "elementwise" / "add_one.json"
    solution_path = tmp_path / "solutions" / "test" / "elementwise" / "one.json"
    workload_path = tmp_path / "workloads" / "elementwise" / "add_one.jsonl"
    definition_path.parent.mkdir(parents=True)
    solution_path.parent.mkdir(parents=True)
    workload_path.parent.mkdir(parents=True)
    definition_path.write_text(definition.model_dump_json(), encoding="utf-8")
    solution_path.write_text(solutions[0].model_dump_json(), encoding="utf-8")
    workload_path.write_text(
        flashinfer.Trace(definition=definition.name, workload=workload).model_dump_json() + "\n",
        encoding="utf-8",
    )

    loaded = flashinfer.TraceSet.from_path(tmp_path)

    assert loaded.definitions[definition.name] == definition
    assert loaded.get_solution("one") == solutions[0]
    assert loaded.workloads[definition.name][0].workload == workload
    assert loaded.summary().model_dump() == {"total": 0, "passed": 0, "failed": 0}


@pytest.mark.skipif(
    os.environ.get("BENCH_GPU_TEST") != "1",
    reason="real FlashInfer adapter test requires BENCH_GPU_TEST=1",
)
def test_real_flashinfer_jobs_execute_on_gpu_and_reuse_server_tensor_cache(tmp_path):
    definition = flashinfer.Definition(
        name="remote_gpu_add_one",
        op_type="elementwise",
        axes={"N": flashinfer.AxisConst(value=256)},
        inputs={"a": flashinfer.TensorSpec(shape=["N"], dtype="float32")},
        outputs={"out": flashinfer.TensorSpec(shape=["N"], dtype="float32")},
        reference="def run(a):\n    return a + 1\n",
    )
    solutions = [
        flashinfer.Solution(
            name=name,
            definition=definition.name,
            author="test",
            spec=flashinfer.BuildSpec(
                language="python",
                target_hardware=["cuda"],
                entry_point="solution.py::run",
                destination_passing_style=False,
            ),
            sources=[flashinfer.SourceFile(path="solution.py", content=definition.reference)],
        )
        for name in ("python_add_one_a", "python_add_one_b")
    ]
    cuda_source = r"""
__global__ void add_one_kernel(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + 1.0f;
}

void add_one(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  int n = static_cast<int>(x.numel());
  add_one_kernel<<<(n + 255) / 256, 256>>>(static_cast<const float*>(x.data_ptr()),
                                           static_cast<float*>(y.data_ptr()), n);
}
"""
    solutions.append(
        flashinfer.Solution(
            name="cuda_add_one",
            definition=definition.name,
            author="test",
            spec=flashinfer.BuildSpec(
                language="cuda",
                target_hardware=["cuda"],
                entry_point="solution.cu::add_one",
                destination_passing_style=True,
                binding="tvm-ffi",
            ),
            sources=[flashinfer.SourceFile(path="solution.cu", content=cuda_source)],
        )
    )
    workload = flashinfer.Workload(
        axes={}, inputs={"a": flashinfer.RandomInput()}, uuid="gpu-cache-workload"
    )
    config = flashinfer.BenchmarkConfig(
        warmup_runs=1,
        iterations=3,
        num_trials=1,
        profile_baseline=True,
    )

    gpu_text = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    gpu_id = int(gpu_text) if gpu_text.isdigit() else 0
    app = create_app(ServerConfig(gpus=[gpu_id]), runtime_factory=gpu_runtime_factory)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("GPU test server did not start")
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]

    cache_attempts = []
    try:
        with Client(f"http://127.0.0.1:{port}") as client:
            original_post = client._post_program

            def record_post(program, options, include_blobs):
                cache_attempts.append(set(include_blobs))
                return original_post(program, options, include_blobs)

            client._post_program = record_post
            benchmark = flashinfer.FlashInferBenchmark(
                client,
                config,
                max_workers=1,
            )
            traces = []
            for solution in solutions:
                traces.extend(
                    benchmark.run(
                        definition,
                        solution,
                        [workload],
                        trace_set_root=tmp_path,
                    )
                )
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert [trace.evaluation.status for trace in traces] == [
        flashinfer.EvaluationStatus.PASSED,
        flashinfer.EvaluationStatus.PASSED,
        flashinfer.EvaluationStatus.PASSED,
    ]
    # First job: key-only miss then one blob resend. Later jobs: key-only warm hits.
    assert len(cache_attempts) == 4
    assert cache_attempts[0] == set()
    assert len(cache_attempts[1]) == 1
    assert cache_attempts[2] == set()
    assert cache_attempts[3] == set()
