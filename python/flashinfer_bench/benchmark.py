"""Client-side FlashInfer Trace orchestration for benchmark-server.

The server remains a generic instruction executor.  This module translates one
FlashInfer ``Solution`` and ``Workload`` into one :class:`Program`, submits many
such programs concurrently, and translates the replies back into bundled
``Trace`` objects.  The adapter and its data model do not depend on the separate
``flashinfer-bench`` package.

The first implementation intentionally covers the common, default numerical
evaluator and single-file Python, Triton, and TVM-FFI CUDA solutions.  The input
provider, solution adapter, program builder, and result mapper are separate
objects so support for other builders and evaluator policies can be added
without changing the wire protocol or the server.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Protocol

import ml_dtypes
import numpy as np

from benchmark_server import __version__
from benchmark_server.client import (
    BenchmarkServerError,
    Client,
    Program,
    ProgramResult,
    Register,
    TransportError,
)

from .models import (
    AxisConst,
    AxisVar,
    BenchmarkConfig,
    BuildSpec,
    Correctness,
    Definition,
    DType,
    Environment,
    EvalConfig,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    ResolvedEvalConfig,
    SafetensorsInput,
    ScalarInput,
    Solution,
    SourceFile,
    SupportedBindings,
    SupportedLanguages,
    TensorSpec,
    Trace,
    TraceSet,
    TraceSetSummary,
    Workload,
)

logger = logging.getLogger(__name__)


class UnsupportedFlashInferFeature(ValueError):
    """The trace uses a feature the default adapter does not implement yet."""


class ProgramExecutor(Protocol):
    """The subset of :class:`Client` used by :class:`FlashInferBenchmark`."""

    def execute(
        self,
        program: Program,
        *,
        timeout_seconds: float | None = None,
        output_limit_bytes: int | None = None,
    ) -> ProgramResult: ...

    def health(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class WorkloadInputProvider(Protocol):
    """Prepare reusable client-side values for one workload."""

    def prepare(
        self,
        definition: Definition,
        workload: Workload,
        *,
        num_trials: int,
        trace_set_root: Path | None,
    ) -> list[list[Any]]: ...


class SolutionAdapter(Protocol):
    """Upload and, when necessary, compile one FlashInfer solution."""

    def supports(self, solution: Any) -> bool: ...

    def add_to_program(self, program: Program, solution: Any) -> Register: ...


@dataclass(frozen=True)
class BenchmarkJob:
    """Everything needed to build one solution-workload server request."""

    definition: Any
    solution: Any
    workload: Any
    trials: list[list[Any]]
    eval_config: Any
    timeout_seconds: float


@dataclass(frozen=True)
class BenchmarkProgram:
    """A program plus the result keys needed to interpret its response."""

    program: Program
    metadata_keys: tuple[str, ...]
    correctness_keys: tuple[str, ...]
    solution_timing_keys: tuple[str, ...]
    reference_timing_keys: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationData:
    """FlashInfer-independent evaluation data produced by a server reply."""

    status: str
    log: str
    correctness: dict[str, Any] | None = None
    performance: dict[str, float] | None = None


class DefaultWorkloadInputProvider:
    """Generate deterministic CPU inputs from a FlashInfer ``Workload``.

    Inputs are prepared once per workload and reused for every solution.  Tensor
    uploads therefore have identical SHA-256 content keys across solution tasks,
    allowing :class:`Client` and the server byte cache to avoid repeat transfer.
    """

    def prepare(
        self,
        definition: Definition,
        workload: Workload,
        *,
        num_trials: int,
        trace_set_root: Path | None,
    ) -> list[list[Any]]:
        shapes = definition.get_input_shapes(workload.axes)
        dtypes = [_enum_value(spec.dtype) for spec in definition.inputs.values()]
        trials: list[list[Any]] = []
        for trial_index in range(num_trials):
            values: list[Any] = []
            for input_index, ((name, _spec), shape, dtype) in enumerate(
                zip(definition.inputs.items(), shapes, dtypes)
            ):
                descriptor = workload.inputs.get(name)
                descriptor_type = getattr(descriptor, "type", "random")
                if descriptor_type == "scalar":
                    values.append(descriptor.value)
                elif descriptor_type == "safetensors":
                    values.append(
                        self._load_safetensor(
                            descriptor,
                            name=name,
                            expected_shape=shape,
                            expected_dtype=dtype,
                            root=trace_set_root,
                        )
                    )
                elif descriptor_type == "random":
                    seed = self._seed(workload.uuid, trial_index, input_index, name)
                    values.append(self._random_value(shape, dtype, seed))
                else:
                    raise UnsupportedFlashInferFeature(
                        f"unsupported workload input type {descriptor_type!r} for {name!r}"
                    )
            trials.append(values)
        return trials

    @staticmethod
    def _seed(workload_uuid: str, trial: int, input_index: int, name: str) -> int:
        value = f"{workload_uuid}\0{trial}\0{input_index}\0{name}".encode()
        return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") & ((1 << 63) - 1)

    @staticmethod
    def _random_value(shape: Any, dtype: str, seed: int) -> Any:
        try:
            numpy_dtype = _NUMPY_DTYPES[dtype]
        except KeyError:
            raise UnsupportedFlashInferFeature(
                f"random input generation does not support dtype {dtype!r}"
            ) from None
        generator = np.random.default_rng(seed)
        tensor_shape = tuple(shape) if shape is not None else ()
        if dtype == "bool":
            value = generator.integers(0, 2, tensor_shape, dtype=np.int8).astype(numpy_dtype)
        elif dtype in _FLOAT_DTYPES:
            value = generator.standard_normal(tensor_shape, dtype=np.float32).astype(numpy_dtype)
        else:
            value = generator.integers(-3, 4, tensor_shape, dtype=numpy_dtype)
        if shape is not None:
            return np.ascontiguousarray(value)
        if dtype == "bool":
            return bool(value)
        if dtype in _FLOAT_DTYPES:
            return float(value)
        return int(value)

    @staticmethod
    def _load_safetensor(
        descriptor: Any,
        *,
        name: str,
        expected_shape: Any,
        expected_dtype: str,
        root: Path | None,
    ) -> Any:
        path = Path(descriptor.path)
        if not path.is_absolute():
            if root is None:
                raise ValueError(f"relative safetensors path for {name!r} needs a TraceSet root")
            path = Path(root) / path
        value = _load_safetensor_array(path, descriptor.tensor_key)
        expected_shape_list = list(expected_shape) if expected_shape is not None else []
        if list(value.shape) != expected_shape_list:
            raise ValueError(
                f"input {name!r} expected shape {expected_shape_list}, got {list(value.shape)}"
            )
        if value.dtype.name != expected_dtype:
            raise ValueError(f"input {name!r} expected dtype {expected_dtype}, got {value.dtype}")
        return np.ascontiguousarray(value)


class DefaultSolutionAdapter:
    """Single-file Python/Triton and TVM-FFI CUDA solution adapter."""

    _LANGUAGES: ClassVar[set[str]] = {"python", "triton", "cuda"}

    def supports(self, solution: Any) -> bool:
        return _enum_value(solution.spec.language) in self._LANGUAGES

    def add_to_program(self, program: Program, solution: Any) -> Register:
        language = _enum_value(solution.spec.language)
        if language not in self._LANGUAGES:
            raise UnsupportedFlashInferFeature(
                f"solution language {language!r} is not supported by the default adapter"
            )
        if len(solution.sources) != 1:
            raise UnsupportedFlashInferFeature(
                "the default adapter supports one source file per solution; "
                "provide another SolutionAdapter for multi-file builds"
            )

        source = solution.sources[0]
        if source.path != str(solution.get_entry_path()):
            raise UnsupportedFlashInferFeature(
                "the single source file must be the solution entry-point file"
            )
        entry = solution.get_entry_symbol()
        if language in {"python", "triton"}:
            return program.upload(id="solution", kind="module", source=source.content, entry=entry)

        binding = getattr(solution.spec, "binding", None)
        if binding is not None and _enum_value(binding) != "tvm-ffi":
            raise UnsupportedFlashInferFeature(
                "CUDA solutions currently require the tvm-ffi binding"
            )
        if solution.spec.dependencies:
            raise UnsupportedFlashInferFeature(
                "CUDA dependency linking is not implemented by the default solution adapter"
            )
        uploaded = program.upload(
            id="solution_source",
            kind="module",
            language="cuda",
            source=source.content,
            entry=entry,
        )
        return program.run(id="solution", fn="builtin.compile_cuda", args=[uploaded])


class SolutionAdapterRegistry:
    """Ordered solution-adapter registry; custom adapters take precedence."""

    def __init__(self, adapters: list[SolutionAdapter] | None = None) -> None:
        self._adapters = [*(adapters or []), DefaultSolutionAdapter()]

    def add_to_program(self, program: Program, solution: Any) -> Register:
        for adapter in self._adapters:
            if adapter.supports(solution):
                return adapter.add_to_program(program, solution)
        language = _enum_value(solution.spec.language)
        raise UnsupportedFlashInferFeature(f"no SolutionAdapter accepts language {language!r}")


class FlashInferProgramBuilder:
    """Build the complete client-organized benchmark program for one job."""

    def __init__(self, adapters: list[SolutionAdapter] | None = None) -> None:
        self._solutions = SolutionAdapterRegistry(adapters)

    def build(self, job: BenchmarkJob) -> BenchmarkProgram:
        if not job.trials:
            raise ValueError("a benchmark job needs at least one input trial")
        program = Program()
        harness = program.upload(
            id="benchmark_harness", kind="module", source=_HARNESS_SOURCE, entry="main"
        )
        reference = program.upload(
            id="reference",
            kind="module",
            source=job.definition.reference,
            entry="run",
        )
        solution = self._solutions.add_to_program(program, job.solution)

        output_count = len(job.definition.outputs)
        if output_count == 0:
            raise UnsupportedFlashInferFeature("definitions without outputs are not supported")
        output_shapes = job.definition.get_output_shapes(job.workload.axes)
        output_dtypes = [_enum_value(spec.dtype) for spec in job.definition.outputs.values()]
        destination_passing = bool(job.solution.spec.destination_passing_style)

        metadata_keys: list[str] = []
        correctness_keys: list[str] = []
        solution_timing_keys: list[str] = []
        reference_timing_keys: list[str] = []
        output_pairs: list[tuple[Register, Register, int, int]] = []
        trial_args: list[list[Any]] = []

        for trial_index, values in enumerate(job.trials):
            if len(values) != len(job.definition.inputs):
                raise ValueError(
                    f"trial {trial_index} has {len(values)} inputs; "
                    f"definition expects {len(job.definition.inputs)}"
                )
            inputs = self._upload_inputs(program, trial_index, values)
            trial_args.append(inputs)
            reference_result = program.run(
                id=f"reference_result_{trial_index}", fn=reference, args=inputs
            )
            expected = [
                program.run(
                    id=f"reference_output_{trial_index}_{output_index}",
                    fn=harness,
                    args=["normalize", reference_result, output_index, output_count, dtype],
                )
                for output_index, dtype in enumerate(output_dtypes)
            ]

            if destination_passing:
                actual = [
                    program.run(
                        id=f"solution_output_{trial_index}_{output_index}",
                        fn="builtin.empty",
                        args=[{"shape": _shape_list(shape), "dtype": dtype}],
                    )
                    for output_index, (shape, dtype) in enumerate(zip(output_shapes, output_dtypes))
                ]
                program.run(
                    id=f"solution_result_{trial_index}",
                    fn=solution,
                    args=[*inputs, *actual],
                )
            else:
                solution_result = program.run(
                    id=f"solution_result_{trial_index}", fn=solution, args=inputs
                )
                actual = [
                    program.run(
                        id=f"solution_output_{trial_index}_{output_index}",
                        fn=harness,
                        args=["normalize", solution_result, output_index, output_count, dtype],
                    )
                    for output_index, dtype in enumerate(output_dtypes)
                ]

            for output_index, (actual_output, expected_output) in enumerate(zip(actual, expected)):
                metadata_key = f"metadata_{trial_index}_{output_index}"
                metadata = program.run(
                    id=metadata_key,
                    fn=harness,
                    args=["metadata", actual_output, expected_output],
                )
                program.return_(key=metadata_key, value=metadata)
                metadata_keys.append(metadata_key)
                output_pairs.append((actual_output, expected_output, trial_index, output_index))

        program.run(
            id="metadata_gate",
            fn=harness,
            args=["require_metadata", *[Register(key) for key in metadata_keys]],
        )

        for actual, expected, trial_index, output_index in output_pairs:
            correctness_key = f"correctness_{trial_index}_{output_index}"
            correctness = program.run(
                id=correctness_key,
                fn=harness,
                args=[
                    "compare",
                    actual,
                    expected,
                    float(job.eval_config.rtol),
                    float(job.eval_config.atol),
                ],
            )
            program.return_(key=correctness_key, value=correctness)
            correctness_keys.append(correctness_key)

        program.run(
            id="correctness_gate",
            fn=harness,
            args=["require_correctness", *[Register(key) for key in correctness_keys]],
        )
        timing_config = {
            "warmup": int(job.eval_config.warmup_runs),
            "repeat": int(job.eval_config.iterations),
            "flush_l2": True,
        }
        for trial_index, inputs in enumerate(trial_args):
            solution_args = list(inputs)
            if destination_passing:
                solution_args.extend(
                    program.run(
                        id=f"timing_output_{trial_index}_{output_index}",
                        fn="builtin.empty",
                        args=[{"shape": _shape_list(shape), "dtype": dtype}],
                    )
                    for output_index, (shape, dtype) in enumerate(zip(output_shapes, output_dtypes))
                )
            solution_timing_key = f"solution_timing_{trial_index}"
            solution_timing = program.run(
                id=solution_timing_key,
                fn="builtin.benchmark",
                args=[solution, *solution_args, timing_config],
            )
            program.return_(key=solution_timing_key, value=solution_timing)
            solution_timing_keys.append(solution_timing_key)

            if job.eval_config.profile_baseline:
                reference_timing_key = f"reference_timing_{trial_index}"
                reference_timing = program.run(
                    id=reference_timing_key,
                    fn="builtin.benchmark",
                    args=[reference, *inputs, timing_config],
                )
                program.return_(key=reference_timing_key, value=reference_timing)
                reference_timing_keys.append(reference_timing_key)

        return BenchmarkProgram(
            program=program,
            metadata_keys=tuple(metadata_keys),
            correctness_keys=tuple(correctness_keys),
            solution_timing_keys=tuple(solution_timing_keys),
            reference_timing_keys=tuple(reference_timing_keys),
        )

    @staticmethod
    def _upload_inputs(program: Program, trial_index: int, values: list[Any]) -> list[Any]:
        inputs: list[Any] = []
        for input_index, value in enumerate(values):
            if _is_tensor_value(value):
                inputs.append(
                    program.upload(
                        id=f"input_{trial_index}_{input_index}", kind="tensor", value=value
                    )
                )
            elif isinstance(value, (bool, int, float)):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("scalar inputs must be finite")
                inputs.append(value)
            else:
                raise UnsupportedFlashInferFeature(
                    f"input {input_index} has unsupported value type {type(value).__name__!r}"
                )
        return inputs


class FlashInferResultMapper:
    """Translate protocol results into FlashInfer-independent evaluation data."""

    def map(self, outcome: ProgramResult, built: BenchmarkProgram) -> EvaluationData:
        log = _outcome_log(outcome)
        metadata = self._required_results(outcome, built.metadata_keys)
        if metadata is None:
            return self._failed_outcome(outcome, log)
        if any(not value.get("shape_matches", False) for value in metadata):
            return EvaluationData(status="INCORRECT_SHAPE", log=log)
        if any(not value.get("dtype_matches", False) for value in metadata):
            return EvaluationData(status="INCORRECT_DTYPE", log=log)

        comparisons = self._required_results(outcome, built.correctness_keys)
        if comparisons is None:
            return self._failed_outcome(outcome, log)
        max_absolute_error = 0.0
        max_relative_error = 0.0
        passed = True
        for comparison in comparisons:
            nonfinite = comparison.get("nonfinite")
            if nonfinite == "nan":
                max_absolute_error = max_relative_error = float("nan")
            elif nonfinite == "inf":
                max_absolute_error = max_relative_error = float("inf")
            elif not (math.isnan(max_absolute_error) or math.isinf(max_absolute_error)):
                max_absolute_error = max(max_absolute_error, float(comparison["max_abs_err"]))
                max_relative_error = max(max_relative_error, float(comparison["max_rel_err"]))
            passed = passed and bool(comparison.get("passed", False))
        correctness = {
            "max_relative_error": max_relative_error,
            "max_absolute_error": max_absolute_error,
            "extra": {"comparisons": len(comparisons)},
        }
        if not passed:
            return EvaluationData(status="INCORRECT_NUMERICAL", log=log, correctness=correctness)
        if not outcome.completed:
            return self._failed_outcome(outcome, log)

        solution_timings = self._required_results(outcome, built.solution_timing_keys)
        if not solution_timings:
            return EvaluationData(status="RUNTIME_ERROR", log=log + "\nmissing timing results")
        solution_latency = _mean_latency(solution_timings)
        if solution_latency <= 0:
            return EvaluationData(status="RUNTIME_ERROR", log=log + "\ninvalid solution latency")
        reference_timings = self._required_results(outcome, built.reference_timing_keys)
        reference_latency = _mean_latency(reference_timings) if reference_timings else 0.0
        performance = {
            "latency_ms": solution_latency,
            "reference_latency_ms": reference_latency,
            "speedup_factor": (
                reference_latency / solution_latency if reference_latency > 0 else 0.0
            ),
        }
        return EvaluationData(
            status="PASSED", log=log, correctness=correctness, performance=performance
        )

    @staticmethod
    def _required_results(
        outcome: ProgramResult, keys: tuple[str, ...]
    ) -> list[dict[str, Any]] | None:
        values: list[dict[str, Any]] = []
        for key in keys:
            value = outcome.results.get(key)
            if not isinstance(value, dict):
                return None
            values.append(value)
        return values

    @staticmethod
    def _failed_outcome(outcome: ProgramResult, log: str) -> EvaluationData:
        error = outcome.error or {}
        kind = error.get("kind")
        instruction_id = error.get("instruction_id")
        unavailable_compile = kind == "unavailable" and instruction_id in {
            "solution",
            "solution_source",
        }
        status = (
            "COMPILE_ERROR"
            if kind in {"parse", "compile"} or unavailable_compile
            else "RUNTIME_ERROR"
        )
        return EvaluationData(status=status, log=log)


class FlashInferBenchmark:
    """Run a FlashInfer ``TraceSet`` through a remote benchmark-server.

    Construct it with the bundled trace set and config, call :meth:`run_all`,
    optionally write traces back to the data set, and call :meth:`close`.  One
    HTTP request is created per solution-workload pair.  Requests are submitted
    concurrently while the server owns GPU assignment and queueing.
    """

    def __init__(
        self,
        trace_set: TraceSet,
        server: str | ProgramExecutor,
        config: BenchmarkConfig | None = None,
        *,
        max_workers: int | None = None,
        input_provider: WorkloadInputProvider | None = None,
        program_builder: FlashInferProgramBuilder | None = None,
        result_mapper: FlashInferResultMapper | None = None,
    ) -> None:
        if max_workers is not None and max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._trace_set = trace_set
        self._config = config if config is not None else BenchmarkConfig.default()
        self._client: ProgramExecutor = Client(server) if isinstance(server, str) else server
        self._owns_client = isinstance(server, str)
        self._max_workers = max_workers
        self._inputs = input_provider or DefaultWorkloadInputProvider()
        self._uses_default_program_builder = program_builder is None
        self._programs = program_builder or FlashInferProgramBuilder()
        self._results = result_mapper or FlashInferResultMapper()

    def __enter__(self) -> FlashInferBenchmark:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def get_trace_set(self) -> TraceSet:
        return self._trace_set

    def run_all(self, dump_traces: bool = True, resume: bool = False) -> TraceSet:
        existing = self._existing_pairs() if resume else set()
        jobs: list[BenchmarkJob] = []
        immediate_traces: list[Trace] = []

        for definition_name, definition in self._selected_definitions():
            solutions = self._selected_solutions(definition_name)
            if not solutions:
                logger.warning(
                    "No solutions found for def=%s, skipping definition", definition_name
                )
                continue
            for workload_trace in self._trace_set.workloads.get(definition_name, []):
                workload = workload_trace.workload
                pending = [
                    solution
                    for solution in solutions
                    if (definition_name, workload.uuid, solution.name) not in existing
                ]
                if not pending:
                    continue
                try:
                    if self._uses_default_program_builder and _requires_specialized_evaluator(
                        definition
                    ):
                        raise UnsupportedFlashInferFeature(
                            f"definition {definition.name!r} needs a specialized evaluator; "
                            "provide a custom FlashInferProgramBuilder"
                        )
                    eval_config = _resolve_eval_config(self._config, definition)
                    if (
                        self._uses_default_program_builder
                        and eval_config.required_matched_ratio is not None
                    ):
                        raise UnsupportedFlashInferFeature(
                            "required_matched_ratio needs a custom program builder"
                        )
                    trials = self._inputs.prepare(
                        definition,
                        workload,
                        num_trials=eval_config.num_trials,
                        trace_set_root=self._trace_set.root,
                    )
                except Exception as exc:
                    data = _exception_data(exc, stage="input preparation")
                    immediate_traces.extend(
                        self._make_trace(definition_name, workload, solution, data)
                        for solution in pending
                    )
                    continue
                jobs.extend(
                    BenchmarkJob(
                        definition=definition,
                        solution=solution,
                        workload=workload,
                        trials=trials,
                        eval_config=eval_config,
                        timeout_seconds=float(self._config.timeout_seconds),
                    )
                    for solution in pending
                )

        traces = list(immediate_traces)
        if jobs:
            worker_count = self._max_workers or self._server_worker_count()
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                traces.extend(executor.map(self._execute_job, jobs))

        if dump_traces and traces:
            self._trace_set.add_traces(traces)
        traces_by_definition: dict[str, list[Trace]] = {}
        for trace in traces:
            traces_by_definition.setdefault(trace.definition, []).append(trace)
        return TraceSet(
            root=self._trace_set.root,
            definitions=self._trace_set.definitions.copy(),
            solutions=self._trace_set.solutions.copy(),
            workloads=self._trace_set.workloads.copy(),
            traces=traces_by_definition,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _execute_job(self, job: BenchmarkJob) -> Trace:
        try:
            built = self._programs.build(job)
            outcome = self._client.execute(
                built.program,
                timeout_seconds=job.timeout_seconds,
            )
            data = self._results.map(outcome, built)
        except BenchmarkServerError as exc:
            status = "TIMEOUT" if exc.kind == "timeout" else "RUNTIME_ERROR"
            data = EvaluationData(status=status, log=f"benchmark-server request failed: {exc}")
        except TransportError as exc:
            data = EvaluationData(status="RUNTIME_ERROR", log=f"transport failed: {exc}")
        except Exception as exc:
            data = _exception_data(exc, stage="program construction or execution")
        return self._make_trace(job.definition.name, job.workload, job.solution, data)

    @staticmethod
    def _make_trace(
        definition_name: str,
        workload: Workload,
        solution: Solution,
        data: EvaluationData,
    ) -> Trace:
        correctness = Correctness(**data.correctness) if data.correctness else None
        performance = Performance(**data.performance) if data.performance else None
        evaluation = Evaluation(
            status=EvaluationStatus(data.status),
            environment=Environment(
                hardware="benchmark-server", libs={"benchmark-server": __version__}
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
            log=data.log,
            correctness=correctness,
            performance=performance,
        )
        return Trace(
            definition=definition_name,
            workload=workload,
            solution=solution.name,
            evaluation=evaluation,
        )

    def _server_worker_count(self) -> int:
        health = self._client.health()
        gpu_count = health.get("gpu_count")
        if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count <= 0:
            raise ValueError("benchmark-server health response has no positive gpu_count")
        return gpu_count

    def _selected_definitions(self) -> list[tuple[str, Any]]:
        items = list(self._trace_set.definitions.items())
        selected = self._config.definitions
        return items if selected is None else [item for item in items if item[0] in selected]

    def _selected_solutions(self, definition_name: str) -> list[Any]:
        solutions = list(self._trace_set.solutions.get(definition_name, []))
        selected = self._config.solutions
        return solutions if selected is None else [s for s in solutions if s.name in selected]

    def _existing_pairs(self) -> set[tuple[str, str, str]]:
        return {
            (definition_name, trace.workload.uuid, trace.solution)
            for definition_name, traces in self._trace_set.traces.items()
            for trace in traces
            if trace.solution is not None and trace.evaluation is not None
        }


def _exception_data(exc: Exception, *, stage: str) -> EvaluationData:
    status = "COMPILE_ERROR" if isinstance(exc, UnsupportedFlashInferFeature) else "RUNTIME_ERROR"
    return EvaluationData(status=status, log=f"{stage} failed: {type(exc).__name__}: {exc}")


def _resolve_eval_config(config: Any, definition: Any) -> Any:
    resolve = getattr(config, "resolve_eval_config", None)
    return resolve(definition) if callable(resolve) else config


def _mean_latency(values: list[dict[str, Any]]) -> float:
    return sum(float(value["latency_ms_median"]) for value in values) / len(values)


def _outcome_log(outcome: ProgramResult) -> str:
    lines = [
        f"request_id={outcome.request_id}",
        f"queue_ms={outcome.queue_ms}",
        f"elapsed_ms={outcome.elapsed_ms}",
    ]
    if outcome.stdout:
        lines.append("stdout:\n" + outcome.stdout)
    if outcome.stderr:
        lines.append("stderr:\n" + outcome.stderr)
    if outcome.error:
        lines.append("error: " + repr(outcome.error))
    return "\n".join(lines)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _shape_list(shape: Any) -> list[int]:
    return [] if shape is None else list(shape)


def _requires_specialized_evaluator(definition: Any) -> bool:
    name = str(definition.name)
    return (
        getattr(definition, "op_type", None) == "sampling"
        or "moe_fp8_block_scale" in name
        or name.startswith("dsa_topk_indexer")
        or name
        in {
            "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1",
            "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
        }
    )


def _is_tensor_value(value: Any) -> bool:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return False
    return (
        hasattr(value, "dtype")
        and hasattr(value, "shape")
        and (hasattr(value, "tobytes") or hasattr(value, "__dlpack__"))
    )


_NUMPY_DTYPES = {
    "bool": np.dtype("bool"),
    "int8": np.dtype("int8"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "float16": np.dtype("float16"),
    "float32": np.dtype("float32"),
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "float8_e4m3fn": np.dtype(ml_dtypes.float8_e4m3fn),
    "float8_e5m2": np.dtype(ml_dtypes.float8_e5m2),
}
_FLOAT_DTYPES = {
    "float16",
    "float32",
    "bfloat16",
    "float8_e4m3fn",
    "float8_e5m2",
}
_SAFETENSORS_DTYPES = {
    "BOOL": _NUMPY_DTYPES["bool"],
    "I8": _NUMPY_DTYPES["int8"],
    "I16": _NUMPY_DTYPES["int16"],
    "I32": _NUMPY_DTYPES["int32"],
    "I64": _NUMPY_DTYPES["int64"],
    "F16": _NUMPY_DTYPES["float16"],
    "F32": _NUMPY_DTYPES["float32"],
    "BF16": _NUMPY_DTYPES["bfloat16"],
    "F8_E4M3": _NUMPY_DTYPES["float8_e4m3fn"],
    "F8_E4M3FN": _NUMPY_DTYPES["float8_e4m3fn"],
    "F8_E5M2": _NUMPY_DTYPES["float8_e5m2"],
}


def _load_safetensor_array(path: Path, tensor_key: str) -> np.ndarray:
    """Read one contiguous array from the documented safetensors layout."""

    with path.open("rb") as file:
        header_size_bytes = file.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"safetensors file {path} has a truncated header")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size > 100 * 1024 * 1024:
            raise ValueError(f"safetensors file {path} has an oversized header")
        header_bytes = file.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"safetensors file {path} has a truncated header")
        try:
            header = json.loads(header_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"safetensors file {path} has an invalid header") from exc
        descriptor = header.get(tensor_key) if isinstance(header, dict) else None
        if not isinstance(descriptor, dict):
            raise ValueError(f"safetensors file {path} has no key {tensor_key!r}")
        try:
            dtype = _SAFETENSORS_DTYPES[descriptor["dtype"]]
            shape = descriptor["shape"]
            start, end = descriptor["data_offsets"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"safetensors key {tensor_key!r} in {path} has invalid metadata"
            ) from exc
        if (
            not isinstance(shape, list)
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
                for dimension in shape
            )
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise ValueError(f"safetensors key {tensor_key!r} in {path} has invalid metadata")
        expected_size = math.prod(shape) * dtype.itemsize
        if end - start != expected_size:
            raise ValueError(f"safetensors key {tensor_key!r} in {path} has invalid byte length")
        file.seek(8 + header_size + start)
        data = file.read(expected_size)
        if len(data) != expected_size:
            raise ValueError(f"safetensors key {tensor_key!r} in {path} is truncated")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


_HARNESS_SOURCE = r"""
def main(operation, *args):
    import torch

    if operation == "normalize":
        value, index, count, dtype_name = args
        if isinstance(value, (tuple, list)):
            if len(value) != count:
                raise ValueError(f"kernel returned {len(value)} values; expected {count}")
            value = value[index]
        elif count != 1:
            raise ValueError(f"kernel returned one value; expected {count}")
        if isinstance(value, torch.Tensor):
            return value
        dtype = getattr(torch, dtype_name)
        return torch.tensor(value, dtype=dtype, device="cuda")

    if operation == "metadata":
        actual, expected = args
        return {
            "shape_matches": tuple(actual.shape) == tuple(expected.shape),
            "dtype_matches": actual.dtype == expected.dtype,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "actual_dtype": str(actual.dtype).removeprefix("torch."),
            "expected_dtype": str(expected.dtype).removeprefix("torch."),
        }

    if operation == "require_metadata":
        for result in args:
            if not result["shape_matches"]:
                raise RuntimeError("solution output shape does not match the reference")
            if not result["dtype_matches"]:
                raise RuntimeError("solution output dtype does not match the reference")
        return True

    if operation == "compare":
        actual, expected, rtol, atol = args
        torch.cuda.synchronize()
        if bool(torch.isnan(actual).any()):
            return {"passed": False, "nonfinite": "nan", "max_abs_err": 0.0,
                    "max_rel_err": 0.0, "rtol": rtol, "atol": atol}
        if bool(torch.isinf(actual).any()):
            return {"passed": False, "nonfinite": "inf", "max_abs_err": 0.0,
                    "max_rel_err": 0.0, "rtol": rtol, "atol": atol}
        actual_float = actual.float()
        expected_float = expected.float()
        diff = (actual_float - expected_float).abs()
        if diff.numel() == 0:
            max_abs = max_rel = 0.0
        else:
            max_abs = float(diff.max())
            nonzero = expected_float != 0
            max_rel = float((diff[nonzero] / expected_float[nonzero].abs()).max()) \
                if bool(nonzero.any()) else 0.0
        return {
            "passed": bool(torch.allclose(actual_float, expected_float, rtol=rtol, atol=atol)),
            "nonfinite": None,
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "rtol": rtol,
            "atol": atol,
        }

    if operation == "require_correctness":
        if not all(result["passed"] for result in args):
            raise RuntimeError("solution output is numerically incorrect")
        return True

    raise ValueError(f"unknown benchmark harness operation: {operation}")
"""


__all__ = [
    "AxisConst",
    "AxisVar",
    "BenchmarkConfig",
    "BenchmarkJob",
    "BenchmarkProgram",
    "BuildSpec",
    "Correctness",
    "DType",
    "DefaultSolutionAdapter",
    "DefaultWorkloadInputProvider",
    "Definition",
    "Environment",
    "EvalConfig",
    "Evaluation",
    "EvaluationData",
    "EvaluationStatus",
    "FlashInferBenchmark",
    "FlashInferProgramBuilder",
    "FlashInferResultMapper",
    "Performance",
    "RandomInput",
    "ResolvedEvalConfig",
    "SafetensorsInput",
    "ScalarInput",
    "Solution",
    "SolutionAdapter",
    "SourceFile",
    "SupportedBindings",
    "SupportedLanguages",
    "TensorSpec",
    "Trace",
    "TraceSet",
    "TraceSetSummary",
    "UnsupportedFlashInferFeature",
    "Workload",
    "WorkloadInputProvider",
]
