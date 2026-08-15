"""Local client that assembles and submits remote FlashInfer Trace programs."""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmark_server import Client, Program
from benchmark_server.client import (
    BenchmarkServerError,
    ProgramResult,
    ProtocolError,
    TransportError,
)

from .schema import (
    BenchmarkConfig,
    Definition,
    Environment,
    Evaluation,
    EvaluationStatus,
    ResolvedEvalConfig,
    Solution,
    Trace,
    Workload,
)


class FlashInferTraceClient:
    """Evaluate in-memory traces through one server program per workload."""

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        *,
        config: BenchmarkConfig | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.config = config or BenchmarkConfig.default()
        self.max_workers = max_workers
        self._client = Client(server_url)
        source_directory = Path(__file__).parent
        self._module_sources = {
            name: (source_directory / f"{name}.py").read_text(encoding="utf-8")
            for name in ("schema", "workload", "compile", "benchmark")
        }

    def __enter__(self) -> FlashInferTraceClient:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        self._client.close()

    def build_program(
        self,
        definition: Definition,
        solution: Solution,
        workload: Workload,
        resolved_config: ResolvedEvalConfig,
        *,
        resource_root: str | Path | None = None,
    ) -> Program:
        """Construct one pure instruction program without submitting it."""

        self._validate_definition_support(definition)
        self._validate_solution_support(solution)
        request_data = {
            "definition": definition.model_dump(mode="json"),
            "solution": solution.model_dump(mode="json"),
            "workload": workload.model_dump(mode="json"),
            "config": resolved_config.model_dump(mode="json"),
        }

        program = Program()
        schema_module = program.upload(
            id="trace_schema",
            kind="module",
            source=self._module_sources["schema"],
            entry="normalize_evaluation",
        )
        workload_module = program.upload(
            id="trace_workload",
            kind="module",
            source=self._module_sources["workload"],
            entry="make_input_generator",
        )
        compile_module = program.upload(
            id="trace_compile",
            kind="module",
            source=self._module_sources["compile"],
            entry="load_callable",
        )
        benchmark_module = program.upload(
            id="trace_benchmark",
            kind="module",
            source=self._module_sources["benchmark"],
            entry="evaluate",
        )

        normalized_handle = program.run(
            id="normalized",
            fn=schema_module,
            args=[
                request_data["definition"],
                request_data["solution"],
                request_data["workload"],
                request_data["config"],
            ],
        )

        resource_paths = sorted(
            {
                input_spec.path
                for input_spec in workload.inputs.values()
                if input_spec.type == "safetensors"
            }
        )
        resource_manifest = {path: index for index, path in enumerate(resource_paths)}
        resource_handles = [
            program.upload(
                id=f"safetensors_{index}",
                kind="bytes",
                value=self._read_resource(path, resource_root),
            )
            for index, path in enumerate(resource_paths)
        ]
        input_generator = program.run(
            id="input_generator",
            fn=workload_module,
            args=[normalized_handle, resource_manifest, *resource_handles],
        )
        reference_source = program.upload(
            id="reference_source",
            kind="module",
            source=definition.reference,
            entry="run",
        )
        reference_callable = program.run(
            id="reference_callable",
            fn=compile_module,
            args=[normalized_handle, "reference", reference_source],
        )

        if solution.spec.language.value == "cuda":
            source = solution.sources[0]
            cuda_source = program.upload(
                id="cuda_source",
                kind="module",
                source=source.content,
                entry=solution.get_entry_symbol(),
                language="cuda",
            )
            compiled_cuda = program.run(
                id="compiled_cuda",
                fn="builtin.compile_cuda",
                args=[cuda_source, resolved_config.extra],
            )
            solution_callable = program.run(
                id="solution_callable",
                fn=compile_module,
                args=[normalized_handle, "solution", compiled_cuda],
            )
        else:
            solution_source = program.upload(
                id="solution_source",
                kind="module",
                source=solution.sources[0].content,
                entry=solution.get_entry_symbol(),
            )
            solution_callable = program.run(
                id="solution_callable",
                fn=compile_module,
                args=[normalized_handle, "solution", solution_source],
            )

        evaluation = program.run(
            id="evaluation",
            fn=benchmark_module,
            args=[
                normalized_handle,
                input_generator,
                reference_callable,
                solution_callable,
            ],
        )
        program.return_(key="evaluation", value=evaluation)
        return program

    def evaluate(
        self,
        definition: Definition,
        solution: Solution,
        trace: Trace,
        *,
        resource_root: str | Path | None = None,
    ) -> Trace:
        """Evaluate one in-memory workload trace and return a completed trace."""

        return self.evaluate_many(
            definition,
            solution,
            [trace],
            resource_root=resource_root,
        )[0]

    def evaluate_many(
        self,
        definition: Definition,
        solution: Solution,
        traces: Sequence[Trace],
        *,
        resource_root: str | Path | None = None,
    ) -> list[Trace]:
        """Evaluate in-memory workload traces concurrently and preserve their order."""

        if solution.definition != definition.name:
            raise ValueError(
                f"solution {solution.name!r} targets "
                f"{solution.definition!r}, expected {definition.name!r}"
            )
        workload_traces = list(traces)
        for trace in workload_traces:
            if trace.definition != definition.name:
                raise ValueError(
                    f"trace targets definition {trace.definition!r}, expected {definition.name!r}"
                )
            if not trace.is_workload_trace():
                raise ValueError("input traces must not contain a solution or evaluation")

        resolved_config = self.config.resolve_eval_config(definition)
        programs = [
            self.build_program(
                definition,
                solution,
                trace.workload,
                resolved_config,
                resource_root=resource_root,
            )
            for trace in workload_traces
        ]
        if not programs:
            return []
        fallback_environment = self._server_environment()

        def submit(item: tuple[Trace, Program]) -> Trace:
            workload_trace, program = item
            evaluation = self._execute(program, fallback_environment)
            return Trace(
                definition=definition.name,
                workload=workload_trace.workload,
                solution=solution.name,
                evaluation=evaluation,
            )

        worker_count = self.max_workers or min(32, len(programs))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(submit, zip(workload_traces, programs, strict=True)))

    def _read_resource(self, resource_path: str, resource_root: str | Path | None) -> bytes:
        relative_path = Path(resource_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"resource path must stay inside the resource root: {resource_path!r}")
        if resource_root is None:
            raise ValueError("a resource root is required for safetensors resources")
        root = Path(resource_root).resolve()
        resolved_path = (root / relative_path).resolve()
        if not resolved_path.is_relative_to(root):
            raise ValueError(f"resource path escapes the resource root: {resource_path!r}")
        if not resolved_path.is_file():
            raise ValueError(f"safetensors resource does not exist: {resource_path!r}")
        return resolved_path.read_bytes()

    @staticmethod
    def _validate_definition_support(definition: Definition) -> None:
        unsupported_dtypes = {
            specification.dtype.value
            for specification in (*definition.inputs.values(), *definition.outputs.values())
            if specification.dtype.value == "float4_e2m1"
        }
        if unsupported_dtypes:
            raise ValueError("remote evaluation does not support float4_e2m1 tensors")
        if definition.op_type == "sampling":
            raise ValueError("remote evaluation does not support sampling correctness")

    @staticmethod
    def _validate_solution_support(solution: Solution) -> None:
        if len(solution.sources) != 1:
            raise ValueError("remote evaluation supports exactly one solution source file")
        if solution.spec.dependencies:
            raise ValueError("remote evaluation does not support solution dependencies")
        language = solution.spec.language.value
        if language not in {"python", "triton", "cuda"}:
            raise ValueError(f"unsupported solution language: {language!r}")
        if language == "cuda" and (
            solution.spec.binding is None or solution.spec.binding.value != "tvm-ffi"
        ):
            raise ValueError("CUDA solutions require the tvm-ffi binding")

    def _server_environment(self) -> Environment:
        try:
            health = self._client.health()
            target = health.get("target", {})
            versions = health.get("versions", {})
            hardware = target.get("arch", "unknown") if isinstance(target, dict) else "unknown"
            libraries = (
                {str(key): str(value) for key, value in versions.items()}
                if isinstance(versions, dict)
                else {}
            )
            return Environment(hardware=hardware or "unknown", libs=libraries)
        except Exception:
            return Environment(hardware="unknown")

    def _execute(self, program: Program, fallback_environment: Environment) -> Evaluation:
        try:
            result = self._client.execute(
                program,
                timeout_seconds=self.config.timeout_seconds,
            )
            if not result.completed:
                return self._failed_program_evaluation(result, fallback_environment)
            if set(result.results) != {"evaluation"}:
                raise ProtocolError("completed trace program returned unexpected result keys")
            evaluation = Evaluation.model_validate(result.results["evaluation"])
            captured_output = "\n".join(value for value in (result.stdout, result.stderr) if value)
            if captured_output:
                evaluation = evaluation.model_copy(
                    update={
                        "log": "\n".join(
                            value for value in (evaluation.log, captured_output) if value
                        )
                    }
                )
            return evaluation
        except BenchmarkServerError as exc:
            status = (
                EvaluationStatus.TIMEOUT
                if exc.status_code == 504 or exc.kind == "timeout"
                else EvaluationStatus.RUNTIME_ERROR
            )
            return self._error_evaluation(status, fallback_environment, str(exc))
        except (ProtocolError, TransportError) as exc:
            return self._error_evaluation(
                EvaluationStatus.RUNTIME_ERROR,
                fallback_environment,
                f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            return self._error_evaluation(
                EvaluationStatus.RUNTIME_ERROR,
                fallback_environment,
                f"{type(exc).__name__}: {exc}",
            )

    def _failed_program_evaluation(
        self,
        result: ProgramResult,
        fallback_environment: Environment,
    ) -> Evaluation:
        error = result.error or {}
        instruction_id = error.get("instruction_id")
        kind = error.get("kind")
        if instruction_id in {
            "reference_callable",
            "solution_callable",
            "compiled_cuda",
        } or kind in {
            "compile",
            "parse",
        }:
            status = EvaluationStatus.COMPILE_ERROR
        elif kind == "correctness":
            status = EvaluationStatus.INCORRECT_NUMERICAL
        else:
            status = EvaluationStatus.RUNTIME_ERROR
        log_parts = [
            str(error.get("message", "remote execution failed")),
            str(error.get("traceback", "")),
            result.stdout,
            result.stderr,
        ]
        log = "\n".join(part for part in log_parts if part)
        if status == EvaluationStatus.INCORRECT_NUMERICAL:
            return Evaluation(
                status=status,
                environment=fallback_environment,
                timestamp=_timestamp(),
                log=log,
                correctness={
                    "max_relative_error": 0.0,
                    "max_absolute_error": 0.0,
                },
            )
        return self._error_evaluation(status, fallback_environment, log)

    @staticmethod
    def _error_evaluation(
        status: EvaluationStatus,
        environment: Environment,
        log: str,
    ) -> Evaluation:
        return Evaluation(
            status=status,
            environment=environment,
            timestamp=_timestamp(),
            log=log,
        )


def _timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


__all__ = ["FlashInferTraceClient"]
