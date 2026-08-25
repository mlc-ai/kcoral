"""Data models for the FlashInfer Trace interchange format."""

from __future__ import annotations

import ast
import math
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

NonEmptyString = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]


class TraceModel(BaseModel):
    """Common configuration for trace JSON models."""

    model_config = ConfigDict(extra="ignore")


class AxisConst(TraceModel):
    """A symbolic axis whose value is fixed by a definition."""

    type: Literal["const"] = "const"
    value: NonNegativeInt
    description: str | None = None


class AxisVar(TraceModel):
    """A symbolic axis whose value is supplied by each workload."""

    type: Literal["var"] = "var"
    description: str | None = None


AxisSpec = Annotated[AxisConst | AxisVar, Field(discriminator="type")]


class DType(str, Enum):
    """Tensor data types represented by the trace format."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT8_E4M3FN = "float8_e4m3fn"
    FLOAT8_E5M2 = "float8_e5m2"
    FLOAT4_E2M1 = "float4_e2m1"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    BOOL = "bool"


class TensorSpec(TraceModel):
    """The symbolic shape and data type of one input or output."""

    shape: list[NonEmptyString] | None
    dtype: DType
    description: str | None = None


class Definition(TraceModel):
    """An operator contract and its Python reference implementation."""

    name: NonEmptyString
    op_type: NonEmptyString
    axes: dict[NonEmptyString, AxisSpec]
    inputs: dict[NonEmptyString, TensorSpec]
    outputs: dict[NonEmptyString, TensorSpec]
    reference: NonEmptyString
    tags: list[NonEmptyString] = Field(default_factory=list)
    description: str | None = None
    constraints: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_definition(self) -> Definition:
        try:
            module = ast.parse(self.reference, mode="exec")
        except SyntaxError as exc:
            raise ValueError(f"reference must be valid Python: {exc}") from exc
        if not any(
            isinstance(node, ast.FunctionDef) and node.name == "run" for node in module.body
        ):
            raise ValueError("reference must define a synchronous top-level 'run' function")

        overlap = set(self.inputs) & set(self.outputs)
        if overlap:
            raise ValueError(f"input and output names overlap: {sorted(overlap)}")

        for constraint in self.constraints:
            try:
                ast.parse(constraint, mode="eval")
            except SyntaxError as exc:
                raise ValueError(f"constraint must be valid Python: {constraint!r}") from exc

        for tensor_name, tensor in {**self.inputs, **self.outputs}.items():
            for axis_name in tensor.shape or []:
                if axis_name not in self.axes:
                    raise ValueError(
                        f"tensor {tensor_name!r} references unknown axis {axis_name!r}"
                    )
        return self

    def _get_shapes(
        self,
        tensors: list[TensorSpec],
        variable_axes: dict[str, int] | None = None,
    ) -> list[tuple[int, ...] | None]:
        values = variable_axes or {}
        shapes: list[tuple[int, ...] | None] = []
        for tensor in tensors:
            if tensor.shape is None:
                shapes.append(None)
                continue

            dimensions: list[int] = []
            for axis_name in tensor.shape:
                axis = self.axes[axis_name]
                if isinstance(axis, AxisConst):
                    dimensions.append(axis.value)
                elif axis_name in values:
                    dimensions.append(values[axis_name])
                else:
                    raise ValueError(f"missing value for variable axis {axis_name!r}")
            shapes.append(tuple(dimensions))
        return shapes

    def get_input_shapes(
        self, variable_axes: dict[str, int] | None = None
    ) -> list[tuple[int, ...] | None]:
        """Resolve input shapes for one workload."""

        return self._get_shapes(list(self.inputs.values()), variable_axes)

    def get_output_shapes(
        self, variable_axes: dict[str, int] | None = None
    ) -> list[tuple[int, ...] | None]:
        """Resolve output shapes for one workload."""

        return self._get_shapes(list(self.outputs.values()), variable_axes)

    @cached_property
    def input_dtypes(self) -> list[str]:
        """Return protocol data type names in input order."""

        return [spec.dtype.value for spec in self.inputs.values()]


class SupportedLanguages(str, Enum):
    """Languages accepted by the trace solution schema."""

    PYTHON = "python"
    TRITON = "triton"
    CPP = "cpp"
    CUDA = "cuda"
    TILELANG = "tilelang"


class SupportedBindings(str, Enum):
    """Bindings accepted by compiled solutions."""

    TVM_FFI = "tvm-ffi"
    TORCH = "torch"


class SourceFile(TraceModel):
    """One source file embedded in a solution."""

    path: NonEmptyString
    content: NonEmptyString

    @model_validator(mode="after")
    def _validate_path(self) -> SourceFile:
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"source path must be relative and contained: {self.path!r}")
        return self


class BuildSpec(TraceModel):
    """How a solution is built and invoked."""

    language: SupportedLanguages
    target_hardware: list[NonEmptyString] = Field(min_length=1)
    entry_point: NonEmptyString
    dependencies: list[NonEmptyString] = Field(default_factory=list)
    destination_passing_style: bool = True
    binding: SupportedBindings | None = None

    @field_validator("entry_point")
    @classmethod
    def _validate_entry_point(cls, value: str) -> str:
        if value.count("::") != 1:
            raise ValueError("entry_point must have the form '<path>::<symbol>'")
        path, symbol = value.split("::")
        if not path or not symbol.isidentifier():
            raise ValueError("entry_point must contain a path and a Python identifier")
        return value


class Solution(TraceModel):
    """A source implementation for one definition."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    name: NonEmptyString
    definition: NonEmptyString
    author: NonEmptyString
    spec: BuildSpec
    sources: list[SourceFile] = Field(min_length=1)
    description: str | None = None

    @model_validator(mode="after")
    def _validate_sources(self) -> Solution:
        paths = [source.path for source in self.sources]
        if len(paths) != len(set(paths)):
            raise ValueError("solution source paths must be unique")
        if str(self.get_entry_path()) not in paths:
            raise ValueError(f"entry source {str(self.get_entry_path())!r} is missing")
        return self

    def get_entry_path(self) -> Path:
        """Return the path component of the entry point."""

        return Path(self.spec.entry_point.split("::", 1)[0])

    def get_entry_symbol(self) -> str:
        """Return the symbol component of the entry point."""

        return self.spec.entry_point.split("::", 1)[1]


class RandomInput(TraceModel):
    """Request deterministic random input generation."""

    type: Literal["random"] = "random"


class ScalarInput(TraceModel):
    """Provide a JSON scalar as an input."""

    type: Literal["scalar"] = "scalar"
    value: int | float | bool


class SafetensorsInput(TraceModel):
    """Load one tensor from a safetensors file."""

    type: Literal["safetensors"] = "safetensors"
    path: NonEmptyString
    tensor_key: NonEmptyString


InputSpec = Annotated[
    RandomInput | ScalarInput | SafetensorsInput,
    Field(discriminator="type"),
]


class Workload(TraceModel):
    """Concrete axis values and input descriptors for one benchmark case."""

    axes: dict[str, NonNegativeInt]
    inputs: dict[str, InputSpec]
    uuid: NonEmptyString


class Correctness(TraceModel):
    """Numerical error measurements."""

    model_config = ConfigDict(extra="ignore", ser_json_inf_nan="strings")

    max_relative_error: float = 0.0
    max_absolute_error: float = 0.0
    extra: dict[str, Any] | None = None

    @field_validator("max_relative_error", "max_absolute_error")
    @classmethod
    def _non_negative_or_nan(cls, value: float) -> float:
        if not math.isnan(value) and value < 0:
            raise ValueError("error measurements must be non-negative or NaN")
        return value


class Performance(TraceModel):
    """Latency and speedup measurements."""

    latency_ms: float = Field(default=0.0, ge=0.0)
    reference_latency_ms: float = Field(default=0.0, ge=0.0)
    speedup_factor: float = Field(default=0.0, ge=0.0)


class Environment(TraceModel):
    """Execution environment recorded with an evaluation."""

    hardware: NonEmptyString
    libs: dict[str, str] = Field(default_factory=dict)


class EvaluationStatus(str, Enum):
    """Possible outcomes for one solution and workload pair."""

    PASSED = "PASSED"
    INCORRECT_SHAPE = "INCORRECT_SHAPE"
    INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
    INCORRECT_DTYPE = "INCORRECT_DTYPE"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    COMPILE_ERROR = "COMPILE_ERROR"
    TIMEOUT = "TIMEOUT"


class Evaluation(TraceModel):
    """The result of evaluating one solution on one workload."""

    status: EvaluationStatus
    environment: Environment
    timestamp: NonEmptyString
    log: str = ""
    correctness: Correctness | None = None
    performance: Performance | None = None

    @model_validator(mode="after")
    def _validate_metrics(self) -> Evaluation:
        if self.status == EvaluationStatus.PASSED:
            if self.correctness is None or self.performance is None:
                raise ValueError("a passed evaluation needs correctness and performance metrics")
        elif self.status == EvaluationStatus.INCORRECT_NUMERICAL:
            if self.correctness is None or self.performance is not None:
                raise ValueError(
                    "an incorrect numerical evaluation needs correctness but no performance"
                )
        elif self.correctness is not None or self.performance is not None:
            raise ValueError("non-numerical failures cannot include correctness or performance")
        return self


class Trace(TraceModel):
    """A workload declaration or a completed solution evaluation."""

    definition: NonEmptyString
    workload: Workload
    solution: str | None = None
    evaluation: Evaluation | None = None

    @model_validator(mode="after")
    def _validate_pair(self) -> Trace:
        if (self.solution is None) != (self.evaluation is None):
            raise ValueError(
                "solution and evaluation must either both be present or both be absent"
            )
        return self

    def is_workload_trace(self) -> bool:
        """Return whether this record only declares a workload."""

        return self.solution is None

    def is_successful(self) -> bool:
        """Return whether this is a passed evaluation trace."""

        return self.evaluation is not None and self.evaluation.status == EvaluationStatus.PASSED


class EvalConfig(TraceModel):
    """Optional evaluation overrides for an operation or definition."""

    warmup_runs: int | None = Field(default=None, ge=0)
    iterations: int | None = Field(default=None, gt=0)
    num_trials: int | None = Field(default=None, gt=0)
    rtol: float | None = Field(default=None, gt=0)
    atol: float | None = Field(default=None, gt=0)
    required_matched_ratio: float | None = Field(default=None, gt=0, le=1)
    extra: dict[str, Any] = Field(default_factory=dict)


class ResolvedEvalConfig(TraceModel):
    """Fully resolved settings consumed by an evaluator."""

    warmup_runs: int = Field(default=10, ge=0)
    iterations: int = Field(default=50, gt=0)
    num_trials: int = Field(default=3, gt=0)
    rtol: float = Field(default=1e-2, gt=0)
    atol: float = Field(default=1e-2, gt=0)
    required_matched_ratio: float | None = Field(default=None, gt=0, le=1)
    profile_baseline: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


def normalize_evaluation(
    definition_data: dict[str, Any],
    solution_data: dict[str, Any],
    workload_data: dict[str, Any],
    resolved_config_data: dict[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one remotely evaluated workload."""

    type_namespace = globals()
    for model in (Definition, Solution, Workload, ResolvedEvalConfig):
        model.model_rebuild(_types_namespace=type_namespace)

    for label, value in (
        ("definition", definition_data),
        ("solution", solution_data),
        ("workload", workload_data),
        ("resolved config", resolved_config_data),
    ):
        if not isinstance(value, dict):
            raise TypeError(f"{label} must be a JSON object")

    definition = Definition.model_validate(definition_data)
    solution = Solution.model_validate(solution_data)
    workload = Workload.model_validate(workload_data)
    resolved_config = ResolvedEvalConfig.model_validate(resolved_config_data)

    if solution.definition != definition.name:
        raise ValueError(
            f"solution targets definition {solution.definition!r}, expected {definition.name!r}"
        )

    variable_axes = {name for name, axis in definition.axes.items() if isinstance(axis, AxisVar)}
    if set(workload.axes) != variable_axes:
        missing = sorted(variable_axes - set(workload.axes))
        unexpected = sorted(set(workload.axes) - variable_axes)
        raise ValueError(
            f"workload axes are incomplete: missing={missing}, unexpected={unexpected}"
        )
    if set(workload.inputs) != set(definition.inputs):
        missing = sorted(set(definition.inputs) - set(workload.inputs))
        unexpected = sorted(set(workload.inputs) - set(definition.inputs))
        raise ValueError(
            f"workload inputs are incomplete: missing={missing}, unexpected={unexpected}"
        )

    axis_values = {
        name: axis.value for name, axis in definition.axes.items() if isinstance(axis, AxisConst)
    }
    axis_values.update(workload.axes)
    for constraint in definition.constraints:
        try:
            satisfied = eval(
                compile(ast.parse(constraint, mode="eval"), "<constraint>", "eval"),
                {"__builtins__": {}},
                axis_values,
            )
        except Exception as exc:
            raise ValueError(f"constraint {constraint!r} could not be evaluated: {exc}") from exc
        if not isinstance(satisfied, bool):
            raise ValueError(f"constraint {constraint!r} must evaluate to a boolean")
        if not satisfied:
            raise ValueError(f"constraint is not satisfied: {constraint!r}")

    return {
        "definition": definition.model_dump(mode="json"),
        "solution": solution.model_dump(mode="json"),
        "workload": workload.model_dump(mode="json"),
        "config": resolved_config.model_dump(mode="json"),
    }


class BenchmarkConfig(TraceModel):
    """Client-side timeout and evaluator settings for one benchmark pair."""

    timeout_seconds: int = Field(default=300, gt=0)
    profile_baseline: bool = True
    warmup_runs: int | None = Field(default=None, ge=0)
    iterations: int | None = Field(default=None, gt=0)
    num_trials: int | None = Field(default=None, gt=0)
    rtol: float | None = Field(default=None, gt=0)
    atol: float | None = Field(default=None, gt=0)
    required_matched_ratio: float | None = Field(default=None, gt=0, le=1)
    op_type_config: dict[str, EvalConfig] = Field(default_factory=dict)
    definition_config: dict[str, EvalConfig] = Field(default_factory=dict)

    @classmethod
    def default(cls, **overrides: Any) -> BenchmarkConfig:
        """Create settings using built-in defaults and explicit overrides."""

        return cls(**overrides)

    def resolve_eval_config(self, definition: Definition) -> ResolvedEvalConfig:
        """Resolve operation, definition, and top-level overrides."""

        merged: dict[str, Any] = {"profile_baseline": self.profile_baseline, "extra": {}}
        for layer in (
            self.op_type_config.get(definition.op_type),
            self.definition_config.get(definition.name),
        ):
            if layer is None:
                continue
            merged.update(
                {
                    key: value
                    for key, value in layer.model_dump(exclude={"extra"}).items()
                    if value is not None
                }
            )
            merged["extra"].update(layer.extra)

        for key in (
            "warmup_runs",
            "iterations",
            "num_trials",
            "rtol",
            "atol",
            "required_matched_ratio",
        ):
            value = getattr(self, key)
            if value is not None:
                merged[key] = value
        return ResolvedEvalConfig(**merged)


__all__ = [
    "AxisConst",
    "AxisVar",
    "BenchmarkConfig",
    "BuildSpec",
    "Correctness",
    "DType",
    "Definition",
    "Environment",
    "EvalConfig",
    "Evaluation",
    "EvaluationStatus",
    "Performance",
    "RandomInput",
    "ResolvedEvalConfig",
    "SafetensorsInput",
    "ScalarInput",
    "Solution",
    "SourceFile",
    "SupportedBindings",
    "SupportedLanguages",
    "TensorSpec",
    "Trace",
    "Workload",
    "normalize_evaluation",
]
