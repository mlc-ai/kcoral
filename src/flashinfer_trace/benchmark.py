"""Self-contained correctness and performance evaluation for uploaded callables."""

from __future__ import annotations

import datetime
from typing import Any


def evaluate(
    normalized: dict[str, Any],
    input_generator,
    reference_callable,
    candidate_callable,
) -> dict[str, Any]:
    """Check every trial before measuring the candidate with CUPTI."""

    import torch

    environment = _environment()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    definition = normalized["definition"]
    solution = normalized["solution"]
    config = normalized["config"]
    output_specs = list(definition["outputs"].values())
    output_shapes = [
        _resolve_shape(specification["shape"], definition["axes"], normalized["workload"]["axes"])
        for specification in output_specs
    ]

    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    minimum_matched_ratio = 1.0
    numerical_failure_log: str | None = None
    try:
        for trial_index in range(config["num_trials"]):
            inputs = input_generator(trial_index)
            with torch.no_grad():
                reference_outputs = _invoke(
                    reference_callable,
                    _clone_inputs(inputs),
                    output_specs,
                    output_shapes,
                    destination_passing=False,
                )
                candidate_outputs = _invoke(
                    candidate_callable,
                    _clone_inputs(inputs),
                    output_specs,
                    output_shapes,
                    destination_passing=solution["spec"]["destination_passing_style"],
                )

            failure = _validate_outputs(
                reference_outputs,
                output_specs,
                output_shapes,
                "reference",
                environment,
                timestamp,
            )
            if failure is not None:
                return failure
            failure = _validate_outputs(
                candidate_outputs,
                output_specs,
                output_shapes,
                "candidate",
                environment,
                timestamp,
            )
            if failure is not None:
                return failure

            for candidate_output, reference_output in zip(
                candidate_outputs, reference_outputs, strict=True
            ):
                if not bool(torch.isfinite(reference_output).all()):
                    return _numerical_failure(
                        environment,
                        timestamp,
                        "reference output contains a non-finite value",
                        maximum_absolute_error,
                        maximum_relative_error,
                        minimum_matched_ratio,
                    )
                if not bool(torch.isfinite(candidate_output).all()):
                    return _numerical_failure(
                        environment,
                        timestamp,
                        "candidate output contains a non-finite value",
                        maximum_absolute_error,
                        maximum_relative_error,
                        minimum_matched_ratio,
                    )

                candidate_float = candidate_output.to(torch.float64)
                reference_float = reference_output.to(torch.float64)
                difference = (candidate_float - reference_float).abs()
                if difference.numel():
                    maximum_absolute_error = max(
                        maximum_absolute_error, float(difference.max().item())
                    )
                    denominator = reference_float.abs() + 1e-8
                    relative = (difference / denominator).clamp_max(torch.finfo(torch.float64).max)
                    maximum_relative_error = max(
                        maximum_relative_error, float(relative.max().item())
                    )
                    matched = difference <= (
                        config["atol"] + config["rtol"] * reference_float.abs()
                    )
                    matched_ratio = float(matched.to(torch.float32).mean().item())
                else:
                    matched_ratio = 1.0
                minimum_matched_ratio = min(minimum_matched_ratio, matched_ratio)
                required_ratio = config["required_matched_ratio"]
                if (
                    matched_ratio < required_ratio
                    if required_ratio is not None
                    else matched_ratio < 1.0
                ):
                    numerical_failure_log = (
                        "candidate output exceeds the configured error tolerance"
                    )
    except Exception as exc:
        return _runtime_failure(environment, timestamp, "correctness", exc)

    if numerical_failure_log is not None:
        return _numerical_failure(
            environment,
            timestamp,
            numerical_failure_log,
            maximum_absolute_error,
            maximum_relative_error,
            minimum_matched_ratio,
        )

    try:
        candidate_latencies: list[float] = []
        reference_latencies: list[float] = []
        for trial_index in range(config["num_trials"]):
            timing_inputs = input_generator(trial_index)
            candidate_timing = _benchmark_callable(
                candidate_callable,
                _clone_inputs(timing_inputs),
                output_specs,
                output_shapes,
                solution["spec"]["destination_passing_style"],
                config,
            )
            candidate_latencies.append(float(candidate_timing["latency_ms_median"]))
            if config["profile_baseline"]:
                reference_timing = _benchmark_callable(
                    reference_callable,
                    _clone_inputs(timing_inputs),
                    output_specs,
                    output_shapes,
                    False,
                    config,
                )
                reference_latencies.append(float(reference_timing["latency_ms_median"]))
        candidate_latency = sum(candidate_latencies) / len(candidate_latencies)
        reference_latency = (
            sum(reference_latencies) / len(reference_latencies) if reference_latencies else 0.0
        )
    except Exception as exc:
        return _runtime_failure(environment, timestamp, "timing", exc)

    speedup = (
        reference_latency / candidate_latency
        if reference_latency > 0.0 and candidate_latency > 0.0
        else 0.0
    )
    return {
        "status": "PASSED",
        "correctness": {
            "max_relative_error": maximum_relative_error,
            "max_absolute_error": maximum_absolute_error,
            "extra": {"matched_ratio": minimum_matched_ratio},
        },
        "performance": {
            "latency_ms": candidate_latency,
            "reference_latency_ms": reference_latency,
            "speedup_factor": speedup,
        },
        "environment": environment,
        "timestamp": timestamp,
        "log": "",
    }


def _invoke(
    callable_object,
    inputs: list[Any],
    output_specs: list[dict[str, Any]],
    output_shapes: list[list[int] | None],
    destination_passing: bool,
) -> list[Any]:
    if destination_passing:
        outputs = _allocate_outputs(output_specs, output_shapes)
        returned = callable_object(*inputs, *outputs)
        if returned is not None:
            raise ValueError("a destination-passing callable must return None")
        return outputs
    return _returned_outputs(callable_object(*inputs), output_specs)


def _returned_outputs(value: Any, output_specs: list[dict[str, Any]]) -> list[Any]:
    import torch

    if value is None and not output_specs:
        return []
    outputs = list(value) if isinstance(value, (tuple, list)) else [value]
    return [
        (
            torch.tensor(output, dtype=_torch_dtype(output_specs[index]["dtype"]), device="cuda")
            if index < len(output_specs)
            and isinstance(output, (bool, int, float))
            and not isinstance(output, torch.Tensor)
            else output
        )
        for index, output in enumerate(outputs)
    ]


def _allocate_outputs(
    output_specs: list[dict[str, Any]], output_shapes: list[list[int] | None]
) -> list[Any]:
    import torch

    return [
        torch.empty(
            [] if shape is None else shape,
            dtype=_torch_dtype(specification["dtype"]),
            device="cuda",
        )
        for specification, shape in zip(output_specs, output_shapes, strict=True)
    ]


def _validate_outputs(
    outputs: list[Any],
    output_specs: list[dict[str, Any]],
    output_shapes: list[list[int] | None],
    owner: str,
    environment: dict[str, Any],
    timestamp: str,
) -> dict[str, Any] | None:
    import torch

    if len(outputs) != len(output_specs):
        return _failure(
            "INCORRECT_SHAPE",
            environment,
            timestamp,
            f"{owner} returned {len(outputs)} outputs, expected {len(output_specs)}",
        )
    for index, (output, specification, shape) in enumerate(
        zip(outputs, output_specs, output_shapes, strict=True)
    ):
        if not isinstance(output, torch.Tensor):
            return _failure(
                "INCORRECT_DTYPE",
                environment,
                timestamp,
                f"{owner} output {index} is not a GPU tensor",
            )
        expected_shape = [] if shape is None else shape
        if list(output.shape) != expected_shape:
            return _failure(
                "INCORRECT_SHAPE",
                environment,
                timestamp,
                f"{owner} output {index} has shape {list(output.shape)}, expected {expected_shape}",
            )
        actual_dtype = str(output.dtype).removeprefix("torch.")
        if actual_dtype != specification["dtype"]:
            return _failure(
                "INCORRECT_DTYPE",
                environment,
                timestamp,
                f"{owner} output {index} has dtype {actual_dtype!r}, "
                f"expected {specification['dtype']!r}",
            )
    return None


def _benchmark_callable(
    callable_object,
    inputs: list[Any],
    output_specs: list[dict[str, Any]],
    output_shapes: list[list[int] | None],
    destination_passing: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from benchmark_server import builtin

    outputs = _allocate_outputs(output_specs, output_shapes) if destination_passing else None

    def call() -> None:
        with torch.no_grad():
            if outputs is None:
                callable_object(*inputs)
            else:
                callable_object(*inputs, *outputs)

    return builtin.benchmark(
        call,
        {
            "warmup": config["warmup_runs"],
            "repeat": config["iterations"],
        },
    )


def _clone_inputs(inputs: list[Any]) -> list[Any]:
    return [value.clone() if hasattr(value, "clone") else value for value in inputs]


def _resolve_shape(
    symbolic_shape: list[str] | None,
    axes: dict[str, dict[str, Any]],
    workload_axes: dict[str, int],
) -> list[int] | None:
    if symbolic_shape is None:
        return None
    return [
        axes[name]["value"] if axes[name]["type"] == "const" else workload_axes[name]
        for name in symbolic_shape
    ]


def _torch_dtype(dtype: str):
    import torch

    if dtype == "float4_e2m1":
        raise ValueError("float4_e2m1 outputs are not supported")
    try:
        return {
            "bool": torch.bool,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unsupported output dtype: {dtype!r}") from exc


def _environment() -> dict[str, Any]:
    import torch

    libraries = {"torch": torch.__version__}
    if torch.version.cuda:
        libraries["cuda"] = torch.version.cuda
    for library_name in ("triton", "tvm_ffi"):
        try:
            version = getattr(__import__(library_name), "__version__")
        except Exception:
            pass
        else:
            libraries[library_name] = str(version)
    return {
        "hardware": torch.cuda.get_device_name(torch.cuda.current_device()),
        "libs": libraries,
    }


def _failure(
    status: str,
    environment: dict[str, Any],
    timestamp: str,
    log: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "correctness": None,
        "performance": None,
        "environment": environment,
        "timestamp": timestamp,
        "log": log,
    }


def _numerical_failure(
    environment: dict[str, Any],
    timestamp: str,
    log: str,
    maximum_absolute_error: float,
    maximum_relative_error: float,
    minimum_matched_ratio: float,
) -> dict[str, Any]:
    return {
        "status": "INCORRECT_NUMERICAL",
        "correctness": {
            "max_relative_error": maximum_relative_error,
            "max_absolute_error": maximum_absolute_error,
            "extra": {"matched_ratio": minimum_matched_ratio},
        },
        "performance": None,
        "environment": environment,
        "timestamp": timestamp,
        "log": log,
    }


def _runtime_failure(
    environment: dict[str, Any],
    timestamp: str,
    phase: str,
    error: Exception,
) -> dict[str, Any]:
    return _failure(
        "RUNTIME_ERROR",
        environment,
        timestamp,
        f"{phase} failed: {type(error).__name__}: {error}",
    )


__all__ = ["evaluate"]
