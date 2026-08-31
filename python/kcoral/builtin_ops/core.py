"""Builtins that need no kernel-language toolchain: tensor creation, timing, and
correctness comparison. They accept any callable a compile builtin returns."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..errors import ExecutionError
from ._common import short, split_cfg, torch_dtype
from ._registry import register_builtin


@register_builtin("randn")
def randn(spec: Any) -> Any:
    import torch

    shape, dtype, seed = _read_spec(spec)
    if not dtype.is_floating_point:
        raise ExecutionError("runtime", "randn requires a floating dtype")
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(int(seed))
    return torch.randn(shape, dtype=dtype, device="cuda", generator=gen)


@register_builtin("empty")
def empty(spec: Any) -> Any:
    import torch

    shape, dtype, _ = _read_spec(spec)
    return torch.empty(shape, dtype=dtype, device="cuda")


@register_builtin("zeros")
def zeros(spec: Any) -> Any:
    import torch

    shape, dtype, _ = _read_spec(spec)
    return torch.zeros(shape, dtype=dtype, device="cuda")


@register_builtin("benchmark")
def benchmark(mod: Any, *rest: Any) -> dict:
    """Per-iteration GPU activity timing from CUPTI. cfg:
    ``warmup_ms``/``repeat_ms`` budgets convert to iteration counts using a
    5-call estimate, or ``warmup``/``repeat`` set explicit counts; ``flush_l2``
    zeroes a 2x-L2 buffer before every call, outside the timed span."""
    import statistics

    import torch

    tensors, cfg = split_cfg(rest)
    if not callable(mod):
        raise ExecutionError("runtime", "benchmark expects a compiled module handle")
    flush_l2 = bool(cfg.get("flush_l2", True))

    def call() -> None:
        mod(*tensors)

    try:
        flush = None
        if flush_l2:
            l2_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
            flush = torch.empty(2 * l2_bytes, dtype=torch.int8, device="cuda")
        warmup, repeat = _iteration_counts(call, cfg, flush)
        for _ in range(warmup):
            if flush is not None:
                flush.zero_()
            call()
        torch.cuda.synchronize()
        times, activities_stable = _time_cupti(call, repeat, flush)
    except RuntimeError as exc:  # a tvm run error or torch "CUDA error: ..."
        raise ExecutionError("runtime", short(exc)) from exc
    return {
        "latency_ms_median": statistics.median(times),
        "latency_ms_mean": statistics.mean(times),
        "latency_ms_min": min(times),
        "latency_ms_max": max(times),
        # False when iterations covered differing work, so the stats mix kernels.
        "activities_stable": activities_stable,
        "flush_l2": flush_l2,
        "warmup": warmup,
        "repeat": repeat,
    }


@register_builtin("check_close")
def check_close(actual: Any, expected: Any, *rest: Any) -> dict:
    import torch

    _, cfg = split_cfg(rest)
    rtol = float(cfg.get("rtol", 1e-2))
    atol = float(cfg.get("atol", 1e-3))
    try:
        torch.cuda.synchronize()
        a = actual.float()
        e = expected.float()
        diff = (a - e).abs()
        max_abs = float(diff.max())
        # max |a-e|/|e| over nonzero e; masking keeps it finite (JSON-safe).
        nz = e != 0
        max_rel = float((diff[nz] / e[nz].abs()).max()) if bool(nz.any()) else 0.0
        passed = bool(torch.allclose(a, e, rtol=rtol, atol=atol))
    except RuntimeError as exc:
        raise ExecutionError("runtime", short(exc)) from exc
    return {
        "passed": passed,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "rtol": rtol,
        "atol": atol,
    }


@register_builtin("assert_close")
def assert_close(actual: Any, expected: Any, *rest: Any) -> dict:
    """Like ``check_close``, but a mismatch is a failure: it raises ``correctness``
    so the instruction FAILs and the rest of the program is skipped."""
    result = check_close(actual, expected, *rest)
    if not result["passed"]:
        raise ExecutionError(
            "correctness",
            f"outputs differ: max_abs_err={result['max_abs_err']}, "
            f"max_rel_err={result['max_rel_err']} exceed "
            f"atol={result['atol']}, rtol={result['rtol']}",
        )
    return result


# --- helpers ----------------------------------------------------------------


def _iteration_counts(call: Callable, cfg: dict, flush: Any) -> tuple[int, int]:
    """Explicit ``warmup``/``repeat`` counts, or counts derived from the
    ``warmup_ms``/``repeat_ms`` budgets and a 5-call runtime estimate."""
    import torch

    warmup, repeat = cfg.get("warmup"), cfg.get("repeat")
    if warmup is not None and repeat is not None:
        return max(int(warmup), 1), max(int(repeat), 1)
    call()  # exclude one-time init from the estimate
    if flush is not None:
        # The buffer is freshly allocated and twice L2 (253 MiB on B200), so its
        # first touch costs ~16x a warm one and would swamp a 5-sample estimate.
        flush.zero_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        if flush is not None:
            flush.zero_()
        call()
    end.record()
    torch.cuda.synchronize()
    est_ms = start.elapsed_time(end) / 5

    def derive(budget_ms: float) -> int:
        return 1000 if est_ms == 0 else max(1, int(budget_ms / est_ms))

    n_warmup = derive(float(cfg.get("warmup_ms", 25))) if warmup is None else max(int(warmup), 1)
    n_repeat = derive(float(cfg.get("repeat_ms", 100))) if repeat is None else max(int(repeat), 1)
    return n_warmup, n_repeat


def _time_cupti(call: Callable, repeat: int, flush: Any) -> tuple[list[float], bool]:
    """Per-call GPU activity spans (ms), and whether every call launched the same
    activities. Iterations are drained, so a span times one isolated call. Adopted
    from flashinfer's ``bench_gpu_time_with_cupti``:
    https://github.com/flashinfer-ai/flashinfer/blob/0659712/flashinfer/testing/utils.py
    """
    import bisect
    import sys
    from collections import Counter

    import torch

    try:
        from cupti import cupti
    except ImportError as exc:
        raise ExecutionError(
            "unavailable",
            "benchmark needs cupti-python in the worker environment",
        ) from exc

    activity_kinds = (
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    )
    launch_records: list[tuple[int, int]] = []
    gpu_records: list[tuple[int, int, int, tuple]] = []
    iteration_windows: list[tuple[int, int]] = []

    def request_buffer() -> tuple[int, int]:
        return 8 * 1024 * 1024, 0

    def complete_buffer(activities: list[Any]) -> None:
        for activity in activities:
            if activity.kind in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER):
                launch_records.append((activity.start, activity.correlation_id))
            elif activity.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                # What counts as "the same" activity across iterations.
                identity = (
                    int(activity.kind),
                    str(getattr(activity, "name", "")),
                    int(getattr(activity, "copy_kind", 0)),
                    int(getattr(activity, "bytes", 0)),
                    int(getattr(activity, "value", 0)),
                )
                gpu_records.append(
                    (activity.start, activity.end, activity.correlation_id, identity)
                )

    enabled_kinds = []
    callbacks_registered = False
    try:
        try:
            for activity_kind in activity_kinds:
                cupti.activity_enable(activity_kind)
                enabled_kinds.append(activity_kind)
            cupti.activity_register_callbacks(request_buffer, complete_buffer)
            callbacks_registered = True
            for _ in range(repeat):
                if flush is not None:
                    flush.zero_()
                torch.cuda.synchronize()
                start = cupti.get_timestamp()
                call()
                end = cupti.get_timestamp()
                torch.cuda.synchronize()
                iteration_windows.append((start, end))
        finally:
            active_exception = sys.exc_info()[0] is not None
            cleanup_errors = []
            if callbacks_registered:
                try:
                    torch.cuda.synchronize()
                except Exception as exc:
                    cleanup_errors.append(exc)
                try:
                    cupti.activity_flush_all(0)
                except Exception as exc:
                    cleanup_errors.append(exc)
            for activity_kind in reversed(enabled_kinds):
                try:
                    cupti.activity_disable(activity_kind)
                except Exception as exc:
                    cleanup_errors.append(exc)
            if enabled_kinds:
                try:
                    cupti.finalize()
                except Exception as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors and not active_exception:
                raise ExecutionError(
                    "unavailable", f"CUPTI cleanup failed: {short(cleanup_errors[0])}"
                )
    except cupti.cuptiError as exc:
        raise ExecutionError("unavailable", f"CUPTI activity tracing failed: {short(exc)}") from exc

    launch_records.sort()
    launch_starts = [record[0] for record in launch_records]
    activities_by_correlation: dict[int, list[tuple[int, int, tuple]]] = {}
    for start, end, correlation_id, identity in gpu_records:
        activities_by_correlation.setdefault(correlation_id, []).append((start, end, identity))

    times = []
    activities_stable = True
    expected_activities = None
    for index, (start, end) in enumerate(iteration_windows):
        left = bisect.bisect_left(launch_starts, start)
        right = bisect.bisect_right(launch_starts, end)
        correlation_ids = {launch_records[position][1] for position in range(left, right)}
        activities = [
            activity
            for correlation_id in correlation_ids
            for activity in activities_by_correlation.get(correlation_id, ())
        ]
        if not activities:
            raise ExecutionError(
                "unavailable", f"CUPTI recorded no GPU activity for benchmark iteration {index}"
            )
        current_activities = Counter(activity[2] for activity in activities)
        if expected_activities is None:
            expected_activities = current_activities
        elif current_activities != expected_activities:
            activities_stable = False
        span_ms = (
            max(activity[1] for activity in activities)
            - min(activity[0] for activity in activities)
        ) / 1e6
        if span_ms <= 0:
            raise ExecutionError(
                "unavailable", f"CUPTI recorded no positive time for benchmark iteration {index}"
            )
        times.append(span_ms)
    return times, activities_stable


def _read_spec(spec: Any) -> tuple[list[int], Any, Any]:
    if not isinstance(spec, dict):
        raise ExecutionError("runtime", "expected a {shape, dtype} spec")
    try:
        shape = [int(d) for d in spec["shape"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ExecutionError("runtime", f"bad 'shape' in spec: {exc}") from exc
    return shape, torch_dtype(spec.get("dtype", "float16")), spec.get("seed")
