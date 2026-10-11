"""Compilation and GPU measurement functions for uploaded Python programs.

Import with ``from kcoral.builtins import compile_tirx, benchmark``. GPU toolchain
imports happen when a function is called.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from kcoral.errors import ExecutionError
from kcoral.support import vendor

__all__ = ["benchmark", "compile_tirx"]

# Bounded because each cached executable retains a loaded GPU module.
_COMPILED: OrderedDict[int, Any] = OrderedDict()
_COMPILED_LIMIT = 32


def compile_tirx(fn: Any, bindings: Any = None) -> Any:
    """Compile a ``@T.jit`` or ``@T.prim_func`` kernel handle. ``bindings`` supplies
    the ``T.constexpr`` values a ``@T.jit`` kernel is specialized on.

    Compiled executables are cached by structural hash, with at most 32 entries.
    This function can call CUDA while compiling and should run with GPU access.
    On ROCm workers it compiles for TVM's ``rocm`` target; TIRx schedules written
    for NVIDIA hardware may not lower there.
    """
    try:
        import tvm
    except ImportError as exc:  # tvm is an optional server dependency
        raise ExecutionError(
            "unavailable",
            "server-side compilation requires tvm, which is not installed on this server",
        ) from exc

    if bindings is not None and not isinstance(bindings, dict):
        raise ExecutionError("compile", "compile_tirx bindings must be a dict of constexpr values")
    kwargs = bindings or {}
    if isinstance(fn, tvm.tirx.PrimFunc):  # a @T.prim_func kernel — already concrete
        if kwargs:
            raise ExecutionError(
                "compile",
                "bindings apply only to @T.jit kernels; this kernel is already a PrimFunc",
            )
        pf = fn
    elif hasattr(fn, "specialize"):  # a @T.jit kernel handle (TIRJit)
        try:
            pf = fn.specialize(**kwargs)  # TIRX parse happens here
        except tvm.error.DiagnosticError as exc:
            raise ExecutionError("parse", _short(exc)) from exc
        except TypeError as exc:  # wrong, missing, or unhashable constexpr bindings
            raise ExecutionError("compile", _short(exc)) from exc
    else:
        raise ExecutionError(
            "compile", "compile_tirx expects a @T.jit or @T.prim_func kernel handle"
        )
    # Specializing is free and hashing costs microseconds; codegen is the 500ms.
    import tvm_ffi

    key = tvm_ffi.structural_hash(pf)
    cached = _COMPILED.get(key)
    if cached is not None:
        _COMPILED.move_to_end(key)
        return cached
    try:
        mod = tvm.IRModule({"main": pf})
        executable = tvm.compile(mod, target=_tvm_target(tvm), tir_pipeline="tirx")
    except tvm.error.InternalError as exc:  # lowering
        raise ExecutionError("compile", _short(exc)) from exc
    except RuntimeError as exc:  # codegen (nvcc/nvrtc, or the ROCm LLVM backend)
        raise ExecutionError("compile", _short(exc)) from exc
    _COMPILED[key] = executable
    if len(_COMPILED) > _COMPILED_LIMIT:
        _COMPILED.popitem(last=False)
    return executable


def _tvm_target(tvm: Any) -> Any:
    if vendor.backend() != "rocm":
        return tvm.target.Target("cuda")
    # The worker recorded its gfx architecture at startup; see describe_target.
    arch = os.environ.get("TVM_FFI_ROCM_ARCH_LIST", "").split()
    return tvm.target.Target({"kind": "rocm", "mcpu": arch[0]} if arch else "rocm")


def benchmark(mod: Any, *rest: Any) -> dict:
    """Per-iteration GPU activity timing from CUPTI (stream events on ROCm). cfg:
    ``warmup_ms``/``repeat_ms`` budgets convert to iteration counts using a
    5-call estimate, or ``warmup``/``repeat`` set explicit counts; ``flush_l2``
    zeroes a 2x-L2 buffer before every call, outside the timed span.

    Pass the callable followed by its arguments and an optional configuration
    dict. Defaults: ``warmup_ms=25``, ``repeat_ms=100``, ``flush_l2=True``.
    Explicit ``warmup`` and ``repeat`` counts override their respective budgets.

    Returns ``latency_ms_median``, ``latency_ms_mean``, ``latency_ms_min`` and
    ``latency_ms_max``, plus the selected counts, ``flush_l2`` and
    ``activities_stable``. A span runs from the first to last GPU activity
    launched by each call; host gaps between those activities are included.
    Requires PyTorch and cupti-python in the execution environment.

    ``timing`` names the method: ``"cupti"`` on NVIDIA, ``"trace"`` on ROCm
    (the same spans, from the PyTorch profiler's ROCprofiler trace). Where that
    trace records no GPU activity it is ``"events"``: each span runs between
    stream events around the call, so it includes launch latency and misses
    other streams, and ``activities_stable`` is None.
    """
    import statistics

    import torch

    tensors, cfg = _split_cfg(rest)
    if not callable(mod):
        raise ExecutionError("runtime", "benchmark expects a compiled module handle")
    flush_l2 = bool(cfg.get("flush_l2", True))

    def call() -> None:
        mod(*tensors)

    try:
        flush = None
        if flush_l2:
            flush = torch.empty(_flush_bytes(), dtype=torch.int8, device="cuda")
        warmup, repeat = _iteration_counts(call, cfg, flush)
        for _ in range(warmup):
            if flush is not None:
                flush.zero_()
            call()
        torch.cuda.synchronize()
        if vendor.backend() == "rocm":
            try:
                timing = "trace"
                times, activities_stable = _time_trace(call, repeat, flush)
            except ExecutionError as exc:
                if exc.kind != "unavailable":
                    raise
                timing, times, activities_stable = "events", _time_events(call, repeat, flush), None
        else:
            timing = "cupti"
            times, activities_stable = _time_cupti(call, repeat, flush)
    except RuntimeError as exc:  # a tvm run error or torch "CUDA error: ..."
        raise ExecutionError("runtime", _short(exc)) from exc
    return {
        "latency_ms_median": statistics.median(times),
        "latency_ms_mean": statistics.mean(times),
        "latency_ms_min": min(times),
        "latency_ms_max": max(times),
        # False when iterations covered differing work, so the stats mix kernels.
        "activities_stable": activities_stable,
        "timing": timing,
        "flush_l2": flush_l2,
        "warmup": warmup,
        "repeat": repeat,
    }


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


# AMD Instinct MI300 parts keep a 256 MiB Infinity Cache behind L2, which the
# reported L2 size does not include. Twice that bounds it on current parts.
_ROCM_MIN_FLUSH_BYTES = 512 * 1024 * 1024


def _flush_bytes() -> int:
    """A buffer large enough that zeroing it evicts the kernel's data from cache."""
    import torch

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    flush_bytes = 2 * int(getattr(properties, "L2_cache_size", 0))
    if vendor.backend() == "rocm":
        flush_bytes = max(flush_bytes, _ROCM_MIN_FLUSH_BYTES)
    return flush_bytes


def _time_events(call: Callable, repeat: int, flush: Any) -> list[float]:
    """Per-call spans (ms) between stream events, where CUPTI is unavailable.

    Same loop shape as :func:`_time_cupti`: flush, drain, then time one isolated
    call, so the two methods measure comparable iterations.
    """
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for index in range(repeat):
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize()
        start.record()
        call()
        end.record()
        end.synchronize()
        span_ms = start.elapsed_time(end)
        if span_ms <= 0:
            raise ExecutionError(
                "unavailable", f"events recorded no positive time for benchmark iteration {index}"
            )
        times.append(span_ms)
    return times


_ITERATION_MARKER = "kcoral::benchmark_iteration"


def _time_trace(call: Callable, repeat: int, flush: Any) -> tuple[list[float], bool]:
    """Per-call GPU activity spans (ms) from the PyTorch profiler, and whether
    every call launched the same activities: :func:`_time_cupti`'s semantics
    where CUPTI is unavailable. On ROCm, Kineto traces through ROCprofiler.

    Each call runs inside a marked CPU range that ends after a synchronize, so
    its GPU activities start inside that range and the flush before it does not.
    """
    from collections import Counter

    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        for _ in range(repeat):
            if flush is not None:
                flush.zero_()
            torch.cuda.synchronize()
            with record_function(_ITERATION_MARKER):
                call()
                torch.cuda.synchronize()
    try:
        events = trace.profiler.kineto_results.events()
    except AttributeError as exc:
        raise ExecutionError("unavailable", "the profiler returned no trace") from exc

    windows: list[tuple[int, int]] = []
    activities: list[tuple[int, int, str]] = []
    for event in events:
        start = int(event.start_ns())
        end = start + int(event.duration_ns())
        if event.device_type() == DeviceType.CPU:
            if event.name() == _ITERATION_MARKER:
                windows.append((start, end))
        elif event.name() != _ITERATION_MARKER:  # not the marker's GPU annotation
            activities.append((start, end, event.name()))
    windows.sort()
    activities.sort()
    if len(windows) != repeat or not activities:
        raise ExecutionError("unavailable", "the profiler trace recorded no GPU activity")

    times = []
    activities_stable = True
    expected_activities = None
    position = 0
    for index, (window_start, window_end) in enumerate(windows):
        while position < len(activities) and activities[position][0] < window_start:
            position += 1  # a flush, or work outside any iteration
        current = []
        while position < len(activities) and activities[position][0] <= window_end:
            current.append(activities[position])
            position += 1
        if not current:
            raise ExecutionError(
                "unavailable", f"the profiler recorded no GPU activity for iteration {index}"
            )
        names = Counter(name for _, _, name in current)
        if expected_activities is None:
            expected_activities = names
        elif names != expected_activities:
            activities_stable = False
        span_ms = (max(end for _, end, _ in current) - min(start for start, _, _ in current)) / 1e6
        if span_ms <= 0:
            raise ExecutionError(
                "unavailable", f"the profiler recorded no positive time for iteration {index}"
            )
        times.append(span_ms)
    return times, activities_stable


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
                    "unavailable", f"CUPTI cleanup failed: {_short(cleanup_errors[0])}"
                )
    except cupti.cuptiError as exc:
        raise ExecutionError(
            "unavailable", f"CUPTI activity tracing failed: {_short(exc)}"
        ) from exc

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


def _short(text: str | Exception, limit: int = 600) -> str:
    """Truncate text keeping both ends: a tvm diagnostic leads with its message,
    a compiler failure ends with its diagnostic."""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    head = limit // 3
    return f"{text[:head]} …[truncated]… {text[head - limit :]}"


def _split_cfg(args: tuple) -> tuple[tuple, dict]:
    """Split a harness's trailing config dict from its leading tensor args."""
    if args and isinstance(args[-1], dict):
        return args[:-1], args[-1]
    return args, {}
