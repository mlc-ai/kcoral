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
    """Per-iteration CUPTI kernel timing (via triton's proton profiler). cfg:
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
        times = _time_proton(call, repeat, flush)
    except RuntimeError as exc:  # a tvm run error or torch "CUDA error: ..."
        raise ExecutionError("runtime", short(exc)) from exc
    return {
        "latency_ms_median": statistics.median(times),
        "latency_ms_mean": statistics.mean(times),
        "latency_ms_min": min(times),
        "latency_ms_max": max(times),
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
        return max(int(warmup), 0), max(int(repeat), 1)
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


def _time_proton(call: Callable, repeat: int, flush: Any) -> list[float]:
    """Per-iteration GPU kernel times (ms) from CUPTI, one proton scope per call."""
    import json
    import os
    import tempfile
    import uuid

    import torch

    try:
        import triton.profiler as proton
    except ImportError as exc:
        raise ExecutionError(
            "unavailable",
            "benchmark needs triton's proton profiler (bundled with CUDA torch builds)",
        ) from exc

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "profile")
        session = proton.start(path, context="shadow", data="tree")
        if session is None:
            raise ExecutionError(
                "unavailable",
                "proton could not start a CUPTI session "
                "(another profiler attached to this process?)",
            )
        prefix = f"bench.{uuid.uuid4().hex}."
        try:  # finalize even on failure, or the leaked session poisons the next one
            for i in range(repeat):
                if flush is not None:
                    flush.zero_()
                with proton.scope(f"{prefix}{i:08d}"):
                    call()
            torch.cuda.synchronize()
        finally:
            proton.finalize(session)
        with open(path + ".hatchet") as f:
            tree = json.load(f)

    times = _proton_scope_times(tree, prefix)
    if len(times) != repeat or not all(t > 0 for t in times):
        raise ExecutionError(
            "unavailable",
            f"proton attributed kernel time to {len(times)}/{repeat} iterations",
        )
    return times


def _proton_scope_times(tree: Any, prefix: str) -> list[float]:
    """Summed GPU kernel ms of every hatchet scope named ``prefix<i>``, in order."""
    found: list[tuple[str, float]] = []

    def leaf_ms(node: dict) -> float:
        children = node.get("children", [])
        if not children:
            return node.get("metrics", {}).get("time (ns)", 0) / 1e6
        return sum(leaf_ms(c) for c in children)

    def visit(node: dict) -> None:
        name = node.get("frame", {}).get("name", "")
        if name.startswith(prefix):
            found.append((name, leaf_ms(node)))
            return
        for c in node.get("children", []):
            visit(c)

    for node in tree:
        if isinstance(node, dict):  # the top level may carry a device_info entry
            visit(node)
    return [t for _, t in sorted(found)]  # zero-padded names sort in iteration order


def _read_spec(spec: Any) -> tuple[list[int], Any, Any]:
    if not isinstance(spec, dict):
        raise ExecutionError("runtime", "expected a {shape, dtype} spec")
    try:
        shape = [int(d) for d in spec["shape"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ExecutionError("runtime", f"bad 'shape' in spec: {exc}") from exc
    return shape, torch_dtype(spec.get("dtype", "float16")), spec.get("seed")
