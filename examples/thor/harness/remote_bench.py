"""Server-side benchmark harness for the Thor GEMM example.

``bench.py`` uploads this file verbatim as a KCoral module and calls
:func:`evaluate` inside a GPU worker. Everything the protocol promises lives
here, so the numbers do not depend on the client machine:

* kernels are compiled once per shape, outside every timed region;
* correctness is checked against an fp32 reference on several seeds, with the
  output poisoned with NaN before every checked call and re-checked after timing;
* timing uses a fixed number of warmup and measured calls per trial, CUDA events
  around each call, an L2 flush between calls, and implementations interleaved
  round-robin in every trial so clock and thermal drift hit all of them alike;
* GPU clock and temperature are sampled around every timed block.

Returned values contain only JSON-compatible, finite numbers.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import statistics
import sys
import tempfile
import time
import traceback

CUBLAS = "cublas"
# Kernel names that indicate a vendor GEMM ran inside a candidate's timed call.
VENDOR_KERNEL_PATTERN = re.compile(r"nvjet|cutlass|cublas|xmma|splitKreduce|ampere_|volta_|turing_")


# ── telemetry ────────────────────────────────────────────────────────────────


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _devfreq_dir(keywords: tuple[str, ...]) -> str | None:
    """First devfreq node whose name contains one of ``keywords`` (in priority order)."""
    names = {
        path: _read(os.path.join(path, "name")) or os.path.basename(path)
        for path in sorted(glob.glob("/sys/class/devfreq/*"))
    }
    for keyword in keywords:
        for path, name in names.items():
            if keyword in name:
                return path
    return None


# Thor exposes the GPU graphics clock as ``gpu-gpc-0`` and the memory clock as ``bwmgr``.
GPU_CLOCK_KEYWORDS = ("gpu-gpc", "gpc", "gpu")
MEM_CLOCK_KEYWORDS = ("bwmgr", "emc")


def _gpu_thermal_zone() -> str | None:
    for path in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        if "gpu" in (_read(os.path.join(path, "type")) or ""):
            return path
    return None


def sample_telemetry() -> dict:
    sample: dict = {"time_s": time.time()}
    for prefix, keywords in (("", GPU_CLOCK_KEYWORDS), ("mem_", MEM_CLOCK_KEYWORDS)):
        devfreq = _devfreq_dir(keywords)
        if devfreq is None:
            continue
        for key in ("cur_freq", "max_freq"):
            value = _read(os.path.join(devfreq, key))
            sample[f"{prefix}{key}_mhz"] = int(value) / 1e6 if value and value.isdigit() else None
        sample[f"{prefix}governor"] = _read(os.path.join(devfreq, "governor"))
    zone = _gpu_thermal_zone()
    if zone is not None:
        value = _read(os.path.join(zone, "temp"))
        sample["temp_c"] = int(value) / 1000 if value and value.lstrip("-").isdigit() else None
    return sample


# ── kernels ──────────────────────────────────────────────────────────────────


def _load_kernel_module(name: str, source: str, directory: str):
    """Import a candidate from source. A real file keeps TVMScript and tracebacks working."""
    path = os.path.join(directory, f"{name}.py")
    with open(path, "w") as f:
        f.write(source)
    module_name = f"thor_kernel_{re.sub(r'[^0-9A-Za-z_]', '_', name)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build", None)):
        raise TypeError(f"{name}: kernel module must define build(M, N, K)")
    return module


def _short_traceback(limit: int = 4000) -> str:
    text = traceback.format_exc()
    return text if len(text) <= limit else "...\n" + text[-limit:]


def _cuda_source(fn) -> str | None:
    """Generated CUDA C of a compiled TVM executable, if ``fn`` is one."""
    try:
        return "\n\n".join(m.inspect_source() for m in fn.mod.imports)
    except Exception:
        return None


def _cublas_fn(A, B, D) -> None:
    import torch

    torch.matmul(A, B.T, out=D)


# ── correctness ──────────────────────────────────────────────────────────────


def check_output(D, ref, rtol: float, atol: float) -> dict:
    import torch

    out = D.float()
    finite = bool(torch.isfinite(out).all())
    err = (out - ref).abs()
    bad = ~(err <= atol + rtol * ref.abs())  # NaN counts as bad
    num_bad = int(bad.sum())
    result = {
        "passed": finite and num_bad == 0,
        "finite": finite,
        "num_bad": num_bad,
        "max_abs_err": float(torch.nan_to_num(err, nan=float("inf")).max().clamp(max=3e38)),
    }
    if num_bad:
        index = int(bad.flatten().nonzero()[0])
        row, col = divmod(index, D.shape[1])
        result["first_bad"] = {
            "row": row,
            "col": col,
            "got": float(torch.nan_to_num(out[row, col], nan=0.0)) if finite else None,
            "expected": float(ref[row, col]),
        }
    return result


def run_checks(fn, inputs, out, rtol: float, atol: float) -> dict:
    """Run ``fn`` once per input set, poisoning the output first, and compare."""
    import torch

    checks = []
    call_ms = []
    for A, B, ref in inputs:
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(A, B, out)
        torch.cuda.synchronize()
        call_ms.append((time.perf_counter() - start) * 1e3)
        checks.append(check_output(out, ref, rtol, atol))
        if not checks[-1]["passed"]:
            break
    return {
        "passed": all(c["passed"] for c in checks),
        "call_ms": round(min(call_ms), 3),
        "runs": checks,
    }


# ── timing ───────────────────────────────────────────────────────────────────


def run_for(fn, args, seconds: float, flush) -> None:
    """Call ``fn`` untimed, L2 flushes included, for ``seconds`` of wall time."""
    import torch

    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if flush is not None:
            flush.zero_()
        fn(*args)
        torch.cuda.synchronize()


def time_calls(fn, args, settle_s: float, warmup: int, repeat: int, flush) -> list[float]:
    """Per-call GPU latency in ms: CUDA events around each call, L2 flushed between calls.

    ``fn`` first runs untimed for ``settle_s`` seconds, then ``warmup`` calls,
    then ``repeat`` timed calls, so the clocks see the same load throughout.
    """
    import torch

    run_for(fn, args, settle_s, flush)
    for _ in range(warmup):
        if flush is not None:
            flush.zero_()
        fn(*args)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        if flush is not None:
            flush.zero_()
        starts[i].record()
        fn(*args)
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def launched_kernels(fn, args) -> list[str] | None:
    """Names of the GPU kernels one call launches (for review; not timed)."""
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(*args)
            torch.cuda.synchronize()
        names = {
            event.name
            for event in prof.events()
            if getattr(event, "device_type", None) == torch.autograd.DeviceType.CUDA
        }
        return sorted(names)
    except Exception:  # profiler support varies across builds; this is informational
        return None


# ── entry point ──────────────────────────────────────────────────────────────


def environment() -> dict:
    import torch

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability()
    env = {
        "device": props.name,
        "arch": f"sm_{major}{minor}a",
        "sm_count": props.multi_processor_count,
        "l2_bytes": props.L2_cache_size,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
    }
    try:
        import tvm

        env["tvm"] = tvm.__version__
    except Exception as exc:
        env["tvm"] = f"unavailable: {exc}"
    return env


def evaluate(task_json: bytes, options_json: bytes, *kernel_sources: bytes) -> dict:
    """Check and time every kernel plus cuBLAS on the task's shapes.

    ``options_json`` holds ``names`` (one per kernel source), ``shapes`` (indices
    into the task's shape list, or null for all) and ``check_only``.
    """
    import torch

    task = json.loads(bytes(task_json))
    options = json.loads(bytes(options_json))
    names = list(options["names"])
    if len(names) != len(kernel_sources) or CUBLAS in names:
        raise ValueError("kernel names must match the sources and must not be 'cublas'")
    check_only = bool(options.get("check_only", False))
    shapes = task["shapes"]
    indices = options.get("shapes")
    selected = [shapes[i] for i in indices] if indices is not None else shapes
    corr = task["correctness"]
    timing = task["timing"]

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    telemetry_before = sample_telemetry()

    workdir = tempfile.mkdtemp(prefix="thor-gemm-")
    modules: dict = {}
    load_errors: dict = {}
    for name, source in zip(names, kernel_sources):
        try:
            modules[name] = _load_kernel_module(name, bytes(source).decode(), workdir)
        except Exception:
            load_errors[name] = _short_traceback()

    flush = None
    if timing.get("flush_l2", True):
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size or (64 << 20)
        flush = torch.empty(2 * l2_bytes, dtype=torch.uint8, device="cuda")

    results = []
    for shape in selected:
        M, N, K = shape["M"], shape["N"], shape["K"]
        entry: dict = {"name": shape.get("name"), "M": M, "N": N, "K": K, "impls": {}}
        results.append(entry)

        inputs = []
        for seed in corr["seeds"]:
            gen = torch.Generator(device="cuda").manual_seed(seed)
            A = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=gen)
            B = torch.randn(N, K, device="cuda", dtype=torch.float16, generator=gen)
            inputs.append((A, B, A.float() @ B.float().T))
        A, B, ref = inputs[0]

        # Build and check each implementation; only passing ones are timed.
        fns = {CUBLAS: _cublas_fn}
        for name in names:
            info: dict = {}
            entry["impls"][name] = info
            if name in load_errors:
                info.update(status="ERROR", error=load_errors[name])
                continue
            try:
                start = time.perf_counter()
                fns[name] = modules[name].build(M, N, K)
                torch.cuda.synchronize()
                info["build_s"] = round(time.perf_counter() - start, 3)
            except Exception:
                info.update(status="ERROR", error=_short_traceback())
        entry["impls"][CUBLAS] = {}
        if options.get("emit_source"):
            for name in names:
                if name in fns:
                    entry["impls"][name]["cuda_source"] = _cuda_source(fns[name])

        outputs = {}
        for name, fn in list(fns.items()):
            info = entry["impls"][name]
            out = torch.empty(M, N, device="cuda", dtype=torch.float16)
            outputs[name] = out
            try:
                info["check"] = run_checks(fn, inputs, out, corr["rtol"], corr["atol"])
            except Exception:
                info.update(status="ERROR", error=_short_traceback())
                del fns[name]
                continue
            if not info["check"]["passed"]:
                info["status"] = "FAIL"
                del fns[name]
        if not entry["impls"][CUBLAS]["check"]["passed"]:
            raise RuntimeError("cuBLAS failed the correctness check: the tolerances are wrong")

        if check_only:
            for name in fns:
                entry["impls"][name]["status"] = "PASS"
            continue

        # Timing. Thor's GPU and memory clocks follow the load and need seconds of
        # sustained memory traffic to reach their top state; idle or compute-bound
        # stretches drop them again. So: time slow (compute-bound, long-call)
        # implementations first on their own, then ramp the clocks with cuBLAS,
        # then time the fast implementations interleaved with no idle gaps.
        # Time on the first seed's buffers refilled in place with the last seed's data, and
        # re-check against that data afterwards: results cached per buffer cannot pass.
        A.copy_(inputs[-1][0])
        B.copy_(inputs[-1][1])
        ref = inputs[-1][2]
        warmup, repeat, trials = int(timing["warmup"]), int(timing["repeat"]), int(timing["trials"])
        slow_ms = float(timing["slow_call_ms"])
        slow = [n for n in fns if n != CUBLAS and entry["impls"][n]["check"]["call_ms"] >= slow_ms]
        fast = [n for n in fns if n not in slow]
        trial_ms: dict = {name: [] for name in fns}
        entry["telemetry"] = [sample_telemetry()]

        def timed(name: str, settle_s: float) -> None:
            out = outputs[name]
            out.fill_(float("nan"))
            try:
                times = time_calls(fns[name], (A, B, out), settle_s, warmup, repeat, flush)
            except Exception:
                entry["impls"][name].update(status="ERROR", error=_short_traceback())
                fns.pop(name, None)
                return
            trial_ms[name].append(statistics.median(times))

        for name in slow:
            entry["impls"][name]["timing"] = "isolated"
            for _ in range(trials):
                if name in fns:
                    timed(name, 0.0)
        if slow:
            entry["telemetry"].append(sample_telemetry())

        run_for(fns[CUBLAS], (A, B, outputs[CUBLAS]), float(timing["ramp_s"]), flush)
        for name in fast:
            entry["impls"][name]["timing"] = "interleaved"
        for trial in range(trials):
            shift = trial % len(fast)
            for name in fast[shift:] + fast[:shift]:
                if name in fns:
                    timed(name, float(timing["settle_s"]))
            entry["telemetry"].append(sample_telemetry())

        flops = 2.0 * M * N * K
        for name in list(fns):
            info = entry["impls"][name]
            after = check_output(outputs[name], ref, corr["rtol"], corr["atol"])
            info["check_after_timing"] = after
            if not after["passed"]:
                info["status"] = "FAIL"
                continue
            medians = trial_ms[name]
            median_ms = statistics.median(medians)
            info.update(
                status="PASS",
                trial_median_ms=[round(t, 5) for t in medians],
                median_ms=round(median_ms, 5),
                spread=round((max(medians) - min(medians)) / median_ms, 4),
                tflops=round(flops / (median_ms * 1e-3) / 1e12, 2),
                kernels=launched_kernels(fns[name], (A, B, outputs[name])),
            )
            if name != CUBLAS and info["kernels"]:
                vendor = [k for k in info["kernels"] if VENDOR_KERNEL_PATTERN.search(k)]
                if vendor:
                    info["status"] = "FAIL"
                    info["error"] = f"timed call launched vendor GEMM kernels: {vendor}"

        inputs, outputs, fns = [], {}, {}  # release this shape's buffers
        torch.cuda.empty_cache()

    return {
        "environment": environment(),
        "task": task.get("name"),
        "check_only": check_only,
        "names": names,
        "shapes": results,
        "telemetry_before": telemetry_before,
        "telemetry_after": sample_telemetry(),
    }
