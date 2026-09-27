"""Check and benchmark GEMM kernels on a remote Thor KCoral server.

Every kernel file passed on the command line, plus cuBLAS as the reference, is
checked and timed in one KCoral request, so all implementations share the same
GPU and the same thermal window. The task is defined by ``definition.json`` and
``workload.jsonl`` (FlashInfer Trace format).

This file runs on both sides. You run it as a script on the client; it then
uploads its own source as a KCoral module, and the server calls
:func:`evaluate` from it inside a GPU worker. Everything the protocol promises
happens there, so the numbers do not depend on the client machine:

* kernels are compiled once per shape, outside every timed region;
* correctness is checked against the definition's reference on several seeds,
  with the output poisoned with NaN before every checked call and re-checked
  after timing;
* timing uses a fixed number of warmup and measured calls per trial, CUDA events
  around each call, an L2 flush between calls, and implementations interleaved
  round-robin in every trial so clock and thermal drift hit all of them alike;
* GPU clock and temperature are sampled around every timed block.

Module scope runs on both sides too, so it imports only the standard library
and ``kcoral``; torch and TVM are imported inside the server-side functions.

Examples::

    uv run python bench.py initial_kernel.py
    uv run python bench.py work/best.py work/new.py
    uv run python bench.py work/new.py --check-only --shapes 0
"""

from __future__ import annotations

import argparse
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
from dataclasses import dataclass
from pathlib import Path

from kcoral import Client, KCoralError, Program, ProtocolError, TransportError

CUBLAS = "cublas"
INITIAL = "initial_kernel"

# Kernel names that indicate a vendor GEMM ran inside a candidate's timed call.
VENDOR_KERNEL_PATTERN = re.compile(r"nvjet|cutlass|cublas|xmma|splitKreduce|ampere_|volta_|turing_")


@dataclass(frozen=True)
class BenchConfig:
    """Correctness, timing and stability settings shared by every run."""

    # Correctness: an element fails only if it is off by more than ``atol`` and by
    # more than ``rtol`` relatively; every element must pass.
    seeds: tuple[int, ...] = (0, 1)
    atol: float = 0.1
    rtol: float = 0.01
    # Timing: per trial, ``warmup`` untimed then ``repeat`` timed calls.
    warmup: int = 10
    repeat: int = 50
    trials: int = 3
    flush_l2: bool = True
    # Thor's clocks follow the load. Implementations slower than ``slow_call_ms`` per
    # call are timed on their own first; then cuBLAS ramps the clocks for ``ramp_s``
    # and each remaining implementation runs untimed for ``settle_s`` before timing.
    slow_call_ms: float = 20.0
    ramp_s: float = 4.0
    settle_s: float = 0.25
    # Stability: the client warns beyond these limits.
    max_spread: float = 0.05
    min_clock_fraction: float = 0.97


# ═════════════════════════════════════════════════════════════════════════════
# Server side: runs inside the KCoral GPU worker, entered through evaluate().
# ═════════════════════════════════════════════════════════════════════════════


# ── telemetry ────────────────────────────────────────────────────────────────


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _find_node(pattern: str, name_file: str, needle: str) -> str | None:
    """First sysfs node matching ``pattern`` whose ``name_file`` contains ``needle``."""
    for path in sorted(glob.glob(pattern)):
        if needle in (_read(os.path.join(path, name_file)) or os.path.basename(path)):
            return path
    return None


def sample_telemetry() -> dict:
    """GPU and memory clocks (Thor's ``gpu-gpc`` and ``bwmgr`` devfreq nodes), GPU temperature."""
    sample: dict = {"time_s": time.time()}
    for prefix, needle in (("", "gpu-gpc"), ("mem_", "bwmgr")):
        devfreq = _find_node("/sys/class/devfreq/*", "name", needle)
        if devfreq is None:
            continue
        for key in ("cur_freq", "max_freq"):
            value = _read(os.path.join(devfreq, key))
            sample[f"{prefix}{key}_mhz"] = int(value) / 1e6 if value and value.isdigit() else None
        sample[f"{prefix}governor"] = _read(os.path.join(devfreq, "governor"))
    zone = _find_node("/sys/class/thermal/thermal_zone*", "type", "gpu")
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


def _cublas_fn(A, B, D) -> None:
    import torch

    torch.matmul(A, B.T, out=D)


# ── task files ───────────────────────────────────────────────────────────────


def parse_workloads(workload_jsonl: str) -> list[dict]:
    """The ``workload`` objects of a FlashInfer Trace ``workload.jsonl``."""
    return [json.loads(line)["workload"] for line in workload_jsonl.splitlines() if line.strip()]


def load_reference(definition: dict):
    """The ``run`` function defined by the definition's ``reference`` source."""
    namespace: dict = {}
    exec(compile(definition["reference"], "definition.json:reference", "exec"), namespace)
    return namespace["run"]


def _tensor_shape(spec: dict, axes: dict) -> list[int]:
    return [axes[axis] for axis in spec["shape"]]


def make_inputs(definition: dict, workload: dict, seed: int) -> list:
    """Input tensors in definition order; only ``random`` input specs are supported."""
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    tensors = []
    for name, spec in definition["inputs"].items():
        kind = workload["inputs"][name]["type"]
        if kind != "random":
            raise ValueError(f"input {name}: unsupported workload input type {kind!r}")
        shape = _tensor_shape(spec, workload["axes"])
        dtype = getattr(torch, spec["dtype"])
        tensors.append(torch.randn(shape, dtype=dtype, device="cuda", generator=gen))
    return tensors


def make_output(definition: dict, workload: dict):
    import torch

    ((_, spec),) = definition["outputs"].items()
    shape = _tensor_shape(spec, workload["axes"])
    return torch.empty(shape, dtype=getattr(torch, spec["dtype"]), device="cuda")


# ── correctness ──────────────────────────────────────────────────────────────


def check_output(D, ref, rtol: float, atol: float) -> dict:
    """FlashInfer's rule: an element fails if its error exceeds both ``atol`` and ``rtol``."""
    import torch

    out = D.float()
    ref = ref.float()
    finite = bool(torch.isfinite(out).all())
    abs_err = (out - ref).abs()
    rel_err = abs_err / (ref.abs() + 1e-8)
    bad = ~((abs_err <= atol) | (rel_err <= rtol))  # NaN counts as bad
    num_bad = int(bad.sum())

    def worst(err) -> float:
        return float(torch.nan_to_num(err, nan=float("inf")).max().clamp(max=3e38))

    result = {
        "passed": finite and num_bad == 0,
        "finite": finite,
        "num_bad": num_bad,
        "max_abs_err": worst(abs_err),
        "max_rel_err": worst(rel_err),
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
    import tvm

    props = torch.cuda.get_device_properties(0)
    return {
        "device": props.name,
        "arch": f"sm_{props.major}{props.minor}a",
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "tvm": tvm.__version__,
    }


def evaluate(
    definition_json: bytes, workload_jsonl: bytes, options_json: bytes, *kernel_sources: bytes
) -> dict:
    """Check and time every kernel plus cuBLAS on the workload's shapes.

    ``options_json`` holds ``names`` (one per kernel source), ``shapes`` (indices
    into the workload rows, or null for all) and ``check_only``.
    """
    import torch

    cfg = BenchConfig()
    definition = json.loads(bytes(definition_json))
    workloads = parse_workloads(bytes(workload_jsonl).decode())
    reference = load_reference(definition)
    options = json.loads(bytes(options_json))
    names = list(options["names"])
    if len(names) != len(kernel_sources) or CUBLAS in names:
        raise ValueError("kernel names must match the sources and must not be 'cublas'")
    check_only = bool(options.get("check_only", False))
    indices = options.get("shapes")
    selected = [workloads[i] for i in indices] if indices is not None else workloads

    # Keep cuBLAS's fp16 reductions in full precision: it is the reference.
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
    if cfg.flush_l2:
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size or (64 << 20)
        flush = torch.empty(2 * l2_bytes, dtype=torch.uint8, device="cuda")

    results = []
    for workload in selected:
        M, N, K = (workload["axes"][axis] for axis in ("M", "N", "K"))
        entry: dict = {"uuid": workload["uuid"], "M": M, "N": N, "K": K, "impls": {}}
        results.append(entry)

        inputs = []
        for seed in cfg.seeds:
            A, B = make_inputs(definition, workload, seed)
            inputs.append((A, B, reference(A, B)))
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

        outputs = {}
        for name, fn in list(fns.items()):
            info = entry["impls"][name]
            out = make_output(definition, workload)
            outputs[name] = out
            try:
                info["check"] = run_checks(fn, inputs, out, cfg.rtol, cfg.atol)
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
        warmup, repeat, trials = cfg.warmup, cfg.repeat, cfg.trials
        slow = [
            n
            for n in fns
            if n != CUBLAS and entry["impls"][n]["check"]["call_ms"] >= cfg.slow_call_ms
        ]
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

        run_for(fns[CUBLAS], (A, B, outputs[CUBLAS]), cfg.ramp_s, flush)
        for name in fast:
            entry["impls"][name]["timing"] = "interleaved"
        for trial in range(trials):
            shift = trial % len(fast)
            for name in fast[shift:] + fast[:shift]:
                if name in fns:
                    timed(name, cfg.settle_s)
            entry["telemetry"].append(sample_telemetry())

        flops = 2.0 * M * N * K
        for name in list(fns):
            info = entry["impls"][name]
            after = check_output(outputs[name], ref, cfg.rtol, cfg.atol)
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
        "definition": definition["name"],
        "check_only": check_only,
        "names": names,
        "shapes": results,
        "telemetry_before": telemetry_before,
        "telemetry_after": sample_telemetry(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Client side: runs on your machine and sends one KCoral request.
# ═════════════════════════════════════════════════════════════════════════════

CLIENT_ERRORS = (KCoralError, TransportError, ProtocolError)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "kernels", nargs="+", type=Path, help="kernel files defining build(M, N, K)"
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("KCORAL_URL", "http://127.0.0.1:8000"),
        help="KCoral server URL (default: $KCORAL_URL or http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--shapes", help="comma-separated row indices of workload.jsonl (default: all)"
    )
    parser.add_argument("--check-only", action="store_true", help="correctness only, no timing")
    parser.add_argument("--timeout", type=float, default=900, help="request timeout in seconds")
    return parser.parse_args()


def kernel_names(paths: list[Path]) -> list[str]:
    names = [path.stem for path in paths]
    if CUBLAS in names or len(set(names)) != len(names):
        raise SystemExit(f"kernel file names must be unique and must not be '{CUBLAS}': {names}")
    return names


def build_program(
    definition: bytes, workloads: bytes, options: dict, sources: list[bytes]
) -> Program:
    program = Program()
    # This very file becomes the server-side module; only evaluate() is called there.
    harness = program.upload(kind="module", source=Path(__file__).read_text())
    evaluate = program.get_function(module=harness, name="evaluate")
    args = [
        program.upload(kind="bytes", value=definition),
        program.upload(kind="bytes", value=workloads),
        program.upload(kind="bytes", value=json.dumps(options).encode()),
        *(program.upload(kind="bytes", value=source) for source in sources),
    ]
    report = program.run(fn=evaluate, args=args)
    program.return_(key="report", value=report)
    return program


def summarize(report: dict) -> tuple[list[str], dict, bool]:
    """Render the per-shape table and aggregate scores; return (lines, summary, all_passed)."""
    names = [*report["names"], CUBLAS]
    lines = []
    header = f"{'shape':<28} {'impl':<16} {'status':<6} {'median ms':>10} {'spread':>7} {'TF/s':>7}"
    header += f" {'x initial':>10} {'x cuBLAS':>9}"
    lines += [header, "-" * len(header)]
    totals = {name: 0.0 for name in names}
    passed_all = {name: True for name in names}
    for shape in report["shapes"]:
        label = shape["uuid"]
        impls = shape["impls"]
        initial_ms = impls.get(INITIAL, {}).get("median_ms")
        cublas_ms = impls.get(CUBLAS, {}).get("median_ms")
        for name in names:
            info = impls.get(name, {})
            status = info.get("status", "?")
            passed_all[name] &= status == "PASS"
            ms = info.get("median_ms")
            row = f"{label:<28} {name:<16} {status:<6}"
            label = ""
            if ms is None:
                lines.append(row)
                continue
            totals[name] += ms
            vs_initial = f"{initial_ms / ms:.2f}" if initial_ms else "-"
            vs_cublas = f"{cublas_ms / ms:.3f}" if cublas_ms else "-"
            lines.append(
                f"{row} {ms:>10.4f} {info['spread']:>7.1%}"
                f" {info['tflops']:>7.1f} {vs_initial:>10} {vs_cublas:>9}"
            )
    all_passed = all(passed_all[name] for name in report["names"])

    summary: dict = {}
    if not report["check_only"]:
        lines += ["", "Aggregate over all shapes (sum of median ms):"]
        for name in names:
            if not passed_all[name]:
                lines.append(f"  {name:<16} not scored (failed or errored on some shape)")
                continue
            entry = {"total_ms": round(totals[name], 4)}
            if passed_all.get(INITIAL) and INITIAL in totals:
                entry["speedup_vs_initial"] = round(totals[INITIAL] / totals[name], 3)
            entry["ratio_vs_cublas"] = round(totals[CUBLAS] / totals[name], 4)
            summary[name] = entry
            extra = "".join(f"  {key}={value}" for key, value in entry.items() if key != "total_ms")
            lines.append(f"  {name:<16} total={entry['total_ms']:.4f} ms{extra}")
    return lines, summary, all_passed


def stability_warnings(report: dict) -> list[str]:
    cfg = BenchConfig()
    warnings = []
    for shape in report["shapes"]:
        for name, info in shape["impls"].items():
            spread = info.get("spread")
            if spread is not None and spread > cfg.max_spread:
                warnings.append(
                    f"{shape['uuid']}: {name} varied {spread:.1%} across trials "
                    f"(limit {cfg.max_spread:.0%}); the measurement was disturbed"
                )
    samples = [s for shape in report["shapes"] for s in shape.get("telemetry", [])[1:]]
    freqs = [(s.get("cur_freq_mhz"), s.get("max_freq_mhz")) for s in samples]
    freqs = [(cur, top) for cur, top in freqs if cur and top]
    if freqs:
        low = min(cur / top for cur, top in freqs)
        if low < cfg.min_clock_fraction:
            warnings.append(
                f"GPU clock sampled after timing fell to {low:.0%} of its maximum; "
                "the governor or thermal throttling lowered it (see README: stable clocks)"
            )
    return warnings


def telemetry_line(report: dict) -> str:
    samples = [s for shape in report["shapes"] for s in shape.get("telemetry", [])]
    samples += [report["telemetry_before"], report["telemetry_after"]]
    parts = []
    for label, prefix in (("GPU clock", ""), ("memory clock", "mem_")):
        cur, top, gov = f"{prefix}cur_freq_mhz", f"{prefix}max_freq_mhz", f"{prefix}governor"
        freqs = [s[cur] for s in samples if s.get(cur)]
        peak = next((s[top] for s in samples if s.get(top)), 0)
        governor = next((s[gov] for s in samples if s.get(gov)), "")
        if freqs:
            low, high = min(freqs), max(freqs)
            parts.append(f"{label} {low:.0f}-{high:.0f} MHz (max {peak:.0f}, {governor})")
    temps = [s["temp_c"] for s in samples if s.get("temp_c") is not None]
    if temps:
        parts.append(f"GPU temp {min(temps):.1f}-{max(temps):.1f} C")
    return "; ".join(parts) if parts else "GPU clock/temperature: unavailable on this server"


def print_failures(report: dict) -> None:
    for shape in report["shapes"]:
        for name, info in shape["impls"].items():
            if info.get("status") == "PASS":
                continue
            label = f"{name} @ {shape['M']}x{shape['N']}x{shape['K']}"
            if "error" in info:
                print(f"\n[{info.get('status')}] {label}:\n{info['error']}")
            for key in ("check", "check_after_timing"):
                runs = info.get(key, {}).get("runs") if key == "check" else [info.get(key)]
                for run in runs or []:
                    if run and not run["passed"]:
                        print(f"\n[FAIL] {label} ({key}): {json.dumps(run)}")


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parent  # __file__ exists only on the client side
    for path in args.kernels:
        if not path.is_file():
            raise SystemExit(f"kernel file not found: {path}")
    names = kernel_names(args.kernels)
    indices = [int(i) for i in args.shapes.split(",")] if args.shapes else None
    options = {
        "names": names,
        "shapes": indices,
        "check_only": args.check_only,
    }
    sources = [p.read_bytes() for p in args.kernels]
    definition = (here / "definition.json").read_bytes()
    workloads = (here / "workload.jsonl").read_bytes()
    program = build_program(definition, workloads, options, sources)

    with Client(args.url) as client:
        try:
            health = client.health()
        except CLIENT_ERRORS as exc:
            raise SystemExit(f"cannot reach KCoral server at {args.url}: {exc}") from exc
        arch = health.get("target", {}).get("arch")
        print(f"server {args.url}: {arch}, versions {health.get('versions')}")
        if arch != "sm_110a":
            print(f"warning: this example targets Thor (sm_110a); the server reports {arch}")
        started = time.time()
        try:
            result = client.execute(program, timeout_seconds=args.timeout)
        except CLIENT_ERRORS as exc:
            raise SystemExit(f"request to {args.url} failed: {exc}") from exc
    if not result.completed:
        print(result.stdout or "", end="")
        print(result.stderr or "", end="", file=sys.stderr)
        raise SystemExit(f"remote execution {result.status}: {result.error}")

    report = result["report"]
    lines, summary, all_passed = summarize(report)
    env = report["environment"]
    print(
        f"device {env['device']} ({env['arch']}, {env['sm_count']} SMs), torch {env['torch']}, "
        f"tvm {env['tvm']}; request {time.time() - started:.1f}s"
    )
    print("\n".join(lines))
    print(telemetry_line(report))
    warnings = [] if report["check_only"] else stability_warnings(report)
    for warning in warnings:
        print(f"warning: {warning}")
    print_failures(report)

    results = here / "results"
    results.mkdir(exist_ok=True)
    out = results / f"{time.strftime('%Y%m%d-%H%M%S')}-{'-'.join(names)}.json"
    record = {
        "kernels": {name: str(path) for name, path in zip(names, args.kernels)},
        "summary": summary,
        "warnings": warnings,
        "request_id": result.request_id,
        "report": report,
    }
    out.write_text(json.dumps(record, indent=2))
    print(f"results: {out.relative_to(here)}")
    print("ALL PASSED" if all_passed else "SOME KERNELS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
