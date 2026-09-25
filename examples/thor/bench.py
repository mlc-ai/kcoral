"""Check and benchmark GEMM kernels on a remote Thor KCoral server.

Every kernel file passed on the command line, plus cuBLAS as the reference, is
checked and timed in one KCoral request, so all implementations share the same
GPU and the same thermal window. The task is defined by ``definition.json`` and
``workload.jsonl`` (FlashInfer Trace format); the protocol is implemented by
``remote_bench.py``, which runs on the server.

Examples::

    uv run python bench.py initial_kernel.py
    uv run python bench.py work/best.py work/new.py
    uv run python bench.py work/new.py --check-only --shapes 0
    uv run python bench.py --health
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from remote_bench import BenchConfig

from kcoral import Client, KCoralError, Program, ProtocolError, TransportError

HERE = Path(__file__).resolve().parent
DEFINITION = HERE / "definition.json"
WORKLOADS = HERE / "workload.jsonl"
HARNESS = HERE / "remote_bench.py"
CLIENT_ERRORS = (KCoralError, TransportError, ProtocolError)
CUBLAS = "cublas"
INITIAL = "initial_kernel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "kernels", nargs="*", type=Path, help="kernel files defining build(M, N, K)"
    )
    parser.add_argument(
        "--health", action="store_true", help="print the server's health report and exit"
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
    parser.add_argument(
        "--emit-source",
        action="store_true",
        help="also save each kernel's generated CUDA C next to the results",
    )
    parser.add_argument(
        "--out", type=Path, default=HERE / "results", help="directory for JSON results"
    )
    parser.add_argument("--timeout", type=float, default=900, help="request timeout in seconds")
    return parser.parse_args()


def kernel_names(paths: list[Path]) -> list[str]:
    names = [path.stem for path in paths]
    if len(set(names)) != len(names):  # fall back to paths when file names collide
        names = [path.with_suffix("").as_posix().replace("/", "_") for path in paths]
    if CUBLAS in names or len(set(names)) != len(names):
        raise SystemExit(f"kernel names must be unique and must not be '{CUBLAS}': {names}")
    return names


def build_program(
    definition: bytes, workloads: bytes, options: dict, sources: list[bytes]
) -> Program:
    program = Program()
    harness = program.upload(kind="module", source=HARNESS.read_text())
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
    if args.health:
        with Client(args.url) as client:
            try:
                print(json.dumps(client.health(), indent=2))
            except CLIENT_ERRORS as exc:
                raise SystemExit(f"cannot reach KCoral server at {args.url}: {exc}") from exc
        return 0
    if not args.kernels:
        raise SystemExit("give at least one kernel file (or --health)")
    for path in args.kernels:
        if not path.is_file():
            raise SystemExit(f"kernel file not found: {path}")
    names = kernel_names(args.kernels)
    indices = [int(i) for i in args.shapes.split(",")] if args.shapes else None
    options = {
        "names": names,
        "shapes": indices,
        "check_only": args.check_only,
        "emit_source": args.emit_source,
    }
    sources = [p.read_bytes() for p in args.kernels]
    program = build_program(DEFINITION.read_bytes(), WORKLOADS.read_bytes(), options, sources)

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

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{'-'.join(names)}"
    out = args.out / f"{stem}.json"
    for shape in report["shapes"]:
        for name, info in shape["impls"].items():
            source = info.pop("cuda_source", None)
            if source:
                path = args.out / f"{stem}-{name}-{shape['M']}x{shape['N']}x{shape['K']}.cu"
                path.write_text(source)
                print(f"generated CUDA: {path}")
    record = {
        "kernels": {name: str(path) for name, path in zip(names, args.kernels)},
        "summary": summary,
        "warnings": warnings,
        "request_id": result.request_id,
        "report": report,
    }
    out.write_text(json.dumps(record, indent=2))
    print(f"results: {out.relative_to(HERE) if out.is_relative_to(HERE) else out}")
    print("ALL PASSED" if all_passed else "SOME KERNELS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
