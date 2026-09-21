"""Run a checkout's pinned TIRx benchmark adapter through KCoral."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .client import Program
from .tool_cli import add_connection_args, execute, require_completed, validate_connection_args


def decode_rows(rows):
    # The wire protocol disallows non-finite JSON numbers. Restore only the
    # harness's numeric result columns before giving its summary function rows.
    columns = (
        "max_abs",
        "max_rel",
        "max_rms_ratio",
        "matched",
        "baseline_ms",
        "kernel_ms",
        "speedup",
    )
    for row in rows:
        for column in columns:
            value = row.get(column)
            if isinstance(value, str) and value in {"NaN", "Infinity", "-Infinity"}:
                row[column] = float(value)
    return rows


def load_adapter(repo):
    if repo is None:
        candidates = [Path.cwd(), *Path.cwd().parents]
    else:
        candidates = [repo.absolute()]
    for root in candidates:
        for path in (root / "kernel-evolution" / "bench_adapter.py", root / "bench_adapter.py"):
            if path.is_file():
                spec = importlib.util.spec_from_file_location("_kcoral_bench_adapter", path)
                adapter = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(adapter)
                return adapter
    raise FileNotFoundError(
        "cannot find kernel-evolution/bench_adapter.py; run inside a TIRx-kernel-agent "
        "checkout or pass --repo PATH (with its flashinfer-bench-evolve submodule initialized)"
    )


def build_request(adapter, source, task, overrides, entry, candidate):
    keys, tensors = adapter.blobs([entry])
    program = Program()
    module = program.upload(id="bundle", kind="module", source=source)
    bundle = program.get_function(id="main", module=module, name="main")
    lowered = (
        None if candidate is None else program.upload(id="lowered", kind="bytes", value=candidate)
    )
    handles = [
        program.upload(id=f"blob{i}", kind="tensor", value=tensor)
        for i, tensor in enumerate(tensors)
    ]
    init = program.run(
        id="init", fn=bundle, args=["init", task, overrides, [entry], keys, lowered, *handles]
    )
    program.return_(key="init", value=init)
    row = program.run(id="run", fn=bundle, args=["run", 0])
    program.return_(key="row", value=row)
    return program


def main(argv):
    parser = argparse.ArgumentParser(
        prog="kcoral bench",
        allow_abbrev=False,
        description="Benchmark a candidate using a checkout's pinned harness.",
    )
    add_connection_args(parser)
    parser.add_argument("workload", help="registered workload, e.g. kda/decode")
    parser.add_argument(
        "version",
        nargs="?",
        default="baseline",
        help="candidate directory, e.g. v0; default: baseline",
    )
    parser.add_argument(
        "--repo", type=Path, help="TIRx-kernel-agent checkout; default: find from current directory"
    )
    parser.add_argument("--warmup", type=int, help="override the workload's warmup count")
    parser.add_argument(
        "--repeat", type=int, help="override the workload's measured iteration count"
    )
    args = parser.parse_args(argv)
    validate_connection_args(parser, args)
    if args.warmup is not None and args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat is not None and args.repeat < 1:
        parser.error("--repeat must be positive")
    try:
        adapter = load_adapter(args.repo)
        key = adapter.workload_key(args.workload)
        task, warmup, repeat, shape_mode = adapter.PACKAGED[key]
        overrides, workloads, candidate = adapter.plan(
            task,
            adapter.KERNEL_EVOLUTION_ROOT / key,
            args.version,
            warmup=warmup if args.warmup is None else args.warmup,
            repeat=repeat if args.repeat is None else args.repeat,
            shape_mode=shape_mode,
        )
        if not workloads:
            raise ValueError("the benchmark adapter selected no workloads")
        source = Path(__file__).with_name("_bench_worker.py").read_text()
        source += "\nSOURCES = " + repr(adapter.harness_sources(task)) + "\n"
        from flashinfer_bench_evolve.benchmark_common import summarize

        rows, seen = [], None
        for index, entry in enumerate(workloads, start=1):
            program = build_request(adapter, source, task, overrides, entry, candidate)
            print(f"benchmark server: {args.url} (workload {index}/{len(workloads)})", flush=True)
            result = execute(args, program)
            require_completed(result)
            worker = result["init"]
            if worker != seen:
                print(f"worker: {json.dumps(worker)}")
                seen = worker
            rows.extend(decode_rows(result["row"]))
        summarize(rows, "")
        return int(any(row.get("passed") is False for row in rows))
    except Exception as exc:
        print(f"kcoral bench: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
