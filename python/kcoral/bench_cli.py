"""Run a checkout's pinned TIRx benchmark adapter through KCoral."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from ._tool_inputs import pack_inputs
from .client import Program
from .tool_cli import execute, require_completed
from .tool_cli import parse_args as parse_tool_args


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
    rows = [dict(row) for row in rows]
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


def build_request(
    adapter, source, task, overrides, entry, candidate, *, inputs, environment, fetch
):
    keys, tensors = adapter.blobs([entry])
    program = Program()
    module = program.upload(id="bundle", kind="module", source=source)
    runner = program.get_function(id="runner", module=module, name="execute")
    files = program.upload(
        id="files", kind="module", source=Path(__file__).with_name("_tool_worker.py").read_text()
    )
    unpack = program.get_function(id="unpack", module=files, name="unpack_inputs")
    collect = program.get_function(id="collect", module=files, name="collect_files")
    archive = program.upload(id="inputs", kind="bytes", value=inputs)
    lowered = (
        None if candidate is None else program.upload(id="lowered", kind="bytes", value=candidate)
    )
    handles = [
        program.upload(id=f"blob{i}", kind="tensor", value=tensor)
        for i, tensor in enumerate(tensors)
    ]
    outcome = program.run(
        id="run",
        fn=runner,
        args=[
            task,
            overrides,
            entry,
            keys,
            lowered,
            archive,
            environment,
            fetch,
            unpack,
            collect,
            *handles,
        ],
    )
    program.return_(key="outcome", value=outcome)
    if fetch:
        program.return_folder(key="artifacts", path="outputs")
    return program


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="kcoral run bench [KCoral options] --",
        allow_abbrev=False,
        description="Benchmark arguments (place these after '--').",
    )
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
    # Native help, like outer help, must work without a configured server.
    if "--" in argv and any(flag in argv[argv.index("--") + 1 :] for flag in ("-h", "--help")):
        parser.parse_args(["--help"])
    args, forwarded = parse_tool_args("bench", argv, epilog=parser.format_help())
    parser.parse_args(forwarded, namespace=args)
    if args.warmup is not None and args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat is not None and args.repeat < 1:
        parser.error("--repeat must be positive")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv):
    args = parse_args(argv)
    summary = {
        "workload": args.workload,
        "version": args.version,
        "completed": False,
        "passed": False,
        "results": [],
        "workloads": [],
        "error": None,
    }
    output_created = False
    code = 1
    try:
        if args.out is not None and os.path.lexists(args.out):
            raise ValueError(f"output already exists; choose a new --out directory: {args.out}")
        inputs = pack_inputs(args.send)
        adapter = load_adapter(args.repo)
        key = adapter.workload_key(args.workload)
        summary["workload"] = key
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

        if args.out is not None:
            args.out.mkdir(parents=True)
            output_created = True
        seen = None
        for index, entry in enumerate(workloads, start=1):
            program = build_request(
                adapter,
                source,
                task,
                overrides,
                entry,
                candidate,
                inputs=inputs,
                environment=args.env,
                fetch=args.fetch,
            )
            print(f"benchmark server: {args.url} (workload {index}/{len(workloads)})", flush=True)
            result = execute(args, program)
            if "outcome" in result.results:
                outcome = {"index": index, **result["outcome"]}
                summary["workloads"].append(outcome)
                summary["results"].extend(outcome["rows"])
                if args.out is not None:
                    directory = args.out / "workloads" / f"{index:04d}"
                    directory.mkdir(parents=True)
                    write_json(directory / "result.json", outcome)
                    if "artifacts" in result.results:
                        result["artifacts"].save(directory / "files")
                if outcome["worker"] is not None and outcome["worker"] != seen:
                    print(f"worker: {json.dumps(outcome['worker'])}")
                    seen = outcome["worker"]
                if outcome["missing"]:
                    print(
                        f"kcoral: missing artifacts: {', '.join(outcome['missing'])}",
                        file=sys.stderr,
                    )
                if outcome["error"]:
                    raise RuntimeError(outcome["error"])
            require_completed(result)
            if "outcome" not in result.results:
                raise RuntimeError("benchmark response has no outcome")
        summarize(decode_rows(summary["results"]), "")
        summary["completed"] = True
        summary["passed"] = not (
            any(row.get("passed") is False for row in summary["results"])
            or any(item["missing"] for item in summary["workloads"])
        )
        code = int(not summary["passed"])
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"kcoral run bench: {summary['error']}", file=sys.stderr)
    finally:
        if output_created:
            try:
                write_json(args.out / "summary.json", summary)
                print(f"kcoral: saved benchmark results to {args.out}", file=sys.stderr)
            except (OSError, TypeError, ValueError) as exc:
                print(f"kcoral run bench: cannot save summary: {exc}", file=sys.stderr)
                code = 1
    return code
