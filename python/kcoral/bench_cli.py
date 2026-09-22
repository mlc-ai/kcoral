"""Run standalone benchmark definitions through KCoral."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from ._tool_inputs import pack_inputs
from .client import Program
from .tool_cli import execute, require_completed
from .tool_cli import parse_args as parse_tool_args


def load_benchmark(workload, version, repo, warmup, repeat):
    directory = ((repo or Path.cwd()) / workload).absolute()
    manifest = directory / "bench.json"
    config = json.loads(manifest.read_text())
    if not isinstance(config, dict):
        raise ValueError("bench.json must be an object")
    unknown = config.keys() - {"cases", "warmup", "repeat", "atol", "rtol"}
    if unknown:
        raise ValueError(f"unknown bench.json settings: {', '.join(sorted(unknown))}")
    cases = config.pop("cases", None)
    if (
        not isinstance(cases, list)
        or not cases
        or not all(isinstance(case, dict) for case in cases)
    ):
        raise ValueError("bench.json cases must be a nonempty list of objects")
    # Reject non-finite numbers anywhere, including case data, before contacting a server.
    json.dumps(cases, allow_nan=False)
    config = {"warmup": 3, "repeat": 50, "atol": 1e-5, "rtol": 1e-5, **config}
    if warmup is not None:
        config["warmup"] = warmup
    if repeat is not None:
        config["repeat"] = repeat
    for key, minimum in (("warmup", 0), ("repeat", 1)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in ("atol", "rtol"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number")
    if not (directory / "bench.py").is_file():
        raise FileNotFoundError(f"benchmark definition not found: {directory / 'bench.py'}")
    candidate = None
    if version != "baseline":
        if Path(version).name != version or version in {".", ".."} or "\\" in version:
            raise ValueError("VERSION must be a filename or name within the benchmark directory")
        candidate = version if version.endswith(".py") else version + ".py"
        if not (directory / candidate).is_file():
            raise FileNotFoundError(f"candidate not found: {directory / candidate}")
    return directory, cases, candidate, config


def summarize(rows):
    print("case | status | baseline_ms | kernel_ms | speedup")
    for index, row in enumerate(rows, start=1):
        values = [row[key] for key in ("baseline_ms", "kernel_ms", "speedup")]
        timing = " | ".join("-" if value is None else f"{value:.6g}" for value in values)
        status = "PASS" if row["passed"] else "FAIL"
        print(f"{row['id'] if row['id'] is not None else index} | {status} | {timing}")
        if row["message"]:
            print(row["message"])
    passed = sum(row["passed"] for row in rows)
    print(f"{passed}/{len(rows)} cases passed")


def build_request(directory, candidate, case, config, *, inputs, environment, fetch):
    program = Program()
    module = program.upload(
        id="bundle", kind="module", source=Path(__file__).with_name("_bench_worker.py").read_text()
    )
    runner = program.get_function(id="runner", module=module, name="execute")
    files = program.upload(
        id="files", kind="module", source=Path(__file__).with_name("_tool_worker.py").read_text()
    )
    unpack = program.get_function(id="unpack", module=files, name="unpack_inputs")
    collect = program.get_function(id="collect", module=files, name="collect_files")
    archive = program.upload(id="inputs", kind="bytes", value=inputs)
    outcome = program.run(
        id="run",
        fn=runner,
        args=[directory, candidate, case, config, archive, environment, fetch, unpack, collect],
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
    parser.add_argument(
        "workload", help="local benchmark directory containing bench.json and bench.py"
    )
    parser.add_argument(
        "version",
        nargs="?",
        default="baseline",
        help="candidate file or stem, e.g. v0 for v0.py; default: baseline",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        help="local root for relative WORKLOAD paths; default: current directory",
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
        benchmark_dir, workloads, candidate, config = load_benchmark(
            args.workload, args.version, args.repo, args.warmup, args.repeat
        )
        inputs = pack_inputs([benchmark_dir, *args.send])
        summary["config"] = config
        if args.out is not None:
            args.out.mkdir(parents=True)
            output_created = True
        seen = None
        for index, entry in enumerate(workloads, start=1):
            program = build_request(
                benchmark_dir.resolve().name,
                candidate,
                entry,
                config,
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
        summarize(summary["results"])
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
