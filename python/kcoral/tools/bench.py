"""Run standalone benchmark definitions through KCoral."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import traceback
from functools import partial
from pathlib import Path


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
    from ..client import Program

    program = Program()
    module = program.upload(id="bundle", kind="module", source=Path(__file__).read_text())
    runner = program.get_function(id="runner", module=module, name="run")
    files = program.upload(
        id="files", kind="module", source=Path(__file__).with_name("_worker.py").read_text()
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
    from ._common import parse_args as parse_tool_args

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
    args, forwarded = parse_tool_args("bench", argv, allow_out=True, epilog=parser.format_help())
    parser.parse_args(forwarded, namespace=args)
    if args.warmup is not None and args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat is not None and args.repeat < 1:
        parser.error("--repeat must be positive")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv):
    from ._common import execute, require_completed
    from ._inputs import pack_inputs

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


def _load(path, name):
    # Treat each entry point as a package so relative helper imports also work.
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _clone(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone(item, torch) for item in value)
    if isinstance(value, list):
        return [_clone(item, torch) for item in value]
    if isinstance(value, dict):
        return {key: _clone(item, torch) for key, item in value.items()}
    return copy.deepcopy(value)


def _measure(call, warmup, repeat):
    from kcoral.builtins import benchmark

    return benchmark(call, {"warmup": warmup, "repeat": repeat, "flush_l2": True})


def _callable(module, name):
    value = getattr(module, name, None)
    if not callable(value):
        raise ValueError(f"{Path(module.__file__).name} must define callable {name}()")
    return value


def _run(directory, candidate, case, config):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("bench requires PyTorch with CUDA on the worker")
    bench = _load(directory / "bench.py", "_kcoral_benchmark")
    reference_fn = _callable(bench, "reference")
    baseline_fn = getattr(bench, "baseline", reference_fn)
    if not callable(baseline_fn):
        raise ValueError("bench.py baseline must be callable")
    candidate_module = _load(directory / candidate, "_kcoral_candidate") if candidate else None
    inputs = _callable(bench, "make_inputs")(case)
    if not isinstance(inputs, (tuple, list)):
        raise ValueError("make_inputs(case) must return a tuple or list of positional arguments")
    with torch.no_grad():
        expected = _clone(reference_fn(*_clone(inputs, torch)), torch)
        if expected is None:
            raise ValueError("reference() must return the outputs to check, not None")
        baseline_inputs = _clone(inputs, torch)
        baseline = partial(baseline_fn, *baseline_inputs)
        if candidate_module is None:
            run = baseline
        else:
            candidate_inputs = _clone(inputs, torch)
            if hasattr(candidate_module, "prepare"):
                run = _callable(candidate_module, "prepare")(*candidate_inputs)
                if not callable(run):
                    raise ValueError("candidate prepare() must return a zero-argument callable")
            else:
                candidate_fn = _callable(candidate_module, "run")
                run = partial(candidate_fn, *candidate_inputs)
        row = {
            "id": case.get("id"),
            "case": case,
            "passed": True,
            "baseline_ms": None,
            "baseline_timing": None,
            "kernel_ms": None,
            "kernel_timing": None,
            "speedup": None,
            "message": None,
        }

        def check(call, label):
            actual = call()
            torch.cuda.synchronize()
            try:
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=config["atol"],
                    rtol=config["rtol"],
                    check_dtype=False,
                    check_device=False,
                    equal_nan=False,
                )
            except AssertionError as exc:
                row.update(passed=False, message=f"{label}: {exc}")
                return False
            return True

        # Repeated checks catch stateful outputs before and after timing.
        if not check(baseline, "baseline") or not check(baseline, "baseline repeat"):
            return row
        if candidate and (not check(run, "candidate") or not check(run, "candidate repeat")):
            return row
        row["baseline_timing"] = _measure(baseline, config["warmup"], config["repeat"])
        row["kernel_timing"] = (
            _measure(run, config["warmup"], config["repeat"])
            if candidate
            else row["baseline_timing"]
        )
        row["baseline_ms"] = row["baseline_timing"]["latency_ms_median"]
        row["kernel_ms"] = row["kernel_timing"]["latency_ms_median"]
        if not check(baseline, "baseline after timing") or not check(run, "candidate after timing"):
            return row
        row["speedup"] = row["baseline_ms"] / row["kernel_ms"]
        return row


def run(
    directory, candidate, case, config, archive, environment, fetch, unpack_inputs, collect_files
):
    """Run one case with request-local files and environment, then collect outputs."""
    workdir = Path("inputs").absolute()
    reports = Path("outputs").absolute()
    workdir.mkdir()
    reports.mkdir()
    original_cwd = Path.cwd()
    original_environment = dict(os.environ)
    original_path = list(sys.path)
    original_modules = dict(sys.modules)
    outcome = {"worker": None, "rows": [], "missing": [], "error": None}
    try:
        try:
            unpack_inputs(archive, workdir)
            os.chdir(workdir)
            directory = workdir / directory
            sys.path[:0] = [str(directory), str(workdir)]
            os.environ.update(environment)
            os.environ["KCORAL_DIR"] = str(workdir)
            os.environ["PATH"] = (
                str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
            )
            import torch

            outcome["worker"] = {"python": sys.version.split()[0], "torch": torch.__version__}
            if torch.cuda.is_available():
                outcome["worker"]["device"] = torch.cuda.get_device_name()
            outcome["rows"] = [_run(directory, candidate, case, config)]
        except Exception:
            outcome["error"] = traceback.format_exc()
        try:
            outcome["missing"] = collect_files(workdir, reports, fetch)
        except Exception:
            outcome["error"] = (outcome["error"] or "") + traceback.format_exc()
        return outcome
    finally:
        for name, module in list(sys.modules.items()):
            namespace = getattr(module, "__dict__", {})
            paths = [namespace.get("__file__"), *namespace.get("__path__", ())]
            if any(isinstance(path, str) and Path(path).is_relative_to(workdir) for path in paths):
                if name in original_modules:
                    sys.modules[name] = original_modules[name]
                else:
                    del sys.modules[name]
        sys.path[:] = original_path
        os.environ.clear()
        os.environ.update(original_environment)
        os.chdir(original_cwd)
