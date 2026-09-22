"""Self-contained worker-side correctness checks and CUPTI timing."""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
import traceback
from functools import partial
from pathlib import Path


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


def execute(
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
