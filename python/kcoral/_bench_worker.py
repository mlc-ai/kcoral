#!/usr/bin/env python3
"""Worker-side benchmark bundle, derived from TIRx-kernel-agent.

The adapter appends ``SOURCES = {...}``: the pinned flashinfer-bench-evolve harness
and one task's modules and definition, verbatim. Inside the worker everything stays in memory —
the harness is imported through an in-memory finder, the workload blobs arrive as
kcoral tensor uploads and are served to the harness's ``load_safetensor``, and the
candidate ``lowered.py`` is exec'd as a module. ``main("init", ...)`` runs once per
request; ``main("run", i)`` scores workload ``i`` through the task's own ``run_suite``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import linecache
import math
import sys
import types
from dataclasses import replace
from functools import cache

SOURCES: dict[str, str] = {}  # package-relative path -> source or JSON, appended by the adapter
_STATE: dict = {}


class _BundleImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, sources):
        self._modules = {}
        for path, source in sources.items():
            if not path.endswith(".py"):
                continue
            parts = path[: -len(".py")].split("/")
            is_package = parts[-1] == "__init__"
            self._modules[".".join(parts[:-1] if is_package else parts)] = (
                path,
                source,
                is_package,
            )

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._modules:
            return None
        path, _, is_package = self._modules[fullname]
        return importlib.machinery.ModuleSpec(
            fullname, self, origin=f"<bundle:{path}>", is_package=is_package
        )

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, source, _ = self._modules[module.__name__]
        module.__file__ = "/bundle/" + path  # never read; keeps Path(__file__) arithmetic alive
        _exec(source, f"<bundle:{path}>", module.__dict__)


def _exec(source: str, filename: str, namespace: dict) -> None:
    _STATE.setdefault("linecache", {}).setdefault(filename, linecache.cache.get(filename))
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    # dont_inherit: this file's `from __future__ import annotations` must not leak into
    # the compiled source — tirx-lite reads live annotation objects at decoration time.
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)


def _sanitize(value):
    """kcoral's JSON refuses NaN/Infinity; carry them as strings."""

    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: _sanitize(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(child) for child in value]
    return value


def _init(task, overrides, workloads, blob_keys, lowered_source, *blob_tensors):
    _STATE["saved_modules"] = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] == "flashinfer_bench_evolve" or name == "lowered"
    }
    for name in _STATE["saved_modules"]:
        del sys.modules[name]  # a reused worker must not keep an earlier bundle's modules
    _STATE["importer"] = _BundleImporter(SOURCES)
    sys.meta_path.insert(0, _STATE["importer"])
    common = importlib.import_module("flashinfer_bench_evolve.benchmark_common")
    blobs = {
        (key["path"], key["tensor_key"]): tensor for key, tensor in zip(blob_keys, blob_tensors)
    }
    common.load_safetensor = lambda spec, device: (
        blobs[(spec["path"], spec["tensor_key"])].to(device).clone()
    )

    @cache
    def load_task_reference(task_name, entrypoint="run"):
        # Keep the pinned oracle in memory, just like the uploaded workload blobs.
        path = f"flashinfer_bench_evolve/tasks/{task_name}/definition.json"
        namespace = {}
        _exec(json.loads(SOURCES[path])["reference"], f"<bundle:{path}>", namespace)
        return namespace[entrypoint]

    common.load_task_reference = load_task_reference
    module = importlib.import_module(
        f"flashinfer_bench_evolve.tasks.{task}.benchmark"
    )  # binds the patched name
    lowered = None
    if lowered_source is not None:
        lowered = sys.modules["lowered"] = types.ModuleType("lowered")
        _exec(lowered_source.decode(), "<lowered.py>", lowered.__dict__)
    _STATE.update(
        module=module,
        config=replace(module.default_config(), **overrides),
        lowered=lowered,
        workloads=workloads,
    )
    kernels = importlib.util.find_spec("tirx_kernels")
    versions = {}
    for name, distribution in (
        ("torch", "torch"),
        ("tvm", "apache-tvm"),
        ("flashinfer", "flashinfer-python"),
    ):
        try:
            versions[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"versions": versions, "tirx_kernels": kernels and kernels.origin}


def _run(index: int):
    module, config, lowered = _STATE["module"], _STATE["config"], _STATE["lowered"]
    workloads = [_STATE["workloads"][index]]
    if lowered is None:
        rows = module.run_suite(config, workloads=workloads)
    else:
        rows = module.run_suite(
            config,
            candidate_fn=module.tirx_run,
            candidate_prepare_fn=lambda *args: module.tirx_prepare(lowered, *args),
            workloads=workloads,
        )
    return _sanitize(rows)


def _cleanup():
    importer = _STATE.get("importer")
    if importer in sys.meta_path:
        sys.meta_path.remove(importer)
    if "saved_modules" in _STATE:
        for name in list(sys.modules):
            if name.split(".")[0] == "flashinfer_bench_evolve" or name == "lowered":
                del sys.modules[name]
        sys.modules.update(_STATE["saved_modules"])
    for filename, previous in _STATE.get("linecache", {}).items():
        if previous is None:
            linecache.cache.pop(filename, None)
        else:
            linecache.cache[filename] = previous
    _STATE.clear()


def main(command, *args):
    if command == "init":
        try:
            return _init(*args)
        except BaseException:
            _cleanup()
            raise
    if command == "run":
        try:
            return _run(*args)
        finally:
            _cleanup()
    raise ValueError(f"unknown driver command {command!r}")
