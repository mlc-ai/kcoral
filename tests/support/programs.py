"""Construct uploaded Python calls for engine and protocol tests."""

from pathlib import Path

from kcoral.schemas import GetFunction, Ref, Run, Upload


def harness_source(name):
    if name in {"poison", "stale_cuda_error", "stale_cuda_error_unavailable"}:
        return f"def main():\n    return _test_runtime._{name}()\n"
    if name in {"compile_tirx", "benchmark"}:
        return f"from kcoral.builtins import {name} as main\n"
    if name.startswith("compile_"):
        module = name.split("_")[1]
    elif name in {"check_close", "assert_close"}:
        module = "core"
    else:
        module = "fake"
    directory = Path(__file__).parent
    source = (directory / "_common.py").read_text() + (directory / f"{module}.py").read_text()
    source = "\n".join(
        line
        for line in source.splitlines()
        if not line.startswith(("from __future__", "from ._common"))
    )
    return "from __future__ import annotations\n" + source + f"\nmain = {name}\n"


def python_call(id, source, args=None, *, cpu_only=False):
    module = f"{id}_harness_module"
    function = f"{id}_harness_function"
    return [
        Upload(module, "module", source=source),
        GetFunction(function, Ref(module), "main", cpu_only=cpu_only),
        Run(id, Ref(function), args or []),
    ]


def python_instructions(id, source, args=None, *, cpu_only=False):
    module, function, _call = python_call(id, source, args, cpu_only=cpu_only)
    return [
        {"op": "upload", "id": module.id, "kind": "module", "source": module.source},
        {
            "op": "get_function",
            "id": function.id,
            "module": {"$ref": module.id},
            "name": "main",
            "cpu_only": function.cpu_only,
        },
        {"op": "run", "id": id, "fn": {"$ref": function.id}, "args": args or []},
    ]


def harness_call(id, name, args=None):
    return python_call(
        id, harness_source(name), args, cpu_only=name in {"cpu_sleep", "compile_cuda_binary"}
    )


def harness_instructions(id, name, args=None):
    return python_instructions(
        id, harness_source(name), args, cpu_only=name in {"cpu_sleep", "compile_cuda_binary"}
    )


def harness_function(program, name, id):
    module = program.upload(id=f"{id}_harness_module", kind="module", source=harness_source(name))
    return program.get_function(
        id=f"{id}_harness_function",
        module=module,
        name="main",
        cpu_only=name in {"cpu_sleep", "compile_cuda_binary"},
    )
