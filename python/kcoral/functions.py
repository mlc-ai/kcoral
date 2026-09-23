"""Source-based remote functions built on the existing execution protocol."""

from __future__ import annotations

import ast
import builtins
import inspect
import json
import symtable
import textwrap
from collections.abc import Callable
from functools import update_wrapper
from typing import Any, Generic, ParamSpec, TypeVar

import numpy as np

from .client import Client, Program, ProgramResult, ProtocolError

_Parameters = ParamSpec("_Parameters")
_ReturnType = TypeVar("_ReturnType")


class RemoteExecutionError(RuntimeError):
    """A decorated function's program failed on the server.

    :param result: Failed execution outcome, retained as ``result`` with the
        request identifier, error details, remote traceback and captured output.

    HTTP and transport failures continue to use the ordinary client exceptions.
    """

    def __init__(self, result: ProgramResult) -> None:
        self.result = result
        error = result.error or {}
        super().__init__(
            f"Remote execution {result.request_id} failed: "
            f"{error.get('kind', 'unknown')}: {error.get('message', 'no error details')}"
        )


class RemoteFunction(Generic[_Parameters, _ReturnType]):
    """A Python function with explicit remote execution methods.

    Construct with :meth:`Client.function`.
    Calling the decorated function normally still executes the original locally.
    Each remote invocation builds a self-contained program; no remote state
    survives between invocations. The function's source is captured at decoration.
    """

    def __init__(
        self,
        fn: Callable[_Parameters, _ReturnType],
        *,
        client: Client,
        timeout: float | None = None,
        output_limit_bytes: int | None = None,
        cpu_only: bool = False,
    ) -> None:
        self._source, self._entry = _function_source(fn)
        self._signature = inspect.signature(fn)
        self._fn = fn
        self._client = client
        self._timeout = timeout
        self._output_limit_bytes = output_limit_bytes
        if not isinstance(cpu_only, bool):
            raise TypeError("cpu_only must be a bool")
        self._cpu_only = cpu_only
        update_wrapper(self, fn, updated=())

    def __call__(self, *args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _ReturnType:
        """Call the original function locally, without contacting the server."""
        return self._fn(*args, **kwargs)

    def build_program(self, *args: _Parameters.args, **kwargs: _Parameters.kwargs) -> Program:
        """Bind arguments and build a program without contacting the server.

        :param args: Positional function arguments.
        :param kwargs: Keyword function arguments.
        :returns: An ordinary :class:`Program` returning the key ``output``.
        :raises TypeError: If binding fails or an argument type is unsupported.
        :raises ValueError: If an argument cannot be serialized.

        Defaults are bound locally and sent explicitly. Arguments may be JSON
        values, bytes-like objects, NumPy arrays, or DLPack-compatible tensors.
        Binary values must be whole arguments, not nested inside JSON containers.
        No pickle deserialization is used.
        """
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        program = Program()
        module = program.upload(id="module", kind="module", source=self._source)
        fn = program.get_function(
            id="function", module=module, name=self._entry, cpu_only=self._cpu_only
        )
        values = []
        literals = []
        for i, value in enumerate((*bound.args, *bound.kwargs.values())):
            if isinstance(value, (bytes, bytearray, memoryview)):
                values.append(program.upload(id=f"arg_{i}", kind="bytes", value=value))
                literals.append(False)
            elif isinstance(value, np.ndarray) or hasattr(value, "__dlpack__"):
                values.append(program.upload(id=f"arg_{i}", kind="tensor", value=value))
                literals.append(False)
            else:
                _validate_literal(value)
                # Wrapping prevents a literal {"$ref": ...} being read as a handle.
                # Round-tripping also snapshots mutable containers for later use.
                values.append({"value": json.loads(json.dumps(value, allow_nan=False))})
                literals.append(True)
        output = program.run(
            id="output",
            fn=fn,
            args=[len(bound.args), list(bound.kwargs), literals, *values],
        )
        program.return_(key="output", value=output)
        return program

    def execute(self, *args: _Parameters.args, **kwargs: _Parameters.kwargs) -> ProgramResult:
        """Run remotely and return the full execution outcome, including logs.

        :param args: Positional function arguments.
        :param kwargs: Keyword function arguments.
        :returns: A :class:`ProgramResult`; instruction failures remain data.

        Uses the same cache negotiation and error handling as ``Client.execute``.
        Reuses the client supplied by ``Client.function`` without closing it.
        """
        program = self.build_program(*args, **kwargs)
        return self._client.execute(
            program,
            timeout_seconds=self._timeout,
            output_limit_bytes=self._output_limit_bytes,
        )

    def remote(self, *args: _Parameters.args, **kwargs: _Parameters.kwargs) -> Any:
        """Run remotely and return the decoded function value.

        :param args: Positional function arguments.
        :param kwargs: Keyword function arguments.
        :returns: The decoded value; tensors become local NumPy arrays.
        :raises RemoteExecutionError: If the program failed. Its ``result``
            preserves the error, traceback, request identifier and captured output.

        Request and transport exceptions propagate unchanged. Use ``execute``
        to inspect successful execution metadata and captured output as well.
        """
        result = self.execute(*args, **kwargs)
        if not result.completed:
            raise RemoteExecutionError(result)
        if "output" not in result.results:
            raise ProtocolError("remote function response is missing its output")
        return result.results["output"]


def _function_source(fn: Callable) -> tuple[str, str]:
    if (
        not inspect.isfunction(fn)
        or inspect.iscoroutinefunction(fn)
        or inspect.isgeneratorfunction(fn)
        or inspect.isasyncgenfunction(fn)
    ):
        raise TypeError("remote functions must be synchronous Python functions, not generators")
    if fn.__closure__:
        raise ValueError("remote functions cannot capture variables; pass them as arguments")
    if hasattr(fn, "__wrapped__"):
        raise ValueError("remote functions cannot have other decorators")
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError) as exc:
        raise ValueError(
            "remote functions require Python source in a file; use Program otherwise"
        ) from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("remote functions require a regular def statement, not a lambda")
    node = tree.body[0]
    if len(node.decorator_list) > 1:
        raise ValueError("remote functions cannot have other decorators")
    node.decorator_list = []
    # Default expressions and annotations may depend on local objects. Defaults
    # are already bound on the client; annotations play no role in execution.
    node.args.defaults = []
    node.args.kw_defaults = [None] * len(node.args.kwonlyargs)
    node.returns = None
    for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        arg.annotation = None
    for arg in (node.args.vararg, node.args.kwarg):
        if arg is not None:
            arg.annotation = None
    source = "from __future__ import annotations\n\n" + ast.unparse(tree) + "\n"
    table = symtable.symtable(source, "<remote function>", "exec")
    pending = list(table.get_children())
    unsupported = set()
    while pending:
        scope = pending.pop()
        pending.extend(scope.get_children())
        for symbol in scope.get_symbols():
            if symbol.is_global() and (symbol.is_referenced() or symbol.is_declared_global()):
                name = symbol.get_name()
                if name == fn.__name__ and not symbol.is_declared_global():
                    continue
                if symbol.is_declared_global() or not hasattr(builtins, name):
                    unsupported.add(name)
                elif name in fn.__globals__ and fn.__globals__[name] is not getattr(builtins, name):
                    unsupported.add(name)
    if unsupported:
        raise ValueError(
            f"remote function references external globals: {', '.join(sorted(unsupported))}; "
            "import dependencies inside the function or pass values as arguments"
        )
    entry = f"_kcoral_invoke_{fn.__name__}"
    source += f"""
def {entry}(nargs, names, literals, *values):
    import builtins as _kcoral_builtins
    values = [value["value"] if literal else value
              for literal, value in _kcoral_builtins.zip(literals, values)]
    return _kcoral_builtins.globals()[{fn.__name__!r}](
        *values[:nargs], **_kcoral_builtins.dict(_kcoral_builtins.zip(names, values[nargs:]))
    )
"""
    return source, entry


def _validate_literal(value: Any, active: set[int] | None = None) -> None:
    if value is None or type(value) in (bool, int, float, str):
        return
    if type(value) not in (list, dict):
        raise TypeError(
            f"unsupported argument type {type(value).__name__}; use JSON values or pass "
            "bytes and tensors as whole arguments"
        )
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError("remote function arguments cannot contain circular containers")
    active.add(id(value))
    try:
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("remote function dictionary arguments require string keys")
            children = value.values()
        else:
            children = value
        for child in children:
            _validate_literal(child, active)
    finally:
        active.remove(id(value))
