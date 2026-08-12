"""Self-contained callable loading uploaded to the benchmark server."""

from __future__ import annotations

from typing import Any


def load_callable(
    normalized: dict[str, Any],
    mode: str,
    prepared_callable: Any = None,
):
    """Validate and return a callable loaded by the instruction protocol."""

    if not isinstance(normalized, dict):
        raise TypeError("normalized evaluation must be an object")
    if mode == "reference":
        if not callable(prepared_callable):
            raise ValueError("the reference implementation did not load as a callable")
        return prepared_callable
    if mode != "solution":
        raise ValueError("mode must be 'reference' or 'solution'")

    solution = normalized["solution"]
    specification = solution["spec"]
    sources = solution["sources"]
    language = specification["language"]
    if len(sources) != 1:
        raise ValueError("remote evaluation supports exactly one solution source file")
    if specification["dependencies"]:
        raise ValueError("remote evaluation does not support solution dependencies")

    entry_path, entry_symbol = specification["entry_point"].split("::", 1)
    if sources[0]["path"] != entry_path:
        raise ValueError("the only solution source must be the entry source")

    if language in {"python", "triton"}:
        if not callable(prepared_callable):
            raise ValueError(f"{language} solution entry {entry_symbol!r} is not callable")
        return prepared_callable
    if language == "cuda":
        if specification["binding"] != "tvm-ffi":
            raise ValueError("CUDA solutions require the tvm-ffi binding")
        if not callable(prepared_callable):
            raise ValueError("CUDA solutions require a separately compiled callable")
        return prepared_callable
    raise ValueError(f"unsupported solution language: {language!r}")


__all__ = ["load_callable"]
