"""Comparison reports used in GPU integration tests."""

from __future__ import annotations

from typing import Any

from kcoral.errors import ExecutionError

from ._common import short, split_cfg


def check_close(actual: Any, expected: Any, *rest: Any) -> dict:
    import torch

    _, cfg = split_cfg(rest)
    rtol = float(cfg.get("rtol", 1e-2))
    atol = float(cfg.get("atol", 1e-3))
    try:
        torch.cuda.synchronize()
        a = actual.float()
        e = expected.float().to(a.device)  # a reference computed on the CPU compares as is
        diff = (a - e).abs()
        max_abs = float(diff.max())
        # max |a-e|/|e| over nonzero e; masking keeps it finite (JSON-safe).
        nz = e != 0
        max_rel = float((diff[nz] / e[nz].abs()).max()) if bool(nz.any()) else 0.0
        passed = bool(torch.allclose(a, e, rtol=rtol, atol=atol))
    except RuntimeError as exc:
        raise ExecutionError("runtime", short(exc)) from exc
    return {
        "passed": passed,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "rtol": rtol,
        "atol": atol,
    }


def assert_close(actual: Any, expected: Any, *rest: Any) -> dict:
    """Like ``check_close``, but a mismatch is a failure: it raises ``correctness``
    so the instruction FAILs and the rest of the program is skipped."""
    result = check_close(actual, expected, *rest)
    if not result["passed"]:
        raise ExecutionError(
            "correctness",
            f"outputs differ: max_abs_err={result['max_abs_err']}, "
            f"max_rel_err={result['max_rel_err']} exceed "
            f"atol={result['atol']}, rtol={result['rtol']}",
        )
    return result
