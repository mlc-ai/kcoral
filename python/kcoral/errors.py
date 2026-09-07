"""Shared exception types.

Instructions raise :class:`ExecutionError` for a submitted-code fault, reported as
a ``FAILED`` instruction; anything else that escapes one is reported ``engine``.
"""

from __future__ import annotations


class ValidationError(ValueError):
    """Malformed or semantically invalid request (maps to HTTP 400)."""


class ExecutionError(Exception):
    """An instruction failure -> a ``FAILED`` instruction result.

    ``kind`` names the cause: ``parse``, ``compile``, ``runtime``, ``gpu_access``
    (a ``cpu_only`` function called CUDA), or ``correctness`` (an assertion the
    kernel failed) — all submitted-code faults — or ``unavailable`` when the
    instruction needs an optional server dependency (e.g. tvm) that is not
    installed.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


class GPUAccessViolation(ExecutionError):
    """A ``cpu_only`` function entered the CUDA runtime or driver API."""

    def __init__(self, cuda_call: str, location: str, call_stack: str, detected_at_ns: int):
        super().__init__("gpu_access", f"cpu_only function called {cuda_call} at {location}")
        self.cuda_call = cuda_call
        self.location = location
        self.call_stack = call_stack
        self.detected_at_ns = detected_at_ns
