"""Shared exception types.

Instructions raise :class:`ExecutionError` for a submitted-code fault, reported as
a ``FAILED`` instruction; anything else that escapes one is reported ``engine``.
"""

from __future__ import annotations


class ValidationError(ValueError):
    """Malformed or semantically invalid request (maps to HTTP 400)."""


class ExecutionError(Exception):
    """An instruction failure -> a ``FAILED`` instruction result.

    ``kind`` names the cause: ``parse``, ``compile``, ``runtime``,
    ``gpu_access``, or ``correctness`` (an assertion the kernel failed) — all
    submitted-code faults — or ``unavailable`` when the instruction needs an
    optional server dependency (e.g. tvm) that is not installed.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


class GPUAccessViolation(ExecutionError):
    """A ``gpu="none"`` instruction called the CUDA runtime or driver."""

    def __init__(
        self,
        cuda_call: str,
        thread_id: int,
        detected_at_ns: int,
        location: str,
        call_traceback: str,
    ) -> None:
        super().__init__(
            "gpu_access",
            f"instruction declared gpu='none' but called {cuda_call}",
        )
        self.cuda_call = cuda_call
        self.thread_id = thread_id
        self.detected_at_ns = detected_at_ns
        self.location = location
        self.call_traceback = call_traceback
