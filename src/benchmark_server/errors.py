"""Shared exception types.

The distinction that matters: a *submitted-code* fault vs an *engine* fault.
Instructions raise :class:`ExecutionError` when the uploaded code/kernel is at
fault; the engine reports that as a ``FAILED`` instruction. Anything else is an
engine fault.
"""

from __future__ import annotations


class ValidationError(ValueError):
    """Malformed or semantically invalid request (maps to HTTP 400)."""


class ExecutionError(Exception):
    """A failure caused by submitted code -> a ``FAILED`` instruction result.

    ``kind`` names the stage: ``parse``, ``compile``, ``runtime``, or
    ``correctness`` (an assertion the kernel failed).
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


class EngineError(Exception):
    """An engine/infrastructure fault (a server bug, OOM, driver fault)."""
