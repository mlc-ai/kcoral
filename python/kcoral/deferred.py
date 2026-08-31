"""Internal values that finish an instruction after the GPU lease is reacquired."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeferredGPUResult:
    """The GPU-finalization phase of work that began outside the GPU lease.

    A ``cpu_only`` builtin may do expensive host compilation after giving up the
    lease, then return this wrapper for the small driver/module-loading phase that
    still touches the GPU.  The engine resolves it only after reacquiring the
    lease, and attributes any failure to the original instruction.
    """

    finalize: Callable[[], Any]

    def resolve(self) -> Any:
        return self.finalize()
