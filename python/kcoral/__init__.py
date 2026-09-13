"""KCoral - a stateless remote execution engine for GPU kernels.

A request is a program (an instruction sequence). See ``docs/client-guide/protocol.md``.
"""

from typing import TYPE_CHECKING, Any

from .artifacts import ReturnedFile, ReturnedFolder
from .client import (
    Client,
    KCoralError,
    Program,
    ProgramResult,
    ProtocolError,
    Register,
    TransportError,
)
from .config import ServerConfig
from .schemas import parse_program

if TYPE_CHECKING:
    from .app import create_app

__all__ = [
    "Client",
    "KCoralError",
    "Program",
    "ProgramResult",
    "ProtocolError",
    "Register",
    "ReturnedFile",
    "ReturnedFolder",
    "ServerConfig",
    "TransportError",
    "create_app",
    "parse_program",
]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
