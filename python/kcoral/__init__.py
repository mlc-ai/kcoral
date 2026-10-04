"""KCoral - a stateless remote execution engine for GPU kernels.

A request is a program (an instruction sequence). See ``docs/client-guide/protocol.md``.
"""

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from kcoral.artifacts import ReturnedFile, ReturnedFolder
from kcoral.client import (
    Client,
    KCoralError,
    Program,
    ProgramResult,
    ProtocolError,
    Register,
    TransportError,
)
from kcoral.client.functions import RemoteExecutionError, RemoteFunction
from kcoral.config import ServerConfig
from kcoral.protocol import parse_program

if TYPE_CHECKING:
    from kcoral.server.app import create_app

__all__ = [
    "Client",
    "KCoralError",
    "Program",
    "ProgramResult",
    "ProtocolError",
    "Register",
    "RemoteExecutionError",
    "RemoteFunction",
    "ReturnedFile",
    "ReturnedFolder",
    "ServerConfig",
    "TransportError",
    "create_app",
    "parse_program",
]
__version__ = version("kcoral")


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from kcoral.server.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
