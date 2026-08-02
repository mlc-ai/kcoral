"""Benchmark Server - a stateless remote execution engine for GPU kernels.

A request is a program (an instruction sequence). See ``docs/protocol.md``.
"""

from .app import create_app
from .client import Client, Program, ProgramResult, Register
from .config import ServerConfig
from .schemas import parse_program

__all__ = [
    "Client",
    "Program",
    "ProgramResult",
    "Register",
    "ServerConfig",
    "create_app",
    "parse_program",
]
__version__ = "0.1.0"
