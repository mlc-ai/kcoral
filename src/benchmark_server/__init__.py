"""Benchmark Server — a stateless remote execution engine for GPU kernels.

A request is a program (an instruction sequence). See ``design.md``.
"""

from .app import create_app
from .config import ServerConfig
from .schemas import Program, parse_program

__all__ = ["create_app", "ServerConfig", "Program", "parse_program"]
__version__ = "0.1.0"
