"""Command-line entry point for KCoral services."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["server"]:
        from .service_cli import server_main

        server_main(argv[1:])
        return
    if argv[:1] == ["router"]:
        from .service_cli import exec_service

        exec_service("kcoral-router", argv[1:])
        return
    parser = argparse.ArgumentParser(prog="kcoral", description="KCoral remote execution services.")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("server", help="start and supervise a local execution service")
    commands.add_parser("router", help="route requests across compute nodes")
    parser.parse_args(argv)
    parser.print_help()


if __name__ == "__main__":
    main()
