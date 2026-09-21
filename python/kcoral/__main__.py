"""Command-line entry point for KCoral servers and routers."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:
        from .tool_cli import COMMANDS, main as tool_main

        parser = argparse.ArgumentParser(prog="kcoral run", allow_abbrev=False)
        commands = parser.add_subparsers(dest="tool", required=True)
        for name in COMMANDS:
            commands.add_parser(name, add_help=False, help=f"run {name} remotely")
        args = parser.parse_args(argv[1:2])
        raise SystemExit(tool_main(args.tool, argv[2:]))
    if argv[:1] == ["server"]:
        from .commands import server_main

        server_main(argv[1:])
        return
    if argv[:1] == ["router"]:
        from .commands import router_main

        router_main(argv[1:])
        return
    parser = argparse.ArgumentParser(prog="kcoral", description="KCoral remote execution.")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("server", help="start a standalone server or join a Router")
    commands.add_parser("router", help="route requests across compute nodes")
    commands.add_parser("run", help="run a tool on a remote server")
    parser.parse_args(argv)
    parser.print_help()


if __name__ == "__main__":
    main()
