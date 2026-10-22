"""Command-line entry point for KCoral services and remote tools."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:
        from kcoral.tools.cli import run_main

        raise SystemExit(run_main(argv[1:]))
    if argv[:1] == ["server"]:
        from kcoral.server.cli import server_main

        server_main(argv[1:])
        return
    if argv[:1] == ["router"]:
        from kcoral.server.cli import router_main

        router_main(argv[1:])
        return
    if argv[:1] == ["slurm"]:
        from kcoral.server.slurm import slurm_main

        raise SystemExit(slurm_main(argv[1:]))
    parser = argparse.ArgumentParser(prog="kcoral", description="KCoral remote execution.")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("server", help="start a standalone server or join a Router")
    commands.add_parser("router", help="route requests across compute nodes")
    commands.add_parser("run", help="run a tool on a remote server")
    commands.add_parser("slurm", help="allocate a Slurm GPU node and join a Router")
    parser.parse_args(argv)
    parser.print_help()


if __name__ == "__main__":
    main()
