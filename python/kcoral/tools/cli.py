"""Dispatch ``kcoral run`` to a built-in tool definition."""

import argparse
from importlib import import_module

from . import COMMANDS


def get_tool(name):
    return import_module(f"{__package__}.{COMMANDS[name]}")


def run_main(argv):
    parser = argparse.ArgumentParser(
        prog="kcoral run", description="Run a tool on a remote server.", allow_abbrev=False
    )
    commands = parser.add_subparsers(dest="tool", required=True)
    for name in COMMANDS:
        commands.add_parser(name, add_help=False, help=f"run {name} remotely")
    args = parser.parse_args(argv[:1])
    return main(args.tool, argv[1:])


def main(tool, argv):
    return get_tool(tool).main(argv)
