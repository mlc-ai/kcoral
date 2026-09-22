"""Run scripts, modules, or inline code with the worker's Python interpreter."""

import sys


def validate_args(parser, arguments):
    remaining = iter(arguments)
    for token in remaining:
        if token == "--":
            if next(remaining, "-") != "-":
                return
            break
        if token == "-":
            break
        if not token.startswith("-"):
            return
        if token == "--check-hash-based-pycs":
            next(remaining, None)
            continue
        if token.startswith("--"):
            continue
        for index, option in enumerate(token[1:], start=1):
            if option == "i":
                parser.error("interactive Python execution is unsupported")
            if option in "cmWX":
                value = token[index + 1 :] or next(remaining, None)
                if value is None:
                    parser.error(f"Python -{option} requires an argument")
                if option in "cm":
                    return
                break
    parser.error("a script, -c command, or -m module is required; stdin execution is unsupported")


def parse_args(argv):
    from ._common import parse_args as parse_common

    return parse_common("python", argv, validate=validate_args)


def main(argv):
    from ._common import run_tool

    return run_tool("python", *parse_args(argv))


def run(arguments, environment, reports, execute):
    return {"returncode": execute([sys.executable, *arguments]), "missing": []}
