"""Run an executable directly in the uploaded working directory."""


def parse_args(argv):
    from kcoral.tools.cli import parse_args as parse_common

    return parse_common("shell", argv)


def main(argv):
    from kcoral.tools.cli import run_tool

    return run_tool("shell", *parse_args(argv))


def run(arguments, environment, reports, execute):
    return {"returncode": execute(arguments), "missing": []}
