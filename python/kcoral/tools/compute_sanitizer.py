"""Run CUDA correctness checks with Compute Sanitizer."""


def parse_args(argv):
    from kcoral.tools.cli import parse_args as parse_common

    return parse_common("compute-sanitizer", argv)


def main(argv):
    from kcoral.tools.cli import run_tool

    return run_tool("compute-sanitizer", *parse_args(argv))


def run(arguments, environment, reports, execute):
    return {"returncode": execute(["compute-sanitizer", *arguments]), "missing": []}
