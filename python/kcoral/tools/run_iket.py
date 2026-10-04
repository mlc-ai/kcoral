"""Capture instrumented kernel execution timelines with run-iket."""


def validate_args(parser, arguments):
    from kcoral.tools.cli import validate_profiler_args

    validate_profiler_args(
        parser,
        arguments,
        managed=("--output-dir", "--working-dir"),
        short=("-o",),
        subcommand="profile",
    )


def parse_args(argv):
    from kcoral.tools.cli import parse_args as parse_common

    return parse_common("run-iket", argv, profiling=True, validate=validate_args)


def main(argv):
    from kcoral.tools.cli import run_tool

    return run_tool("run-iket", *parse_args(argv))


def run(arguments, environment, reports, execute):
    # IKET creates its output directory and prompts if it already exists.
    code = execute(["run-iket", "--output-dir", str(reports), *arguments], create_reports=False)
    missing = [] if any(path.is_file() for path in reports.rglob("*")) else ["run-iket output"]
    return {"returncode": code, "missing": missing}
