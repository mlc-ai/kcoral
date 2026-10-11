"""Capture AMD ROCprofiler traces and kernel statistics with rocprofv3."""

# Without tracing options rocprofv3 records nothing; default to a kernel trace
# with per-kernel statistics in CSV, which needs no ROCm tools to read locally.
DEFAULT_OPTIONS = ("--kernel-trace", "--stats", "--output-format", "csv")


def validate_args(parser, arguments):
    from kcoral.tools.cli import validate_profiler_args

    validate_profiler_args(
        parser,
        arguments,
        managed=("--output-directory", "--output-file"),
        short=("-d", "-o"),
    )


def parse_args(argv):
    from kcoral.tools.cli import parse_args as parse_common

    return parse_common("rocprofv3", argv, profiling=True, validate=validate_args)


def main(argv):
    from kcoral.tools.cli import run_tool

    return run_tool("rocprofv3", *parse_args(argv))


def run(arguments, environment, reports, execute):
    boundary = arguments.index("--")
    options = arguments[:boundary] or list(DEFAULT_OPTIONS)
    code = execute(
        [
            "rocprofv3",
            "--output-directory",
            str(reports),
            "--output-file",
            "capture",
            *options,
            "--",
            *arguments[boundary + 1 :],
        ]
    )
    missing = [] if any(path.is_file() for path in reports.rglob("*")) else ["rocprofv3 output"]
    return {"returncode": code, "missing": missing}
