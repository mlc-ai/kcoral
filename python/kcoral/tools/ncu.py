"""Capture NVIDIA Nsight Compute reports in a managed remote output directory."""


def validate_args(parser, arguments):
    from ._common import validate_profiler_args

    validate_profiler_args(
        parser,
        arguments,
        managed=("--export", "--import", "--mode", "--config-file", "--config-file-path"),
        short=("-o", "-i"),
    )


def parse_args(argv):
    from ._common import parse_args as parse_common

    return parse_common("ncu", argv, profiling=True, validate=validate_args)


def main(argv):
    from ._common import run_tool

    return run_tool("ncu", *parse_args(argv))


def run(arguments, environment, reports, execute):
    boundary = arguments.index("--")
    environment.setdefault("NCU_PROFILE", "1")
    code = execute(
        [
            "ncu",
            "--config-file",
            "0",
            "--export",
            str(reports / "capture.ncu-rep"),
            *arguments[:boundary],
            *arguments[boundary + 1 :],
        ]
    )
    missing = [] if (reports / "capture.ncu-rep").is_file() else ["capture.ncu-rep"]
    return {"returncode": code, "missing": missing}
