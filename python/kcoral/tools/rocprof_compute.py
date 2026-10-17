"""Collect and analyze AMD hardware-counter profiles with ROCm Compute Profiler."""

# Roofline analysis adds benchmark passes and plotting dependencies; opt in by
# passing native options, which replace this default.
DEFAULT_OPTIONS = ("--no-roof",)
# Before ROCm 6.2 the tool was called Omniperf; both accept the same commands.
EXECUTABLES = ("rocprof-compute", "omniperf")


def validate_args(parser, arguments):
    from kcoral.tools.cli import validate_profiler_args

    validate_profiler_args(parser, arguments, managed=("--name", "--path"), short=("-n", "-p"))


def parse_args(argv):
    from kcoral.tools.cli import parse_args as parse_common

    return parse_common("rocprof-compute", argv, profiling=True, validate=validate_args)


def main(argv):
    from kcoral.tools.cli import run_tool

    return run_tool("rocprof-compute", *parse_args(argv))


def run(arguments, environment, reports, execute):
    import shutil
    import subprocess

    tool = next(
        (name for name in EXECUTABLES if shutil.which(name, path=environment.get("PATH"))),
        EXECUTABLES[0],  # absent: execute reports it as not installed
    )
    boundary = arguments.index("--")
    options = arguments[:boundary] or list(DEFAULT_OPTIONS)
    workload = reports / "workload"
    code = execute(
        [
            tool,
            "profile",
            "-n",
            "capture",
            "-p",
            str(workload),
            *options,
            "--",
            *arguments[boundary + 1 :],
        ]
    )
    # Versions differ in how deep under --path they write; analyze the directory
    # that holds the counter data.
    data = sorted(path.parent for path in workload.rglob("pmc_perf.csv"))
    missing = [] if data else ["rocprof-compute workload"]
    analysis = reports / "analysis.txt"
    if code == 0 and data:
        # Analyze prints its report; keep it as a file next to the raw workload.
        with analysis.open("w") as output:
            subprocess.run(
                [shutil.which(tool, path=environment.get("PATH")), "analyze", "-p", str(data[0])],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        if analysis.stat().st_size == 0:
            missing.append("analysis.txt")
    return {"returncode": code, "missing": missing}
