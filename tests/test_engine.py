from benchmark_server.engine import execute
from benchmark_server.keys import canonical_bytes
from benchmark_server.schemas import Program, Run, Upload
from benchmark_server.testing import FakeRuntime


def prog(*instrs):
    return Program(instructions=list(instrs))


def fn(id, src="def main(x):\n    return x\n"):
    return Upload(id=id, kind="function", key="sha256:x", inline={"source": src})


def run(program, runtime=None):
    """Run a program, filling upload bytes from each upload's inline (as the
    front-end would from the canonical form)."""
    program.upload_bytes = {
        i.id: canonical_bytes(i.kind, i.inline) for i in program.instructions if i.op == "upload"
    }
    return execute(program, runtime or FakeRuntime())


def test_full_pass_threads_handles():
    res = run(
        prog(
            Run("x", "builtin.opaque", []),
            Run("y", "builtin.opaque", []),
            Run("r", "builtin.structural", [{"$ref": "x"}, {"$ref": "y"}]),
        )
    )
    assert [r.status for r in res] == ["OK"] * 3
    assert res[0].value == {"handle": "x"}  # an opaque result comes back as a handle
    assert res[-1].value == {"ok": True}  # a structural result passes through


def test_failure_short_circuits_rest_skipped():
    res = run(
        prog(
            Run("bad", "builtin.nope", []),  # unknown builtin -> FAILED
            Run("after", "builtin.opaque", []),  # -> SKIPPED
        )
    )
    assert res[0].status == "FAILED"
    assert res[1].status == "SKIPPED" and res[1].error["reason"] == "predecessor_failed"


def test_unknown_builtin_is_failed_runtime():
    res = run(prog(Run("y", "builtin.nope", [])))
    assert res[0].status == "FAILED" and res[0].error["kind"] == "runtime"


def test_function_syntax_error_is_parse_failure():
    res = run(prog(fn("k", src="def bad(:\n  pass\n")))
    assert res[0].status == "FAILED" and res[0].error["kind"] == "parse"


def test_call_uploaded_function_by_handle():
    res = run(prog(fn("f", "def f(a):\n    return a + 1\n"), Run("y", {"$ref": "f"}, [41])))
    assert res[1].status == "OK" and res[1].value == 42


def test_ref_to_non_callable_fn_is_failed():
    res = run(
        prog(
            Run("x", "builtin.opaque", []),
            Run("y", {"$ref": "x"}, []),  # x is an opaque value, not callable
        )
    )
    assert res[1].status == "FAILED" and res[1].error["kind"] == "runtime"


# --- per-instruction stdout/stderr capture ----------------------------------


def test_stdout_stderr_captured_per_instruction():
    src = (
        "import sys\n"
        "def f(x):\n"
        "    print('to stdout')\n"
        "    print('to stderr', file=sys.stderr)\n"
        "    return x\n"
    )
    res = run(prog(fn("f", src), Run("y", {"$ref": "f"}, [1]), Run("z", "builtin.structural", [])))
    assert res[1].stdout == "to stdout\n" and res[1].stderr == "to stderr\n"
    assert res[2].stdout == "" and res[2].stderr == ""  # capture is per instruction


def test_fd_level_output_is_captured():
    # os.write bypasses sys.stdout, like output from a C extension would.
    src = "import os\ndef f(x):\n    os.write(1, b'raw bytes out')\n    return x\n"
    res = run(prog(fn("f", src), Run("y", {"$ref": "f"}, [0])))
    assert res[1].stdout == "raw bytes out"


def test_failed_instruction_keeps_captured_output():
    src = "def f():\n    print('before failure')\n    raise ValueError('boom')\n"
    res = run(prog(fn("f", src), Run("y", {"$ref": "f"}, [])))
    assert res[1].status == "FAILED"
    assert res[1].stdout == "before failure\n"


def test_output_truncated_to_limit():
    program = prog(
        fn("f", "def f(x):\n    print('x' * 100)\n    return x\n"),
        Run("y", {"$ref": "f"}, [0]),
    )
    program.options = {"output_limit_bytes": 10}
    res = run(program)
    assert res[1].stdout == "x" * 10 and res[1].stdout_truncated
    assert res[1].stderr == "" and not res[1].stderr_truncated


def test_capture_disabled_with_non_positive_limit():
    program = prog(
        fn("f", "def f(x):\n    print('hello')\n    return x\n"),
        Run("y", {"$ref": "f"}, [0]),
    )
    program.options = {"output_limit_bytes": 0}
    res = run(program)
    assert res[1].stdout == "" and not res[1].stdout_truncated
