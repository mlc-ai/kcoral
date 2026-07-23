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
