"""Send one benchmark request to a running server.

    python examples/example_client.py [PORT]      # default 8000

Standalone: needs only ``httpx``. It uploads an input tensor, a TIRx kernel, and a
torch reference, compiles the kernel, runs it on the uploaded tensor, asserts the
result matches the reference (``assert_close`` fails the request if they differ),
and benchmarks it, then prints the reply.

Each upload carries a content key = ``sha256`` of the exact bytes the server will
materialize — a ``function``'s UTF-8 source, or a ``tensor``'s compact
``{dtype,shape}`` header + a NUL byte + raw row-major bytes — computed inline below.
"""

import array
import base64
import hashlib
import json
import sys

import httpx

KERNEL = """from __future__ import annotations
from tvm.script import tirx as T

@T.jit
def main(A: T.Buffer((N,), "float32"), B: T.Buffer((N,), "float32"), *, N: T.constexpr):
    T.device_entry()
    i = T.cta_id([N])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0
"""
REF = "def main(a):\n    return a + 1.0\n"


def upload(id, source):
    key = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    return {"id": id, "op": "upload", "kind": "function", "key": key,
            "inline": {"source": source}}


def upload_tensor(id, values):
    raw = array.array("f", values).tobytes()  # row-major float32 bytes
    shape = [len(values)]
    header = json.dumps({"dtype": "float32", "shape": shape}, separators=(",", ":")).encode()
    key = "sha256:" + hashlib.sha256(header + b"\x00" + raw).hexdigest()
    return {"id": id, "op": "upload", "kind": "tensor", "key": key,
            "inline": {"dtype": "float32", "shape": shape,
                       "data_b64": base64.b64encode(raw).decode()}}


program = {"instructions": [
    upload("kernel", KERNEL),
    upload("reffn", REF),
    upload_tensor("a", [float(i) for i in range(256)]),
    {"id": "out", "op": "run", "fn": "builtin.empty",
     "args": [{"shape": [256], "dtype": "float32"}]},
    {"id": "mod", "op": "run", "fn": "builtin.compile_tirx",
     "args": [{"$ref": "kernel"}, {"N": 256}]},
    {"id": "run", "op": "run", "fn": {"$ref": "mod"}, "args": [{"$ref": "a"}, {"$ref": "out"}]},
    {"id": "ref", "op": "run", "fn": {"$ref": "reffn"}, "args": [{"$ref": "a"}]},
    {"id": "chk", "op": "run", "fn": "builtin.assert_close",
     "args": [{"$ref": "out"}, {"$ref": "ref"}]},
    {"id": "perf", "op": "run", "fn": "builtin.benchmark",
     "args": [{"$ref": "mod"}, {"$ref": "a"}, {"$ref": "out"}, {"warmup": 10, "repeat": 50}]},
], "options": {"timeout_seconds": 120}}


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "8000"
    resp = httpx.post(f"http://127.0.0.1:{port}/benchmark", json=program, timeout=180)
    print("HTTP", resp.status_code)
    print(resp.text)


if __name__ == "__main__":
    main()
