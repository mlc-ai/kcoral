"""Regenerate checked-in Python gateway bindings with the pinned compiler.

Run: uv run --no-project --with grpcio-tools==1.66.2 --with protobuf==5.27.2 \\
    python scripts/generate_gateway.py [--check]
"""

import argparse
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if version("grpcio-tools") != "1.66.2":
        raise SystemExit("generation requires grpcio-tools==1.66.2")
    root = Path(__file__).resolve().parents[1]
    proto = root / "rust/kcoral/proto"
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{proto}",
                f"--python_out={output}",
                f"--grpc_python_out={output}",
                str(proto / "kcoral_gateway.proto"),
            ],
            check=True,
        )
        grpc_file = output / "kcoral_gateway_pb2_grpc.py"
        grpc_file.write_text(
            grpc_file.read_text().replace(
                "import kcoral_gateway_pb2 as", "from . import kcoral_gateway_pb2 as"
            )
        )
        for source in sorted(output.glob("*.py")):
            source.write_text(source.read_text().rstrip() + "\n")
            target = root / "python/kcoral/server/_generated" / source.name
            if args.check:
                if not target.exists() or target.read_bytes() != source.read_bytes():
                    raise SystemExit(f"gateway bindings need regeneration: {target}")
            else:
                target.write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
