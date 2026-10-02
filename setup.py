"""Wheels containing Python modules and, optionally, standalone Rust executables.

Set KCORAL_BUILD_RUST=1 to build the Rust router and node supervisor, which run
on Linux only. By default the wheel is pure Python.
"""

import os

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools_rust import RustBin, Strip


class BinaryWheel(bdist_wheel):
    def get_tag(self):
        _, _, platform = super().get_tag()
        # The executables do not link against Python or depend on its ABI.
        return "py3", "none", platform


def build_rust() -> bool:
    value = os.environ.get("KCORAL_BUILD_RUST") or "0"
    if value not in ("0", "1"):
        raise SystemExit(f"KCORAL_BUILD_RUST must be 0 or 1, not {value!r}")
    return value == "1"


def rust_bin(target: str) -> RustBin:
    return RustBin(
        target,
        path="rust/kcoral/Cargo.toml",
        cargo_manifest_args=["--locked"],
        strip=Strip.All,
    )


setup(
    cmdclass={"bdist_wheel": BinaryWheel},
    rust_extensions=[rust_bin("kcoral-router"), rust_bin("kcoral-node")] if build_rust() else [],
)
