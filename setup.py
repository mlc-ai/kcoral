"""Platform wheels containing Python modules and standalone Rust executables."""

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel


class BinaryWheel(bdist_wheel):
    def get_tag(self):
        _, _, platform = super().get_tag()
        # The executables do not link against Python or depend on its ABI.
        return "py3", "none", platform


setup(cmdclass={"bdist_wheel": BinaryWheel})
