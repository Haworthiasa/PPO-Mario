"""Build the upstream nes-py 8.2.1 source archive with MinGW-w64 on Windows."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


WINDOWS_BUILDER = """
from pathlib import Path
import subprocess
from setuptools.command.build_ext import build_ext


class BuildMinGW(build_ext):
    def build_extension(self, ext):
        output = Path(self.get_ext_fullpath(ext.name)).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "g++", *ext.extra_compile_args, "-shared", "-static",
            *(f"-I{directory}" for directory in ext.include_dirs),
            *ext.sources, "-o", str(output),
        ], check=True)


setup(
    cmdclass={"build_ext": BuildMinGW},
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=Path(".wheels"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as scratch:
        with tarfile.open(args.archive, "r:gz") as source:
            source.extractall(scratch, filter="data")
        source_dir = Path(scratch) / "nes_py-8.2.1"
        setup_file = source_dir / "setup.py"
        setup = setup_file.read_text(encoding="utf-8")
        setup_file.write_text(setup.replace("setup(\n", WINDOWS_BUILDER, 1), encoding="utf-8")
        # Pip's nested wheel build directories can exceed MAX_PATH on Windows.
        build_env = dict(os.environ, TEMP=scratch, TMP=scratch)
        subprocess.run([
            sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
            "--wheel-dir", str(args.out.resolve()), str(source_dir),
        ], check=True, env=build_env)


if __name__ == "__main__":
    main()
