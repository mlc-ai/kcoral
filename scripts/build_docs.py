#!/usr/bin/env python3
"""Build the project homepage and versioned documentation into a static website.

Requires Python 3.12 and uv. Each ref gets its own non-editable package install
so autodoc reads that version's API, not the API installed for another ref.
"""

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARKER = ".kcoral-docs-site"


def run(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def release_tags():
    tags = subprocess.check_output(["git", "tag", "--list"], cwd=REPO, text=True).splitlines()
    return sorted(
        (tag for tag in tags if re.fullmatch(r"v\d+\.\d+\.\d+", tag)),
        key=lambda tag: tuple(map(int, tag[1:].split("."))),
        reverse=True,
    )


def redirect(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
        f"<script>location.replace({json.dumps(target)}"
        " + location.search + location.hash);</script>\n"
        f'<noscript><meta http-equiv="refresh" content="0; url={target}"></noscript>\n'
        "<title>KCoral documentation</title></head>\n"
        f'<body><noscript><a href="{target}">KCoral documentation</a></noscript></body></html>\n'
    )


def build_version(source, destination, name, versions, environment):
    run("uv", "venv", "--python", sys.executable, environment)
    python = environment / "bin" / "python"
    run(
        "uv",
        "pip",
        "install",
        "--python",
        python,
        "-r",
        source / "docs" / "requirements.txt",
        f"{source}[server]",
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        KCORAL_DOC_VERSION=name,
        KCORAL_DOC_VERSIONS=json.dumps(versions),
        KCORAL_DOC_REF="main" if name == "latest" else name,
    )
    run(
        python,
        "-m",
        "sphinx",
        "-b",
        "html",
        "-n",
        "-W",
        "--keep-going",
        source / "docs",
        destination,
        cwd=source,
        env=env,
    )
    # Sphinx's pickled build cache is not part of the public website.
    shutil.rmtree(destination / ".doctrees")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-only", action="store_true", help="Skip release tags (PR checks).")
    parser.add_argument("--output", type=Path, default=REPO / "_site")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == REPO or output in REPO.parents:
        parser.error("The output directory must not contain the repository.")
    if output.exists() and not (output / MARKER).is_file():
        parser.error(f"Refusing to replace {output}: it is not a generated documentation site.")

    versions = ["latest", *(release_tags() if not args.latest_only else [])]
    print(f"Building documentation versions: {', '.join(versions)}", flush=True)
    # Build in a staging directory; a failed build leaves the previous site intact.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kcoral-docs-") as temporary:
        work = Path(temporary)
        site = work / "site"
        for name in versions:
            source = REPO
            if name != "latest":
                source = work / name
                source.mkdir()
                archive = subprocess.check_output(
                    ["git", "archive", "--format=tar", f"refs/tags/{name}"], cwd=REPO
                )
                with tarfile.open(fileobj=io.BytesIO(archive)) as files:
                    files.extractall(source, filter="data")
            build_version(source, site / "docs" / name, name, versions, work / f"env-{name}")

        shutil.copytree(REPO / "website", site, dirs_exist_ok=True)
        shutil.copytree(REPO / "docs" / "_static" / "brand", site / "assets" / "brand")
        redirect(site / "docs" / "index.html", "latest/")
        (site / ".nojekyll").touch()
        (site / MARKER).touch()
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(site, output)
    print(f"Website ready at {output}", flush=True)


if __name__ == "__main__":
    main()
