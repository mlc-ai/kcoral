<a id="development"></a>
<a id="build-and-preview-the-documentation"></a>

# Build the Docs

Run these commands from the repository root with Python 3.12, `uv`, and the
[source build tools](../getting-started/installation.md#get-the-source) installed.
No GPU or CUDA toolkit is needed.

```bash
uv venv --python 3.12 .docs-venv
uv pip install --python .docs-venv/bin/python -r docs/requirements.txt '.[server]'
.docs-venv/bin/python -m sphinx -b html -n -W --keep-going docs docs/_build/html
.docs-venv/bin/python -m http.server 8008 --bind 127.0.0.1 --directory docs/_build/html
```

Open <http://127.0.0.1:8008>. After editing pages, rerun Sphinx and reload the
browser. After changing Python docstrings, reinstall KCoral before rebuilding:

```bash
uv pip install --python .docs-venv/bin/python --reinstall-package kcoral '.[server]'
```

Use a non-editable install so the API reference reads the installed package.
The Sphinx flags check references and fail on warnings. For external link checks,
use `-b linkcheck` with a separate output directory.

## Maintain the documentation

- Add pages under the appropriate section and update `docs/index.md`.
- When moving pages, update links and source includes; preserve linked headings
  or add explicit anchors.
- Document public APIs in Python docstrings and keep runnable examples in
  `examples/`, included with `literalinclude`.

To update documentation dependencies, edit `docs/requirements.in` and regenerate
the lock file:

```bash
uv pip compile --python-version 3.12 docs/requirements.in -o docs/requirements.txt
```

## Build the versioned website

The website at <https://kcoral.mlc.ai/docs/> serves `main` at `/docs/latest/` and
stable `vMAJOR.MINOR.PATCH` tags at paths such as `/docs/v1.2.3/`.
Keep published tags immutable and include the documentation and its dependencies
in each tag. Other tag formats are excluded from the website.

```bash
git fetch origin --tags
.docs-venv/bin/python scripts/build_docs.py
.docs-venv/bin/python -m http.server 8008 --bind 127.0.0.1 --directory _site
```

Open <http://127.0.0.1:8008/docs/>. The builder uses the current checkout for
`latest` and installs each tag in a separate environment for its API reference.
Add `--latest-only` to skip tag builds. Output goes to `_site/`; a failed build
preserves the previous site.

## Publish the website

The `Documentation` workflow publishes pushes to `main`, version tag pushes, and
manual runs on `main` to [kcoral-docs](https://github.com/mlc-ai/kcoral-docs).
Pull requests and manual runs on other branches only produce HTML artifacts.

Hosting setup:

- Set `DOCS_DEPLOY_KEY` to an SSH deploy key with write access to `kcoral-docs`.
- In that repository, serve GitHub Pages from `main` at `/`, set the custom domain
  to `kcoral.mlc.ai`, and enable **Enforce HTTPS** once the certificate is ready.
- Add DNS record `CNAME kcoral mlc-ai.github.io`.

The workflow maintains `CNAME` and `.nojekyll`. Change or revert documentation in
the source repository; deployment replaces the generated website.

## Package versions

`setuptools-scm` derives package versions from Git tags: `v1.2.3` builds as
`1.2.3`, `v1.2.3.post1` as `1.2.3.post1`, and later commits as development
versions. Fetch history and tags before building; unshallow the clone if needed.
No version constants need updating. The Rust crate has no separate release version.

`Build wheels` checks Linux x86_64 and aarch64 wheels on every PR, push to `main`,
and version tag push. It also accepts a tag, branch, or SHA through its manual
`ref` input.

## Publish a release to PyPI

Only publishing a GitHub Release triggers a PyPI upload. `Publish to PyPI` calls
`Build wheels` for the release tag and uploads both wheels after checks pass.

Before the first release, create the `pypi` GitHub environment and configure a
Trusted Publisher for `kcoral` on PyPI (a pending publisher for a new project):

- Owner / repository: `mlc-ai` / `kcoral`
- Workflow: `publish_pypi.yml`
- Environment: `pypi`

No PyPI API token is required. Each version can be published only once.

## Run project checks

Run tests and Python formatting checks:

```bash
uv run --no-editable --group test pytest -q
uvx ruff check python tests
uvx ruff format --check python tests
```

Enable GPU integration tests on a machine with a configured GPU:

```bash
KCORAL_GPU_TEST=1 uv run --no-editable --group test pytest -q
```

CPU compilation tests run when `nvcc`, `ninja`, and a host C++ compiler are
available, and skip otherwise. They do not require a GPU.
