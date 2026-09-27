<a id="development"></a>
<a id="build-and-preview-the-documentation"></a>

# Build the Docs

Sphinx builds the website; MyST reads Markdown, and Furo supplies the theme.
Only the Python interface reference uses Sphinx's native reStructuredText format.
The documentation environment uses Python 3.12 and needs no GPU toolchain.

Run from the repository checkout. Set `DOC_ENV` and `DOC_OUTPUT` to directories
for this checkout; in an isolated task, keep both inside the task directory.

```bash
DOC_REPO="$PWD"
DOC_ENV="$DOC_REPO/.docs-venv"
DOC_OUTPUT="$DOC_REPO/docs/_build/html"
uv venv --python 3.12 "$DOC_ENV"
uv pip install --python "$DOC_ENV/bin/python" \
  -r "$DOC_REPO/docs/requirements.txt" "${DOC_REPO}[server]"
"$DOC_ENV/bin/python" -m sphinx -b html -n -W --keep-going \
  "$DOC_REPO/docs" "$DOC_OUTPUT"
"$DOC_ENV/bin/python" -m http.server 8008 --bind 127.0.0.1 \
  --directory "$DOC_OUTPUT"
```

Open `http://127.0.0.1:8008`. Stop the foreground server with Ctrl+C. Rebuild after
editing pages and reload the browser. After changing Python documentation
strings, reinstall with `uv pip install --reinstall-package kcoral` using the
same environment and checkout path before rebuilding. Do not use an editable
installation for the documentation build.

`-n` checks object references, `-W` fails on warnings, and `--keep-going` reports
as many issues as possible. To check external links, replace `-b html` with
`-b linkcheck` and choose a separate output directory. External checks need a
network connection. The HTML build uses a bundled theme and does not download
fonts or execute remote kernel examples.

## Maintain the documentation

Place each page in the directory for its navigation section:

```text
docs/
├── getting-started/
├── client-guide/
├── server-guide/
├── tutorials/
├── development-guide/
└── python-api/
```

The root `index.md` defines the section navigation. Sphinx configuration,
dependency files and shared static assets stay at the documentation root.

- Update navigation, relative links, source includes and skill references when
  moving a page. Preserve headings when other pages link to them; add an explicit
  anchor before renaming one.
- Explain public parameters, results and errors in the Python documentation
  strings. The reference page lists public objects explicitly.
- Keep complete runnable scripts in `examples/`. Their documentation pages use
  `literalinclude`, a Sphinx directive that displays the source file, and offer
  the same file for download.
- Update `docs/requirements.in`, then regenerate its lock file on Python 3.12:

  ```bash
  uv pip compile --python-version 3.12 docs/requirements.in -o docs/requirements.txt
  ```

- Pull requests build the documentation and upload an HTML artifact for review.
  They do not publish a preview site.

## Package versions

Git tags are the source of package versions. `setuptools-scm` derives the version
at build time: a clean checkout of `v1.2.3` produces version `1.2.3`, and commits
after that tag produce development versions. Post-release tags such as
`v1.2.3.post1` are also supported. There are no version constants to update before
tagging a release. The bundled Rust crate is unpublished and has no separate
release version.

Use a checkout with Git history and tags available (`git fetch --tags`; unshallow
the clone first if needed). Every pull request, push to `main`, and version tag
push runs the `Build wheels` workflow; it can also be run manually with the tag
as its `ref`. The installed package metadata, `kcoral.__version__`, and the HTTP
API version all use the derived version.
Source distributions preserve it for builds without Git; `git archive` exports
carry version metadata through `.git_archival.txt`.

## Publish a release to PyPI

Publishing a GitHub Release triggers `Publish to PyPI`. It calls `Build wheels`
for the release tag, then uploads the checked Linux x86_64 and aarch64 wheels
after both builds succeed. Pull requests, pushes to `main`, tag pushes, and manual
wheel builds only produce GitHub Actions artifacts; they do not upload to PyPI.
Draft releases do not publish either.

Before the first release, configure a GitHub Trusted Publisher for the `kcoral`
project on PyPI (a pending publisher if the project does not yet exist):

- Owner: `mlc-ai`
- Repository: `kcoral`
- Workflow: `publish_pypi.yml`
- Environment: `pypi`

Create the matching `pypi` GitHub environment. The publish job uses GitHub's OIDC
token, so no PyPI API token secret is required. Publish each package version
once; PyPI does not allow replacing an uploaded file.

## Build the versioned website

The public website lives at <https://kcoral.mlc.ai/docs/>. Its sidebar includes
a version menu. `latest` follows `main`; stable release tags such as `v0.1.0`
have their own permanent URLs:

```text
/docs/                 redirects to /docs/latest/
/docs/latest/          documentation from main
/docs/v0.1.0/          documentation from tag v0.1.0
```

Switching versions opens that version's documentation home page. Tags must use
the form `vMAJOR.MINOR.PATCH` and contain the documentation configuration and
dependencies. Prerelease tags are not published. Treat published tags as
immutable: changing or removing a tag changes the site on the next deployment.

Build and serve the complete website from the repository root with Python 3.12
and `uv` available:

```bash
git fetch origin --tags
python scripts/build_docs.py
python -m http.server 8008 --bind 127.0.0.1 --directory _site
```

Open `http://127.0.0.1:8008/docs/`. The builder uses the current checkout for
`latest` and extracts each stable tag into a temporary directory. Each version
gets a separate Python environment, its own locked documentation dependencies,
and a non-editable installation of its own KCoral package. This ensures that
the API reference describes the selected release. No GPU toolchain is needed.

Use `python scripts/build_docs.py --latest-only` to skip release builds while
editing. `_site/` is generated output and is not committed to this repository.
A failed build preserves the previous output. The usual single-version Sphinx
command above remains available for quick local edits.

## Publish the website

The `Documentation` workflow rebuilds the website after a push to `main`, a
version tag push, or a manual run on `main`. It publishes the HTML artifact to
the public [kcoral-docs repository](https://github.com/mlc-ai/kcoral-docs), which
serves it through GitHub Pages. This keeps the source repository private while
allowing public documentation on GitHub Free. Only generated documentation,
including the documented example downloads and page sources, is published.

The deployment job uses the `DOCS_DEPLOY_KEY` Actions secret: an SSH deploy key
with write access only to `mlc-ai/kcoral-docs`. Pull requests and manual runs on
other branches only build an artifact; they cannot publish.

The hosting repository's Pages source is `main` at `/`, with custom domain
`kcoral.mlc.ai`. Its root `CNAME` and `.nojekyll` files are maintained by the
workflow. The `mlc.ai` DNS zone needs this record:

```text
CNAME  kcoral  mlc-ai.github.io
```

After GitHub issues the domain's certificate, enable **Enforce HTTPS** in the
hosting repository's Pages settings. To undo a documentation change, revert it
in the source repository and let the workflow publish again. Do not edit
generated HTML in the hosting repository: the next deployment replaces it.

## Run project checks

`--group test` adds pytest to the server environment from [installation](../getting-started/installation.md):

```bash
uv sync --no-editable --group test
pytest -q
ruff check python tests
ruff format --check python tests
```

The GPU integration tests are opt-in, and need the server dependencies and a GPU:

```bash
KCORAL_GPU_TEST=1 pytest -q
```

The CPU compilation integration test needs the server extra and a CUDA
toolchain, but no GPU. It runs whenever `nvcc`, `ninja` and a host C++ compiler
are present, and skips itself otherwise:

```bash
uv run --no-editable --group test pytest -q
```
