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

- The documentation workflow builds every page and uploads the website for
  review. Public deployment can be added after a hosting destination is chosen.

## Run project checks

`--group test` adds pytest to either environment from [installation](../getting-started/installation.md):

```bash
uv sync --no-editable --group test             # or --group test --group gpu
pytest -q
ruff check python tests
ruff format --check python tests
```

The GPU integration tests are opt-in, and need the GPU environment:

```bash
KCORAL_GPU_TEST=1 pytest -q
```

The CPU compilation integration test needs the compiler group and a CUDA
toolchain, but no GPU. It runs whenever `nvcc`, `ninja` and a host C++ compiler
are present, and skips itself otherwise:

```bash
uv run --no-editable --group test --group compiler pytest -q
```
