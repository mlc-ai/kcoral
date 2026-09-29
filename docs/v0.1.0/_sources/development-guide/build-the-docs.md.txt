<a id="development"></a>
<a id="build-and-preview-the-documentation"></a>

# Build the Docs

Sphinx builds the site from the pages under `docs/`, using MyST to parse Markdown
and Furo for the theme. `docs/index.md` defines navigation; `docs/conf.py`
configures the build, with templates and assets in `docs/_templates/` and
`docs/_static/`.

The API reference in `docs/python-api/index.rst` uses autodoc to read docstrings
from the installed KCoral package. `scripts/build_docs.py` assembles builds of
the current checkout and release tags into the versioned website.

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

To update documentation dependencies, edit `docs/requirements.in` and regenerate
the lock file:

```bash
uv pip compile --python-version 3.12 docs/requirements.in -o docs/requirements.txt
```

## Versioned site

To build the current checkout and all stable `vMAJOR.MINOR.PATCH` tags:

```bash
git fetch origin --tags
.docs-venv/bin/python scripts/build_docs.py
.docs-venv/bin/python -m http.server 8008 --bind 127.0.0.1 --directory _site
```

Open <http://127.0.0.1:8008/docs/>. The builder uses the current checkout for
`/docs/latest/` and each tag for paths such as `/docs/v1.2.3/`. Each tag must
contain the documentation and its dependencies; it gets a separate environment
so the API reference matches that version. Add `--latest-only` to skip tag builds.
Output goes to `_site/`; a failed build preserves the previous site.
