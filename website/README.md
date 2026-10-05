# Project website

The homepage at <https://kcoral.mlc.ai/> uses plain HTML, CSS, and an optional
clipboard script. Edit `index.html` and `assets/`; shared brand images come from
`docs/_static/brand/`. Only `index.html` and `assets/` are copied into the site.

## Build and preview

With Python 3.12 and [uv](https://docs.astral.sh/uv/) installed, run from the
repository root:

```bash
python scripts/build_docs.py --latest-only
python -m http.server 8008 --bind 127.0.0.1 --directory _site
```

Open <http://127.0.0.1:8008/>. Omit `--latest-only` to also build all stable
`vMAJOR.MINOR.PATCH` release tags. Each documentation version gets its own
environment and package installation; no GPU is required to build the site.

The generated layout preserves the versioned documentation URLs:

```text
/                      project homepage
/docs/                 redirects to /docs/latest/
/docs/latest/          documentation from the current checkout
/docs/v0.1.0/          documentation from tag v0.1.0
```

Homepage documentation links go directly to `/docs/latest/`. The `/docs/` alias
retains its meta-refresh redirect and works without JavaScript.

The Documentation workflow checks that pull requests build successfully.
After a merge to `main`, a stable version tag push, or a manual run on `main`, it publishes the
site to this repository's `gh-pages` branch and requests a GitHub Pages build.
Pull requests do not publish the site.
