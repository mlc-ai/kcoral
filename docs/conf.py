"""Sphinx configuration for the KCoral documentation."""

from importlib.metadata import version as package_version

project = "KCoral"
author = "KCoral contributors"
release = package_version("kcoral")
version = release
language = "en"

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.intersphinx"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "requirements.*", ".DS_Store"]
myst_heading_anchors = 5

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
# FastAPI has no Sphinx inventory; the API page links its reference explicitly.
nitpick_ignore = [("py:class", "fastapi.applications.FastAPI")]

html_theme = "furo"
html_title = "KCoral documentation"
html_favicon = "_static/brand/kcoral-icon.png"
html_theme_options = {
    "light_logo": "brand/kcoral-logo-light.png",
    "dark_logo": "brand/kcoral-logo-dark.png",
    "source_repository": "https://github.com/mlc-ai/kcoral/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#0f766e",
        "color-brand-content": "#0f766e",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5eead4",
        "color-brand-content": "#5eead4",
    },
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True
html_copy_source = True
html_last_updated_fmt = None
html_show_copyright = False
linkcheck_ignore = [r"http://(?:localhost|127\.0\.0\.1|server)(?::\d+)?(?:/.*)?$"]
