"""Sphinx configuration for the M3Resp documentation website."""

from __future__ import annotations

import os

import m3resp

project = "M3Resp"
author = "M3Resp contributors"
copyright = "2026, M3Resp contributors"
version = m3resp.__version__
release = version

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
root_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_preserve_defaults = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 4
myst_substitutions = {
    "m3resp_version": release,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}
intersphinx_disabled_reftypes = ["std:doc"]

# Keep the gallery ready for future examples without creating an empty page.
# Adding the first ``.py`` file to ``docs/gallery_examples`` enables generation.
docs_dir = os.path.dirname(os.path.abspath(__file__))
gallery_examples_dir = os.path.join(docs_dir, "gallery_examples")
gallery_has_examples = os.path.isdir(gallery_examples_dir) and any(
    filename.endswith(".py") for filename in os.listdir(gallery_examples_dir)
)
if gallery_has_examples:
    extensions.append("sphinx_gallery.gen_gallery")
    sphinx_gallery_conf = {
        "examples_dirs": "gallery_examples",
        "gallery_dirs": os.path.join("generated", "gallery"),
        "filename_pattern": r".*\.py",
        "download_all_examples": False,
        "remove_config_comments": True,
    }

html_theme = "pydata_sphinx_theme"
html_title = f"{project} documentation"
html_show_sourcelink = False
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_sidebars = {
    "getting-started": [],
    "pipelines": [],
}
html_theme_options = {
    "navbar_align": "content",
    "navbar_end": [
        "theme-switcher",
        "navbar-icon-links",
    ],
    "header_links_before_dropdown": 4,
    "navigation_with_keys": True,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/M3RESP/m3resp",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}
html_context = {
    "github_user": "M3RESP",
    "github_repo": "m3resp",
    "github_version": os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main"),
    "doc_path": "docs",
}

readthedocs_version = os.environ.get("READTHEDOCS_VERSION_NAME")
if readthedocs_version:
    myst_substitutions["m3resp_version"] = readthedocs_version
