"""Sphinx configuration for the M3Resp documentation website."""

from __future__ import annotations

import os
import shutil
import sys

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
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]
templates_path = ["_templates"]

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
napoleon_google_docstring = True
# List the attributes of a class in its description. The member list documents each attribute once.
napoleon_use_ivar = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
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
    "matplotlib": ("https://matplotlib.org/stable", None),
    "plotly": ("https://plotly.com/python-api-reference", None),
}
intersphinx_disabled_reftypes = ["std:doc"]

# Copy only the code of a code block, without the ``$`` and ``>>>`` prompts.
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True

# Tutorials are Python scripts in ``docs/tutorials_src``. Each build runs every
# script and writes the pages, with the printed output and the figures, to
# ``docs/tutorials``. The scripts read the synthetic recordings in ``tests/data``.
docs_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["M3RESP_EXAMPLE_DATA"] = os.path.join(
    os.path.dirname(docs_dir), "tests", "data"
)

# The tutorials draw interactive plotly figures. ``_ext/plotly_gallery.py``
# puts them on the pages. The import makes ``fig.show()`` keep each figure
# for the page.
sys.path.insert(0, os.path.join(docs_dir, "_ext"))
import plotly_gallery  # noqa: F401

# Each build writes one page per function or class of the API reference to
# ``docs/api/generated``. The folder is emptied first, so it holds only the pages
# of the names that m3resp exports now, also after a switch between branches.
shutil.rmtree(os.path.join(docs_dir, "api", "generated"), ignore_errors=True)
sphinx_gallery_conf = {
    "examples_dirs": "tutorials_src",
    "gallery_dirs": "tutorials",
    "filename_pattern": r".*\.py",
    "within_subsection_order": "FileNameSortKey",
    # Each API page lists the tutorials that use that function or class.
    "backreferences_dir": os.path.join("api", "backreferences"),
    "doc_module": ("m3resp",),
    "image_scrapers": ("matplotlib", "plotly_gallery.plotly_scraper"),
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_signature": False,
    "write_computation_times": False,
    # Show only what the scripts print, not the value of the last line of each block.
    "capture_repr": (),
}

html_theme = "pydata_sphinx_theme"
html_title = f"{project} documentation"
html_show_sourcelink = False
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,400;0,500;0,700;1,400&family=Roboto+Mono:wght@400;500;700&display=swap",
    "custom.css",
]
html_js_files = ["m3-page-toc.js", "m3-plotly-figures.js"]
# No left sidebar. The right sidebar has the sections of the page and the
# other pages of the same part of the documentation.
html_sidebars = {"**": []}
html_theme_options = {
    "navbar_align": "content",
    "navbar_end": [
        "theme-switcher",
        "navbar-icon-links",
    ],
    "header_links_before_dropdown": 5,
    "navigation_with_keys": True,
    "show_nav_level": 2,
    "show_toc_level": 2,
    # The other pages of the same folder, then the sections of this page.
    "secondary_sidebar_items": ["section-nav", "m3-page-toc"],
    # One footer line: the copyright at the left, the Sphinx and theme versions at the right.
    "footer_start": ["copyright"],
    "footer_end": ["m3-built-with"],
    "pygments_light_style": "a11y-high-contrast-light",
    "pygments_dark_style": "github-dark",
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
