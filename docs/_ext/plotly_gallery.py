"""Show the plotly figures of the tutorials on the documentation pages.

The tutorials call ``fig.show()``. During the documentation build, the
``m3resp_gallery`` renderer keeps each figure in a list. After each code
block, sphinx-gallery calls :func:`plotly_scraper`. It writes each kept figure
twice: as a PNG image with a white background, which sphinx-gallery uses for
the thumbnail of the tutorial, and as an HTML file with a transparent
background for the interactive figure on the page. ``_static/m3-plotly-figures.js``
gives the figure on the page the colours of the light or the dark theme. The
PNG image needs kaleido and a Chrome browser.
"""

from __future__ import annotations

import os
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio
from plotly.io.base_renderers import ExternalRenderer

# Width of the PNG image, in pixels. The height comes from the figure.
PNG_WIDTH = 1100

# Runs after plotly draws the figure on the page. ``m3PlotlyFigure`` is in
# ``_static/m3-plotly-figures.js``.
PAGE_SCRIPT = "m3PlotlyFigure(document.getElementById('{plot_id}'));"
TRANSPARENT = "rgba(0, 0, 0, 0)"


class GalleryRenderer(ExternalRenderer):
    """Keep each figure that ``fig.show()`` gets, in the order of the calls."""

    def __init__(self) -> None:
        self.figures: list[dict[str, Any]] = []

    def render(self, fig_dict: dict[str, Any]) -> None:
        self.figures.append(fig_dict)


renderer = GalleryRenderer()
pio.renderers["m3resp_gallery"] = renderer
pio.renderers.default = "m3resp_gallery"


def plotly_scraper(
    block: Any, block_vars: dict[str, Any], gallery_conf: dict[str, Any]
) -> str:
    """Write the figures of one code block and return the text for the page.

    Args:
        block (Any): The code block. Not used.
        block_vars (dict[str, Any]): The sphinx-gallery data of the tutorial.
            ``image_path_iterator`` gives the path of each new image.
        gallery_conf (dict[str, Any]): The sphinx-gallery configuration. Not
            used.

    Returns:
        str: One ``raw:: html`` directive for each figure, in reStructuredText.
    """

    rst = ""
    while renderer.figures:
        figure = go.Figure(renderer.figures.pop(0))
        png_path = next(block_vars["image_path_iterator"])
        figure.write_image(png_path, width=PNG_WIDTH, height=figure.layout.height)
        html_path = os.path.splitext(png_path)[0] + ".html"
        figure.update_layout(paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT)
        figure.write_html(
            html_path,
            include_plotlyjs="cdn",
            full_html=False,
            default_width="100%",
            post_script=PAGE_SCRIPT,
        )
        # The tutorial page is one folder above the images folder.
        gallery_dir = os.path.dirname(os.path.dirname(png_path))
        relative_path = os.path.relpath(html_path, gallery_dir).replace(os.sep, "/")
        rst += f"\n.. raw:: html\n    :file: {relative_path}\n\n"
    return rst
