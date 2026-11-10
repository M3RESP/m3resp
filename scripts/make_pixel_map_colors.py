"""Write the colour tables of the pixel map styles to a data file.

``m3resp.visualization.pixel_map_style`` reads the colours of each pixel map
style from ``src/m3resp/visualization/pixel_map_colors.json``, so it gives the
same colours to matplotlib figures and to the GUI without matplotlib. This
script makes that file: it builds the colour maps of eitprocessing 1.8.7
(``plotting/pixelmap.py``) with matplotlib and stores, for each one, the
colour of every step (red, green and blue from 0 to 1), the colours for
values below and above the scale and for NaN, and the background colour of
the axes.

Run it again after a matplotlib update and check the test
``tests/test_pixel_map_style.py``.

Usage: ``python scripts/make_pixel_map_colors.py [--output FILE]``
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib as mpl
import numpy as np
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    to_rgba,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(
    REPO_ROOT, "src", "m3resp", "visualization", "pixel_map_colors.json"
)

#: Colour of the axes background, which shows through NaN pixels.
BACKGROUND = "darkgrey"


def _tiv_colors() -> Colormap:
    """Reversed ``Blues`` with purple below the scale (eitprocessing TIV map)."""

    colormap = mpl.colormaps["Blues"].reversed()
    colormap.set_under("purple")
    return colormap


def color_maps() -> dict[str, Colormap]:
    """Return the colour maps of the pixel map styles, by name.

    Returns:
        dict[str, Colormap]: One matplotlib colour map per name, made as in
            the plot settings of eitprocessing 1.8.7.
    """

    return {
        "viridis": mpl.colormaps["viridis"],
        "blues_reversed": _tiv_colors(),
        "odcl": LinearSegmentedColormap.from_list(
            "ODCL", ["white", "black", "darkorange"]
        ),
        "vanimo": mpl.colormaps["vanimo"],
        "perfusion": LinearSegmentedColormap.from_list("Perfusion", ["black", "red"]),
        "pendelluft": LinearSegmentedColormap.from_list(
            "Pendelluft", ["black", "forestgreen"]
        ),
        "signed_pendelluft": LinearSegmentedColormap.from_list(
            "SignedPendelluft", ["deeppink", "black", "forestgreen"]
        ),
        "tab20": mpl.colormaps["tab20"],
        "mask": ListedColormap(["tab:blue"]),
    }


def color_table(colormap: Colormap) -> dict[str, list]:
    """Return the colours of one colour map as lists of numbers.

    Args:
        colormap (Colormap): A matplotlib colour map.

    Returns:
        dict[str, list]: ``"colors"`` (red, green, blue of each step),
            ``"under"``, ``"over"`` and ``"bad"`` (red, green, blue, alpha).

    Raises:
        ValueError: If a step of the colour map is partly transparent.
    """

    steps = colormap(np.arange(colormap.N))
    if not np.all(steps[:, 3] == 1):
        raise ValueError(f"Colour map {colormap.name!r} has transparent steps.")
    return {
        "colors": steps[:, :3].tolist(),
        "under": [float(v) for v in colormap.get_under()],
        "over": [float(v) for v in colormap.get_over()],
        "bad": [float(v) for v in colormap.get_bad()],
    }


def main() -> None:
    """Write the colour tables to the output file."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    output = os.path.abspath(parser.parse_args().output)
    data = {
        "matplotlib_version": mpl.__version__,
        "background": [float(v) for v in to_rgba(BACKGROUND)],
        "color_maps": {
            name: color_table(colormap) for name, colormap in color_maps().items()
        },
    }
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(_as_text(data))
    print(f"Wrote {output}")


def _as_text(data: dict) -> str:
    """JSON text with one colour per line, so a change shows per colour."""

    lines = [
        "{",
        f' "matplotlib_version": {json.dumps(data["matplotlib_version"])},',
        f' "background": {json.dumps(data["background"])},',
        ' "color_maps": {',
    ]
    names = list(data["color_maps"])
    for name in names:
        table = data["color_maps"][name]
        colors = ",\n".join(f"    {json.dumps(color)}" for color in table["colors"])
        lines += [
            f"  {json.dumps(name)}: {{",
            f'   "under": {json.dumps(table["under"])},',
            f'   "over": {json.dumps(table["over"])},',
            f'   "bad": {json.dumps(table["bad"])},',
            '   "colors": [',
            colors,
            "   ]",
            "  }" + ("," if name != names[-1] else ""),
        ]
    lines += [" }", "}", ""]
    text = "\n".join(lines)
    if json.loads(text) != data:
        raise RuntimeError("The colour table text does not read back the same.")
    return text


if __name__ == "__main__":
    main()
