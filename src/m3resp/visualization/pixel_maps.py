"""Matplotlib pictures of pixel maps and masks.

:func:`plot_pixel_map` draws a ``PixelMap`` or a ``PixelMask`` as an image,
with the colours of its style (:mod:`m3resp.visualization.pixel_map_style`)
and a colour bar. :func:`add_region_labels` writes the number of each region
of a map of region numbers (kind ``"integer"``) inside the region.

Both need matplotlib; it is imported when a function runs.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/plotting/pixelmap.py::PixelMapPlotting.imshow;
                PixelMapPlotting._create_colorbar;
                PixelMapPlotting.add_region_markers
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The methods `imshow` and `add_region_markers` of `PixelMap.plotting`
      are functions, `plot_pixel_map` and `add_region_labels`, that take the
      map or the mask.
    - The colour map and the norm are made from a `PixelMapStyle`; the
      argument `style` replaces the plot config of the map.
    - `colorbar_kwargs` is named `colorbar_options` and `facecolor` is
      named `background`.
    - `add_region_labels` takes a map of kind "integer" and raises TypeError
      for other maps (eitprocessing: NotImplementedError). The labels are
      whole numbers ("1", the values are stored as floats); `labels` maps a
      region number to a label. All `text_options` go to `Axes.text`
      (eitprocessing uses only `color`, `fontsize` and `bbox`). Region 0,
      the background, gets no label (eitprocessing labels it).
    - `contour` and `surface` are left out.
    - Docstrings are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from m3resp.data.pixel_maps import PixelMap, PixelMask
from m3resp.visualization.pixel_map_style import (
    PixelMapStyle,
    normalized_values,
    scale_limits,
    style_of,
)
from m3resp.visualization.plot_data import region_label_positions

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.image import AxesImage
    from matplotlib.text import Text

#: Box behind each region label: white, partly see-through.
_LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 2}


def plot_pixel_map(
    pixel_map: PixelMap | PixelMask,
    ax: Axes | None = None,
    *,
    colorbar: bool | None = None,
    normalize: bool | None = None,
    percentage: bool | None = None,
    absolute: bool | None = None,
    hide_axes: bool | None = None,
    background: Any = None,
    colorbar_options: Mapping[str, Any] | None = None,
    style: PixelMapStyle | None = None,
    **imshow_options: Any,
) -> AxesImage:
    """Draw a pixel map or mask as an image, with a colour bar.

    The colours, the scale and the colour bar follow the style of the map
    (its ``kind``) or of the mask. An argument that is None takes the value
    of the style.

    Args:
        pixel_map (PixelMap | PixelMask): The map or the mask.
        ax (Axes | None): The matplotlib axes to draw on. None draws on the
            current axes.
        colorbar (bool | None): Whether to add a colour bar.
        normalize (bool | None): Whether to divide the values by the largest
            value (or by 1 when the largest value is below 1) before they
            are drawn.
        percentage (bool | None): Whether the colour bar shows the values as
            a percentage (1 is 100 %).
        absolute (bool | None): Whether the colour bar shows the values
            without their sign.
        hide_axes (bool | None): Whether to hide the ticks and labels of the
            axes. The background stays visible.
        background (Any): Colour of the axes background, which shows through
            NaN pixels; any matplotlib colour.
        colorbar_options (Mapping | None): Options for
            ``matplotlib.pyplot.colorbar``, such as ``label``; they replace
            the label, ticks, number format and arrow ends of the style.
        style (PixelMapStyle | None): The style. None takes the style of the
            map kind, or the mask style.
        **imshow_options: Options for ``Axes.imshow``. ``cmap`` and ``norm``
            replace the colours and the scale of the style; ``vmin`` and
            ``vmax`` set the ends of the scale.

    Returns:
        AxesImage: The image. Its colour bar, if any, is ``image.colorbar``.

    Raises:
        TypeError: If ``pixel_map`` is not a ``PixelMap`` or a ``PixelMask``.
        ImportError: If matplotlib is not installed.
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    style = style_of(pixel_map) if style is None else style
    colorbar = style.colorbar if colorbar is None else colorbar
    normalize = style.normalize if normalize is None else normalize
    percentage = style.percentage if percentage is None else percentage
    absolute = style.absolute if absolute is None else absolute
    hide_axes = style.hide_axes if hide_axes is None else hide_axes

    values = pixel_map.values
    if normalize:
        values = normalized_values(values)

    options = dict(imshow_options)
    ax = plt.gca() if ax is None else ax
    options.setdefault("cmap", matplotlib_colormap(style))
    norm = options.setdefault("norm", matplotlib_norm(style))
    if isinstance(norm, Normalize):
        vmin = options.pop("vmin", None)
        vmax = options.pop("vmax", None)
        if vmin is not None:
            norm.vmin = vmin
        if vmax is not None:
            norm.vmax = vmax
        if style.scale == "from_zero" and norm.vmin == 0 and norm.vmax is None:
            low, high = scale_limits(style, values)
            if low < 0:
                # Only negative values: matplotlib would take the largest
                # value as the high end, below the low end of 0.
                norm.vmin, norm.vmax = low, high
    image = ax.imshow(values, **options)

    if colorbar:
        _add_colorbar(style, image, ax, percentage, absolute, colorbar_options)
    if hide_axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    ax.set(facecolor=style.background if background is None else background)
    return image


def matplotlib_colormap(style: PixelMapStyle) -> Colormap:
    """Return the matplotlib colour map of a style.

    Args:
        style (PixelMapStyle): The style.

    Returns:
        Colormap: A ``ListedColormap`` with the colour table, and the colours
            below and above the scale and for NaN.

    Raises:
        ImportError: If matplotlib is not installed.
    """

    from matplotlib.colors import ListedColormap

    colormap = ListedColormap(list(style.colors), name=style.name)
    colormap.set_under(style.under_color)
    colormap.set_over(style.over_color)
    colormap.set_bad(style.bad_color)
    return colormap


def matplotlib_norm(style: PixelMapStyle) -> Normalize:
    """Return a new matplotlib norm for the scale rule of a style.

    Args:
        style (PixelMapStyle): The style.

    Returns:
        Normalize: ``Normalize`` (``"linear"``), ``Normalize(vmin=0)``
            (``"from_zero"``), ``CenteredNorm(vcenter=0)`` with the half
            range (``"centered"``), or ``BoundaryNorm`` with the boundaries
            (``"whole_numbers"`` clips values outside to the first and the
            last colour; ``"mask"`` does not).

    Raises:
        ImportError: If matplotlib is not installed.
    """

    from matplotlib.colors import BoundaryNorm, CenteredNorm, Normalize

    if style.scale == "from_zero":
        return Normalize(vmin=0)
    if style.scale == "centered":
        return CenteredNorm(vcenter=0, halfrange=style.half_range)
    if style.boundaries is not None:
        return BoundaryNorm(
            list(style.boundaries),
            ncolors=len(style.colors),
            extend="neither",
            clip=style.scale == "whole_numbers",
        )
    return Normalize()


def _add_colorbar(
    style: PixelMapStyle,
    image: AxesImage,
    ax: Axes,
    percentage: bool,
    absolute: bool,
    colorbar_options: Mapping[str, Any] | None,
) -> Any:
    """Add the colour bar of the style next to the image."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from m3resp.visualization._tick_formats import (
        AbsolutePercentFormatter,
        AbsoluteScalarFormatter,
    )

    options: dict[str, Any] = {}
    if style.colorbar_label is not None:
        options["label"] = style.colorbar_label
    if style.colorbar_ticks is not None:
        options["ticks"] = list(style.colorbar_ticks)
    options.update(colorbar_options or {})
    if "format" not in options:
        if absolute and percentage:
            options["format"] = AbsolutePercentFormatter(xmax=1, decimals=0)
        elif percentage:
            options["format"] = PercentFormatter(xmax=1, decimals=0)
        elif absolute:
            options["format"] = AbsoluteScalarFormatter()
    options.setdefault("extend", style.extend)
    return plt.colorbar(image, ax=ax, **options)


def add_region_labels(
    regions: PixelMap,
    ax: Axes,
    labels: Mapping[int, str] | None = None,
    **text_options: Any,
) -> list[Text]:
    """Write the number of each region inside the region.

    Each label sits at the centre of mass of its region, in a white box. A
    region made of separate parts gets its label in the largest part. When
    the centre of mass lies on the edge or outside the region (a C shape,
    for example), the label goes to the pixel farthest from the edge.
    Region 0 is the background and gets no label.

    Args:
        regions (PixelMap): A map of kind ``"integer"`` with one whole number
            per region, such as the capture ``"watershed regions"`` of
            :func:`~m3resp.eit.roi.watershed_lungspace`.
        ax (Axes): The axes the map is drawn on.
        labels (Mapping[int, str] | None): The text for some or all region
            numbers. A region without a text gets its number.
        **text_options: Options for ``Axes.text``. The defaults are
            ``color="black"``, ``fontsize=10``, centred text, and the white
            box; a ``bbox`` dictionary changes parts of the box.

    Returns:
        list[Text]: The labels, in the order of the region numbers.

    Raises:
        TypeError: If ``regions`` is not a ``PixelMap`` of kind ``"integer"``.
    """

    if not isinstance(regions, PixelMap) or regions.kind != "integer":
        raise TypeError(
            "add_region_labels needs a PixelMap of kind 'integer' (region "
            f"numbers), got {_describe(regions)}."
        )
    labels = labels or {}
    options = dict(text_options)
    box = _LABEL_BOX | options.pop("bbox", {})
    options.setdefault("color", "black")
    options.setdefault("fontsize", 10)
    options.setdefault("ha", "center")
    options.setdefault("va", "center")

    texts = []
    for number, (row, column) in region_label_positions(regions).items():
        text = labels.get(number, str(number))
        texts.append(ax.text(column, row, text, bbox=box, **options))
    return texts


def _describe(item: Any) -> str:
    """Type and kind of an object, for an error message."""

    if isinstance(item, PixelMap):
        return f"a PixelMap of kind {item.kind!r}"
    return type(item).__name__


__all__ = [
    "add_region_labels",
    "matplotlib_colormap",
    "matplotlib_norm",
    "plot_pixel_map",
]
