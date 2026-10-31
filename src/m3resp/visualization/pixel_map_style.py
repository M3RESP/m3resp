"""Colours and colour scale of each kind of pixel map.

A :class:`PixelMapStyle` says how a pixel map is drawn: the colour table, the
colours for values below and above the scale and for NaN pixels, the rule
that sets the ends of the colour scale, the background colour, and the colour
bar. :func:`pixel_map_style` gives the style of each ``PixelMap.kind``, of a
plain map (kind None) and of a ``PixelMask`` (``"mask"``):

========================  ====================================  ==============
Style                     Colours                               Scale
========================  ====================================  ==============
None (plain map)          ``viridis``                           data range
``"tiv"``                 reversed ``Blues`` (dark blue = 0,    from 0
                          white = most), purple below 0
``"amplitude"``           as ``"tiv"``                          from 0
``"odcl"``                white - black - dark orange           centred on 0,
                                                                -1 to 1
``"difference"``          ``vanimo``                            centred on 0
``"perfusion"``           black to red                          from 0
``"pendelluft"``          black to forest green                 from 0
``"signed_pendelluft"``   deep pink - black - forest green      centred on 0
``"integer"``             ``tab20`` (20 colours)                one colour per
                                                                whole number
``"mask"``                one colour (``tab:blue``)             0 to 2
========================  ====================================  ==============

The colour tables come from matplotlib, stored once in
``pixel_map_colors.json`` by ``scripts/make_pixel_map_colors.py``. This
module reads that file, so the PNG figures
(``m3resp.visualization.pixel_maps``) and the GUI plot data
(``m3resp.visualization.plot_data``) use the same colours.

:func:`scale_limits` gives the two ends of the colour scale for a grid of
values, and :func:`pixel_colors` gives the colour of each pixel, with the
rules of matplotlib.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/plotting/pixelmap.py::PixelMapPlotConfig;
                TIVMapPlotConfig; AmplitudeMapPlotConfig; ODCLMapPlotConfig;
                DifferenceMapPlotConfig; PerfusionMapPlotConfig;
                PendelluftMapPlotConfig; SignedPendelluftMapPlotConfig;
                IntegerMapPlotConfig; PixelMaskPlotConfig;
                PixelMapPlotting.imshow (the division by the largest value)
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The 10 config classes are one class, `PixelMapStyle`, with one style
      per `PixelMap.kind`, for kind None and for masks
      (`pixel_map_style`).
    - The colour map is a table of colours (read from a data file) with the
      colours below and above the scale and for NaN; the norm is a scale
      rule ("linear", "from_zero", "centered", "whole_numbers", "mask")
      with a half range or boundaries.
    - The colour bar label and ticks are fields (`colorbar_label`,
      `colorbar_ticks`); `extra_kwargs` is left out.
    - The registry that changes the settings of a kind for the whole program
      (`plotting/__init__.py`) is left out; a figure call takes a changed
      style.
    - `scale_limits` and `pixel_colors` give the scale ends and the pixel
      colours that matplotlib gives with the colour map and the norm.
    - A map with the scale "from_zero" and only negative values gets the
      scale from its largest value to 0. In eitprocessing the scale goes
      from 0 down to the largest value: matplotlib raises ValueError, or,
      with a colour bar, turns the scale around and widens it by 10 %.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np

from m3resp.data.pixel_maps import PIXEL_MAP_KINDS, PixelMap, PixelMask

#: Name of the style of a ``PixelMask``.
MASK = "mask"

#: The rules for the two ends of the colour scale:
#:
#: - ``"linear"``: smallest to largest value of the map.
#: - ``"from_zero"``: 0 to the largest value, or the largest value to 0
#:   when every value is below 0.
#: - ``"centered"``: centred on 0; from ``-half_range`` to ``half_range``,
#:   or, with ``half_range`` None, out to the value farthest from 0.
#: - ``"whole_numbers"``: one colour per whole number, between
#:   ``boundaries``; values outside get the first or the last colour.
#: - ``"mask"``: one colour for the values between the two ``boundaries``.
SCALE_RULES = ("linear", "from_zero", "centered", "whole_numbers", "mask")

ScaleRule = Literal["linear", "from_zero", "centered", "whole_numbers", "mask"]
Color = tuple[float, float, float, float]

_COLOR_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pixel_map_colors.json"
)


@dataclass(frozen=True)
class PixelMapStyle:
    """How a pixel map or mask is drawn.

    Colours are red, green, blue (and alpha, the opacity) from 0 to 1.

    Attributes:
        name: The style: a ``PixelMap.kind``, ``"plain"`` for kind None, or
            ``"mask"``.
        colors: The colour table, from the low end of the scale to the high
            end: red, green, blue of each step.
        under_color: Colour of values below the low end of the scale.
        over_color: Colour of values above the high end of the scale.
        bad_color: Colour of NaN pixels; (0, 0, 0, 0) is transparent, so the
            background shows.
        scale: The rule for the ends of the colour scale, one of
            :data:`SCALE_RULES`.
        background: Colour of the axes background.
        half_range: For ``scale="centered"``: half the width of the scale, or
            None to take the value farthest from 0.
        boundaries: For ``scale="whole_numbers"`` and ``"mask"``: the edges
            between the colours, one more than the number of colours.
        colorbar: Whether a figure shows a colour bar.
        colorbar_label: Label of the colour bar, or None.
        colorbar_ticks: Values with a tick on the colour bar, or None for
            the ticks that matplotlib picks.
        normalize: Whether the values are divided by the largest value
            before they are drawn (see :func:`normalized_values`).
        percentage: Whether the colour bar shows the values as a percentage
            (1 is 100 %).
        absolute: Whether the colour bar shows the values without their sign.
        hide_axes: Whether a figure hides the ticks and labels of the axes.

    Raises:
        ValueError: If ``scale`` is unknown, ``colors`` is empty, a colour
            does not have 3 numbers (4 for the other colours), or
            ``boundaries`` or ``half_range`` does not fit ``scale`` and
            ``colors``.
    """

    name: str
    colors: tuple[tuple[float, float, float], ...]
    under_color: Color
    over_color: Color
    bad_color: Color
    scale: ScaleRule
    background: Color
    half_range: float | None = None
    boundaries: tuple[float, ...] | None = None
    colorbar: bool = True
    colorbar_label: str | None = None
    colorbar_ticks: tuple[float, ...] | None = None
    normalize: bool = False
    percentage: bool = False
    absolute: bool = False
    hide_axes: bool = True

    def __post_init__(self) -> None:
        """Check the scale rule, the colours and the boundaries."""

        if self.scale not in SCALE_RULES:
            raise ValueError(
                f"PixelMapStyle.scale must be one of {SCALE_RULES}, got {self.scale!r}."
            )
        if len(self.colors) == 0:
            raise ValueError("PixelMapStyle.colors needs at least one colour.")
        if any(len(color) != 3 for color in self.colors):
            raise ValueError("Each colour of PixelMapStyle.colors needs 3 numbers.")
        for field_name in ("under_color", "over_color", "bad_color", "background"):
            if len(getattr(self, field_name)) != 4:
                raise ValueError(f"PixelMapStyle.{field_name} needs 4 numbers.")
        needs_boundaries = self.scale in ("whole_numbers", "mask")
        if needs_boundaries != (self.boundaries is not None):
            raise ValueError(
                "PixelMapStyle.boundaries is needed for the scales "
                "'whole_numbers' and 'mask', and only for these."
            )
        if self.boundaries is not None:
            edges = np.asarray(self.boundaries, dtype=float)
            if len(edges) != len(self.colors) + 1 or np.any(np.diff(edges) <= 0):
                raise ValueError(
                    "PixelMapStyle.boundaries must rise and hold one more "
                    f"value than the {len(self.colors)} colour(s)."
                )
        if self.half_range is not None and self.scale != "centered":
            raise ValueError("PixelMapStyle.half_range is only for scale='centered'.")

    @property
    def extend(self) -> Literal["neither", "min", "max", "both"]:
        """Ends of the colour bar with an arrow.

        An end has an arrow when the colour below (or above) the scale
        differs from the first (or last) colour of the table.
        """

        low = tuple(self.under_color) != (*self.colors[0], 1.0)
        high = tuple(self.over_color) != (*self.colors[-1], 1.0)
        if low and high:
            return "both"
        if low:
            return "min"
        if high:
            return "max"
        return "neither"

    @property
    def colorbar_format(
        self,
    ) -> Literal["number", "percent", "absolute", "absolute percent"]:
        """How the colour bar writes its numbers.

        ``"percent"``: 0.5 is "50%", with no decimals. ``"absolute"``: -2 is
        "2". ``"absolute percent"``: -0.5 is "50%". ``"number"``: as is.
        """

        if self.absolute and self.percentage:
            return "absolute percent"
        if self.percentage:
            return "percent"
        if self.absolute:
            return "absolute"
        return "number"

    def to_dict(self) -> dict[str, Any]:
        """Return the style as numbers, text, lists and None.

        Returns:
            dict[str, Any]: All fields, with tuples as lists, and the
                properties ``"extend"`` and ``"colorbar_format"``.
        """

        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, tuple):
                data[key] = [
                    list(item) if isinstance(item, tuple) else item for item in value
                ]
        data["extend"] = self.extend
        data["colorbar_format"] = self.colorbar_format
        return data


@functools.cache
def _color_tables() -> dict[str, Any]:
    """The colour tables of ``pixel_map_colors.json``."""

    with open(_COLOR_FILE, encoding="utf-8") as handle:
        return json.load(handle)


def _style(name: str, color_map: str, scale: ScaleRule, **fields: Any) -> PixelMapStyle:
    """A style with the colours of one table of ``pixel_map_colors.json``."""

    tables = _color_tables()
    table = tables["color_maps"][color_map]
    return PixelMapStyle(
        name=name,
        colors=tuple(tuple(color) for color in table["colors"]),
        under_color=tuple(table["under"]),
        over_color=tuple(table["over"]),
        bad_color=tuple(table["bad"]),
        scale=scale,
        background=tuple(tables["background"]),
        **fields,
    )


@functools.cache
def _styles() -> dict[str | None, PixelMapStyle]:
    """The style of each kind, of kind None and of masks."""

    return {
        None: _style("plain", "viridis", "linear"),
        "tiv": _style(
            "tiv", "blues_reversed", "from_zero", colorbar_label="TIV (a.u.)"
        ),
        "amplitude": _style(
            "amplitude",
            "blues_reversed",
            "from_zero",
            colorbar_label="Amplitude (a.u.)",
        ),
        "odcl": _style(
            "odcl",
            "odcl",
            "centered",
            half_range=1.0,
            percentage=True,
            absolute=True,
            colorbar_label="Collapse/Overdistention (%)",
        ),
        "difference": _style(
            "difference", "vanimo", "centered", colorbar_label="Difference"
        ),
        "perfusion": _style(
            "perfusion", "perfusion", "from_zero", colorbar_label="Perfusion"
        ),
        "pendelluft": _style(
            "pendelluft", "pendelluft", "from_zero", colorbar_label="Pendelluft"
        ),
        "signed_pendelluft": _style(
            "signed_pendelluft",
            "signed_pendelluft",
            "centered",
            absolute=True,
            colorbar_label="Pendelluft",
        ),
        "integer": _style(
            "integer",
            "tab20",
            "whole_numbers",
            boundaries=tuple(float(edge) for edge in np.arange(-0.5, 20.5, 1)),
            colorbar_label="Label",
            colorbar_ticks=tuple(float(tick) for tick in range(0, 20, 3)),
        ),
        MASK: _style(
            MASK,
            "mask",
            "mask",
            boundaries=(0.0, 2.0),
            colorbar=False,
            colorbar_label="Mask",
        ),
    }


def pixel_map_style(kind: str | None = None, **changes: Any) -> PixelMapStyle:
    """Return the style of a kind of pixel map.

    Args:
        kind (str | None): A ``PixelMap.kind`` (see
            :data:`~m3resp.data.pixel_maps.PIXEL_MAP_KINDS`), None for a
            plain map, or ``"mask"`` for a ``PixelMask``.
        **changes: Fields of :class:`PixelMapStyle` to change, for example
            ``colorbar_label="TIV (mL)"``.

    Returns:
        PixelMapStyle: The style.

    Raises:
        ValueError: If ``kind`` is unknown, or a changed field is not valid.
        TypeError: If a name in ``changes`` is not a field.
    """

    styles = _styles()
    if kind not in styles:
        raise ValueError(
            f"pixel_map_style needs None, 'mask' or one of {PIXEL_MAP_KINDS}, "
            f"got {kind!r}."
        )
    return replace(styles[kind], **changes) if changes else styles[kind]


def style_of(item: PixelMap | PixelMask) -> PixelMapStyle:
    """Return the style of a ``PixelMap`` (by its kind) or of a ``PixelMask``.

    Args:
        item (PixelMap | PixelMask): The map or the mask.

    Returns:
        PixelMapStyle: The style.

    Raises:
        TypeError: If ``item`` is not a ``PixelMap`` or a ``PixelMask``.
    """

    if isinstance(item, PixelMask):
        return pixel_map_style(MASK)
    if isinstance(item, PixelMap):
        return pixel_map_style(item.kind)
    raise TypeError(f"Expected a PixelMap or a PixelMask, got {type(item).__name__}.")


def normalized_values(values: Any) -> np.ndarray:
    """Return the values divided by their largest value, as eitprocessing does.

    The divisor is the largest value or 1, whichever is larger, so a map
    whose largest value is below 1 keeps its values. NaN pixels stay NaN.

    Args:
        values (Any): A 2D grid of numbers.

    Returns:
        numpy.ndarray: The divided values, as floats.
    """

    grid = np.asarray(values, dtype=float)
    return grid / np.nanmax(grid, initial=1)


def scale_limits(style: PixelMapStyle, values: Any) -> tuple[float, float]:
    """Return the two ends of the colour scale for a grid of values.

    The ends follow the scale rule of the style (see :data:`SCALE_RULES`) and
    leave out NaN pixels, as matplotlib sets them for ``imshow``. An end
    that depends on the values is 0 for a grid that is NaN everywhere, as in
    matplotlib. With the scale ``"from_zero"`` and only negative values, the
    scale goes from the largest value to 0.

    Args:
        style (PixelMapStyle): The style.
        values (Any): The values that are drawn, a grid of numbers.

    Returns:
        tuple[float, float]: The low end and the high end.
    """

    grid = np.asarray(values, dtype=float)
    finite = grid[~np.isnan(grid)]
    if style.boundaries is not None:
        return float(style.boundaries[0]), float(style.boundaries[-1])
    if style.scale == "centered":
        if style.half_range is not None:
            half = abs(float(style.half_range))
        elif finite.size:
            half = float(max(-finite.min(), finite.max()))
        else:
            half = 0.0
        return 0.0 - half, half
    high = float(finite.max()) if finite.size else 0.0
    if style.scale == "from_zero":
        return min(high, 0.0), max(high, 0.0)
    low = float(finite.min()) if finite.size else 0.0
    return low, high


def pixel_colors(
    style: PixelMapStyle,
    values: Any,
    limits: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return the colour of each pixel, as matplotlib draws it.

    A value is placed on the scale between the two ends and gets the colour
    of that step of the table. Values below or above the scale get
    ``under_color`` or ``over_color`` (for ``scale="whole_numbers"``: the
    first or the last colour), and NaN pixels get ``bad_color``. When both
    ends are the same, every value gets the first colour.

    Args:
        style (PixelMapStyle): The style.
        values (Any): The values that are drawn, a grid of numbers.
        limits (tuple | None): The low end and the high end of the scale.
            None takes :func:`scale_limits`. The scales ``"whole_numbers"``
            and ``"mask"`` always use ``boundaries``.

    Returns:
        numpy.ndarray: Red, green, blue and alpha of each pixel, from 0 to 1,
            with shape ``values.shape + (4,)``.

    Raises:
        ValueError: If the low end is above the high end.
    """

    grid = np.asarray(values, dtype=float)
    bad = np.isnan(grid)
    count = len(style.colors)
    lookup = np.vstack(
        [
            np.column_stack([np.asarray(style.colors), np.ones(count)]),
            style.under_color,
            style.over_color,
            style.bad_color,
        ]
    )
    under, over, nan = count, count + 1, count + 2

    if style.boundaries is not None:
        index = _boundary_index(style, grid)
    else:
        low, high = scale_limits(style, grid) if limits is None else limits
        if low > high:
            raise ValueError("The low end of the scale must not be above the high end.")
        if low == high:
            position = np.zeros(grid.shape)
        else:
            position = (grid - low) / (high - low)
        steps = position * count
        steps[steps == count] = count - 1
        with np.errstate(invalid="ignore"):
            index = np.where(bad, 0, steps).astype(int)
        index[steps < 0] = under
        index[steps >= count] = over
    index[bad] = nan
    return lookup[index]


def _boundary_index(style: PixelMapStyle, grid: np.ndarray) -> np.ndarray:
    """Colour index of each pixel for the scales with boundaries.

    Values below the first edge get the index of ``under_color`` and values
    from the last edge up get the index of ``over_color``; for
    ``"whole_numbers"`` they get the first and the last colour instead.
    """

    count = len(style.colors)
    edges = np.asarray(style.boundaries, dtype=float)
    low, high = edges[0], edges[-1]
    clip = style.scale == "whole_numbers"
    filled = np.where(np.isnan(grid), high + 1, grid)
    if clip:
        filled = np.clip(filled, low, high)
    index = np.digitize(filled, edges) - 1
    index[filled < low] = count
    index[filled >= high] = count - 1 if clip else count + 1
    return index
