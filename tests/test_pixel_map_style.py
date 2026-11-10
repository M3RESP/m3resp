"""Pixel map styles of `m3resp.visualization.pixel_map_style`.

Each style must equal the plot config of eitprocessing 1.8.7 for its kind:
the colours of every step of the colour map, the colours below and above the
scale and for NaN, the background colour, the colour bar label and ticks, and
the switches `colorbar`, `normalize`, `percentage`, `absolute` and
`hide_axes`. On seeded maps (NaN pixels, negative values, zeros, flat and
all-NaN maps; binary, weighted and all-NaN masks), `scale_limits` must equal
the norm limits of eitprocessing `imshow`, and `pixel_colors` the colours of
the image (`AxesImage.to_rgba`), with and without `normalize`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from pixel_map_cases import STYLES, seeded_grids, upstream_classes, upstream_object

from m3resp.data.pixel_maps import PixelMap, PixelMask
from m3resp.visualization.pixel_map_style import (
    MASK,
    PixelMapStyle,
    normalized_values,
    pixel_colors,
    pixel_map_style,
    scale_limits,
    style_of,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _upstream_config(kind):
    plotting = pytest.importorskip("eitprocessing.plotting")
    return plotting.get_plot_config(upstream_classes()[kind])


@pytest.fixture(scope="module")
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot


@pytest.mark.parametrize("kind", STYLES, ids=str)
def test_the_colours_equal_the_colour_map_of_eitprocessing(kind):
    import matplotlib as mpl

    config = _upstream_config(kind)
    cmap = config.cmap
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    style = pixel_map_style(kind)

    steps = cmap(np.arange(cmap.N))
    assert np.array_equal(np.asarray(style.colors), steps[:, :3])
    assert np.all(steps[:, 3] == 1)
    assert np.array_equal(style.under_color, cmap.get_under())
    assert np.array_equal(style.over_color, cmap.get_over())
    assert np.array_equal(style.bad_color, cmap.get_bad())


@pytest.mark.parametrize("kind", STYLES, ids=str)
def test_the_settings_equal_the_plot_config_of_eitprocessing(kind):
    from matplotlib.colors import to_rgba

    config = _upstream_config(kind)
    style = pixel_map_style(kind)

    assert style.background == to_rgba(config.facecolor)
    assert style.colorbar is config.colorbar
    assert style.normalize is config.normalize
    assert style.percentage is config.percentage
    assert style.absolute is config.absolute
    assert style.hide_axes is config.hide_axes
    assert style.colorbar_label == config.colorbar_kwargs.get("label")
    ticks = config.colorbar_kwargs.get("ticks")
    if ticks is None:
        assert style.colorbar_ticks is None
    else:
        assert np.array_equal(style.colorbar_ticks, ticks)
    assert dict(config.extra_kwargs) == {}


@pytest.mark.parametrize("kind", STYLES, ids=str)
def test_the_scale_rule_equals_the_norm_of_eitprocessing(kind):
    from matplotlib.colors import BoundaryNorm, CenteredNorm, Normalize

    norm = _upstream_config(kind).norm
    style = pixel_map_style(kind)
    if style.scale == "linear":
        assert norm == "linear"
    elif style.scale == "from_zero":
        assert type(norm) is Normalize
        assert (norm.vmin, norm.vmax) == (0, None)
    elif style.scale == "centered":
        assert isinstance(norm, CenteredNorm)
        assert norm.vcenter == 0
        assert norm.halfrange == style.half_range
    else:
        assert isinstance(norm, BoundaryNorm)
        assert np.array_equal(norm.boundaries, style.boundaries)
        assert norm.Ncmap == len(style.colors)
        assert norm.clip is (style.scale == "whole_numbers")
        assert norm.extend == "neither"


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("kind", STYLES, ids=str)
def test_scale_ends_and_colours_equal_imshow_of_eitprocessing(plt, kind, normalize):
    style = pixel_map_style(kind)
    for case, grid in seeded_grids(kind).items():
        upstream = upstream_object(kind, grid)
        values = np.asarray(upstream.values, dtype=float)
        if normalize:
            values = normalized_values(values)
        fig, ax = plt.subplots()
        image = upstream.plotting.imshow(ax=ax, colorbar=False, normalize=normalize)
        try:
            expected_colors = image.to_rgba(image.get_array())
        except ValueError:
            # A scale from 0 with only negative values: matplotlib refuses it
            # without a colour bar. With a colour bar it swaps the two ends.
            image.norm.vmin, image.norm.vmax = image.norm.vmax, image.norm.vmin
            expected_colors = image.to_rgba(image.get_array())
        limits = (image.norm.vmin, image.norm.vmax)
        assert scale_limits(style, values) == limits, case
        assert np.array_equal(pixel_colors(style, values), expected_colors), case
        plt.close(fig)


def test_normalized_values_divide_by_the_largest_value_or_by_1():
    grid = np.array([[2.0, np.nan], [4.0, -1.0]])
    expected = np.array([[0.5, np.nan], [1.0, -0.25]])
    assert np.array_equal(normalized_values(grid), expected, equal_nan=True)
    small = np.array([[0.2, 0.5]])
    assert np.array_equal(normalized_values(small), small)


def test_pixel_colors_with_given_limits_and_outside_values():
    style = pixel_map_style("tiv")
    values = np.array([[-1.0, 0.0, 10.0, 20.0, np.nan]])
    colors = pixel_colors(style, values, (0, 10))
    assert np.array_equal(colors[0, 0], style.under_color)
    assert np.array_equal(colors[0, 1], (*style.colors[0], 1.0))
    assert np.array_equal(colors[0, 2], (*style.colors[-1], 1.0))
    assert np.array_equal(colors[0, 3], style.over_color)
    assert np.array_equal(colors[0, 4], style.bad_color)


def test_arrow_ends_and_number_formats():
    assert pixel_map_style("tiv").extend == "min"
    assert pixel_map_style(None).extend == "neither"
    assert pixel_map_style("odcl").colorbar_format == "absolute percent"
    assert pixel_map_style("signed_pendelluft").colorbar_format == "absolute"
    assert pixel_map_style("tiv", percentage=True).colorbar_format == "percent"
    assert pixel_map_style("tiv").colorbar_format == "number"


def test_pixel_map_style_takes_changes_and_refuses_unknown_kinds():
    changed = pixel_map_style("tiv", colorbar_label="TIV (mL)")
    assert changed.colorbar_label == "TIV (mL)"
    assert pixel_map_style("tiv").colorbar_label == "TIV (a.u.)"
    with pytest.raises(ValueError, match="pixel_map_style needs"):
        pixel_map_style("pressure")
    with pytest.raises(TypeError):
        pixel_map_style("tiv", colour="red")


def test_style_of_maps_and_masks():
    tiv_map = PixelMap(name="tiv", values=np.ones((2, 2)), kind="tiv")
    mask = PixelMask(name="mask", values=np.ones((2, 2)))
    assert style_of(tiv_map).name == "tiv"
    assert style_of(PixelMap(name="plain", values=np.ones((2, 2)))).name == "plain"
    assert style_of(mask).name == MASK
    with pytest.raises(TypeError, match="PixelMap or a PixelMask"):
        style_of(np.ones((2, 2)))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"scale": "log"}, "scale must be one of"),
        ({"colors": ()}, "at least one colour"),
        ({"colors": ((0.0, 0.0),)}, "3 numbers"),
        ({"under_color": (0.0, 0.0, 0.0)}, "under_color needs 4 numbers"),
        ({"boundaries": (0.0, 1.0)}, "boundaries is needed"),
        ({"half_range": 2.0}, "half_range is only"),
    ],
)
def test_pixel_map_style_refuses_wrong_fields(changes, message):
    with pytest.raises(ValueError, match=message):
        pixel_map_style("tiv", **changes)


def test_boundaries_must_fit_the_colours():
    with pytest.raises(ValueError, match="one more value"):
        pixel_map_style("integer", boundaries=(0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="one more value"):
        pixel_map_style(MASK, boundaries=(2.0, 0.0))


@pytest.mark.parametrize("kind", STYLES, ids=str)
def test_to_dict_is_json_and_gives_the_style_back(kind):
    style = pixel_map_style(kind)
    data = json.loads(json.dumps(style.to_dict(), allow_nan=False))
    assert data["extend"] == style.extend
    assert data["colorbar_format"] == style.colorbar_format
    fields = {
        key: tuple(tuple(v) if isinstance(v, list) else v for v in value)
        if isinstance(value, list)
        else value
        for key, value in data.items()
        if key not in ("extend", "colorbar_format")
    }
    assert PixelMapStyle(**fields) == style


def test_the_colour_file_matches_the_script():
    """The stored colour tables equal what the script makes now."""

    pytest.importorskip("matplotlib")
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    try:
        import make_pixel_map_colors as script
    finally:
        sys.path.pop(0)
    path = os.path.join(
        REPO_ROOT, "src", "m3resp", "visualization", "pixel_map_colors.json"
    )
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    assert set(stored["color_maps"]) == set(script.color_maps())
    for name, colormap in script.color_maps().items():
        assert stored["color_maps"][name] == script.color_table(colormap), name


def test_the_style_modules_load_no_matplotlib_and_no_eitprocessing():
    code = (
        "import sys\n"
        "import m3resp.visualization.pixel_map_style\n"
        "import m3resp.visualization.plot_data\n"
        "loaded = [name for name in sys.modules\n"
        "          if name.split('.')[0] in ('matplotlib', 'eitprocessing')]\n"
        "print(loaded)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
