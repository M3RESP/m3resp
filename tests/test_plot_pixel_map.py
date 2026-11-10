"""Figures of `m3resp.visualization.pixel_maps`: `plot_pixel_map`, `add_region_labels`.

`plot_pixel_map` must draw what eitprocessing 1.8.7 `PixelMap.plotting.imshow`
draws, on the seeded maps and masks of `test_pixel_map_style.py`, for every
style and with the flags `colorbar`, `normalize`, `percentage`, `absolute`,
`hide_axes` and `facecolor` set: the image array, the norm limits, the colours
(`AxesImage.to_rgba`), the colour bar label, ticks, tick texts and arrow ends,
the background colour and the hidden axes. `add_region_labels` must place
the same texts at the same points as `add_region_markers`, on the watershed
regions of the 3 golden test files and on seeded region maps.
"""

from __future__ import annotations

import itertools
import json
import os
import warnings

import numpy as np
import pytest
from pixel_map_cases import (
    STYLES,
    seeded_grids,
    upstream_classes,
    upstream_object,
)

from m3resp.data.pixel_maps import PixelMap, PixelMask
from m3resp.visualization.pixel_map_style import MASK, pixel_map_style

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from m3resp.visualization.pixel_maps import (
    add_region_labels,
    plot_pixel_map,
)

pytest.importorskip("eitprocessing")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")
with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as _handle:
    DATASETS = sorted(json.load(_handle)["inputs"])

FLAG_SETS = [
    dict(zip(("normalize", "percentage", "absolute"), flags, strict=True))
    for flags in itertools.product(
        (None, True), (None, True, False), (None, True, False)
    )
]


def m3resp_object(kind, values):
    """The m3resp map or mask of a style, warnings switched off."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == MASK:
            return PixelMask(name="mask", values=values, suppress_all_nan_warning=True)
        return PixelMap(
            name="map", values=values, kind=kind, suppress_negative_warning=True
        )


def _draw(draw):
    """Draw on new axes; give the image, the colour bar texts and the axes."""

    fig, ax = plt.subplots()
    try:
        image = draw(ax)
        fig.canvas.draw()
        state = {
            "array": image.get_array(),
            "limits": (image.norm.vmin, image.norm.vmax),
            "colors": image.to_rgba(image.get_array()),
            "background": ax.get_facecolor(),
            "axes visible": (ax.xaxis.get_visible(), ax.yaxis.get_visible()),
            "colorbar": None,
        }
        colorbar = image.colorbar
        if colorbar is not None:
            state["colorbar"] = {
                "label": colorbar.ax.get_ylabel(),
                "ticks": list(colorbar.get_ticks()),
                "tick texts": [t.get_text() for t in colorbar.ax.get_yticklabels()],
                "extend": colorbar.extend,
            }
        return state
    except ValueError as error:
        return {"error": str(error)}
    finally:
        plt.close(fig)


def _assert_same(ours, expected, case):
    if "error" in expected:
        assert ours == expected, case
        return
    assert np.array_equal(ours["array"].mask, expected["array"].mask), case
    assert np.array_equal(
        ours["array"].filled(np.nan), expected["array"].filled(np.nan), equal_nan=True
    ), case
    assert ours["limits"] == expected["limits"], case
    assert np.array_equal(ours["colors"], expected["colors"]), case
    assert ours["background"] == expected["background"], case
    assert ours["axes visible"] == expected["axes visible"], case
    assert ours["colorbar"] == expected["colorbar"], case


@pytest.mark.parametrize("kind", STYLES, ids=str)
def test_default_figures_equal_imshow_of_eitprocessing(kind):
    for case, grid in seeded_grids(kind).items():
        upstream = upstream_object(kind, grid)
        ours = m3resp_object(kind, np.asarray(upstream.values))
        found = _draw(lambda ax, o=ours: plot_pixel_map(o, ax))
        finite = ours.values[~np.isnan(ours.values)]
        largest = finite.max() if finite.size else 0.0
        if pixel_map_style(kind).scale == "from_zero" and largest < 0:
            # eitprocessing: matplotlib turns the scale 0 to the largest value
            # around and widens it by 10 %.
            assert found["limits"] == (largest, 0.0), case
            continue
        expected = _draw(lambda ax, u=upstream: u.plotting.imshow(ax=ax))
        _assert_same(found, expected, case)


@pytest.mark.parametrize("flags", FLAG_SETS, ids=str)
@pytest.mark.parametrize("kind", STYLES, ids=str)
def test_figures_with_flags_equal_imshow_of_eitprocessing(kind, flags):
    grids = seeded_grids(kind, seed=1)
    for case in list(grids)[:2]:
        upstream = upstream_object(kind, grids[case])
        ours = m3resp_object(kind, np.asarray(upstream.values))
        options = dict(flags, colorbar=True)
        expected = _draw(
            lambda ax, u=upstream, opt=options: u.plotting.imshow(ax=ax, **opt)
        )
        found = _draw(lambda ax, o=ours, opt=options: plot_pixel_map(o, ax, **opt))
        _assert_same(found, expected, case)


@pytest.mark.parametrize("kind", ["tiv", "odcl", MASK])
def test_colorbar_axes_background_and_imshow_options_equal_eitprocessing(kind):
    grid = next(iter(seeded_grids(kind, seed=2).values()))
    upstream = upstream_object(kind, grid)
    ours = m3resp_object(kind, np.asarray(upstream.values))
    cases = [
        ({"colorbar": False}, {"colorbar": False}),
        ({"hide_axes": False}, {"hide_axes": False}),
        ({"facecolor": "white"}, {"background": "white"}),
        ({"vmin": -0.5, "vmax": 2.0}, {"vmin": -0.5, "vmax": 2.0}),
        ({"cmap": "magma", "colorbar": True}, {"cmap": "magma", "colorbar": True}),
        (
            {"colorbar": True, "colorbar_kwargs": {"label": "x", "shrink": 0.5}},
            {"colorbar": True, "colorbar_options": {"label": "x", "shrink": 0.5}},
        ),
    ]
    for upstream_options, our_options in cases:
        expected = _draw(
            lambda ax, u=upstream, opt=upstream_options: u.plotting.imshow(ax=ax, **opt)
        )
        found = _draw(lambda ax, o=ours, opt=our_options: plot_pixel_map(o, ax, **opt))
        _assert_same(found, expected, our_options)


def test_plot_pixel_map_draws_on_the_current_axes_and_takes_a_style():
    fig, ax = plt.subplots()
    tiv = PixelMap(name="tiv", values=np.arange(6.0).reshape(2, 3), kind="tiv")
    image = plot_pixel_map(tiv, style=pixel_map_style(None))
    assert image.axes is ax
    assert image.cmap.name == "plain"
    assert image.colorbar.ax.get_ylabel() == ""
    plt.close(fig)


def test_plot_pixel_map_refuses_other_objects():
    with pytest.raises(TypeError, match="PixelMap or a PixelMask"):
        plot_pixel_map(np.ones((2, 2)))


def _golden_regions(dataset):
    path = os.path.join(GOLDEN_DIR, dataset, "roi.npz")
    with np.load(path) as data:
        return data["watershed_lungspace/captures/watershed regions"]


def _seeded_regions():
    rng = np.random.default_rng(3)
    c_shape = np.zeros((9, 9), dtype=int)
    c_shape[1:8, 1:4] = 1
    c_shape[1:3, 1:8] = 1
    c_shape[6:8, 1:8] = 1
    two_parts = np.zeros((8, 8), dtype=int)
    two_parts[0:2, 0:2] = 2
    two_parts[4:8, 4:8] = 2
    thin = np.zeros((6, 6), dtype=int)
    thin[2, :] = 4
    return {
        "random": rng.integers(0, 5, (10, 8)),
        "C shape": c_shape,
        "two parts": two_parts,
        "thin line": thin,
    }


def _labels(regions, draw):
    fig, ax = plt.subplots()
    ax.imshow(regions)
    texts = draw(ax)
    found = [
        {
            "position": tuple(float(v) for v in text.get_position()),
            "text": text.get_text(),
            "color": text.get_color(),
            "fontsize": text.get_fontsize(),
            "box": text.get_bbox_patch().get_facecolor(),
            "alignment": (text.get_ha(), text.get_va()),
        }
        for text in texts
    ]
    plt.close(fig)
    return found


REGION_CASES = [("golden", d) for d in DATASETS] + [
    ("seeded", name) for name in _seeded_regions()
]


@pytest.mark.parametrize(("source", "name"), REGION_CASES)
def test_region_labels_equal_add_region_markers_of_eitprocessing(source, name):
    regions = _golden_regions(name) if source == "golden" else _seeded_regions()[name]
    regions = np.asarray(regions).astype(int)
    upstream = upstream_classes()["integer"](regions)
    ours = PixelMap(name="regions", values=regions, kind="integer")

    # eitprocessing labels region 0 (the background) first; m3resp skips it.
    skip = 1 if 0 in regions else 0
    expected = _labels(regions, lambda ax: upstream.plotting.add_region_markers(ax))
    found = _labels(regions, lambda ax: add_region_labels(ours, ax))
    assert found == expected[skip:]

    label_map = {int(regions.max()): "last", 99: "unused"}
    style = {"color": "red", "fontsize": 7, "bbox": {"facecolor": "yellow"}}
    expected = _labels(
        regions,
        lambda ax: upstream.plotting.add_region_markers(ax, label_map, **style),
    )
    found = _labels(regions, lambda ax: add_region_labels(ours, ax, label_map, **style))
    assert found == expected[skip:]


def test_add_region_labels_passes_other_text_options():
    fig, ax = plt.subplots()
    regions = PixelMap(name="regions", values=np.eye(3), kind="integer")
    texts = add_region_labels(regions, ax, fontweight="bold")
    assert [text.get_text() for text in texts] == ["1"]
    assert all(text.get_fontweight() == "bold" for text in texts)
    plt.close(fig)


def test_add_region_labels_needs_a_map_of_region_numbers():
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="kind 'integer'"):
        add_region_labels(PixelMap(name="tiv", values=np.eye(3), kind="tiv"), ax)
    with pytest.raises(TypeError, match="kind 'integer'"):
        add_region_labels(np.eye(3), ax)
    plt.close(fig)
