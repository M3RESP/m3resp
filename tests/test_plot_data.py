"""GUI plot data of `m3resp.visualization.plot_data` and the minute image.

Every result must go through `json.dumps(..., allow_nan=False)`, with None for
NaN, and hold the values, shapes, times and units of its input. On the 3
golden test files: the EIT image at a time equals the pixel `Signal` at the
nearest sample; the minute image equals the mean pixel TIV over the breaths
inside the window; the lung-space plot data equals the captures of the
step 7 functions; the rate plot data equals the captures of `detect_rates`.
`plot_data` and `pixel_map_style` import no matplotlib, no eitprocessing and
no `m3resp.adapters`.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import warnings

import numpy as np
import pytest

from m3resp.data import BreathEvent, PixelMap, PixelMask, Signal
from m3resp.eit.processing import (
    BreathDetectionSettings,
    RateDetectionSettings,
    detect_rates,
    find_breaths,
    minute_image,
    pixel_tidal_impedance_variation,
)
from m3resp.eit.roi import (
    amplitude_lungspace,
    filter_mask_by_size,
    tiv_lungspace,
    watershed_lungspace,
)
from m3resp.visualization.pixel_map_style import (
    pixel_map_style,
    scale_limits,
)
from m3resp.visualization.plot_data import (
    eit_image_plot_data,
    eit_images_plot_data,
    grid_plot_data,
    lungspace_plot_data,
    pixel_map_plot_data,
    pixel_mask_plot_data,
    rate_spectrum_plot_data,
    region_label_positions,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")
with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as _handle:
    MANIFEST = json.load(_handle)
DATASETS = sorted(MANIFEST["inputs"])


def _json(data):
    """The data after a trip through JSON, which must refuse NaN."""

    return json.loads(json.dumps(data, allow_nan=False))


def _from_plot(grid):
    """A plot data grid as floats, None as NaN."""

    return np.array(
        [[np.nan if v is None else v for v in row] for row in grid], dtype=float
    )


def _golden(dataset, name):
    with np.load(os.path.join(GOLDEN_DIR, dataset, f"{name}.npz")) as data:
        return {key: data[key] for key in data.files}


@functools.cache
def _file(dataset):
    """Pixels, global impedance and breaths of a golden file (raw data)."""

    reader = _golden(dataset, "reader")
    time = reader["time"]
    pixels = Signal(
        values=reader["pixel_impedance"].astype(float),
        time=time,
        sample_frequency=1 / float(np.median(np.diff(time))),
        unit="AU",
        name="pixels",
        modality="eit",
        category="impedance",
    )
    waveform = _golden(dataset, "global_impedance")
    signal = Signal(
        values=waveform["raw/values"],
        time=waveform["raw/time"],
        sample_frequency=pixels.sample_frequency,
        unit="AU",
        name="global_impedance",
        modality="eit",
        category="impedance",
    )
    breaths = find_breaths(signal, BreathDetectionSettings())
    return pixels, signal, breaths


def _seeded_map(seed=0, kind=None):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(6, 5)) * 3
    values[1, 1] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PixelMap(
            name="map",
            values=values,
            kind=kind,
            unit="AU",
            category="impedance",
            suppress_negative_warning=True,
        )


@pytest.mark.parametrize("kind", [None, "tiv", "difference", "odcl"])
def test_pixel_map_plot_data_holds_the_map_and_its_style(kind):
    pixel_map = _seeded_map(kind=kind)
    data = _json(pixel_map_plot_data(pixel_map))

    assert data["name"] == "map"
    assert data["kind"] == kind
    assert data["unit"] == pixel_map.unit
    assert data["category"] == pixel_map.category
    assert data["shape"] == [6, 5]
    assert data["values"][1][1] is None
    assert np.array_equal(_from_plot(data["values"]), pixel_map.values, equal_nan=True)
    assert data["normalized"] is False
    style = pixel_map_style(kind)
    assert data["style"]["name"] == style.name
    assert data["style"]["colors"] == [list(c) for c in style.colors]
    assert data["style"]["limits"] == list(scale_limits(style, pixel_map.values))
    assert data["style"]["extend"] == style.extend


def test_pixel_map_plot_data_divides_by_the_largest_value_when_asked():
    pixel_map = PixelMap(name="m", values=np.array([[2.0, 4.0], [np.nan, 1.0]]))
    data = _json(pixel_map_plot_data(pixel_map, normalize=True))
    assert data["values"] == [[0.5, 1.0], [None, 0.25]]
    assert data["normalized"] is True
    assert data["style"]["limits"] == [0.25, 1.0]


def test_pixel_map_plot_data_takes_a_style_and_refuses_infinity():
    pixel_map = _seeded_map(kind="tiv")
    data = pixel_map_plot_data(pixel_map, style=pixel_map_style(None))
    assert data["style"]["name"] == "plain"
    infinite = PixelMap(name="inf", values=np.array([[1.0, np.inf]]))
    with pytest.raises(ValueError, match="infinite"):
        pixel_map_plot_data(infinite)
    with pytest.raises(TypeError, match="needs a PixelMap"):
        pixel_map_plot_data(np.ones((2, 2)))


def test_pixel_mask_plot_data_holds_the_mask():
    values = np.array([[1.0, np.nan, 0.5], [np.nan, 1.0, 1.0]])
    mask = PixelMask(name="lungs", values=values)
    data = _json(pixel_mask_plot_data(mask))
    assert data["name"] == "lungs"
    assert data["shape"] == [2, 3]
    assert data["values"] == [[1.0, None, 0.5], [None, 1.0, 1.0]]
    assert data["included_pixel_count"] == 4
    assert data["included_pixel_fraction"] == 4 / 6
    assert data["is_weighted"] is True
    assert data["style"]["name"] == "mask"
    assert data["style"]["limits"] == [0.0, 2.0]
    with pytest.raises(TypeError, match="needs a PixelMask"):
        pixel_mask_plot_data(_seeded_map())


@pytest.mark.parametrize("dataset", DATASETS)
def test_eit_image_at_a_time_is_the_nearest_sample(dataset):
    pixels, _, _ = _file(dataset)
    time = np.asarray(pixels.time)
    rng = np.random.default_rng(4)
    picks = [time[0], time[-1], time[5], (time[7] + time[8]) / 2 + 1e-6]
    picks += list(rng.uniform(time[0], time[-1], 5))
    for value in picks:
        data = _json(eit_image_plot_data(pixels, float(value)))
        sample = int(np.argmin(np.abs(time - value)))
        assert data["sample"] == sample
        assert data["time"] == float(time[sample])
        assert data["unit"] == pixels.unit
        assert data["shape"] == list(pixels.values.shape[1:])
        image = np.asarray(pixels.values)[sample]
        assert np.array_equal(_from_plot(data["values"]), image, equal_nan=True)
        assert data["style"]["limits"] == list(scale_limits(pixel_map_style(), image))


def test_eit_image_halfway_between_two_samples_takes_the_first():
    pixels = Signal(
        values=np.arange(12.0).reshape(3, 2, 2),
        time=np.array([0.0, 1.0, 2.0]),
        sample_frequency=1.0,
        name="pixels",
        modality="eit",
    )
    assert eit_image_plot_data(pixels, 0.5)["sample"] == 0
    assert eit_image_plot_data(pixels, 0.5, kind="tiv")["style"]["name"] == "tiv"
    with pytest.raises(ValueError, match="outside the data"):
        eit_image_plot_data(pixels, 2.5)
    with pytest.raises(ValueError, match="shape"):
        eit_image_plot_data(
            Signal(values=np.arange(3.0), time=np.arange(3.0), name="w"), 1.0
        )


@pytest.mark.parametrize("dataset", DATASETS)
def test_eit_images_from_start_to_end_share_one_scale(dataset):
    pixels, _, _ = _file(dataset)
    time = np.asarray(pixels.time)
    start, end, step = float(time[3]), float(time[3]) + 2.0, 0.1
    data = _json(eit_images_plot_data(pixels, start, end, step))

    requested = start + step * np.arange(21)
    assert np.allclose(data["requested_times"], requested, rtol=0, atol=1e-12)
    samples = [int(np.argmin(np.abs(time - value))) for value in requested]
    assert data["samples"] == samples
    assert data["times"] == [float(time[s]) for s in samples]
    images = np.asarray(pixels.values)[samples]
    assert len(data["images"]) == 21
    for grid, image in zip(data["images"], images, strict=True):
        assert np.array_equal(_from_plot(grid), image, equal_nan=True)
    assert data["style"]["limits"] == list(scale_limits(pixel_map_style(), images))


def test_eit_images_refuse_wrong_times():
    pixels, _, _ = _file(DATASETS[0])
    with pytest.raises(ValueError, match="step > 0"):
        eit_images_plot_data(pixels, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="start <= end"):
        eit_images_plot_data(pixels, 2.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="outside the data"):
        eit_images_plot_data(pixels, float(pixels.time[0]) - 1, 1.0, 0.5)


def _small_images(time):
    return Signal(
        values=np.arange(len(time) * 4.0).reshape(len(time), 2, 2),
        time=np.asarray(time),
        sample_frequency=10.0,
        name="pixels",
        modality="eit",
    )


def test_eit_images_reach_the_last_sample():
    # 0.1 + 0.1 * 2 is 0.30000000000000004, just after the last sample.
    pixels = _small_images([0.0, 0.1, 0.2, 0.3])
    data = eit_images_plot_data(pixels, 0.1, 0.3, 0.1)
    assert data["requested_times"][-1] == 0.3
    assert data["samples"] == [1, 2, 3]


def test_eit_images_halfway_between_two_samples_take_the_first():
    pixels = _small_images([0.0, 1.0, 2.0, 3.0])
    data = eit_images_plot_data(pixels, 0.0, 3.0, 0.5)
    assert data["samples"] == [0, 0, 1, 1, 2, 2, 3]
    assert data["images"][1] == [[0.0, 1.0], [2.0, 3.0]]


def test_eit_images_need_a_rising_time_axis():
    pixels = _small_images([0.0, 0.2, 0.1, 0.3])
    with pytest.raises(ValueError, match="time axis that rises"):
        eit_image_plot_data(pixels, 0.1)
    with pytest.raises(ValueError, match="time axis that rises"):
        eit_images_plot_data(pixels, 0.0, 0.3, 0.1)


@pytest.mark.parametrize("kind", ["tiv", "amplitude", "perfusion", "pendelluft"])
def test_scale_from_zero_of_a_map_with_only_negative_values(kind):
    pixel_map = PixelMap(
        name="map",
        values=[[-1.0, -3.0], [np.nan, -2.0]],
        kind=kind,
        suppress_negative_warning=True,
    )
    assert pixel_map_plot_data(pixel_map)["style"]["limits"] == [-1.0, 0.0]


@pytest.mark.parametrize("dataset", DATASETS)
def test_minute_image_is_the_mean_pixel_tiv_of_the_breaths_inside(dataset):
    pixels, _, breaths = _file(dataset)
    start = breaths[1].start_time - 0.01
    end = breaths[4].end_time + 0.01
    found = minute_image(pixels, breaths, start=start, end=end)

    inside = breaths[1:5]
    per_breath = pixel_tidal_impedance_variation(pixels, inside)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.nanmean(np.stack([m.values for m in per_breath.values]), axis=0)
    assert np.array_equal(found.values, expected, equal_nan=True)
    assert found.kind == "tiv"
    assert found.unit == pixels.unit
    assert found.metadata == {
        "start": start,
        "end": end,
        "breath_count": 4,
        "tiv_method": "inspiratory",
    }
    data = _json(pixel_map_plot_data(found))
    assert np.array_equal(_from_plot(data["values"]), expected, equal_nan=True)


@pytest.mark.parametrize("dataset", DATASETS)
def test_minute_image_of_the_whole_data_is_the_mean_tiv_of_tiv_lungspace(dataset):
    pixels, _, breaths = _file(dataset)
    captures = {}
    tiv_lungspace(pixels, breaths, captures=captures)
    found = minute_image(pixels, breaths)
    assert np.array_equal(found.values, captures["mean TIV"].values, equal_nan=True)
    assert found.metadata["breath_count"] == len(breaths)


def test_minute_image_leaves_out_breaths_partly_outside_and_checks_input():
    pixels, _, breaths = _file(DATASETS[0])
    halfway = (breaths[2].start_time + breaths[2].end_time) / 2
    found = minute_image(pixels, breaths, start=halfway, end=breaths[5].end_time)
    assert found.metadata["breath_count"] == 3
    found = minute_image(
        pixels, breaths, start=breaths[0].start_time, end=breaths[0].end_time
    )
    assert found.metadata["breath_count"] == 1
    with pytest.raises(ValueError, match="No breath lies fully inside"):
        minute_image(pixels, breaths, start=halfway, end=halfway + 0.1)
    with pytest.raises(ValueError, match="start <= end"):
        minute_image(pixels, breaths, start=5.0, end=4.0)
    with pytest.raises(TypeError, match="BreathEvent"):
        minute_image(pixels, [(0.0, 1.0)])
    end = float(pixels.time[-1])
    late = BreathEvent(
        modality="eit",
        start_time=end + 1.0,
        end_time=end + 5.0,
        extremum_time=end + 3.0,
    )
    with pytest.warns(UserWarning, match="partly outside the data"):
        found = minute_image(pixels, [breaths[0], late], start=0.0, end=end + 10.0)
    assert found.metadata["breath_count"] == 1
    with (
        pytest.warns(UserWarning, match="partly outside the data"),
        pytest.raises(ValueError, match="has NaN TIV in all pixels"),
    ):
        minute_image(pixels, [late], start=0.0, end=end + 10.0)
    expiratory = minute_image(pixels, breaths, tiv_method="expiratory")
    assert expiratory.metadata["tiv_method"] == "expiratory"


@pytest.mark.parametrize("dataset", DATASETS)
def test_lungspace_plot_data_equals_the_watershed_captures(dataset):
    pixels, signal, breaths = _file(dataset)
    captures = {}
    with warnings.catch_warnings():
        # The Timpel file has pixels without breaths; the method warns.
        warnings.simplefilter("ignore", RuntimeWarning)
        mask = watershed_lungspace(pixels, signal, breaths, captures=captures)
    data = _json(lungspace_plot_data(captures, mask=mask))

    assert set(data["maps"]) == {
        "mean TIV of shared breaths",
        "mean amplitude of shared breaths",
        "included watershed regions",
    }
    for key, plotted in data["maps"].items():
        expected = captures[key].values
        assert np.array_equal(_from_plot(plotted["values"]), expected, equal_nan=True)
        assert plotted["kind"] == captures[key].kind
    assert set(data["masks"]) == {
        "functional tiv mask",
        "functional amplitude mask",
        "included region",
    }
    for key, plotted in data["masks"].items():
        expected = captures[key].values
        assert np.array_equal(_from_plot(plotted["values"]), expected, equal_nan=True)
    regions = captures["watershed regions"]
    plotted = data["regions"]["watershed regions"]
    assert np.array_equal(_from_plot(plotted["values"]), regions.values)
    assert plotted["style"]["name"] == "integer"
    assert plotted["label_positions"] == [
        {"region": number, "row": row, "column": column}
        for number, (row, column) in region_label_positions(regions).items()
    ]
    assert 0 not in [label["region"] for label in plotted["label_positions"]]
    for key in ("local peaks", "included peaks", "excluded peaks"):
        assert data["points"][key] == np.asarray(captures[key]).tolist()
    assert data["numbers"]["included marker indices"] == (
        np.asarray(captures["included marker indices"]).tolist()
    )
    assert np.array_equal(
        _from_plot(data["mask"]["values"]), mask.values, equal_nan=True
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_lungspace_plot_data_of_tiv_amplitude_and_size_filter(dataset):
    pixels, signal, breaths = _file(dataset)
    for function, key in (
        (lambda c: tiv_lungspace(pixels, breaths, captures=c), "TIV"),
        (
            lambda c: amplitude_lungspace(pixels, signal, breaths, captures=c),
            "amplitude",
        ),
    ):
        captures = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = function(captures)
        data = _json(lungspace_plot_data(captures))
        mean = captures[f"mean {key}"]
        plotted = data["maps"][f"mean {key}"]
        assert np.array_equal(
            _from_plot(plotted["values"]), mean.values, equal_nan=True
        )
        assert data["breath_counts"] == {
            f"{key} per breath": len(captures[f"{key} per breath"].intervals)
        }
        assert data["mask"] is None

    captures = {}
    filter_mask_by_size(mask, min_region_size=1, captures=captures)
    data = _json(lungspace_plot_data(captures))
    assert np.array_equal(
        _from_plot(data["regions"]["regions"]["values"]), captures["regions"].values
    )
    assert data["numbers"]["region sizes"] == captures["region sizes"]
    assert data["numbers"]["kept regions"] == captures["kept regions"]


def test_lungspace_plot_data_refuses_unknown_and_wrong_captures():
    with pytest.raises(ValueError, match="does not know the capture 'other'"):
        lungspace_plot_data({"other": 1})
    with pytest.raises(TypeError, match="must be a PixelMap"):
        lungspace_plot_data({"mean TIV": np.ones((2, 2))})


@pytest.mark.parametrize("dataset", DATASETS)
def test_rate_spectrum_plot_data_equals_the_captures(dataset):
    pixels, _, _ = _file(dataset)
    settings = RateDetectionSettings("adult")
    captures = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        respiratory_rate, heart_rate = detect_rates(pixels, settings, captures=captures)
    data = _json(rate_spectrum_plot_data(captures, settings))

    assert data["frequency_unit"] == "Hz"
    for key in (
        "frequencies",
        "normalized_total_power",
        "average_normalized_pixel_power",
        "diff_total_averaged_power",
    ):
        assert np.array_equal(
            np.array([np.nan if v is None else v for v in data[key]]),
            np.asarray(captures[key], dtype=float),
            equal_nan=True,
        ), key
    assert data["respiratory_rate"] == respiratory_rate
    assert data["heart_rate"] == heart_rate
    assert data["respiratory_rate_per_minute"] == respiratory_rate * 60
    assert data["heart_rate_per_minute"] == heart_rate * 60
    assert data["respiratory_rate_range"] == [
        settings.min_respiratory_rate,
        settings.max_respiratory_rate,
    ]
    assert data["heart_rate_range"] == [
        settings.min_heart_rate,
        settings.max_heart_rate,
    ]
    assert rate_spectrum_plot_data(captures)["heart_rate_range"] is None


def test_region_label_positions_need_region_numbers():
    with pytest.raises(TypeError, match="kind 'integer'"):
        region_label_positions(_seeded_map())
    positions = region_label_positions(
        PixelMap(name="r", values=np.array([[1, 1, 2], [1, 1, 2]]), kind="integer")
    )
    assert list(positions) == [1, 2]
    positions = region_label_positions(
        PixelMap(name="r", values=np.array([[0, 1, 1], [0, 1, 1]]), kind="integer")
    )
    assert list(positions) == [1]
    assert all(isinstance(v, float) for point in positions.values() for v in point)


def test_grid_plot_data_turns_nan_into_none():
    assert grid_plot_data(np.array([[1.0, np.nan]]), owner="x") == [[1.0, None]]
    with pytest.raises(ValueError, match="'x' holds an infinite value"):
        grid_plot_data(np.array([[-np.inf]]), owner="x")


@pytest.mark.parametrize("module", ["plot_data", "pixel_map_style"])
def test_the_gui_modules_import_no_adapters_matplotlib_or_eitprocessing(module):
    path = os.path.join(REPO_ROOT, "src", "m3resp", "visualization", f"{module}.py")
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    for name in names:
        assert not name.startswith(("matplotlib", "eitprocessing", "m3resp.adapters"))


def test_breath_events_of_the_test_files_have_breaths():
    """The test files give enough breaths for the minute image tests."""

    for dataset in DATASETS:
        _, _, breaths = _file(dataset)
        assert len(breaths) >= 6
        assert all(isinstance(b, BreathEvent) for b in breaths)
