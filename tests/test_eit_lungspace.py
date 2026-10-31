"""Lung-space masks of `m3resp.eit.roi`: TIV, amplitude, watershed, size filter.

The masks and captures must equal eitprocessing 1.8.7: the golden files
``roi.npz`` of the 3 test files (default settings), and `TIVLungspace`,
`AmplitudeLungspace`, `WatershedLungspace` and `FilterROIBySize` called on the
same data with other settings: the 3 files raw and MDN-filtered, seeded
synthetic images (flat, NaN, upside-down and phase-shifted pixels, one or
many maxima) and seeded random masks. Masks, peaks and region numbers must be
equal; mean maps may differ by 1e-9 of their largest value.

The watershed regions differ from eitprocessing in two ways. Pixels with a NaN
mean amplitude belong to no region (number 0), and "included watershed
regions" holds the included regions only. The expected regions, the expected
"included region" and the expected watershed mask come from the eitprocessing
captures with these two rules.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import warnings
from typing import Any

import numpy as np
import pytest

from m3resp.data import BreathEvent, PixelMap, PixelMask, Signal
from m3resp.eit.processing import BreathDetectionSettings, find_breaths
from m3resp.eit.roi import (
    amplitude_lungspace,
    filter_mask_by_size,
    tiv_lungspace,
    watershed_lungspace,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")
with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as _handle:
    MANIFEST = json.load(_handle)
DATASETS = sorted(MANIFEST["inputs"])
TIMPEL = "timpel_healthy_volunteer_1_60-120s"

#: Watershed capture names that m3resp renamed (Q4 of the step 7 plan).
UPSTREAM_KEYS = {
    "mean TIV of shared breaths": "mean tiv",
    "mean amplitude of shared breaths": "mean amplitude",
}
EXACT_CAPTURES = {
    "functional tiv mask",
    "functional amplitude mask",
    "local peaks",
    "included marker indices",
    "included region",
    "watershed regions",
    "included peaks",
    "excluded peaks",
    "included watershed regions",
}


def _watershed_expected(
    mean_amplitude: Any,
    local_peaks: Any,
    included_markers: Any,
    amplitude_mask: Any,
) -> dict[str, np.ndarray]:
    """The watershed regions, "included region", "included watershed regions"
    and the mask, from the eitprocessing captures, with the pixels of NaN mean
    amplitude left out of the regions."""

    from scipy import ndimage
    from skimage import segmentation

    amplitude = _values(mean_amplitude)
    is_peak = np.zeros(amplitude.shape, dtype=bool)
    is_peak[tuple(np.asarray(local_peaks).T)] = True
    markers, _ = ndimage.label(is_peak)
    regions = segmentation.watershed(
        -np.nan_to_num(amplitude), markers=markers, mask=~np.isnan(amplitude)
    )
    included = np.isin(regions, included_markers)
    return {
        "watershed regions": regions,
        "included region": np.where(included, 1.0, np.nan),
        "included watershed regions": np.where(included, regions, np.nan),
        "mask": np.where(included & (_values(amplitude_mask) == 1), 1.0, np.nan),
    }


def _golden(dataset: str, name: str) -> dict[str, np.ndarray]:
    with np.load(os.path.join(GOLDEN_DIR, dataset, f"{name}.npz")) as data:
        return {key: data[key] for key in data.files}


def _signal(
    values: np.ndarray, time: np.ndarray, *, name: str, source: str | None = None
) -> Signal:
    return Signal(
        values=np.asarray(values, dtype=float),
        time=np.asarray(time, dtype=float),
        sample_frequency=1 / float(np.median(np.diff(time))),
        unit="AU",
        name=name,
        modality="eit",
        category="impedance",
        source=source,
    )


@functools.cache
def _file_data(dataset: str, kind: str) -> tuple[Signal, Signal]:
    """The raw or MDN-filtered pixels and global impedance of a golden file."""

    reader = _golden(dataset, "reader")
    pixels = reader["pixel_impedance"]
    if kind == "filtered":
        pixels = _golden(dataset, "mdn_filter")["pixel_impedance"]
    waveform = _golden(dataset, "global_impedance")
    return (
        _signal(pixels, reader["time"], name="pixels", source=str(reader["vendor"])),
        _signal(
            waveform[f"{kind}/values"],
            waveform[f"{kind}/time"],
            name="global_impedance",
        ),
    )


@functools.cache
def _synthetic(seed: int, case: str) -> tuple[Signal, Signal]:
    """Seeded 12 x 10 EIT images of 40 s at 20 Hz, with breaths of 4 s.

    ``case`` picks the amplitude map and the special pixels:
    ``"one maximum"`` (one smooth bump), ``"many maxima"`` (random bumps),
    ``"upside-down"``, ``"phase-shifted"``, ``"flat"`` and ``"nan"`` (some
    pixels of these kinds on a map of random bumps).
    """

    rng = np.random.default_rng(seed)
    shape = (12, 10)
    time = np.arange(0, 40, 1 / 20)
    rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]

    def bump(row: float, column: float, width: float) -> np.ndarray:
        return np.exp(-((rows - row) ** 2 + (columns - column) ** 2) / (2 * width**2))

    if case == "one maximum":
        amplitude = bump(5.5, 4.5, 2.5)
    else:
        amplitude = sum(
            rng.uniform(0.3, 1.0)
            * bump(
                rng.uniform(0, shape[0]), rng.uniform(0, shape[1]), rng.uniform(1, 2.5)
            )
            for _ in range(5)
        )
    lag = np.zeros(shape)
    sign = np.ones(shape)
    special = rng.random(shape) < 0.15
    if case == "upside-down":
        sign[special] = -1
    if case == "phase-shifted":
        lag[special] = rng.uniform(0.2, 0.8, size=special.sum())
    if case == "flat":
        amplitude[special] = 0
    breathing = 0.5 * (1 - np.cos(2 * np.pi * (time[:, None, None] - lag) / 4))
    values = (
        sign * amplitude * breathing
        + 0.01 * rng.normal(size=(len(time), *shape))
        + rng.uniform(5, 10, size=shape)
    )
    if case == "nan":
        values[:, special] = np.nan
    pixels = _signal(values, time, name="pixels", source="simulated")
    return pixels, _signal(
        np.nansum(values, axis=(1, 2)), time, name="global_impedance"
    )


#: Breaths found per waveform object; the waveforms are cached above.
_BREATHS: dict[int, list[BreathEvent]] = {}


def _breaths(data: tuple[Signal, Signal]) -> list[BreathEvent]:
    """The breaths of the waveform, with the default settings (as eitprocessing)."""

    key = id(data[1])
    if key not in _BREATHS:
        _BREATHS[key] = find_breaths(data[1], BreathDetectionSettings())
    return _BREATHS[key]


def _quiet(function: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function()


def _run(function: Any) -> tuple[Any, type | None]:
    try:
        return _quiet(function), None
    except Exception as error:  # noqa: BLE001 - the error type is compared
        return None, type(error)


def _upstream(method: str, data: tuple[Signal, Signal], **settings: Any) -> Any:
    """Run an eitprocessing lung-space method: (mask, error type, captures)."""

    pytest.importorskip("eitprocessing")
    from eitprocessing.roi.amplitude import AmplitudeLungspace
    from eitprocessing.roi.tiv import TIVLungspace
    from eitprocessing.roi.watershed import WatershedLungspace

    from m3resp.adapters.eitprocessing_adapter._signals import (
        as_continuous_data,
        as_eit_data,
    )

    classes = {
        "tiv": TIVLungspace,
        "amplitude": AmplitudeLungspace,
        "watershed": WatershedLungspace,
    }
    captures: dict[str, Any] = {}
    pixels, waveform = data
    mask, error = _run(
        lambda: classes[method](**settings).apply(
            as_eit_data(pixels),
            timing_data=as_continuous_data(waveform),
            captures=captures,
        )
    )
    return mask, error, captures


def _ours(method: str, data: tuple[Signal, Signal], **settings: Any) -> Any:
    """Run an m3resp lung-space function: (mask, error type, captures)."""

    pixels, waveform = data
    breaths = _breaths(data)
    captures: dict[str, Any] = {}
    calls = {
        "tiv": lambda: tiv_lungspace(pixels, breaths, captures=captures, **settings),
        "amplitude": lambda: amplitude_lungspace(
            pixels, waveform, breaths, captures=captures, **settings
        ),
        "watershed": lambda: watershed_lungspace(
            pixels, waveform, breaths, captures=captures, **settings
        ),
    }
    mask, error = _run(calls[method])
    return mask, error, captures


def _setting(method: str, value: Any) -> dict[str, Any]:
    return {"threshold_fraction" if method == "watershed" else "threshold": value}


def _values(item: Any) -> np.ndarray:
    if hasattr(item, "mask"):  # eitprocessing PixelMask
        return np.asarray(item.mask)
    return np.asarray(getattr(item, "values", item), dtype=float)


def _assert_close(ours: Any, expected: Any) -> None:
    ours, expected = _values(ours), _values(expected)
    np.testing.assert_array_equal(np.isnan(ours), np.isnan(expected))
    scale = np.nanmax(np.abs(expected)) if np.any(~np.isnan(expected)) else 1.0
    np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-9 * scale)


def _amplitude_fix_pixels(
    data: tuple[Signal, Signal], ours: np.ndarray, upstream: np.ndarray
) -> np.ndarray:
    """Return the pixels where the amplitude per breath differs, after checking
    that each difference is the step 6 fix of the "phase shift" pixel breath
    search: eitprocessing keeps a pixel breath whose end of inspiration the
    lag moves outside the data, in the second or the second-to-last breath;
    m3resp leaves it out (see ``tests/test_eit_pixel_breath.py``).
    """

    same = (ours == upstream) | (np.isnan(ours) & np.isnan(upstream))
    differs = ~same
    assert np.isnan(ours[differs]).all()
    count = len(_breaths(data))
    assert set(np.flatnonzero(differs.any(axis=(1, 2)))) <= {1, count - 2}
    return differs.any(axis=0)


@functools.cache
def _upstream_amplitude(data_key: tuple[str, ...]) -> np.ndarray:
    """eitprocessing amplitude per breath (pixel timing, "phase shift")."""

    from eitprocessing.features.pixel_breath import PixelBreath
    from eitprocessing.parameters.tidal_impedance_variation import TIV

    from m3resp.adapters.eitprocessing_adapter._signals import (
        as_continuous_data,
        as_eit_data,
    )

    pixels, waveform = _DATA[data_key]
    result = _quiet(
        lambda: TIV(
            pixel_breath=PixelBreath(phase_correction_mode="phase shift")
        ).compute_pixel_parameter(
            as_eit_data(pixels),
            as_continuous_data(waveform),
            None,
            tiv_timing="pixel",
            store=False,
        )
    )
    return np.asarray(result.values, dtype=float)


#: The data of each test case, by key, for the cached eitprocessing runs.
_DATA: dict[tuple[str, ...], tuple[Signal, Signal]] = {}


def _assert_same_as_upstream(
    method: str, data_key: tuple[str, ...], data: tuple[Signal, Signal], value: float
) -> None:
    _DATA[data_key] = data
    settings = _setting(method, value)
    up_mask, up_error, up_captures = _upstream(method, data, **settings)
    mask, error, captures = _ours(method, data, **settings)
    assert error is up_error
    if up_error is not None:
        return
    expected_mask = up_mask.mask
    if method == "watershed":
        expected = _watershed_expected(
            up_captures["mean amplitude"],
            up_captures["local peaks"],
            up_captures["included marker indices"],
            up_captures["functional amplitude mask"],
        )
        expected_mask = expected.pop("mask")
        up_captures = {**up_captures, **expected}
    np.testing.assert_array_equal(mask.values, expected_mask)

    fixed = np.zeros(mask.shape, dtype=bool)
    if method != "tiv":
        from m3resp.eit.roi.amplitude import pixel_amplitude_per_breath

        ours_amplitude = _quiet(
            lambda: pixel_amplitude_per_breath(*data, _breaths(data))
        )
        fixed = _amplitude_fix_pixels(
            data,
            np.stack([m.values for m in ours_amplitude.values]),
            _upstream_amplitude(data_key),
        )

    for key, item in captures.items():
        expected = up_captures[UPSTREAM_KEYS.get(key, key)]
        if key in EXACT_CAPTURES:
            np.testing.assert_array_equal(_values(item), _values(expected))
        elif key == "amplitude per breath":
            ours_grids = np.stack([m.values for m in item.values])
            expected_grids = np.asarray(expected.values, dtype=float)
            _amplitude_fix_pixels(data, ours_grids, expected_grids)
            _assert_close(ours_grids[:, ~fixed], expected_grids[:, ~fixed])
        elif key.endswith("per breath"):
            _assert_close(
                np.stack([m.values for m in item.values]),
                np.asarray(expected.values, dtype=float),
            )
        elif "amplitude" in key:
            _assert_close(_values(item)[~fixed], _values(expected)[~fixed])
        else:
            _assert_close(item, expected)


# --- golden files ---------------------------------------------------------------


@pytest.mark.parametrize("dataset", DATASETS)
def test_masks_and_captures_match_golden(dataset: str) -> None:
    data = _file_data(dataset, "filtered")
    pixels, waveform = data
    breaths = _breaths(data)
    golden = _golden(dataset, "roi")
    captures: dict[str, dict[str, Any]] = {"tiv": {}, "amplitude": {}, "watershed": {}}

    masks = _quiet(
        lambda: {
            "tiv_lungspace": tiv_lungspace(pixels, breaths, captures=captures["tiv"]),
            "amplitude_lungspace": amplitude_lungspace(
                pixels, waveform, breaths, captures=captures["amplitude"]
            ),
            "watershed_lungspace": watershed_lungspace(
                pixels, waveform, breaths, captures=captures["watershed"]
            ),
        }
    )
    masks["filter_by_size"] = filter_mask_by_size(
        masks["watershed_lungspace"],
        min_region_size=MANIFEST["settings"]["filter_by_size"]["min_region_size"],
    )

    for name, mask in masks.items():
        np.testing.assert_array_equal(mask.values, golden[f"{name}/mask"])
    for method, key, mean_key in (
        ("tiv", "TIV per breath", "mean TIV"),
        ("amplitude", "amplitude per breath", "mean amplitude"),
    ):
        np.testing.assert_array_equal(
            np.stack([m.values for m in captures[method][key].values]),
            golden[f"{method}_lungspace/captures/{key}/values"],
        )
        _assert_close(
            captures[method][mean_key],
            golden[f"{method}_lungspace/captures/{mean_key}"],
        )
    assert set(captures["watershed"]) == {*UPSTREAM_KEYS, *EXACT_CAPTURES}
    golden_watershed = {
        key.removeprefix("watershed_lungspace/captures/"): value
        for key, value in golden.items()
        if key.startswith("watershed_lungspace/captures/")
    }
    fixed = _watershed_expected(
        golden_watershed["mean amplitude"],
        golden_watershed["local peaks"],
        golden_watershed["included marker indices"],
        golden_watershed["functional amplitude mask"],
    )
    np.testing.assert_array_equal(
        masks["watershed_lungspace"].values, fixed.pop("mask")
    )
    golden_watershed.update(fixed)
    for key, item in captures["watershed"].items():
        expected = golden_watershed[UPSTREAM_KEYS.get(key, key)]
        if key in EXACT_CAPTURES:
            np.testing.assert_array_equal(_values(item), expected)
        else:
            _assert_close(item, expected)


@pytest.mark.parametrize("dataset", DATASETS)
def test_capture_types(dataset: str) -> None:
    data = _file_data(dataset, "filtered")
    captures: dict[str, Any] = {}
    _quiet(lambda: watershed_lungspace(*data, _breaths(data), captures=captures))

    assert isinstance(captures["mean TIV of shared breaths"], PixelMap)
    assert captures["mean TIV of shared breaths"].kind == "tiv"
    assert captures["mean amplitude of shared breaths"].kind == "amplitude"
    assert captures["watershed regions"].kind == "integer"
    for key in ("functional tiv mask", "functional amplitude mask", "included region"):
        assert isinstance(captures[key], PixelMask)
    for key in ("local peaks", "included peaks", "excluded peaks"):
        assert captures[key].dtype.kind == "i"
        assert captures[key].shape[1] == 2


@pytest.mark.parametrize("dataset", DATASETS)
def test_mask_fields(dataset: str) -> None:
    data = _file_data(dataset, "filtered")
    mask = _quiet(lambda: watershed_lungspace(*data, _breaths(data), name="lung"))

    included = ~np.isnan(mask.values)
    assert mask.name == "lung"
    assert mask.method == "m3resp.eit.roi.watershed_lungspace"
    assert mask.metadata["threshold_fraction"] == 0.15
    assert mask.metadata["included_pixel_count"] == included.sum()
    assert mask.metadata["included_pixel_fraction"] == included.mean()
    # The first and the last breath have no amplitude.
    assert mask.metadata["breath_count"] == len(_breaths(data)) - 2


# --- eitprocessing with other settings -------------------------------------------


@pytest.mark.parametrize("value", [0.05, 0.15, 0.5, 0.9])
@pytest.mark.parametrize("method", ["tiv", "amplitude", "watershed"])
@pytest.mark.parametrize("kind", ["raw", "filtered"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_files_equal_eitprocessing(dataset, kind, method, value) -> None:
    _assert_same_as_upstream(method, (dataset, kind), _file_data(dataset, kind), value)


SYNTHETIC_CASES = [
    "one maximum",
    "many maxima",
    "upside-down",
    "phase-shifted",
    "flat",
    "nan",
]


@pytest.mark.parametrize("value", [0.15, 0.5])
@pytest.mark.parametrize("method", ["tiv", "amplitude", "watershed"])
@pytest.mark.parametrize("case", SYNTHETIC_CASES)
@pytest.mark.parametrize("seed", [1, 2])
def test_synthetic_images_equal_eitprocessing(seed, case, method, value) -> None:
    _assert_same_as_upstream(method, (str(seed), case), _synthetic(seed, case), value)


@pytest.mark.parametrize("method", ["tiv", "amplitude", "watershed"])
@pytest.mark.parametrize("value", [0.15, 0.0, 1.0, 1, "0.5"])
def test_threshold_checks_equal_eitprocessing(method, value) -> None:
    data = _synthetic(1, "one maximum")
    up_mask, up_error, _ = _upstream(method, data, **_setting(method, value))
    mask, error, _ = _ours(method, data, **_setting(method, value))

    assert error is up_error
    if value == 0.15:
        assert error is None
        np.testing.assert_array_equal(mask.values, up_mask.mask)


@pytest.mark.parametrize("method", ["tiv", "amplitude", "watershed"])
def test_no_breaths_is_an_error(method) -> None:
    pixels, waveform = _synthetic(1, "one maximum")
    calls = {
        "tiv": lambda: tiv_lungspace(pixels, []),
        "amplitude": lambda: amplitude_lungspace(pixels, waveform, []),
        "watershed": lambda: watershed_lungspace(pixels, waveform, []),
    }

    with pytest.raises(ValueError, match="breaths"):
        _quiet(calls[method])


def test_watershed_gives_the_errors_of_the_pixel_values_unchanged() -> None:
    pixels, waveform = _synthetic(1, "one maximum")
    breaths = _breaths((pixels, waveform))
    flat_pixels = dataclasses.replace(pixels, values=pixels.values[:, :, 0])

    with pytest.raises(ValueError, match=r"shape \(time, row, column\)"):
        _quiet(lambda: watershed_lungspace(flat_pixels, waveform, breaths))


# --- the size filter ---------------------------------------------------------------


def _random_masks() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    masks = {
        f"density {density}": np.where(rng.random((32, 32)) < density, 1.0, np.nan)
        for density in (0.3, 0.5, 0.7)
    }
    weights = rng.uniform(0.1, 1.0, size=(32, 32))
    weights[rng.random((32, 32)) < 0.5] = np.nan
    masks["weights"] = weights
    masks["empty"] = np.full((32, 32), np.nan)
    for dataset in DATASETS:
        masks[dataset] = _golden(dataset, "roi")["watershed_lungspace/mask"]
        masks[f"{dataset} tiv"] = _golden(dataset, "roi")["tiv_lungspace/mask"]
    return masks


RANDOM_MASKS = _random_masks()
CONNECTIVITIES = {
    "1": 1,
    "2": 2,
    "cross": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    "rows only": np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
    "3": 3,
    "text": "2",
}


@pytest.mark.parametrize("size", [1, 10, 50])
@pytest.mark.parametrize("connectivity", sorted(CONNECTIVITIES))
@pytest.mark.parametrize("mask_name", sorted(RANDOM_MASKS))
def test_size_filter_equals_eitprocessing(mask_name, connectivity, size) -> None:
    pytest.importorskip("eitprocessing")
    from eitprocessing.roi import PixelMask as UpstreamPixelMask
    from eitprocessing.roi.filter_by_size import FilterROIBySize

    grid = RANDOM_MASKS[mask_name]
    structure = CONNECTIVITIES[connectivity]
    expected, up_error = _run(
        lambda: FilterROIBySize(min_region_size=size, connectivity=structure).apply(
            UpstreamPixelMask(grid)
        )
    )
    mask = _quiet(lambda: PixelMask(name="lung", values=grid))
    result, error = _run(
        lambda: filter_mask_by_size(mask, min_region_size=size, connectivity=structure)
    )

    assert error is up_error
    if error is None:
        np.testing.assert_array_equal(result.values, expected.mask)
        assert result.name == "lung_size_filtered"
        assert result.metadata["min_region_size"] == size


def test_size_filter_captures_the_groups() -> None:
    grid = np.full((4, 6), np.nan)
    grid[0, 0:3] = 1.0  # 3 pixels
    grid[2:4, 4:6] = 0.5  # 4 pixels, weight lost (issue I6)
    grid[3, 0] = 1.0  # 1 pixel
    captures: dict[str, Any] = {}

    result = filter_mask_by_size(
        PixelMask(name="m", values=grid),
        min_region_size=3,
        name="big",
        captures=captures,
    )

    assert captures["region sizes"] == [3, 4, 1]
    assert captures["kept regions"] == [1, 2]
    assert captures["regions"].kind == "integer"
    expected = np.where(np.isnan(grid), np.nan, 1.0)
    expected[3, 0] = np.nan
    np.testing.assert_array_equal(result.values, expected)
    assert result.name == "big"


def test_size_filter_takes_a_pixel_mask_only() -> None:
    with pytest.raises(TypeError, match="PixelMask"):
        filter_mask_by_size(np.ones((2, 2)))


# --- documented upstream behaviour -------------------------------------------------


def test_watershed_tiv_mask_uses_the_shared_breaths_only() -> None:
    """The TIV mask inside the watershed method leaves out the first and the
    last breath; tiv_lungspace on those breaths gives the same mask."""

    data = _file_data(TIMPEL, "filtered")
    breaths = _breaths(data)
    captures: dict[str, Any] = {}
    _quiet(lambda: watershed_lungspace(*data, breaths, captures=captures))
    shared = _quiet(lambda: tiv_lungspace(data[0], breaths[1:-1]))

    np.testing.assert_array_equal(captures["functional tiv mask"].values, shared.values)


@pytest.mark.parametrize("dataset", DATASETS)
def test_pixels_without_amplitude_belong_to_no_region(dataset: str) -> None:
    data = _file_data(dataset, "filtered")
    captures: dict[str, Any] = {}
    mask = _quiet(lambda: watershed_lungspace(*data, _breaths(data), captures=captures))

    no_amplitude = np.isnan(captures["mean amplitude of shared breaths"].values)
    assert no_amplitude.any()
    np.testing.assert_array_equal(
        captures["watershed regions"].values == 0, no_amplitude
    )
    assert np.isnan(captures["included region"].values[no_amplitude]).all()
    assert np.isnan(mask.values[no_amplitude]).all()


@pytest.mark.parametrize("dataset", DATASETS)
def test_included_watershed_regions_hold_the_included_regions_only(
    dataset: str,
) -> None:
    data = _file_data(dataset, "filtered")
    captures: dict[str, Any] = {}
    _quiet(lambda: watershed_lungspace(*data, _breaths(data), captures=captures))

    regions = captures["watershed regions"].values
    included = captures["included region"].values == 1
    np.testing.assert_array_equal(
        captures["included watershed regions"].values,
        np.where(included, regions, np.nan),
    )
    np.testing.assert_array_equal(
        np.unique(regions[included]), np.unique(captures["included marker indices"])
    )
    assert np.any(~included & (regions > 0))
