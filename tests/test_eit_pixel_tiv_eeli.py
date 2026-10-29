"""Pixel TIV and pixel EELI of EIT images, computed by m3resp code.

``pixel_tidal_impedance_variation`` must give the numbers of eitprocessing
1.8.7: the golden files ``pixel.npz``, and ``TIV.compute_pixel_parameter``
with both timings and the three TIV methods on the same data.
``pixel_end_expiratory_lung_impedance`` has no eitprocessing version; it is
checked against a hand calculation and against the EELI of the global
impedance.
"""

from __future__ import annotations

import functools
import json
import os
import warnings
from typing import Any

import numpy as np
import pytest

from m3resp.data import BreathEvent, PixelBreaths, Signal
from m3resp.eit.processing import (
    TIV_METHODS,
    BreathDetectionSettings,
    end_expiratory_lung_impedance,
    find_breaths,
    global_impedance,
    pixel_end_expiratory_lung_impedance,
    pixel_tidal_impedance_variation,
    tidal_impedance_variation,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")
with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as _handle:
    MANIFEST = json.load(_handle)
DATASETS = sorted(MANIFEST["inputs"])
MINIMUM_DURATION = MANIFEST["settings"]["breath_detection"]["minimum_duration_s"]
TIMINGS = ("continuous", "pixel")
TIMPEL = "timpel_healthy_volunteer_1_60-120s"


def _golden(dataset: str, name: str) -> dict[str, np.ndarray]:
    """Return all arrays of one golden ``.npz`` file."""

    with np.load(os.path.join(GOLDEN_DIR, dataset, f"{name}.npz")) as data:
        return {key: data[key] for key in data.files}


def _signal(
    values: np.ndarray, time: np.ndarray, *, name: str, source: Any = None
) -> Signal:
    return Signal(
        values=np.asarray(values, dtype=float),
        time=np.asarray(time, dtype=float),
        sample_frequency=1 / float(np.median(np.diff(time))),
        unit="AU",
        name=name,
        modality="eit",
        category="impedance",
        source=None if source is None else str(source),
    )


@functools.cache
def _golden_data(dataset: str, kind: str) -> tuple[Signal, Signal]:
    """The raw or the MDN-filtered pixels and global impedance of a golden file."""

    reader = _golden(dataset, "reader")
    pixels = reader["pixel_impedance"]
    if kind == "filtered":
        pixels = _golden(dataset, "mdn_filter")["pixel_impedance"]
    waveform = _golden(dataset, "global_impedance")
    return (
        _signal(pixels, reader["time"], name="pixels", source=reader["vendor"]),
        _signal(
            waveform[f"{kind}/values"],
            waveform[f"{kind}/time"],
            name="global_impedance",
        ),
    )


@functools.cache
def _breaths(dataset: str, kind: str) -> list[BreathEvent]:
    _, waveform = _golden_data(dataset, kind)
    settings = BreathDetectionSettings(minimum_duration=MINIMUM_DURATION)
    return find_breaths(waveform, settings)


@functools.cache
def _upstream_pixel_breaths(dataset: str, kind: str) -> PixelBreaths:
    """The breaths of eitprocessing ``PixelBreath``, as ``PixelBreaths``."""

    pytest.importorskip("eitprocessing")
    from eitprocessing.features.breath_detection import BreathDetection
    from eitprocessing.features.pixel_breath import PixelBreath

    from m3resp.adapters.eitprocessing_adapter._signals import (
        as_continuous_data,
        as_eit_data,
    )

    pixels, waveform = _golden_data(dataset, kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = PixelBreath(
            breath_detection=BreathDetection(minimum_duration=MINIMUM_DURATION)
        ).find_pixel_breaths(
            as_eit_data(pixels), as_continuous_data(waveform), store=False
        )
    breaths = _breaths(dataset, kind)
    assert [(b.start_time, b.end_time) for b in breaths] == [
        (float(start), float(end)) for start, end in result.intervals
    ]
    per_pixel = np.asarray(result.values, dtype=object)
    times = np.full((*per_pixel.shape, 3), np.nan)
    for index, breath in np.ndenumerate(per_pixel):
        if breath is not None:
            times[index] = (breath.start_time, breath.middle_time, breath.end_time)
    return PixelBreaths(
        name="pixel_breaths",
        breaths=breaths,
        start_time=times[..., 0],
        extremum_time=times[..., 1],
        end_time=times[..., 2],
        method="eitprocessing.PixelBreath",
    )


def _upstream_pixel_tiv(dataset: str, kind: str, method: str, timing: str) -> Any:
    """eitprocessing ``TIV.compute_pixel_parameter`` on the same data."""

    pytest.importorskip("eitprocessing")
    from eitprocessing.features.breath_detection import BreathDetection
    from eitprocessing.parameters.tidal_impedance_variation import TIV

    from m3resp.adapters.eitprocessing_adapter._signals import (
        as_continuous_data,
        as_eit_data,
    )

    pixels, waveform = _golden_data(dataset, kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TIV(
            breath_detection=BreathDetection(minimum_duration=MINIMUM_DURATION)
        ).compute_pixel_parameter(
            as_eit_data(pixels),
            as_continuous_data(waveform),
            None,
            tiv_method=method,
            tiv_timing=timing,
            store=False,
        )


def _maps(result: Any) -> np.ndarray:
    return np.stack([pixel_map.values for pixel_map in result.values])


def _no_pixel_breaths(
    breaths: list[BreathEvent], shape: tuple[int, int]
) -> PixelBreaths:
    empty = np.full((len(breaths), *shape), np.nan)
    return PixelBreaths(
        name="pixel_breaths",
        breaths=breaths,
        start_time=empty,
        extremum_time=empty,
        end_time=empty,
        method="test",
    )


# Golden files and eitprocessing


@pytest.mark.parametrize("dataset", DATASETS)
def test_pixel_tiv_matches_golden(dataset: str) -> None:
    pixels, _ = _golden_data(dataset, "filtered")
    breaths = _breaths(dataset, "filtered")
    golden = _golden(dataset, "pixel")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pixel_tidal_impedance_variation(pixels, breaths)

    np.testing.assert_array_equal(_maps(result), golden["tiv/values"])
    # With global timing every pixel of a breath has the time of the breath.
    np.testing.assert_array_equal(
        [breath.extremum_time for breath in result.intervals],
        golden["tiv/time"][:, 0, 0],
    )


@pytest.mark.parametrize("timing", TIMINGS)
@pytest.mark.parametrize("kind", ["raw", "filtered"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_pixel_tiv_equals_eitprocessing(dataset: str, kind: str, timing: str) -> None:
    pixels, _ = _golden_data(dataset, kind)
    breaths: list[BreathEvent] | PixelBreaths = _breaths(dataset, kind)
    if timing == "pixel":
        breaths = _upstream_pixel_breaths(dataset, kind)

    for method in TIV_METHODS:
        result = pixel_tidal_impedance_variation(pixels, breaths, method=method)
        upstream = _upstream_pixel_tiv(dataset, kind, method, timing)
        np.testing.assert_array_equal(
            _maps(result), np.asarray(upstream.values, dtype=float)
        )
        assert result.metadata["tiv_timing"] == timing


# Sums over the pixels


@pytest.mark.parametrize("kind", ["raw", "filtered"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_sum_over_the_pixels_equals_the_global_value(dataset: str, kind: str) -> None:
    """The global impedance is the sum of the pixels, so with the same breath
    times the sum of the pixel values is the value of the global impedance."""

    pixels, _ = _golden_data(dataset, kind)
    breaths = _breaths(dataset, kind)
    summed = global_impedance(pixels)
    scale = float(np.nanmax(np.abs(summed.values)))

    for method in TIV_METHODS:
        pixel_tiv = pixel_tidal_impedance_variation(pixels, breaths, method=method)
        tiv = tidal_impedance_variation(summed, breaths, method=method)
        np.testing.assert_allclose(
            np.nansum(_maps(pixel_tiv), axis=(1, 2)),
            np.asarray(tiv.values),
            rtol=0,
            atol=1e-9 * scale,
        )
    pixel_eeli = pixel_end_expiratory_lung_impedance(pixels, breaths)
    eeli = end_expiratory_lung_impedance(summed, breaths)
    np.testing.assert_allclose(
        np.nansum(_maps(pixel_eeli), axis=(1, 2)),
        np.asarray(eeli.values),
        rtol=0,
        atol=1e-9 * scale,
    )


# Calculations by hand


def test_pixel_eeli_by_hand() -> None:
    # 1 x 2 image at 1 Hz. Pixel (0, 0) is time * 10, pixel (0, 1) time * 100.
    time = np.arange(10.0)
    values = np.stack([time * 10, time * 100], axis=-1)[:, np.newaxis, :]
    pixels = _signal(values, time, name="pixels")
    breaths = [
        BreathEvent("eit", 1.0, 4.0, extremum_time=2.0),
        BreathEvent("eit", 4.0, 7.0, extremum_time=5.0),
    ]
    pixel_breaths = PixelBreaths(
        name="pixel_breaths",
        breaths=breaths,
        start_time=np.array([[[1.0, 2.0]], [[4.0, np.nan]]]),
        extremum_time=np.array([[[2.0, 3.0]], [[5.0, np.nan]]]),
        end_time=np.array([[[3.0, 5.0]], [[8.0, np.nan]]]),
    )

    continuous = pixel_end_expiratory_lung_impedance(pixels, breaths)
    per_pixel = pixel_end_expiratory_lung_impedance(pixels, pixel_breaths)

    np.testing.assert_array_equal(_maps(continuous), [[[40.0, 400.0]], [[70.0, 700.0]]])
    np.testing.assert_array_equal(_maps(per_pixel), [[[30.0, 500.0]], [[80.0, np.nan]]])
    assert continuous.metadata["eeli_timing"] == "continuous"
    assert per_pixel.metadata["eeli_timing"] == "pixel"


def test_pixel_tiv_by_hand() -> None:
    # 1 x 2 image at 1 Hz. Pixel (0, 0) is time squared, pixel (0, 1) -time.
    time = np.arange(10.0)
    values = np.stack([time**2, -time], axis=-1)[:, np.newaxis, :]
    pixels = _signal(values, time, name="pixels")
    breaths = [BreathEvent("eit", 1.0, 4.0, extremum_time=2.0)]

    expected = {
        "inspiratory": [[4.0 - 1.0, -2.0 + 1.0]],
        "expiratory": [[4.0 - 16.0, -2.0 + 4.0]],
        "mean": [[4.0 - 8.5, -2.0 + 2.5]],
    }
    for method, grid in expected.items():
        result = pixel_tidal_impedance_variation(pixels, breaths, method=method)
        np.testing.assert_array_equal(_maps(result)[0], grid)


# The result


def test_the_result_keeps_the_global_breaths_and_describes_the_maps() -> None:
    pixels, _ = _golden_data(DATASETS[0], "filtered")
    pixel_breaths = _no_pixel_breaths(_breaths(DATASETS[0], "filtered"), (32, 32))

    for function, name in (
        (pixel_tidal_impedance_variation, "pixel_tivs"),
        (pixel_end_expiratory_lung_impedance, "pixel_eelis"),
    ):
        result = function(pixels, pixel_breaths)
        assert result.name == name
        assert result.intervals == pixel_breaths.breaths
        assert result.unit == pixels.unit == "a.u."
        assert result.category == "impedance"
        assert result.metadata["signal_name"] == "pixels"
        assert result.metadata["breath_source"] == "test"
        assert all(
            pixel_map.name == name
            and pixel_map.unit == "a.u."
            and pixel_map.category == "impedance"
            and pixel_map.shape == (32, 32)
            for pixel_map in result.values
        )
        assert np.isnan(_maps(result)).all()


@pytest.mark.parametrize("timing", TIMINGS)
def test_all_nan_pixels_give_nan_and_no_warning(timing: str) -> None:
    pixels, _ = _golden_data(TIMPEL, "filtered")
    all_nan = np.isnan(pixels.values).all(axis=0)
    assert all_nan.sum() == 501
    breaths: list[BreathEvent] | PixelBreaths = _breaths(TIMPEL, "filtered")
    if timing == "pixel":
        breaths = _upstream_pixel_breaths(TIMPEL, "filtered")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tiv = _maps(pixel_tidal_impedance_variation(pixels, breaths))
        eeli = _maps(pixel_end_expiratory_lung_impedance(pixels, breaths))

    assert np.isnan(tiv[:, all_nan]).all()
    assert np.isnan(eeli[:, all_nan]).all()
    assert not np.isnan(eeli[1:-1, ~all_nan]).all()


def test_no_breaths_give_an_empty_result() -> None:
    pixels, _ = _golden_data(DATASETS[0], "filtered")

    for breaths in ([], _no_pixel_breaths([], (32, 32))):
        assert pixel_tidal_impedance_variation(pixels, breaths).values == []
        assert pixel_end_expiratory_lung_impedance(pixels, breaths).values == []


def test_global_breaths_partly_outside_the_data_give_nan_with_one_warning() -> None:
    pixels = _signal(np.ones((10, 2, 2)), np.arange(10.0), name="pixels")
    breaths = [
        BreathEvent("eit", 1.0, 4.0, extremum_time=2.0),
        BreathEvent("eit", 8.0, 12.0, extremum_time=9.0),
    ]

    with pytest.warns(UserWarning, match="1 of 2 breaths lie partly outside"):
        result = pixel_tidal_impedance_variation(pixels, breaths)

    assert (_maps(result)[0] == 0).all()
    assert np.isnan(_maps(result)[1]).all()


def test_the_input_is_left_unchanged() -> None:
    pixels, _ = _golden_data(DATASETS[0], "filtered")
    values = pixels.values.copy()
    breaths = _breaths(DATASETS[0], "filtered")

    pixel_tidal_impedance_variation(pixels, breaths)
    pixel_end_expiratory_lung_impedance(pixels, breaths)

    np.testing.assert_array_equal(pixels.values, values)


# Wrong input


@pytest.mark.parametrize(
    "function", [pixel_tidal_impedance_variation, pixel_end_expiratory_lung_impedance]
)
def test_wrong_input_is_refused(function: Any) -> None:
    pixels, waveform = _golden_data(DATASETS[0], "filtered")
    breaths = _breaths(DATASETS[0], "filtered")
    pressure = Signal(
        values=pixels.values,
        time=pixels.time,
        sample_frequency=pixels.sample_frequency,
        name="pressure",
        category="pressure",
    )

    with pytest.raises(ValueError, match=r"shape \(time, row, column\)"):
        function(waveform, breaths)
    with pytest.raises(ValueError, match="2 x 2 grid"):
        function(pixels, _no_pixel_breaths(breaths, (2, 2)))
    with pytest.raises(ValueError, match="2 x 2 grid"):
        function(pixels, _no_pixel_breaths([], (2, 2)))
    with pytest.raises(TypeError, match="needs BreathEvent objects"):
        function(pixels, [(0.0, 1.0, 2.0)])
    with pytest.raises(TypeError, match="needs a Signal"):
        function(pixels.values, breaths)
    with pytest.raises(ValueError, match="impedance data only"):
        function(pressure, breaths)


def test_a_wrong_tiv_method_is_refused() -> None:
    pixels, _ = _golden_data(DATASETS[0], "filtered")
    with pytest.raises(ValueError, match="Unknown TIV method"):
        pixel_tidal_impedance_variation(pixels, [], method="peak")
