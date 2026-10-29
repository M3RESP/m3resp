"""TIV and EELI of an impedance waveform, computed by m3resp code.

With the breaths of ``find_breaths``, ``tidal_impedance_variation`` and
``end_expiratory_lung_impedance`` must give the numbers of eitprocessing
1.8.7: the golden files ``tiv_eeli.npz``, and ``TIV`` and ``EELI`` on the
same data. Breaths from another source are read at the nearest sample.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pytest

from m3resp.data import BreathEvent, IntervalData, Signal
from m3resp.eit.processing import (
    TIV_METHODS,
    BreathDetectionSettings,
    end_expiratory_lung_impedance,
    find_breaths,
    tidal_impedance_variation,
)
from m3resp.eit.processing._breath_samples import NO_SAMPLE, nearest_samples

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")
with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as _handle:
    MANIFEST = json.load(_handle)
DATASETS = sorted(MANIFEST["inputs"])
MINIMUM_DURATION = MANIFEST["settings"]["breath_detection"]["minimum_duration_s"]
SEEDS = range(20)


def _golden(dataset: str, name: str) -> dict[str, np.ndarray]:
    """Return all arrays of one golden ``.npz`` file."""

    with np.load(os.path.join(GOLDEN_DIR, dataset, f"{name}.npz")) as data:
        return {key: data[key] for key in data.files}


def _waveform(
    values: np.ndarray,
    time: np.ndarray | None = None,
    *,
    fs: float = 20.0,
    category: str | None = "impedance",
) -> Signal:
    values = np.asarray(values, dtype=float)
    return Signal(
        values=values,
        time=np.arange(len(values)) / fs if time is None else np.asarray(time),
        sample_frequency=fs,
        unit="AU",
        name="global_impedance",
        modality="eit",
        category=category,
    )


def _golden_waveform(dataset: str, kind: str) -> Signal:
    """The raw or the MDN-filtered global impedance of a golden file."""

    data = _golden(dataset, "global_impedance")
    time = data[f"{kind}/time"]
    return _waveform(data[f"{kind}/values"], time, fs=1 / np.median(np.diff(time)))


def _random_waveform(seed: int) -> Signal:
    """A breathing-like waveform with drift, noise, and some runs of NaN."""

    rng = np.random.default_rng(seed)
    fs = float(rng.choice([20.0, 50.0, 50.2]))
    seconds = rng.uniform(20.0, 120.0)
    time = 1000.0 + np.arange(int(seconds * fs)) / fs
    rate = rng.uniform(0.15, 0.8)
    amplitude = rng.uniform(0.5, 5.0)
    values = amplitude * np.sin(2 * np.pi * rate * time + rng.uniform(0, 2 * np.pi))
    values += rng.uniform(-1.0, 1.0) * amplitude * (time - time[0]) / seconds
    values += rng.normal(0.0, rng.uniform(0.01, 0.3) * amplitude, len(time))
    if rng.random() < 0.4:
        start = int(rng.integers(0, len(time) - 10))
        values[start : start + int(rng.integers(1, 2 * fs))] = np.nan
    return _waveform(values, time, fs=fs)


def _settings() -> BreathDetectionSettings:
    return BreathDetectionSettings(minimum_duration=MINIMUM_DURATION)


def _upstream(signal: Signal, tiv_method: str = "inspiratory") -> tuple:
    """eitprocessing ``TIV`` and ``EELI`` on the same waveform."""

    pytest.importorskip("eitprocessing")
    from eitprocessing.features.breath_detection import BreathDetection
    from eitprocessing.parameters.eeli import EELI
    from eitprocessing.parameters.tidal_impedance_variation import TIV

    from m3resp.adapters.eitprocessing_adapter._signals import as_continuous_data

    detector = BreathDetection(minimum_duration=MINIMUM_DURATION)
    data = as_continuous_data(signal)
    tiv = TIV(breath_detection=detector).compute_parameter(
        data, tiv_method=tiv_method, store=False
    )
    eeli = EELI(breath_detection=detector).compute_parameter(data, store=False)
    return tiv, eeli


def _breath(start: float, extremum: float | None, end: float) -> BreathEvent:
    return BreathEvent("eit", start, end, extremum_time=extremum)


# Golden files and eitprocessing


@pytest.mark.parametrize("dataset", DATASETS)
def test_tiv_and_eeli_match_golden(dataset: str) -> None:
    signal = _golden_waveform(dataset, "filtered")
    breaths = find_breaths(signal, _settings())
    golden = _golden(dataset, "tiv_eeli")

    tiv = tidal_impedance_variation(signal, breaths)
    eeli = end_expiratory_lung_impedance(signal, breaths)

    np.testing.assert_array_equal(tiv.values, golden["tiv/values"])
    np.testing.assert_array_equal(eeli.values, golden["eeli/values"])
    # eitprocessing places TIV at the middle of each breath, EELI at its end.
    np.testing.assert_array_equal(
        [breath.extremum_time for breath in tiv.intervals], golden["tiv/time"]
    )
    np.testing.assert_array_equal(
        [breath.end_time for breath in eeli.intervals], golden["eeli/time"]
    )


@pytest.mark.parametrize("method", TIV_METHODS)
@pytest.mark.parametrize("kind", ["raw", "filtered"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_golden_waveforms_equal_eitprocessing(
    dataset: str, kind: str, method: str
) -> None:
    signal = _golden_waveform(dataset, kind)
    breaths = find_breaths(signal, _settings())
    upstream_tiv, upstream_eeli = _upstream(signal, method)

    tiv = tidal_impedance_variation(signal, breaths, method=method)
    eeli = end_expiratory_lung_impedance(signal, breaths)

    np.testing.assert_array_equal(tiv.values, np.asarray(upstream_tiv.values))
    np.testing.assert_array_equal(eeli.values, np.asarray(upstream_eeli.values))
    np.testing.assert_array_equal(
        [breath.extremum_time for breath in tiv.intervals], upstream_tiv.time
    )
    np.testing.assert_array_equal(
        [breath.end_time for breath in eeli.intervals], upstream_eeli.time
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_random_waveforms_equal_eitprocessing(seed: int) -> None:
    signal = _random_waveform(seed)
    breaths = find_breaths(signal, _settings())
    for method in TIV_METHODS:
        upstream_tiv, upstream_eeli = _upstream(signal, method)
        tiv = tidal_impedance_variation(signal, breaths, method=method)
        np.testing.assert_array_equal(tiv.values, np.asarray(upstream_tiv.values))
    np.testing.assert_array_equal(
        end_expiratory_lung_impedance(signal, breaths).values,
        np.asarray(upstream_eeli.values),
    )


def test_random_waveforms_cover_nan_and_several_breaths() -> None:
    """At least one random waveform has NaN, and each has breaths."""

    signals = [_random_waveform(seed) for seed in SEEDS]
    assert any(np.isnan(signal.values).any() for signal in signals)
    assert all(len(find_breaths(signal, _settings())) > 1 for signal in signals)


def test_nan_at_a_breath_time_gives_nan_for_that_breath() -> None:
    signal = _golden_waveform(DATASETS[0], "filtered")
    breaths = find_breaths(signal, _settings())
    values = np.array(signal.values, dtype=float)
    values[breaths[2].extremum_index] = np.nan
    values[breaths[4].end_index] = np.nan
    with_nan = _waveform(values, signal.time, fs=signal.sample_frequency)

    tiv = tidal_impedance_variation(with_nan, breaths)
    eeli = end_expiratory_lung_impedance(with_nan, breaths)

    # The end of breath 4 is the start of breath 5.
    assert np.flatnonzero(np.isnan(tiv.values)).tolist() == [2, 5]
    assert np.flatnonzero(np.isnan(eeli.values)).tolist() == [4]
    clean_tiv = tidal_impedance_variation(signal, breaths).values
    np.testing.assert_array_equal(
        np.delete(tiv.values, [2, 5]), np.delete(clean_tiv, [2, 5])
    )


# No breaths, one breath, the result


def test_no_breaths_give_an_empty_result() -> None:
    signal = _waveform(np.ones(100))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tiv = tidal_impedance_variation(signal, [])
        eeli = end_expiratory_lung_impedance(signal, [])
    assert len(tiv) == 0 and len(eeli) == 0
    assert np.asarray(tiv.values).shape == (0,)
    assert tiv.metadata["breath_modality"] is None


def test_one_breath_gives_its_values() -> None:
    values = np.array([1.0, 2.0, 5.0, 3.0, 0.5])
    signal = _waveform(values, np.arange(5.0), fs=1.0)
    breath = _breath(0.0, 2.0, 4.0)

    assert tidal_impedance_variation(signal, [breath]).values[0] == 4.0
    expiratory = tidal_impedance_variation(signal, [breath], method="expiratory")
    assert expiratory.values[0] == 4.5
    assert tidal_impedance_variation(signal, [breath], method="mean").values[0] == 4.25
    assert end_expiratory_lung_impedance(signal, [breath]).values[0] == 0.5


def test_the_result_keeps_the_breaths_and_describes_the_values() -> None:
    signal = _golden_waveform(DATASETS[0], "filtered")
    breaths = find_breaths(signal, _settings())

    tiv = tidal_impedance_variation(signal, breaths, method="mean")
    eeli = end_expiratory_lung_impedance(signal, breaths, name="my_eeli")

    assert isinstance(tiv, IntervalData) and isinstance(eeli, IntervalData)
    assert all(a is b for a, b in zip(tiv.intervals, breaths, strict=True))
    assert (tiv.name, eeli.name) == ("continuous_tivs", "my_eeli")
    assert tiv.modality == eeli.modality == "eit"
    assert tiv.category == eeli.category == "impedance"
    assert tiv.unit == eeli.unit == signal.unit
    assert tiv.method == "m3resp.eit.processing.tidal_impedance_variation"
    assert eeli.method == "m3resp.eit.processing.end_expiratory_lung_impedance"
    assert tiv.metadata == {
        "tiv_method": "mean",
        "signal_name": "global_impedance",
        "breath_modality": "eit",
        "breath_source": "m3resp.eit.processing.find_breaths",
    }
    assert eeli.metadata["breath_source"] == "m3resp.eit.processing.find_breaths"


def test_breaths_of_several_sources_are_listed() -> None:
    signal = _waveform(np.arange(100.0))
    breaths = [
        BreathEvent("ventilator", 0.5, 1.5, extremum_time=1.0, source="b"),
        BreathEvent("emg", 2.0, 3.0, extremum_time=2.5, source="a"),
    ]
    metadata = end_expiratory_lung_impedance(signal, breaths).metadata
    assert metadata["breath_modality"] == ["emg", "ventilator"]
    assert metadata["breath_source"] == ["a", "b"]


def test_the_input_is_left_unchanged() -> None:
    signal = _golden_waveform(DATASETS[0], "filtered")
    breaths = find_breaths(signal, _settings())
    values = np.copy(signal.values)
    times = [(b.start_time, b.extremum_time, b.end_time) for b in breaths]

    tidal_impedance_variation(signal, breaths)
    end_expiratory_lung_impedance(signal, breaths)

    np.testing.assert_array_equal(signal.values, values)
    assert [(b.start_time, b.extremum_time, b.end_time) for b in breaths] == times


# Breaths between samples and outside the data


def test_nearest_samples() -> None:
    time = np.array([10.0, 10.5, 11.0, 11.5])
    indices = nearest_samples(
        time, [10.0, 10.2, 10.25, 10.3, 11.5, 11.74, 11.76, 9.76, 9.74, None, np.nan]
    )
    assert indices.tolist() == [0, 0, 1, 1, 3, 3, NO_SAMPLE, 0] + [NO_SAMPLE] * 3


def test_times_in_a_gap_of_the_time_axis_have_no_sample() -> None:
    # Samples every 0.5 s, with a gap from 11 s to 20 s as between two files.
    time = np.array([10.0, 10.5, 11.0, 20.0, 20.5])
    indices = nearest_samples(time, [11.2, 11.25, 11.3, 15.0, 19.7, 19.75, 20.2])
    assert indices.tolist() == [2, 2, NO_SAMPLE, NO_SAMPLE, NO_SAMPLE, 3, 3]


def test_a_step_up_to_one_and_a_half_samples_is_no_gap() -> None:
    # The median step is 0.5 s; the step of 0.7 s from 1.0 s to 1.7 s is no gap.
    time = np.array([0.0, 0.5, 1.0, 1.7, 2.2, 2.7])
    assert nearest_samples(time, [1.3, 1.35, 1.4]).tolist() == [2, 3, 3]


def test_breaths_in_a_gap_of_the_time_axis_give_nan_with_a_warning() -> None:
    # Two parts of 10 samples, 1 s apart, with a gap from 9 s to 70 s.
    time = np.concatenate([np.arange(10.0), 70.0 + np.arange(10.0)])
    signal = _waveform(np.arange(20.0), time, fs=1.0)
    breaths = [
        _breath(2.0, 4.0, 6.0),
        _breath(7.0, 8.0, 30.0),
        _breath(71.0, 72.0, 74.0),
    ]

    with pytest.warns(UserWarning, match="1 of 3 breaths lie .* in a gap"):
        tiv = tidal_impedance_variation(signal, breaths)
    with pytest.warns(UserWarning, match="1 of 3 breaths lie .* in a gap"):
        eeli = end_expiratory_lung_impedance(signal, breaths)

    np.testing.assert_array_equal(tiv.values, [2.0, np.nan, 1.0])
    np.testing.assert_array_equal(eeli.values, [6.0, np.nan, 14.0])


def test_breaths_between_samples_are_read_at_the_nearest_sample() -> None:
    values = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    signal = _waveform(values, np.arange(7.0), fs=1.0)
    breath = _breath(0.6, 2.4, 5.5)  # samples 1, 2 and 6 (halfway: the later)

    assert tidal_impedance_variation(signal, [breath]).values[0] == 3.0
    assert end_expiratory_lung_impedance(signal, [breath]).values[0] == 36.0


def test_breaths_partly_outside_the_data_give_nan_with_one_warning() -> None:
    signal = _waveform(np.arange(10.0), np.arange(10.0), fs=1.0)
    breaths = [_breath(-2.0, 1.0, 3.0), _breath(2.0, 4.0, 6.0), _breath(8.0, 9.0, 12.0)]
    message = "2 of 3 breaths lie partly outside"

    with pytest.warns(UserWarning, match=message) as tiv_warnings:
        tiv = tidal_impedance_variation(signal, breaths)
    with pytest.warns(UserWarning, match=message):
        eeli = end_expiratory_lung_impedance(signal, breaths)

    assert len(tiv_warnings) == 1
    np.testing.assert_array_equal(tiv.values, [np.nan, 2.0, np.nan])
    np.testing.assert_array_equal(eeli.values, [np.nan, 6.0, np.nan])


def test_a_breath_with_no_extremum_has_no_tiv_but_an_eeli() -> None:
    signal = _waveform(np.arange(10.0), np.arange(10.0), fs=1.0)
    breaths = [_breath(1.0, None, 3.0), _breath(3.0, 4.0, 6.0)]

    with pytest.warns(UserWarning, match="1 of 2 breaths have no extremum_time"):
        tiv = tidal_impedance_variation(signal, breaths)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eeli = end_expiratory_lung_impedance(signal, breaths)

    np.testing.assert_array_equal(tiv.values, [np.nan, 1.0])
    np.testing.assert_array_equal(eeli.values, [3.0, 6.0])


# Wrong input


def test_a_wrong_tiv_method_is_refused_as_by_eitprocessing() -> None:
    signal = _golden_waveform(DATASETS[0], "filtered")
    with pytest.raises(ValueError, match="Unknown TIV method 'peak'"):
        tidal_impedance_variation(signal, [], method="peak")
    with pytest.raises(ValueError):
        _upstream(signal, "peak")


def test_data_that_is_not_impedance_is_refused() -> None:
    pressure = _waveform(np.arange(10.0), category="airway_pressure")
    with pytest.raises(ValueError, match="impedance data only"):
        tidal_impedance_variation(pressure, [])
    with pytest.raises(ValueError, match="impedance data only"):
        end_expiratory_lung_impedance(pressure, [])

    pytest.importorskip("eitprocessing")
    from eitprocessing.parameters.eeli import EELI

    from m3resp.adapters.eitprocessing_adapter._signals import as_continuous_data

    with pytest.raises(ValueError):
        EELI().compute_parameter(as_continuous_data(pressure))


@pytest.mark.parametrize(
    "category", [None, "impedance difference", "relative impedance"]
)
def test_impedance_categories_and_no_category_pass(category: str | None) -> None:
    signal = _waveform(np.arange(10.0), category=category)
    assert len(end_expiratory_lung_impedance(signal, [_breath(0.0, 0.1, 0.2)])) == 1


def test_wrong_types_and_shapes_are_refused() -> None:
    with pytest.raises(TypeError, match="needs a Signal"):
        tidal_impedance_variation(np.arange(10.0), [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="needs BreathEvent objects"):
        end_expiratory_lung_impedance(_waveform(np.arange(10.0)), [(0.0, 1.0, 2.0)])
    images = Signal(
        values=np.zeros((10, 2, 2)),
        time=np.arange(10.0),
        sample_frequency=1.0,
        name="images",
        modality="eit",
    )
    with pytest.raises(ValueError, match="one value per sample"):
        tidal_impedance_variation(images, [])
