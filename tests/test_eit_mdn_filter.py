"""MDN filter of EIT data, run by m3resp code.

``mdn_filter`` must give the numbers of eitprocessing 1.8.7: the golden files
``mdn_filter.npz`` and ``MDNFilter.apply`` on the same data.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pytest

from m3resp.data import Signal
from m3resp.eit.io import read_eit_file
from m3resp.eit.processing import mdn_filter, mdn_frequency_bands

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")
with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as _handle:
    MANIFEST = json.load(_handle)
DATASETS = sorted(MANIFEST["inputs"])
MINUTE = 60

#: Rates and settings in Hz, each with a different number of removed harmonics
#: or another setting. The keys are the eitprocessing setting names.
SETTINGS = {
    "3 harmonics": {"respiratory_rate": 0.25, "heart_rate": 1.2},
    "2 harmonics": {"respiratory_rate": 0.25, "heart_rate": 1.9},
    "4 harmonics": {"respiratory_rate": 0.3, "heart_rate": 0.9},
    "1 harmonic": {"respiratory_rate": 0.25, "heart_rate": 3.5},
    "first-harmonic rule": {"respiratory_rate": 1.0, "heart_rate": 1.2},
    "other limit, notch and order": {
        "respiratory_rate": 0.2,
        "heart_rate": 0.8,
        "noise_frequency_limit": 2.0,
        "notch_distance": 0.1,
        "order": 4,
    },
}


@pytest.fixture(scope="module")
def images() -> dict[str, Signal]:
    """The EIT images of the 3 test files, read by m3resp."""

    return {
        dataset: read_eit_file(
            os.path.join(REPO_ROOT, MANIFEST["inputs"][dataset]["file"]),
            MANIFEST["inputs"][dataset]["vendor"],
        ).pixel_impedance
        for dataset in DATASETS
    }


def _golden(dataset: str, name: str) -> dict[str, np.ndarray]:
    """Return all arrays of one golden ``.npz`` file."""

    with np.load(os.path.join(GOLDEN_DIR, dataset, f"{name}.npz")) as data:
        return {key: data[key] for key in data.files}


def _synthetic(
    shape: tuple[int, ...] = (6, 6),
    *,
    fs: float = 20.0,
    seconds: float = 60.0,
    breathing_hz: float = 0.25,
    heart_hz: float = 1.2,
) -> Signal:
    """A breathing signal, a heart signal with 4 harmonics, and noise.

    ``shape`` is the shape of one sample: () for one waveform, (rows,
    columns) for EIT images.
    """

    time = np.arange(int(seconds * fs)) / fs
    rng = np.random.default_rng(7)
    breathing = np.sin(2 * np.pi * breathing_hz * time)
    heart = sum(np.sin(2 * np.pi * k * heart_hz * time) / k for k in range(1, 5))
    weights = rng.uniform(0.2, 1.0, size=shape)
    values = (
        np.multiply.outer(breathing, weights)
        + 0.2 * np.multiply.outer(heart, np.ones(shape))
        + 0.05 * rng.standard_normal((len(time), *shape))
        + 3.0
    )
    return Signal(
        values=values,
        time=time,
        sample_frequency=fs,
        unit="AU",
        name="synthetic",
        modality="eit",
        source="draeger",
        metadata={"note": "synthetic"},
    )


def _upstream(signal: Signal, captures: dict | None = None, **settings) -> np.ndarray:
    """eitprocessing ``MDNFilter.apply`` on the same values."""

    pytest.importorskip("eitprocessing")
    from eitprocessing.filters.mdn import MDNFilter

    return MDNFilter(**settings).apply(
        np.asarray(signal.values),
        sample_frequency=signal.sample_frequency,
        captures=captures,
    )


def _band_settings(settings: dict) -> dict:
    """The settings that ``mdn_frequency_bands`` takes (all except the order)."""

    return {key: value for key, value in settings.items() if key != "order"}


# Golden files


@pytest.mark.parametrize("dataset", DATASETS)
def test_filtered_images_and_captures_match_golden(dataset: str, images: dict) -> None:
    rates = _golden(dataset, "rates")
    golden = _golden(dataset, "mdn_filter")
    captures: dict = {}

    result = mdn_filter(
        images[dataset],
        respiratory_rate=float(rates["respiratory_rate_hz"]),
        heart_rate=float(rates["heart_rate_hz"]),
        captures=captures,
    )

    np.testing.assert_array_equal(result.values, golden["pixel_impedance"])
    assert captures["low_pass_frequency"] == golden["captures/low_pass_frequency"]
    assert captures["n_harmonics"] == golden["captures/n_harmonics"]
    np.testing.assert_array_equal(
        np.asarray(captures["frequency_bands"]), golden["captures/frequency_bands"]
    )
    assert captures["unfiltered_data"] is images[dataset]
    assert captures["filtered_data"] is result


# Same numbers as eitprocessing


@pytest.mark.parametrize("fs", [20.0, 50.0])
@pytest.mark.parametrize("shape", [(), (6, 6)], ids=["waveform", "images"])
@pytest.mark.parametrize("case", sorted(SETTINGS))
def test_values_and_captures_equal_eitprocessing(
    case: str, shape: tuple[int, ...], fs: float
) -> None:
    settings = SETTINGS[case]
    signal = _synthetic(shape, fs=fs, heart_hz=settings["heart_rate"])
    expected_captures: dict = {}
    captures: dict = {}

    expected = _upstream(signal, expected_captures, **settings)
    result = mdn_filter(signal, captures=captures, **settings)

    np.testing.assert_array_equal(result.values, expected)
    assert list(captures) == list(expected_captures)
    for key in ("low_pass_frequency", "n_harmonics", "frequency_bands"):
        assert captures[key] == expected_captures[key], key
        assert type(captures[key]) is type(expected_captures[key]), key
    np.testing.assert_array_equal(
        captures["unfiltered_data"].values, expected_captures["unfiltered_data"]
    )
    np.testing.assert_array_equal(
        captures["filtered_data"].values, expected_captures["filtered_data"]
    )


def test_no_captures_give_the_same_result() -> None:
    signal = _synthetic((3, 3))

    with_captures = mdn_filter(signal, captures={}, **SETTINGS["3 harmonics"])
    without_captures = mdn_filter(signal, **SETTINGS["3 harmonics"])

    np.testing.assert_array_equal(without_captures.values, with_captures.values)


# Frequency bands


def test_the_first_band_starts_halfway_between_the_rates_when_they_are_close() -> None:
    bands = mdn_frequency_bands(respiratory_rate=1.0, heart_rate=1.2)

    assert bands[0] == pytest.approx((1.1, 1.2 + 10 / MINUTE))
    assert bands[1] == pytest.approx((2.4 - 10 / MINUTE, 2.4 + 10 / MINUTE))


@pytest.mark.parametrize("case", sorted(SETTINGS))
def test_frequency_bands_are_the_bands_the_filter_removes(case: str) -> None:
    settings = SETTINGS[case]
    captures: dict = {}

    mdn_filter(_synthetic((3, 3)), captures=captures, **settings)

    assert (
        mdn_frequency_bands(**_band_settings(settings)) == captures["frequency_bands"]
    )


def test_a_heart_rate_above_the_limit_removes_no_harmonic() -> None:
    signal = _synthetic((3, 3), fs=50.0)
    settings = {"respiratory_rate": 0.25, "heart_rate": 4.0}
    expected_captures: dict = {}
    captures: dict = {}

    with pytest.warns(UserWarning):
        expected = _upstream(signal, expected_captures, **settings)
    with pytest.warns(UserWarning):
        result = mdn_filter(signal, captures=captures, **settings)

    np.testing.assert_array_equal(result.values, expected)
    assert captures["n_harmonics"] == expected_captures["n_harmonics"] == 0
    assert "frequency_bands" not in captures
    assert "frequency_bands" not in expected_captures
    with pytest.warns(UserWarning):
        assert mdn_frequency_bands(**settings) == []


# Missing values


def test_a_pixel_missing_at_every_sample_stays_missing_with_no_warning() -> None:
    signal = _synthetic((4, 4))
    signal.values[:, 0, 0] = np.nan
    settings = SETTINGS["3 harmonics"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = mdn_filter(signal, **settings)

    np.testing.assert_array_equal(result.values, _upstream(signal, **settings))
    assert np.isnan(result.values[:, 0, 0]).all()
    assert np.isfinite(result.values[:, 1:, 1:]).all()


def test_a_pixel_missing_at_some_samples_becomes_missing_with_a_warning() -> None:
    signal = _synthetic((4, 4))
    signal.values[:, 0, 0] = np.nan
    signal.values[100:103, 2, 1] = np.nan
    settings = SETTINGS["3 harmonics"]

    with pytest.warns(
        UserWarning, match=r"1 pixel\(s\).*\(2, 1\): 3 samples"
    ) as record:
        result = mdn_filter(signal, **settings)

    assert len(record) == 1
    assert record[0].filename == __file__
    np.testing.assert_array_equal(result.values, _upstream(signal, **settings))
    assert np.isnan(result.values[:, 2, 1]).all()
    assert np.isfinite(result.values[:, 3, 3]).all()


def test_a_waveform_missing_at_some_samples_gives_a_warning() -> None:
    signal = _synthetic(())
    signal.values[10] = np.nan

    with pytest.warns(UserWarning, match="the waveform has NaN at 1 of 1200 samples"):
        result = mdn_filter(signal, **SETTINGS["3 harmonics"])

    assert np.isnan(result.values).all()


# Warnings and errors


def test_high_rates_warn_as_eitprocessing() -> None:
    signal = _synthetic((3, 3), fs=50.0)
    settings = {"respiratory_rate": 90 / MINUTE, "heart_rate": 216 / MINUTE}

    with warnings.catch_warnings(record=True) as expected:
        warnings.simplefilter("always")
        _upstream(signal, **settings)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mdn_filter(signal, **settings)

    assert [w.category for w in record] == [w.category for w in expected]
    assert len(record) == 2
    assert "breathing rate" in str(record[0].message)
    assert "heart rate" in str(record[1].message)
    assert record[0].filename == __file__


@pytest.mark.filterwarnings(
    "ignore:The (provided respiratory|breathing) rate:UserWarning"
)
@pytest.mark.parametrize(
    "settings",
    [
        {"respiratory_rate": 0.0, "heart_rate": 1.2},
        {"respiratory_rate": -0.1, "heart_rate": 1.2},
        {"respiratory_rate": 0.25, "heart_rate": 0.0},
        {"respiratory_rate": 1.2, "heart_rate": 1.2},
        {"respiratory_rate": 1.5, "heart_rate": 1.2},
    ],
)
def test_wrong_rates_are_refused_as_by_eitprocessing(settings: dict) -> None:
    signal = _synthetic((3, 3))

    with pytest.raises(ValueError):
        _upstream(signal, **settings)
    with pytest.raises(ValueError, match="rate"):
        mdn_filter(signal, **settings)
    with pytest.raises(ValueError, match="rate"):
        mdn_frequency_bands(**settings)


def test_a_too_low_sample_frequency_is_refused_as_by_eitprocessing() -> None:
    signal = _synthetic((3, 3), fs=7.0)

    with pytest.raises(ValueError):
        _upstream(signal, **SETTINGS["3 harmonics"])
    with pytest.raises(ValueError):
        mdn_filter(signal, **SETTINGS["3 harmonics"])


def test_data_without_sample_frequency_is_refused() -> None:
    signal = _synthetic((3, 3))
    signal.sample_frequency = None

    with pytest.raises(ValueError, match="sample_frequency"):
        mdn_filter(signal, **SETTINGS["3 harmonics"])


# The result


def test_the_result_keeps_the_data_description_and_leaves_the_input_unchanged() -> None:
    signal = _synthetic((4, 4))
    before = np.copy(signal.values)

    result = mdn_filter(signal, **SETTINGS["3 harmonics"])

    np.testing.assert_array_equal(signal.values, before)
    assert result.values.shape == signal.values.shape
    np.testing.assert_array_equal(result.time, signal.time)
    assert result.time is not signal.time
    assert result.sample_frequency == signal.sample_frequency
    assert result.unit == signal.unit
    assert result.source == "draeger"
    assert result.modality == "eit"
    assert result.name == "synthetic"
    assert result.metadata == {"note": "synthetic"}
    assert result.metadata is not signal.metadata
    assert result.processing_state == "intermediate"
    assert result.method == "m3resp.eit.processing.mdn_filter"


def test_a_given_name_names_the_result() -> None:
    result = mdn_filter(
        _synthetic((3, 3)), **SETTINGS["3 harmonics"], name="mdn_filtered"
    )

    assert result.name == "mdn_filtered"


def test_the_heart_signal_is_removed_and_the_breathing_signal_stays() -> None:
    fs = 20.0
    signal = _synthetic((), fs=fs, breathing_hz=0.25, heart_hz=1.2)

    result = mdn_filter(signal, respiratory_rate=0.25, heart_rate=1.2)

    frequencies = np.fft.rfftfreq(len(signal.values), 1 / fs)
    before = np.abs(np.fft.rfft(signal.values - signal.values.mean()))
    after = np.abs(np.fft.rfft(result.values - result.values.mean()))
    for frequency in (1.2, 2.4, 3.6):
        index = int(np.argmin(np.abs(frequencies - frequency)))
        assert after[index] < 0.02 * before[index]
    breathing = int(np.argmin(np.abs(frequencies - 0.25)))
    assert after[breathing] == pytest.approx(before[breathing], rel=0.01)
