"""Breathing rate and heart rate from EIT images, found by m3resp code.

``detect_rates`` must give the numbers of eitprocessing 1.8.7: the golden
files ``rates.npz`` and ``RateDetection.apply`` on the same images.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pytest

from m3resp.data import Signal
from m3resp.eit.io import read_eit_file
from m3resp.eit.processing.rate_detection import (
    DEFAULT_RATE_LIMITS,
    RateDetectionSettings,
    detect_rates,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")
with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as _handle:
    MANIFEST = json.load(_handle)
DATASETS = sorted(MANIFEST["inputs"])
RELATIVE_TOLERANCE = 1e-9
CAPTURE_KEYS = (
    "frequencies",
    "normalized_total_power",
    "average_normalized_pixel_power",
    "diff_total_averaged_power",
    "estimated_respiratory_rate",
    "estimated_heart_rate",
)


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


def _synthetic_images(
    breathing_hz: float = 0.25, heart_hz: float = 1.2, seconds: float = 60.0
) -> Signal:
    """Images of 8 x 8 pixels: a large breathing signal and a small heart signal.

    The pixels of the first two rows have a weak breathing signal, so the
    heart signal is relatively strong there. The last column is 0 in every
    frame, as pixels outside the measured area.
    """

    fs = 20.0
    time = np.arange(int(seconds * fs)) / fs
    breathing = np.sin(2 * np.pi * breathing_hz * time)
    heart = np.sin(2 * np.pi * heart_hz * time)
    weights = np.ones((8, 8))
    weights[:2] = 0.05
    rng = np.random.default_rng(3)
    values = (
        breathing[:, None, None] * weights
        + 0.2 * heart[:, None, None]
        + 0.01 * rng.standard_normal((len(time), 8, 8))
        + 5.0
    )
    values[:, :, -1] = 0.0
    return Signal(
        values=values,
        time=time,
        sample_frequency=fs,
        name="synthetic_images",
        modality="eit",
        source="draeger",
    )


def _upstream_rates(
    signal: Signal, captures: dict | None = None, **settings
) -> tuple[float, float]:
    """eitprocessing ``RateDetection.apply`` on the same images."""

    pytest.importorskip("eitprocessing")
    from eitprocessing.features.rate_detection import RateDetection

    from m3resp.adapters.eitprocessing_adapter._signals import as_eitprocessing

    subject_type = settings.pop("subject_type")
    detector = RateDetection(subject_type, **settings)
    return detector.apply(as_eitprocessing(signal), captures=captures)


def _assert_same(actual: dict, expected: dict) -> None:
    """Check two capture dictionaries value by value and type by type."""

    assert set(actual) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value, err_msg=key)
        assert np.asarray(actual[key]).dtype == np.asarray(value).dtype, key


# Golden files


@pytest.mark.parametrize("dataset", DATASETS)
def test_rates_and_captures_match_golden(dataset: str, images: dict) -> None:
    golden_path = os.path.join(GOLDEN_DIR, dataset, "rates.npz")
    with np.load(golden_path) as golden_file:
        golden = {key: golden_file[key] for key in golden_file.files}
    captures: dict = {}
    respiratory_rate, heart_rate = detect_rates(
        images[dataset],
        RateDetectionSettings(MANIFEST["settings"]["rate_detection"]["subject_type"]),
        captures=captures,
        suppress_length_warnings=True,
        suppress_edge_case_warning=True,
    )
    actual = {"respiratory_rate_hz": respiratory_rate, "heart_rate_hz": heart_rate}
    actual |= {f"captures/{key}": value for key, value in captures.items()}
    assert set(actual) == set(golden)
    for key, expected in golden.items():
        value = np.asarray(actual[key])
        assert value.shape == expected.shape, key
        assert value.dtype == expected.dtype, key
        finite = expected[np.isfinite(expected)]
        scale = float(np.max(np.abs(finite))) if finite.size else 0.0
        np.testing.assert_allclose(
            value,
            expected,
            rtol=0,
            atol=RELATIVE_TOLERANCE * scale,
            equal_nan=True,
            err_msg=key,
        )


# eitprocessing on the same images


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("subject_type", ["adult", "neonate"])
@pytest.mark.parametrize("welch_window", [10.0, 20.0, 30.0])
def test_rates_equal_eitprocessing(
    dataset: str, subject_type: str, welch_window: float, images: dict
) -> None:
    expected_captures: dict = {}
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = _upstream_rates(
            images[dataset],
            expected_captures,
            subject_type=subject_type,
            welch_window=welch_window,
        )
    actual_captures: dict = {}
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = detect_rates(
            images[dataset],
            RateDetectionSettings(subject_type, welch_window=welch_window),
            captures=actual_captures,
        )

    assert actual == expected
    _assert_same(actual_captures, expected_captures)
    assert [w.category for w in actual_warnings] == [
        w.category for w in expected_warnings
    ]


@pytest.mark.parametrize("refine", [True, False])
@pytest.mark.parametrize("welch_overlap", [0.0, 0.5, 0.9])
def test_settings_give_the_rates_of_eitprocessing(
    refine: bool, welch_overlap: float
) -> None:
    signal = _synthetic_images()
    expected = _upstream_rates(
        signal,
        subject_type="adult",
        welch_window=20.0,
        welch_overlap=welch_overlap,
        refine_estimated_frequency=refine,
    )
    actual = detect_rates(
        signal,
        RateDetectionSettings(
            "adult",
            welch_window=20.0,
            welch_overlap=welch_overlap,
            refine_estimated_frequency=refine,
        ),
    )
    assert actual == expected


def test_synthetic_rates_are_found() -> None:
    respiratory_rate, heart_rate = detect_rates(
        _synthetic_images(), RateDetectionSettings("adult", welch_window=20.0)
    )
    assert respiratory_rate == pytest.approx(0.25, abs=0.01)
    assert heart_rate == pytest.approx(1.2, abs=0.01)


def test_data_shorter_than_the_window_warns_as_eitprocessing() -> None:
    signal = _synthetic_images(seconds=15.0)
    with pytest.warns(UserWarning, match="longer than the data"):
        expected = _upstream_rates(signal, subject_type="adult", welch_window=30.0)
    with pytest.warns(UserWarning, match="longer than the data"):
        actual = detect_rates(signal, RateDetectionSettings("adult", welch_window=30.0))
    assert actual == expected

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        detect_rates(
            signal,
            RateDetectionSettings("adult", welch_window=30.0),
            suppress_length_warnings=True,
        )


def test_peak_at_the_edge_of_the_range_warns_as_eitprocessing() -> None:
    # The breathing rate (0.5 Hz) lies above the highest allowed rate.
    signal = _synthetic_images(breathing_hz=0.5)
    limits = {"min_respiratory_rate": 0.1, "max_respiratory_rate": 0.3}
    with pytest.warns(RuntimeWarning, match="edge"):
        expected = _upstream_rates(signal, subject_type="adult", **limits)
    with pytest.warns(RuntimeWarning, match="edge of the range"):
        actual = detect_rates(signal, RateDetectionSettings("adult", **limits))
    assert actual == expected
    # Without refinement the rate is a frequency step of the spectrum at an
    # edge of the range (steps of 1/30 Hz for the 30 s window).
    assert actual[0] in (pytest.approx(0.1), pytest.approx(0.3))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        detect_rates(
            signal,
            RateDetectionSettings("adult", **limits),
            suppress_edge_case_warning=True,
        )


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"welch_overlap": 1.0}, "less than 1"),
        ({"welch_overlap": -0.1}, "at least 0"),
        ({"welch_window": 9.9}, "at least 10"),
        ({"subject_type": "child"}, "subject type"),
    ],
)
def test_wrong_settings_are_refused_as_by_eitprocessing(
    settings: dict, message: str
) -> None:
    pytest.importorskip("eitprocessing")
    from eitprocessing.features.rate_detection import RateDetection

    arguments = {"subject_type": "adult"} | settings
    with pytest.raises(ValueError):
        RateDetection(**arguments)
    with pytest.raises(ValueError, match=message):
        RateDetectionSettings(**arguments)


@pytest.mark.parametrize("subject_type", ["adult", "neonate"])
def test_missing_limits_take_the_default_of_the_subject_type(
    subject_type: str,
) -> None:
    settings = RateDetectionSettings(subject_type, max_heart_rate=3.0)
    assert settings.max_heart_rate == 3.0
    for name in ("min_heart_rate", "min_respiratory_rate", "max_respiratory_rate"):
        assert getattr(settings, name) == DEFAULT_RATE_LIMITS[name][subject_type]


def test_a_limit_of_zero_is_kept() -> None:
    assert (
        RateDetectionSettings("adult", min_respiratory_rate=0.0).min_respiratory_rate
        == 0.0
    )


def test_a_range_without_any_frequency_is_refused() -> None:
    # The frequency step of a 20 s window is 0.05 Hz; 0.26 to 0.27 Hz has none.
    settings = RateDetectionSettings(
        "adult", welch_window=20.0, min_respiratory_rate=0.26, max_respiratory_rate=0.27
    )
    with pytest.raises(ValueError, match="breathing-rate range 15.6 to 16.2"):
        detect_rates(_synthetic_images(), settings)


def test_images_without_three_axes_or_frame_rate_are_refused() -> None:
    signal = _synthetic_images()
    waveform = Signal(
        values=signal.values[:, 0, 0], time=signal.time, sample_frequency=20.0
    )
    with pytest.raises(ValueError, match="shape"):
        detect_rates(waveform, RateDetectionSettings("adult"))
    no_rate = Signal(values=signal.values, time=signal.time, name="images")
    with pytest.raises(ValueError, match="frames per second"):
        detect_rates(no_rate, RateDetectionSettings("adult"))


def test_no_captures_store_nothing_and_leave_the_images_unchanged() -> None:
    signal = _synthetic_images()
    before = signal.values.copy()
    captures: dict = {}
    with_captures = detect_rates(
        signal, RateDetectionSettings("adult", welch_window=20.0), captures=captures
    )
    without = detect_rates(signal, RateDetectionSettings("adult", welch_window=20.0))
    assert with_captures == without
    assert tuple(captures) == CAPTURE_KEYS
    np.testing.assert_array_equal(signal.values, before)


# The rate figure


@pytest.mark.parametrize("subject_type", ["adult", "neonate"])
def test_rate_figure_draws_the_picture_of_eitprocessing(subject_type: str) -> None:
    pytest.importorskip("eitprocessing")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from eitprocessing.features.rate_detection import RateDetection
    from matplotlib import pyplot as plt

    from m3resp.visualization import plot_rate_detection

    settings = RateDetectionSettings(subject_type, welch_window=20.0)
    captures: dict = {}
    detect_rates(_synthetic_images(), settings, captures=captures)

    def pixels(figure) -> np.ndarray:
        figure.canvas.draw()
        picture = np.asarray(figure.canvas.buffer_rgba()).copy()
        plt.close(figure)
        return picture

    expected = pixels(
        RateDetection(subject_type, welch_window=20.0).plotting.plot(**captures)
    )
    actual = pixels(plot_rate_detection(settings, **captures))
    np.testing.assert_array_equal(actual, expected)
