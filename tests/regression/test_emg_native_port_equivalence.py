"""EMG native-port regression tests.

`m3resp.emg.processing.ReSurfEMG` is a native port of the ReSurfEMG algorithms;
`m3resp.adapters.resurfemg_adapter.ReSurfEMGNative` is the thin adapter that
calls the upstream `resurfemg` package. For every method the two classes share,
these tests feed both implementations the same inputs and require identical
outputs.

Tolerance: exact equality (`assert_array_equal`, NaN == NaN). The port is meant
to be a line-for-line reimplementation, so any divergence - a reordered float
operation, a different default, a swapped return slot - is a regression.

Each method is exercised twice:

- with every parameter passed explicitly, which pins the algorithm itself;
- with only the required arguments, which pins that the defaults agree too.

All derived inputs (baseline, breath peaks, on/offsets, time products, ...) are
computed once with upstream `resurfemg` functions so both implementations see
exactly the same data.

Known divergences are marked `xfail(strict=True)` with the cause as the reason:
the suite stays green while they are open, and fails as soon as one is fixed so
the marker gets removed and the case becomes a regular regression check.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("resurfemg")
pytest.importorskip("pywt")

from m3resp.adapters.resurfemg_adapter import ReSurfEMGAdapter
from m3resp.adapters.resurfemg_adapter import _signals as adapter_signals
from m3resp.core.events import BreathEvent
from m3resp.emg import ReSurfEMG as ReSurfEMGNative
from m3resp.emg.processing import signals as native_signals

EMG_FS = 2048
VENT_FS = 100
ECG_FS = 2048


def _known_divergence(reason: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(condition=True, strict=False, reason=reason, run=True)


GATE_INTERP_AT_END = _known_divergence(
    "native `_gate_fill_interp` clips the post-gate index to `max_samples` "
    "instead of `max_samples - 1` (IndexError when a gate reaches the end of "
    "the signal) and treats index 0 as out of bounds via `np.where(pre, ...)`."
)
WAVELET_THRESHOLD = _known_divergence(
    "native `wavelet_denoise_ecg` computes `fixed_threshold & std_estimate` "
    "(bitwise and, TypeError) instead of `*`, and never fills `thresholds`."
)
# FIXED
SLOPESUM_BASELINE = _known_divergence(
    "native `slopesum_baseline` omits upstream's 1.2 scale factor on the "
    "baseline and computes the running mean/std over `moving_average_samples` "
    "instead of `window_samples`."
)
# FIXED
SLOPESUM_DEFAULT_PERCENTILE_WINDOW = _known_divergence(
    "native `slopesum_baseline` defaults `percentile_window_samples` to "
    "`fs // 2`; upstream defaults `perc_window` to `fs` (plus the "
    "SLOPESUM_BASELINE divergence)."
)
# FIXED
BELL_CURVE_RETURN_ORDER = _known_divergence(
    "`evaluate_bell_curve_error` return slots 1 and 2 are swapped: upstream "
    "(and hence the adapter) returns (valid, bell_error, percentage, ...), the "
    "native port returns (valid, percentage, bell_error, ...)."
)
AUB_REFERENCE_SIGNAL = _known_divergence(
    "upstream `percentage_under_baseline` ignores `ref_signal` (it passes "
    "`ref_signal=signal` to `area_under_baseline`); the native port honours "
    "`reference_signal`."
)
EVENT_TIMING_DEFAULTS = _known_divergence(
    "`evaluate_event_timing` defaults differ: adapter (min_delta=-0.5, "
    "max_delta=2.0), native (min_delta=0.0, max_delta=None)."
)
VENTILATOR_SIGNALS_HELPER = _known_divergence(
    "the adapter's `ventilator_signals` delegates to "
    "`ventilator_adapter.split_channels` (label-based resolution, extra "
    "`origins` key); the native copy still indexes fixed columns."
)


# -- comparison helper -------------------------------------------------------


def assert_identical(actual: Any, expected: Any, path: str = "result") -> None:
    """Recursively assert that two results are exactly equal (NaN == NaN)."""

    if isinstance(expected, tuple | list):
        assert isinstance(actual, tuple | list), f"{path}: expected a sequence"
        assert len(actual) == len(expected), f"{path}: length differs"
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_identical(a, e, f"{path}[{index}]")
        return
    if isinstance(expected, pd.Series):
        assert isinstance(actual, pd.Series), f"{path}: expected a pandas.Series"
        pd.testing.assert_series_equal(actual, expected, check_exact=True, obj=path)
        return
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        actual_array = np.asarray(actual)
        expected_array = np.asarray(expected)
        assert actual_array.shape == expected_array.shape, (
            f"{path}: shape {actual_array.shape} != {expected_array.shape}"
        )
        np.testing.assert_array_equal(actual_array, expected_array, err_msg=path)
        return
    if isinstance(expected, float) and np.isnan(expected):
        assert isinstance(actual, float) and np.isnan(actual), f"{path}: not NaN"
        return
    assert actual == expected, f"{path}: {actual!r} != {expected!r}"
    assert type(actual) is type(expected), (
        f"{path}: type {type(actual).__name__} != {type(expected).__name__}"
    )


@pytest.fixture(scope="module")
def adapter() -> ReSurfEMGAdapter:
    return ReSurfEMGAdapter()


@pytest.fixture(scope="module")
def native() -> ReSurfEMGNative:
    return ReSurfEMGNative()


# -- synthetic inputs --------------------------------------------------------


@dataclass(frozen=True)
class Breaths:
    """An EMG envelope plus everything upstream derives from it."""

    fs: int
    envelope: np.ndarray
    baseline: np.ndarray
    peaks: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    time_products: np.ndarray
    aubs: np.ndarray


def _synthetic_envelope(fs: int, duration_seconds: float = 60.0) -> np.ndarray:
    """Bell-shaped breaths on a drifting, noisy quiet level.

    Amplitudes and widths vary breath to breath, and one breath is an outlier,
    so the percentile-based quality checks produce both True and False flags.
    """

    rng = np.random.default_rng(seed=7)
    time = np.arange(int(duration_seconds * fs)) / fs
    quiet_level = 0.2 + 0.15 * time / duration_seconds
    envelope = quiet_level + np.abs(rng.normal(scale=0.02, size=time.shape))
    centers = np.arange(2.0, duration_seconds - 1.5, 4.0)
    centers = centers + rng.uniform(-0.3, 0.3, size=centers.shape)
    amplitudes = rng.uniform(0.6, 1.4, size=centers.shape)
    amplitudes[len(amplitudes) // 2] = 6.0  # outlier breath
    widths = rng.uniform(0.3, 0.5, size=centers.shape)
    for center, amplitude, width in zip(centers, amplitudes, widths, strict=True):
        envelope += amplitude * np.exp(-((time - center) ** 2) / width**2)
    return envelope


@pytest.fixture(scope="module")
def breaths() -> Breaths:
    from resurfemg.postprocessing import features
    from resurfemg.postprocessing.baseline import moving_baseline
    from resurfemg.postprocessing.event_detection import (
        detect_emg_breaths,
        onoffpeak_baseline_crossing,
    )

    fs = EMG_FS
    envelope = _synthetic_envelope(fs)
    baseline = moving_baseline(envelope, 10 * fs, fs // 2, set_percentile=33.0)
    peaks = np.asarray(
        detect_emg_breaths(envelope, baseline, min_peak_width_s=fs // 4), dtype=int
    )
    starts, ends, *_ = onoffpeak_baseline_crossing(envelope, baseline, peaks)
    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    time_products = np.asarray(
        features.time_product(envelope, fs, starts, ends, baseline), dtype=float
    )
    aubs, _ = features.area_under_baseline(
        envelope, fs, peaks, starts, ends, 5 * fs, baseline
    )

    assert len(peaks) >= 10, "fixture should contain a realistic number of breaths"
    return Breaths(
        fs=fs,
        envelope=envelope,
        baseline=np.asarray(baseline, dtype=float),
        peaks=peaks,
        starts=starts,
        ends=ends,
        time_products=time_products,
        aubs=np.asarray(aubs, dtype=float),
    )


def _ecg_contaminated_signal(fs: int, duration_seconds: float) -> np.ndarray:
    neurokit2 = pytest.importorskip("neurokit2")
    ecg = np.asarray(
        neurokit2.ecg_simulate(
            duration=duration_seconds, sampling_rate=fs, heart_rate=75, random_state=42
        ),
        dtype=float,
    )
    time = np.arange(len(ecg)) / fs
    rng = np.random.default_rng(seed=3)
    emg_like = 0.05 * np.sin(2 * np.pi * 100 * time) + rng.normal(
        scale=0.01, size=time.shape
    )
    return ecg + emg_like


@pytest.fixture(scope="module")
def ecg_signal() -> np.ndarray:
    return _ecg_contaminated_signal(ECG_FS, 20.0)


@pytest.fixture(scope="module")
def ecg_signal_padded() -> np.ndarray:
    # >= 15 s for wavelet_denoising's noise window; a sample count that is not
    # a multiple of 2**levels so the zero-padding branch is exercised.
    return _ecg_contaminated_signal(ECG_FS, 20.002)


@pytest.fixture(scope="module")
def ecg_peaks(ecg_signal: np.ndarray) -> np.ndarray:
    from resurfemg.preprocessing.ecg_removal import detect_ecg_peaks

    return np.asarray(detect_ecg_peaks(ecg_signal, ECG_FS), dtype=int)


@dataclass(frozen=True)
class Occlusions:
    pressure: np.ndarray
    peaks: np.ndarray
    ends: np.ndarray
    time_products: np.ndarray


@pytest.fixture(scope="module")
def occlusions() -> Occlusions:
    """A PEEP-level pressure trace with negative occlusion dips.

    Dip depth and release speed vary, so `pocc_quality` rejects some of them.
    """

    rng = np.random.default_rng(seed=11)
    fs = VENT_FS
    peep = 5.0
    pressure = peep + rng.normal(scale=0.05, size=60 * fs)
    peaks, ends = [], []
    # (onset s, depth cmH2O, release duration s) - the fast releases fail.
    for onset, depth, release in [
        (5.0, 12.0, 1.5),
        (15.0, 8.0, 0.2),
        (25.0, 15.0, 1.2),
        (35.0, 3.0, 2.0),
        (45.0, 10.0, 0.3),
    ]:
        start = int(onset * fs)
        fall = int(0.4 * fs)
        rise = max(3, int(release * fs))
        pressure[start : start + fall] -= np.linspace(0.0, depth, fall)
        pressure[start + fall : start + fall + rise] -= np.linspace(depth, 0.0, rise)
        peaks.append(start + fall)
        ends.append(start + fall + rise)
    peaks_array = np.asarray(peaks, dtype=int)
    ends_array = np.asarray(ends, dtype=int)
    time_products = np.asarray(
        [
            np.sum(peep - pressure[p - int(0.4 * fs) : e]) / fs
            for p, e in zip(peaks_array, ends_array, strict=True)
        ]
    )
    return Occlusions(pressure, peaks_array, ends_array, time_products)


# -- ECG ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("peak_fraction", "peak_width_samples", "peak_distance_samples", "bandpass"),
    [
        (0.4, 2, 682, True),
        (0.4, 2, 682, False),
        (0.25, 5, 1000, True),
        (0.6, 1, 400, False),
    ],
)
def test_detect_ecg_peaks_explicit(
    adapter,
    native,
    ecg_signal,
    peak_fraction,
    peak_width_samples,
    peak_distance_samples,
    bandpass,
):
    common = {
        "sample_frequency": ECG_FS,
        "peak_fraction": peak_fraction,
        "peak_width_samples": peak_width_samples,
        "peak_distance_samples": peak_distance_samples,
    }
    expected = adapter.detect_ecg_peaks(ecg_signal, bandpass_filter=bandpass, **common)
    actual = native.detect_ecg_peaks(
        ecg_signal, apply_bandpass_filter=bandpass, **common
    )

    assert len(expected) > 0
    assert_identical(actual, expected)


def test_detect_ecg_peaks_defaults(adapter, native, ecg_signal):
    expected = adapter.detect_ecg_peaks(ecg_signal, sample_frequency=ECG_FS)
    actual = native.detect_ecg_peaks(ecg_signal, sample_frequency=ECG_FS)

    assert_identical(actual, expected)


_GATE_PEAK_LAYOUTS = {
    "interior": lambda n: np.array([4096, 8192, 12288, 16384]),
    "near_boundaries": lambda n: np.array([50, 4096, 8192, n - 60]),
    "at_boundaries": lambda n: np.array([0, 8192, n - 1]),
    "overlapping_gates": lambda n: np.array([6000, 6100, 6150, 20000]),
    "single_peak": lambda n: np.array([10000]),
    # peak - gate_width // 2 - 1 == 0: the sample before the gate is index 0
    "pre_gate_at_index_0": lambda n: np.array([103, 8192]),
}


def _gate_cases() -> list[Any]:
    cases = []
    for fill_method in (0, 1, 2, 3):
        for gate_width_samples in (205, 204):
            for layout in sorted(_GATE_PEAK_LAYOUTS):
                boundary = layout in (
                    "near_boundaries",
                    "at_boundaries",
                    "pre_gate_at_index_0",
                )
                marks = [GATE_INTERP_AT_END] if fill_method == 1 and boundary else []
                cases.append(
                    pytest.param(
                        layout,
                        gate_width_samples,
                        fill_method,
                        marks=marks,
                        id=f"fill{fill_method}-w{gate_width_samples}-{layout}",
                    )
                )
    return cases


@pytest.mark.parametrize(("layout", "gate_width_samples", "fill_method"), _gate_cases())
def test_gate_ecg_explicit(
    adapter, native, ecg_signal, layout, gate_width_samples, fill_method
):
    peaks = _GATE_PEAK_LAYOUTS[layout](len(ecg_signal))
    kwargs = {"gate_width_samples": gate_width_samples, "fill_method": fill_method}

    expected = adapter.gate_ecg(ecg_signal, peaks, **kwargs)
    actual = native.gate_ecg(ecg_signal, peaks, **kwargs)

    if fill_method == 3:
        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0)
    else:
        assert_identical(actual, expected)


@GATE_INTERP_AT_END  # default fill_method=1; the last detected peak is near the end
def test_gate_ecg_detected_peaks_defaults(adapter, native, ecg_signal, ecg_peaks):
    expected = adapter.gate_ecg(ecg_signal, ecg_peaks)
    actual = native.gate_ecg(ecg_signal, ecg_peaks)

    np.testing.assert_almost_equal(actual, expected, decimal=15)


@pytest.mark.parametrize("fill_method", [0, 2, 3])
def test_gate_ecg_detected_peaks_other_fill_methods(
    adapter, native, ecg_signal, ecg_peaks, fill_method
):
    expected = adapter.gate_ecg(ecg_signal, ecg_peaks, fill_method=fill_method)
    actual = native.gate_ecg(ecg_signal, ecg_peaks, fill_method=fill_method)

    if fill_method == 3:
        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0)
    else:
        assert_identical(actual, expected)


@WAVELET_THRESHOLD
@pytest.mark.parametrize("hard_thresholding", [True, False])
@pytest.mark.parametrize(
    ("levels", "wavelet_type", "fixed_threshold"),
    [(4, "db2", 4.5), (5, "db2", 3.0), (3, "sym4", 4.5)],
)
def test_wavelet_denoise_ecg_explicit(
    adapter,
    native,
    ecg_signal_padded,
    hard_thresholding,
    levels,
    wavelet_type,
    fixed_threshold,
):
    peaks = np.array([50, 4096, 8192, 20000, len(ecg_signal_padded) - 60])
    kwargs = {
        "sample_frequency": ECG_FS,
        "hard_thresholding": hard_thresholding,
        "levels": levels,
        "wavelet_type": wavelet_type,
        "fixed_threshold": fixed_threshold,
    }

    expected = adapter.wavelet_denoise_ecg(ecg_signal_padded, peaks, **kwargs)
    actual = native.wavelet_denoise_ecg(ecg_signal_padded, peaks, **kwargs)

    assert_identical(actual, expected)


@WAVELET_THRESHOLD
def test_wavelet_denoise_ecg_defaults(adapter, native, ecg_signal_padded, ecg_peaks):
    peaks = ecg_peaks[ecg_peaks < len(ecg_signal_padded)]

    expected = adapter.wavelet_denoise_ecg(
        ecg_signal_padded, peaks, sample_frequency=ECG_FS
    )
    actual = native.wavelet_denoise_ecg(
        ecg_signal_padded, peaks, sample_frequency=ECG_FS
    )

    assert_identical(actual, expected)


# -- baseline ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("window_samples", "step_samples", "percentile"),
    [
        (5 * EMG_FS, EMG_FS // 4, 33.0),
        (10 * EMG_FS + 1, EMG_FS, 50.0),  # odd window
        (10 * EMG_FS, 3 * EMG_FS + 7, 33.0),  # step not dividing the length
        (3 * EMG_FS, EMG_FS // 2, 0.0),
        (3 * EMG_FS, EMG_FS // 2, 100.0),
        (10**7, EMG_FS, 25.0),  # window longer than the signal
    ],
)
def test_moving_baseline_explicit(
    adapter, native, breaths, window_samples, step_samples, percentile
):
    kwargs = {
        "window_samples": window_samples,
        "step_samples": step_samples,
        "percentile": percentile,
    }

    expected = adapter.moving_baseline(breaths.envelope, **kwargs)
    actual = native.moving_baseline(breaths.envelope, **kwargs)

    assert_identical(actual, expected)


def test_moving_baseline_defaults(adapter, native, breaths):
    kwargs = {"window_samples": 5 * EMG_FS, "step_samples": EMG_FS // 2}

    expected = adapter.moving_baseline(breaths.envelope, **kwargs)
    actual = native.moving_baseline(breaths.envelope, **kwargs)

    assert_identical(actual, expected)


@SLOPESUM_BASELINE
@pytest.mark.parametrize(
    (
        "window_samples",
        "step_samples",
        "percentile",
        "augmented_percentile",
        "moving_average_samples",
        "percentile_window_samples",
    ),
    [
        (5 * EMG_FS, EMG_FS // 2, 33.0, 25.0, EMG_FS // 2, EMG_FS),
        (10 * EMG_FS, EMG_FS, 50.0, 10.0, EMG_FS // 4, EMG_FS // 2),
        (7 * EMG_FS + 3, EMG_FS // 3, 20.0, 40.0, 101, 3 * EMG_FS + 7),
    ],
)
def test_slopesum_baseline_explicit(
    adapter,
    native,
    breaths,
    window_samples,
    step_samples,
    percentile,
    augmented_percentile,
    moving_average_samples,
    percentile_window_samples,
):
    kwargs = {
        "window_samples": window_samples,
        "step_samples": step_samples,
        "sample_frequency": EMG_FS,
        "percentile": percentile,
        "augmented_percentile": augmented_percentile,
        "moving_average_samples": moving_average_samples,
        "percentile_window_samples": percentile_window_samples,
    }

    expected = adapter.slopesum_baseline(breaths.envelope, **kwargs)
    actual = native.slopesum_baseline(breaths.envelope, **kwargs)

    assert_identical(actual, expected)


@SLOPESUM_DEFAULT_PERCENTILE_WINDOW
def test_slopesum_baseline_defaults(adapter, native, breaths):
    kwargs = {
        "window_samples": 5 * EMG_FS,
        "step_samples": EMG_FS // 2,
        "sample_frequency": EMG_FS,
    }

    expected = adapter.slopesum_baseline(breaths.envelope, **kwargs)
    actual = native.slopesum_baseline(breaths.envelope, **kwargs)

    assert_identical(actual, expected)


# -- quality assessment ------------------------------------------------------


def test_snr_pseudo_detected_breaths(adapter, native, breaths):
    args = (breaths.envelope, breaths.peaks, breaths.baseline)

    expected = adapter.snr_pseudo(*args, sample_frequency=breaths.fs)
    actual = native.snr_pseudo(*args, sample_frequency=breaths.fs)

    assert_identical(actual, expected)


def test_snr_pseudo_peaks_at_signal_edges(adapter, native, breaths):
    n = len(breaths.envelope)
    peaks = np.array([0, 10, n // 2, n - 10, n - 1])
    args = (breaths.envelope, peaks, breaths.baseline)

    expected = adapter.snr_pseudo(*args, sample_frequency=breaths.fs)
    actual = native.snr_pseudo(*args, sample_frequency=breaths.fs)

    assert_identical(actual, expected)


@pytest.mark.parametrize(
    ("dp_up_10_threshold", "dp_up_90_threshold", "dp_up_90_norm_threshold"),
    [(0.0, 2.0, 0.8), (-0.1, 0.5, 0.2), (0.05, 5.0, 3.0)],
)
def test_pocc_quality_explicit(
    adapter,
    native,
    occlusions,
    dp_up_10_threshold,
    dp_up_90_threshold,
    dp_up_90_norm_threshold,
):
    args = (
        occlusions.pressure,
        occlusions.peaks,
        occlusions.ends,
        occlusions.time_products,
    )
    kwargs = {
        "dp_up_10_threshold": dp_up_10_threshold,
        "dp_up_90_threshold": dp_up_90_threshold,
        "dp_up_90_norm_threshold": dp_up_90_norm_threshold,
    }

    expected = adapter.pocc_quality(*args, **kwargs)
    actual = native.pocc_quality(*args, **kwargs)

    assert_identical(actual, expected)


def test_pocc_quality_defaults(adapter, native, occlusions):
    args = (
        occlusions.pressure,
        occlusions.peaks,
        occlusions.ends,
        occlusions.time_products,
    )

    expected = adapter.pocc_quality(*args)
    actual = native.pocc_quality(*args)

    # The fixture should exercise both outcomes of the default criteria.
    assert set(expected[0].tolist()) == {True, False}
    assert_identical(actual, expected)


@pytest.mark.parametrize("threshold", [0.5, 1.1, 5.0, 50.0])
def test_interpeak_distance_explicit(adapter, native, ecg_peaks, breaths, threshold):
    expected = adapter.interpeak_distance(ecg_peaks, breaths.peaks, threshold=threshold)
    actual = native.interpeak_distance(ecg_peaks, breaths.peaks, threshold=threshold)

    assert_identical(actual, expected)


def test_interpeak_distance_defaults(adapter, native, ecg_peaks, breaths):
    expected = adapter.interpeak_distance(ecg_peaks, breaths.peaks)
    actual = native.interpeak_distance(ecg_peaks, breaths.peaks)

    assert_identical(actual, expected)


@pytest.mark.parametrize(
    ("aub_window_samples", "aub_threshold"),
    [(5 * EMG_FS, 40.0), (EMG_FS, 40.0), (2 * EMG_FS, 5.0)],
)
def test_percentage_under_baseline_explicit(
    adapter, native, breaths, aub_window_samples, aub_threshold
):
    args = (
        breaths.envelope,
        breaths.peaks,
        breaths.starts,
        breaths.ends,
        breaths.baseline,
    )
    kwargs = {
        "sample_frequency": breaths.fs,
        "aub_window_samples": aub_window_samples,
        "aub_threshold": aub_threshold,
    }

    expected = adapter.percentage_under_baseline(*args, **kwargs)
    actual = native.percentage_under_baseline(*args, **kwargs)

    assert_identical(actual, expected)


@AUB_REFERENCE_SIGNAL
def test_percentage_under_baseline_with_reference_signal(adapter, native, breaths):
    """A reference signal distinct from `signal` must be treated identically."""

    rng = np.random.default_rng(seed=5)
    reference = breaths.envelope * 0.5 - np.abs(
        rng.normal(scale=0.1, size=breaths.envelope.shape)
    )
    args = (
        breaths.envelope,
        breaths.peaks,
        breaths.starts,
        breaths.ends,
        breaths.baseline,
    )
    kwargs = {"sample_frequency": breaths.fs, "reference_signal": reference}

    expected = adapter.percentage_under_baseline(*args, **kwargs)
    actual = native.percentage_under_baseline(*args, **kwargs)

    assert_identical(actual, expected)


def test_percentage_under_baseline_defaults(adapter, native, breaths):
    args = (
        breaths.envelope,
        breaths.peaks,
        breaths.starts,
        breaths.ends,
        breaths.baseline,
    )

    expected = adapter.percentage_under_baseline(*args, sample_frequency=breaths.fs)
    actual = native.percentage_under_baseline(*args, sample_frequency=breaths.fs)

    assert_identical(actual, expected)


@pytest.mark.parametrize(
    ("threshold_percentile", "threshold_factor"),
    [(75.0, 4.0), (50.0, 1.5), (90.0, 1.0), (0.0, 2.0), (100.0, 1.0)],
)
def test_detect_local_high_aub_explicit(
    adapter, native, breaths, threshold_percentile, threshold_factor
):
    kwargs = {
        "threshold_percentile": threshold_percentile,
        "threshold_factor": threshold_factor,
    }

    expected = adapter.detect_local_high_aub(breaths.aubs, **kwargs)
    actual = native.detect_local_high_aub(breaths.aubs, **kwargs)

    assert_identical(actual, expected)


def test_detect_local_high_aub_defaults_with_outlier(adapter, native, breaths):
    aubs = breaths.aubs.copy()
    aubs[len(aubs) // 3] = 100 * np.max(np.abs(aubs)) + 1.0

    expected = adapter.detect_local_high_aub(aubs)
    actual = native.detect_local_high_aub(aubs)

    assert not expected.all(), "the outlier should be flagged"
    assert_identical(actual, expected)


@pytest.mark.parametrize(
    ("upper_percentile", "upper_factor", "lower_percentile", "lower_factor"),
    [
        (95.0, 10.0, 5.0, 0.1),
        (80.0, 1.2, 20.0, 0.9),
        (100.0, 1.0, 0.0, 1.0),
    ],
)
def test_detect_extreme_time_products_explicit(
    adapter,
    native,
    breaths,
    upper_percentile,
    upper_factor,
    lower_percentile,
    lower_factor,
):
    kwargs = {
        "upper_percentile": upper_percentile,
        "upper_factor": upper_factor,
        "lower_percentile": lower_percentile,
        "lower_factor": lower_factor,
    }

    expected = adapter.detect_extreme_time_products(breaths.time_products, **kwargs)
    actual = native.detect_extreme_time_products(breaths.time_products, **kwargs)

    assert_identical(actual, expected)


def test_detect_extreme_time_products_defaults_with_outliers(adapter, native, breaths):
    time_products = breaths.time_products.copy()
    time_products[1] = 1e-6
    time_products[-2] = 1e6

    expected = adapter.detect_extreme_time_products(time_products)
    actual = native.detect_extreme_time_products(time_products)

    assert not expected.all(), "the outliers should be flagged"
    assert_identical(actual, expected)


@pytest.mark.parametrize(
    ("ventilator_breaths", "manoeuvres"),
    [
        # every manoeuvre separated by a supported breath
        ([100, 300, 500, 700, 900], [200, 400, 600, 800]),
        # "double dip": no supported breath between the 2nd and 3rd manoeuvre
        ([100, 300, 700, 900], [200, 400, 600, 800]),
        # no supported breaths at all
        ([], [200, 400, 600]),
        # single manoeuvre
        ([100, 300], [200]),
        # breaths coinciding with manoeuvre indices
        ([200, 400, 500], [200, 400, 600]),
    ],
    ids=["separated", "double_dip", "no_breaths", "single", "coinciding"],
)
def test_detect_non_consecutive_manoeuvres(
    adapter, native, ventilator_breaths, manoeuvres
):
    breaths = np.asarray(ventilator_breaths, dtype=int)
    poccs = np.asarray(manoeuvres, dtype=int)

    expected = adapter.detect_non_consecutive_manoeuvres(breaths, poccs)
    actual = native.detect_non_consecutive_manoeuvres(breaths, poccs)

    assert_identical(actual, expected)


def _bell_curve_errors(implementation, breaths: Breaths, **kwargs):
    with warnings.catch_warnings():
        # scipy's curve_fit may warn about the covariance on some breaths.
        warnings.simplefilter("ignore")
        return implementation.evaluate_bell_curve_error(
            breaths.peaks,
            breaths.starts,
            breaths.ends,
            breaths.envelope,
            breaths.time_products,
            sample_frequency=breaths.fs,
            **kwargs,
        )


@BELL_CURVE_RETURN_ORDER
@pytest.mark.parametrize(
    ("bell_window_samples", "bell_threshold"),
    [(5 * EMG_FS, 40.0), (EMG_FS, 40.0), (2 * EMG_FS, 5.0)],
)
def test_evaluate_bell_curve_error_explicit(
    adapter, native, breaths, bell_window_samples, bell_threshold
):
    kwargs = {
        "bell_window_samples": bell_window_samples,
        "bell_threshold": bell_threshold,
    }

    expected = _bell_curve_errors(adapter, breaths, **kwargs)
    actual = _bell_curve_errors(native, breaths, **kwargs)

    assert_identical(actual, expected)


@BELL_CURVE_RETURN_ORDER
def test_evaluate_bell_curve_error_defaults(adapter, native, breaths):
    expected = _bell_curve_errors(adapter, breaths)
    actual = _bell_curve_errors(native, breaths)

    assert_identical(actual, expected)


@BELL_CURVE_RETURN_ORDER
def test_evaluate_bell_curve_error_short_breath_padding(adapter, native, breaths):
    """Breaths shorter than 3 samples take the `plus_index` padding branch."""

    peaks = breaths.peaks[:3]
    starts = peaks - 1
    ends = peaks + 1
    time_products = breaths.time_products[:3]
    args = (peaks, starts, ends, breaths.envelope, time_products)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = adapter.evaluate_bell_curve_error(*args, sample_frequency=breaths.fs)
        actual = native.evaluate_bell_curve_error(*args, sample_frequency=breaths.fs)

    assert_identical(actual, expected)


_EVENT_TIMES_FIRST = np.array([1.0, 5.0, 9.2, 13.0, 17.5, 21.0])
_EVENT_TIMES_SECOND = np.array([1.3, 4.6, 9.2, 15.5, 17.9, 20.2])


@pytest.mark.parametrize(
    ("min_delta", "max_delta"),
    [(0.0, None), (-0.5, 2.0), (0.1, 0.5), (-1.0, None), (0.0, 0.0)],
)
def test_evaluate_event_timing_explicit(adapter, native, min_delta, max_delta):
    kwargs = {"min_delta": min_delta, "max_delta": max_delta}

    expected = adapter.evaluate_event_timing(
        _EVENT_TIMES_FIRST, _EVENT_TIMES_SECOND, **kwargs
    )
    actual = native.evaluate_event_timing(
        _EVENT_TIMES_FIRST, _EVENT_TIMES_SECOND, **kwargs
    )

    assert_identical(actual, expected)


@EVENT_TIMING_DEFAULTS
def test_evaluate_event_timing_defaults(adapter, native):
    expected = adapter.evaluate_event_timing(
        _EVENT_TIMES_FIRST, _EVENT_TIMES_SECOND, min_delta=-0.5, max_delta=2.0
    )
    actual = native.evaluate_event_timing(
        _EVENT_TIMES_FIRST, _EVENT_TIMES_SECOND, min_delta=-0.5, max_delta=2.0
    )

    assert_identical(actual, expected)


@pytest.mark.parametrize(
    ("recording_duration_seconds", "ventilator_respiratory_rate", "minimum_fraction"),
    [(60.0, 15.0, 0.1), (60.0, 15.0, 0.99), (37.5, 22.0, 0.5), (600.0, 30.0, 0.1)],
)
def test_evaluate_respiratory_rates_explicit(
    adapter,
    native,
    breaths,
    recording_duration_seconds,
    ventilator_respiratory_rate,
    minimum_fraction,
):
    args = (breaths.peaks, recording_duration_seconds, ventilator_respiratory_rate)

    expected = adapter.evaluate_respiratory_rates(
        *args, minimum_fraction=minimum_fraction
    )
    actual = native.evaluate_respiratory_rates(*args, minimum_fraction=minimum_fraction)

    assert_identical(actual, expected)


def test_evaluate_respiratory_rates_defaults(adapter, native, breaths):
    args = (breaths.peaks, 60.0, 15.0)

    expected = adapter.evaluate_respiratory_rates(*args)
    actual = native.evaluate_respiratory_rates(*args)

    assert_identical(actual, expected)


# -- signal-shaping helpers --------------------------------------------------


@VENTILATOR_SIGNALS_HELPER
def test_ventilator_signals_helper_is_identical():
    rng = np.random.default_rng(seed=1)
    recording = {"array": rng.normal(size=(3, 500)), "metadata": {"fs": 100.0}}
    kwargs = {"pressure_channel": 0, "flow_channel": 1, "volume_channel": 2}

    for fs in (None, 250.0):
        expected = adapter_signals.ventilator_signals(recording, fs=fs, **kwargs)
        actual = native_signals.ventilator_signals(recording, fs=fs, **kwargs)

        assert actual.keys() == expected.keys()
        for key in ("pressure", "flow", "volume", "fs"):
            assert_identical(actual[key], expected[key], key)
        assert actual["metadata"] == expected["metadata"]


def test_peak_indices_from_events_helper_is_identical():
    events = [
        BreathEvent("emg", 0.0, 1.0, peak_time=0.4999),
        BreathEvent("emg", 1.0, 2.0, peak_time=1.5),
        BreathEvent("emg", 2.0, 3.0, peak_time=None),
    ]

    for fs in (100.0, 2048.0):
        assert_identical(
            native_signals.peak_indices_from_events(events, fs),
            adapter_signals.peak_indices_from_events(events, fs),
        )
    assert_identical(
        native_signals.peak_indices_from_events(None, 100.0),
        adapter_signals.peak_indices_from_events(None, 100.0),
    )
