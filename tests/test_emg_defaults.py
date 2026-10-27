"""Tests for `m3resp.emg.processing.defaults`, the native EMG fallback pipeline.

The postprocessing tests use a synthetic envelope with known breath peaks.
Two seams are patched with ``monkeypatch``:

- ``peak_indices_from_events``: tests pass plain peak indices as ``events``
  (except ``test_postprocess_accepts_detect_breaths_output``, which uses real
  detector output);
- ``ventilator_signals``: the ``ventilator`` fixture supplies synthetic
  volume and pressure traces.

Some tests pin behaviour that the current implementation gets wrong. They
will fail until the corresponding bug is fixed; the docstrings say which.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from m3resp.core.exceptions import UnsupportedWorkflowError
from m3resp.emg.processing import defaults
from m3resp.processing import intervals
from m3resp.data.events import BreathEvent, coerce_breath_events
# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #

FS = 100.0  # Hz, envelope sampling rate
DURATION_S = 60.0
BREATH_PERIOD_S = 4.0  # 15 breaths per minute
BREATH_SIGMA_S = 0.3
BASELINE_LEVEL = 1.0
PEAK_HEIGHT = 2.0

VENT_FS = 50.0
PEEP_CMH2O = 5.0

ALL_SELECTIONS = (
    defaults.BASELINE_FUNCTIONS,
    defaults.EMG_EVENT_DETECTION_FUNCTIONS,
    defaults.VENTILATOR_EVENT_DETECTION_FUNCTIONS,
    defaults.VENTILATOR_QUALITY_ASSESSMENT_FUNCTIONS,
    defaults.FEATURES_FUNCTIONS,
    defaults.EMG_QUALITY_ASSESSMENT_FUNCTIONS,
    defaults.MULTIMODAL_QUALITY_ASSESSMENT_FUNCTIONS,
)

EMG_FEATURES = {
    "amplitude",
    "area_under_baseline",
    "pseudo_slope",
    "time_to_peak",
    "time_product",
    "respiratory_rate",
}
VENTILATOR_ONLY = {
    "event_detection.detect_ventilator_breath",
    "event_detection.find_occluded_breaths",
    "quality_assessment.detect_non_consecutive_manoeuvres",
    "quality_assessment.ventilator_respiratory_rate",
}


def _breath_peaks(
    fs: float = FS, duration: float = DURATION_S, period: float = BREATH_PERIOD_S
) -> np.ndarray:
    times = np.arange(period / 2, duration - period / 2, period)
    return np.round(times * fs).astype(int)


def _envelope(
    peaks: np.ndarray, fs: float = FS, duration: float = DURATION_S
) -> np.ndarray:
    n = int(duration * fs)
    t = np.arange(n)
    sigma = BREATH_SIGMA_S * fs
    env = np.full(n, BASELINE_LEVEL)
    for p in peaks:
        env += PEAK_HEIGHT * np.exp(-0.5 * ((t - p) / sigma) ** 2)
    return env + np.random.default_rng(0).normal(0.0, 0.02, n)


def _default_selection() -> dict[str, dict[str, bool]]:
    return defaults._merge_selections(*ALL_SELECTIONS)


def _selected_names(selection: dict[str, dict[str, bool]]) -> set[str]:
    return {f"{c}.{n}" for c, fns in selection.items() for n, on in fns.items() if on}


def _computed_names(result: dict) -> set[str]:
    return {f"{c}.{n}" for c, fns in result["computed"].items() for n in fns}


def _run(processed_emg: dict, events, **kwargs) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return defaults._postprocess_default(processed_emg, events, **kwargs)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def peaks() -> np.ndarray:
    return _breath_peaks()


@pytest.fixture
def processed_emg(peaks: np.ndarray) -> dict:
    return {"envelope": _envelope(peaks), "fs": FS, "channel": 0}


@pytest.fixture
def inject_peaks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let tests pass plain peak indices as `events`."""
    monkeypatch.setattr(
        defaults,
        "peak_indices_from_events",
        lambda events, fs: np.asarray(
            [] if events is None else list(events), dtype=int
        ),
    )


@pytest.fixture
def ventilator(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Synthetic ventilator traces, returned whenever `ventilator=` is passed.

    Volume peaks at the same times as the EMG breaths. Pressure sits at PEEP
    for half of each cycle, so its median equals PEEP and no occlusions exist.
    """
    n = int(DURATION_S * VENT_FS)
    t = np.arange(n) / VENT_FS
    cycle = 2 * np.pi * t / BREATH_PERIOD_S
    volume = 0.25 * (1 - np.cos(cycle))
    pressure = PEEP_CMH2O + 10.0 * np.clip(np.sin(cycle), 0.0, None)
    signals = {
        "fs": VENT_FS,
        "volume": volume,
        "airway_pressure": pressure,
        "flow": np.gradient(volume) * VENT_FS,
    }

    def fake_ventilator_signals(**kwargs):
        return signals if kwargs.get("ventilator") is not None else None

    monkeypatch.setattr(defaults, "ventilator_signals", fake_ventilator_signals)
    return signals


@pytest.fixture
def recording() -> dict:
    fs = 2048.0
    rng = np.random.default_rng(1)
    return {
        "array": rng.normal(0.0, 1.0, size=(2, int(10 * fs))),
        "metadata": {"fs": fs, "labels": ["EMG di", "ECG"]},
    }


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def test_default_low_pass_hz_caps_at_500():
    assert defaults.default_low_pass_hz(2048.0) == 500


def test_default_low_pass_hz_stays_below_nyquist_for_low_rates():
    assert defaults.default_low_pass_hz(500.0) == pytest.approx(237.5)


@pytest.mark.parametrize("max_frequency", [None, 0])
def test_notch_max_frequency_defaults_to_nyquist(max_frequency):
    assert defaults.notch_max_frequency_hz(1000.0, max_frequency) == 500.0


def test_notch_max_frequency_keeps_valid_value():
    assert defaults.notch_max_frequency_hz(1000.0, 300.0) == 300.0


def test_notch_max_frequency_above_nyquist_warns_and_clips():
    with pytest.warns(UserWarning, match="Nyquist"):
        assert defaults.notch_max_frequency_hz(1000.0, 800.0) == 500.0


# --------------------------------------------------------------------------- #
# Selections
# --------------------------------------------------------------------------- #


def test_merge_selections_keeps_functions_from_every_dict():
    merged = _default_selection()
    quality = merged["quality_assessment"]
    for name in (
        "detect_non_consecutive_manoeuvres",  # ventilator QA
        "evaluate_event_timing",  # multimodal QA
        "snr_pseudo",  # EMG QA
        "ventilator_respiratory_rate",
    ):
        assert name in quality


def test_merge_selections_does_not_mutate_inputs():
    before = copy.deepcopy(ALL_SELECTIONS)
    merged = _default_selection()
    merged["quality_assessment"]["snr_pseudo"] = False
    assert ALL_SELECTIONS == before


def test_ventilator_event_selection_has_only_ventilator_functions():
    """Bug: the ventilator dict still lists the EMG on/offset detectors."""
    assert set(defaults.VENTILATOR_EVENT_DETECTION_FUNCTIONS["event_detection"]) == {
        "detect_ventilator_breath",
        "find_occluded_breaths",
    }


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


def test_build_context_without_ventilator(peaks, processed_emg):
    context = defaults._build_context(processed_emg["envelope"], FS, peaks)
    assert context.ventilator_signals is None
    assert context.ventilator_fs is None
    assert context.vent_width_samples is None
    assert np.isnan(context.peep)
    assert context.baseline is None


def test_build_context_converts_windows_to_samples(peaks, processed_emg):
    context = defaults._build_context(
        processed_emg["envelope"], FS, peaks, baseline_window_seconds=5.0
    )
    assert context.window_samples == int(5.0 * FS)
    assert context.step_samples == int(0.2 * FS)


# --------------------------------------------------------------------------- #
# Disabling unavailable functions
# --------------------------------------------------------------------------- #


def _context_with_baseline(processed_emg, peaks, **kwargs):
    context = defaults._build_context(processed_emg["envelope"], FS, peaks, **kwargs)
    context.baseline = np.full_like(processed_emg["envelope"], BASELINE_LEVEL)
    return context


def test_disable_skips_selected_ventilator_functions_without_ventilator(
    processed_emg, peaks
):
    """Bug: the membership check never matches, so nothing gets disabled."""
    context = _context_with_baseline(processed_emg, peaks)
    defaults._disable_unavailable_postprocessing(context)
    assert VENTILATOR_ONLY <= set(context.skipped)
    for key in VENTILATOR_ONLY:
        category, name = key.split(".")
        assert context.selected_functions[category][name] is False


def test_disable_ignores_unselected_functions(processed_emg, peaks):
    selection = {"baseline": {"moving_baseline": True}}
    context = _context_with_baseline(processed_emg, peaks, selected_functions=selection)
    defaults._disable_unavailable_postprocessing(context)
    assert not (set(context.skipped) & VENTILATOR_ONLY)


def test_quality_skips_functions_needing_time_products(processed_emg, peaks):
    selection = {
        "quality_assessment": {
            "detect_extreme_time_products": True,
            "evaluate_bell_curve_error": True,
        },
    }
    context = _context_with_baseline(processed_emg, peaks, selected_functions=selection)
    defaults._compute_emg_quality_default(context)  # no features computed
    assert "quality_assessment.detect_extreme_time_products" in context.skipped
    assert "quality_assessment.evaluate_bell_curve_error" in context.skipped
    assert context.computed["quality_assessment"] == {}


def test_quality_skips_local_high_aub_without_area_under_baseline(processed_emg, peaks):
    selection = {"quality_assessment": {"detect_local_high_aub": True}}
    context = _context_with_baseline(processed_emg, peaks, selected_functions=selection)
    defaults._compute_emg_quality_default(context)
    assert "quality_assessment.detect_local_high_aub" in context.skipped
    assert context.computed["quality_assessment"] == {}


# --------------------------------------------------------------------------- #
# Breath detection
# --------------------------------------------------------------------------- #


def test_detect_breaths_rejects_unprocessed_input():
    with pytest.raises(UnsupportedWorkflowError):
        defaults._detect_breaths_default({"fs": FS})


def test_detect_breaths_warns_without_baseline(processed_emg):
    with pytest.warns(UserWarning, match="baseline"):
        defaults._detect_breaths_default(processed_emg)


def test_detect_breaths_returns_point_events_near_true_peaks(processed_emg, peaks):
    baseline = np.full_like(processed_emg["envelope"], BASELINE_LEVEL)
    events = defaults._detect_breaths_default(processed_emg, baseline=baseline)

    assert events, "no breaths detected on a clean synthetic envelope"
    tolerance = int(0.2 * FS)
    for event in events:
        index = event["extremum_index"]
        assert event["start_index"] == event["end_index"] == index
        assert event["extremum_time"] == pytest.approx(index / FS)
        assert event["sample_frequency"] == FS
        assert np.min(np.abs(peaks - index)) <= tolerance


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_preprocess_returns_expected_fields(recording):
    result = defaults._preprocess_default(recording, channel=0, remove_ecg=False)
    n = recording["array"].shape[1]
    for key in (
        "channel",
        "fs",
        "raw_channel",
        "filtered",
        "clean",
        "envelope",
        "filter",
    ):
        assert key in result
    assert result["channel"] == 0
    assert result["fs"] == recording["metadata"]["fs"]
    assert len(result["raw_channel"]) == len(result["filtered"]) == n
    assert len(result["envelope"]) == n


def test_preprocess_without_ecg_removal_keeps_filtered_signal(recording):
    result = defaults._preprocess_default(recording, channel=0, remove_ecg=False)
    np.testing.assert_array_equal(result["clean"], result["filtered"])


def test_preprocess_can_skip_envelope(recording):
    result = defaults._preprocess_default(
        recording, channel=0, remove_ecg=False, compute_envelope=False
    )
    assert result["envelope"] is None


def test_preprocess_rejects_unknown_ecg_method(recording):
    with pytest.raises(ValueError, match="Unknown ECG removal method"):
        defaults._preprocess_default(recording, channel=0, ecg_removal_method="nope")


def test_preprocess_records_filter_settings(recording):
    result = defaults._preprocess_default(
        recording,
        channel=0,
        remove_ecg=False,
        high_pass_hz=10.0,
        low_pass_hz=400.0,
        notch_base_frequency=50.0,
    )
    settings = result["filter"]
    assert settings["high_pass_hz"] == 10.0
    assert settings["low_pass_hz"] == 400.0
    assert settings["notch_base_frequency"] == 50.0
    assert settings["notch_quality_factor"] == 30.0
    assert settings["notch_before_bandpass"] is False


def test_preprocess_default_low_pass_matches_helper(recording):
    result = defaults._preprocess_default(recording, channel=0, remove_ecg=False)
    fs = recording["metadata"]["fs"]
    assert result["filter"]["low_pass_hz"] == defaults.default_low_pass_hz(fs)


def test_preprocess_does_not_mutate_recording(recording):
    before = copy.deepcopy(recording["metadata"])
    raw_before = recording["array"].copy()
    defaults._preprocess_default(recording, channel=0, remove_ecg=False)
    assert recording["metadata"] == before
    np.testing.assert_array_equal(recording["array"], raw_before)


# --------------------------------------------------------------------------- #
# Postprocessing: end to end
# --------------------------------------------------------------------------- #


def test_postprocess_rejects_unprocessed_input():
    with pytest.raises(UnsupportedWorkflowError):
        defaults._postprocess_default({"fs": FS}, [])


@pytest.mark.usefixtures("inject_peaks")
@pytest.mark.parametrize("with_ventilator", [False, True])
def test_every_selected_function_is_either_computed_or_skipped(
    request, processed_emg, peaks, with_ventilator
):
    """Core contract: no selected function silently disappears or appears twice."""
    kwargs = {"ecg_peak_indices": peaks + 50}
    if with_ventilator:
        request.getfixturevalue("ventilator")
        kwargs["ventilator"] = object()

    result = _run(processed_emg, peaks, **kwargs)

    computed = _computed_names(result)
    skipped = set(result["skipped"])
    selected = _selected_names(_default_selection())

    assert not (computed & skipped), f"both computed and skipped: {computed & skipped}"
    assert selected <= computed | skipped, f"missing: {selected - computed - skipped}"
    assert skipped <= selected, f"skip reasons for unselected: {skipped - selected}"


@pytest.mark.usefixtures("inject_peaks")
def test_postprocess_computes_emg_features_with_peaks_and_baseline(
    processed_emg, peaks
):
    """Guards the call order: disabling must not run before the baseline exists."""
    result = _run(processed_emg, peaks, ecg_peak_indices=peaks + 50)
    assert set(result["computed"]["baseline"]) == {
        "moving_baseline",
        "slopesum_baseline",
    }
    assert EMG_FEATURES <= set(result["computed"]["features"])
    for name in ("snr_pseudo", "percentage_under_baseline"):
        assert name in result["computed"]["quality_assessment"]


@pytest.mark.usefixtures("inject_peaks")
def test_postprocess_returns_peak_indices_and_settings(processed_emg, peaks):
    result = _run(
        processed_emg, peaks, ecg_peak_indices=peaks + 50, baseline_window_seconds=5.0
    )
    np.testing.assert_array_equal(result["peak_indices"], peaks)
    assert result["settings"]["baseline_window_seconds"] == 5.0
    assert np.isnan(result["settings"]["peep"])


@pytest.mark.usefixtures("inject_peaks")
def test_postprocess_does_not_mutate_caller_selection(processed_emg, peaks):
    selection = _default_selection()
    before = copy.deepcopy(selection)
    _run(processed_emg, peaks, selected_functions=selection)
    assert selection == before


@pytest.mark.usefixtures("inject_peaks")
def test_postprocess_without_ecg_peaks_skips_interpeak_distance(processed_emg, peaks):
    """Bug: interpeak_distance is called with ecg_peak_indices=None."""
    result = _run(processed_emg, peaks)
    assert "quality_assessment.interpeak_distance" in result["skipped"]
    assert "interpeak_distance" not in result["computed"]["quality_assessment"]


@pytest.mark.usefixtures("inject_peaks")
@pytest.mark.parametrize("n_peaks", [0, 1])
def test_postprocess_without_enough_peaks_skips_instead_of_raising(
    processed_emg, peaks, n_peaks
):
    result = _run(processed_emg, peaks[:n_peaks], ecg_peak_indices=peaks + 50)

    assert result["computed"]["features"] == {}
    for name in EMG_FEATURES:
        assert f"features.{name}" in result["skipped"]
    for name in ("onoffpeak_baseline_crossing", "onoffpeak_slope_extrapolation"):
        assert f"event_detection.{name}" in result["skipped"]
    # The baseline does not need peaks.
    assert result["computed"]["baseline"]


@pytest.mark.usefixtures("inject_peaks")
def test_postprocess_without_ventilator_reports_ventilator_skips(processed_emg, peaks):
    result = _run(processed_emg, peaks, ecg_peak_indices=peaks + 50)
    assert VENTILATOR_ONLY <= set(result["skipped"])
    assert "detect_ventilator_breath" not in result["computed"]["event_detection"]


@pytest.mark.usefixtures("inject_peaks", "ventilator")
def test_postprocess_with_ventilator_computes_ventilator_results(processed_emg, peaks):
    result = _run(
        processed_emg, peaks, ecg_peak_indices=peaks + 50, ventilator=object()
    )

    vent_breaths = result["computed"]["event_detection"]["detect_ventilator_breath"]
    assert abs(len(vent_breaths) - len(peaks)) <= 1

    quality = result["computed"]["quality_assessment"]
    assert "ventilator_respiratory_rate" in quality
    assert "evaluate_respiratory_rates" in quality
    assert "evaluate_event_timing" in quality


@pytest.mark.usefixtures("inject_peaks", "ventilator")
def test_ventilator_rate_follows_its_own_selection(processed_emg, peaks):
    """Bug: the ventilator rate is gated by 'evaluate_respiratory_rates'."""
    selection = {
        "event_detection": {"detect_ventilator_breath": True},
        "quality_assessment": {"ventilator_respiratory_rate": True},
    }
    result = _run(
        processed_emg, peaks, selected_functions=selection, ventilator=object()
    )
    assert "ventilator_respiratory_rate" in result["computed"]["quality_assessment"]


@pytest.mark.usefixtures("inject_peaks")
def test_missing_onoffsets_skip_dependent_functions_without_error(
    monkeypatch, processed_emg, peaks
):
    """Bug: quality functions read features that were skipped -> KeyError."""
    empty = np.asarray([], dtype=int)
    no_bounds = (empty, empty, empty, empty, np.asarray([], dtype=bool))
    monkeypatch.setattr(
        intervals, "onoff_from_baseline_crossings", lambda **_: no_bounds
    )
    monkeypatch.setattr(intervals, "onoff_from_slope", lambda **_: no_bounds)
    result = _run(processed_emg, peaks, ecg_peak_indices=peaks + 50)

    for name in ("time_to_peak", "pseudo_slope", "time_product", "area_under_baseline"):
        assert result["skipped"][f"features.{name}"] == defaults._BOUNDS_REASON
    for name in (
        "percentage_under_baseline",
        "detect_local_high_aub",
        "detect_extreme_time_products",
        "evaluate_bell_curve_error",
    ):
        assert f"quality_assessment.{name}" in result["skipped"]
    # Features that only need peaks are still computed.
    assert {"amplitude", "respiratory_rate"} <= set(result["computed"]["features"])


def test_postprocess_accepts_detect_breaths_output(processed_emg):
    """Contract between the detector and postprocessing, without patching."""
    baseline = np.full_like(processed_emg["envelope"], BASELINE_LEVEL)
    # _detect_breaths_default returns a list of dictionaries, which are converted into BreathEvent objects in the wrapper function in _CoreMixin.
    # here, we'll coerce them into BreathEvent objects to simulate the output of the detector.
    events = defaults._detect_breaths_default(processed_emg, baseline=baseline)

    result = _run(
        processed_emg,
        coerce_breath_events(
            events,
            modality="emg",
            source="resurfemg.defaults._detect_breaths_default",
        ),
    )
    np.testing.assert_array_equal(
        result["peak_indices"], [event["extremum_index"] for event in events]
    )
