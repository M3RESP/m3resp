"""Stage 2 ReSurfEMG gap migration, Phase 0.2 (plan/stage2/
2_resurfemg_gap_migration_implementation_plan.md): a ground-truth ECG
detection-accuracy fixture.

The committed `data/source/data_from_repo/emg_data_synth_quiet_breathing.Poly5`
fixture is useful for wrapper-equivalence tests (same call, same result), but
its "true" R-peak locations are not independently known - it can only prove
the adapter forwards its call correctly, not that the detector actually finds
real peaks. This module generates a small, clean, deterministic ECG signal
via `neurokit2.ecg_simulate(random_state=<seed>)` directly (bypassing the
noisy EMG mixer resurfemg's own synthetic generator applies), and uses
`neurokit2`'s own peak detector (`nk.ecg_peaks`) - a different algorithm from
`resurfemg.preprocessing.ecg_removal.detect_ecg_peaks` - as an independent
reference, so comparing the two is a real accuracy check, not a circular
"does the wrapper call the function" check.

Tolerance: peaks must match within 50 ms (`_MATCH_TOLERANCE_SECONDS`). This is
generous relative to a 72-90 bpm heart rate (beat period ~0.7-0.8 s) - it is
wide enough to absorb the two algorithms' differing peak-alignment
conventions (raw ECG max vs a bandpass-filtered/QRS-derivative-based pick),
while still catching a genuinely missed or spurious beat.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("resurfemg")
neurokit2 = pytest.importorskip("neurokit2")

_MATCH_TOLERANCE_SECONDS = 0.05
_BOUNDARY_MARGIN_SAMPLES = 40


def _reference_ecg(
    *,
    sample_frequency: float,
    heart_rate_bpm: float,
    seed: int,
    duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """A clean synthetic ECG and its independently-detected reference R-peak
    sample indices (from `neurokit2`'s own detector, not `resurfemg`'s)."""

    ecg = np.asarray(
        neurokit2.ecg_simulate(
            duration=duration_seconds,
            sampling_rate=sample_frequency,
            heart_rate=heart_rate_bpm,
            random_state=seed,
        ),
        dtype=float,
    )
    _signals, info = neurokit2.ecg_peaks(ecg, sampling_rate=sample_frequency)
    reference_peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    return ecg, reference_peaks


def _boundary_hugging_fixture(
    *,
    sample_frequency: float = 2048.0,
    heart_rate_bpm: float = 75.0,
    seed: int = 42,
    duration_seconds: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop `_reference_ecg`'s output so the first/last reference peaks sit
    close to (but inside) the signal boundaries, per the plan's "peaks near
    but not outside signal boundaries" requirement."""

    ecg, reference_peaks = _reference_ecg(
        sample_frequency=sample_frequency,
        heart_rate_bpm=heart_rate_bpm,
        seed=seed,
        duration_seconds=duration_seconds,
    )
    assert len(reference_peaks) >= 3, "fixture must contain multiple ECG peaks"

    start = max(0, int(reference_peaks[0]) - _BOUNDARY_MARGIN_SAMPLES)
    end = min(len(ecg), int(reference_peaks[-1]) + _BOUNDARY_MARGIN_SAMPLES)
    cropped_ecg = ecg[start:end]
    cropped_peaks = (
        reference_peaks[(reference_peaks >= start) & (reference_peaks < end)] - start
    )
    return cropped_ecg, cropped_peaks


def test_ground_truth_fixture_generation_is_deterministic():
    ecg_a, peaks_a = _boundary_hugging_fixture()
    ecg_b, peaks_b = _boundary_hugging_fixture()

    np.testing.assert_array_equal(ecg_a, ecg_b)
    np.testing.assert_array_equal(peaks_a, peaks_b)


def test_fixture_reference_peaks_sit_near_but_inside_signal_boundaries():
    ecg, reference_peaks = _boundary_hugging_fixture(
        sample_frequency=2048.0,
    )

    fs = 2048.0
    first_peak_margin_seconds = reference_peaks[0] / fs
    last_peak_margin_seconds = (len(ecg) - 1 - reference_peaks[-1]) / fs

    assert 0.0 < first_peak_margin_seconds < _MATCH_TOLERANCE_SECONDS
    assert 0.0 < last_peak_margin_seconds < _MATCH_TOLERANCE_SECONDS


def test_resurfemg_detector_finds_every_reference_peak_within_tolerance():
    from resurfemg.preprocessing.ecg_removal import detect_ecg_peaks

    fs = 2048.0
    ecg, reference_peaks = _boundary_hugging_fixture(sample_frequency=fs)

    detected_peaks = np.asarray(detect_ecg_peaks(ecg, int(fs)), dtype=int)
    tolerance_samples = round(_MATCH_TOLERANCE_SECONDS * fs)

    unmatched = [
        int(reference)
        for reference in reference_peaks
        if not np.any(np.abs(detected_peaks - reference) <= tolerance_samples)
    ]
    assert unmatched == [], (
        f"reference peaks {unmatched} had no detected peak within "
        f"{_MATCH_TOLERANCE_SECONDS * 1000:.0f} ms"
    )


def test_resurfemg_detector_does_not_grossly_over_detect():
    """A coarse false-positive guard: the detector should not report
    substantially more peaks than the known heart rate implies (it may
    report a few extra edge-effect peaks, which the accuracy test above
    does not penalize, but a large excess would indicate a broken detector
    parameter, not benign edge noise)."""

    from resurfemg.preprocessing.ecg_removal import detect_ecg_peaks

    fs = 2048.0
    ecg, reference_peaks = _boundary_hugging_fixture(sample_frequency=fs)

    detected_peaks = detect_ecg_peaks(ecg, int(fs))

    assert len(reference_peaks) <= len(detected_peaks) <= len(reference_peaks) + 2


def test_upstream_ecg_detector_rejects_an_integer_valued_float_sample_frequency():
    """Characterization case for a Stage 3 correction (plan Phase 0.2):
    `resurfemg.preprocessing.ecg_removal.detect_ecg_peaks` passes `fs // 200`
    straight into a pandas rolling window, which requires a Python/NumPy
    integer. An integer-*valued* float (`2048.0`, as loaders commonly
    produce) is rejected even though no precision is lost - only a
    genuinely fractional `fs` should be a real error.

    A Stage 2 adapter method may convert `2048.0` -> `2048` after an exact
    integer-value check (`float(fs).is_integer()`), but must still reject or
    clearly report a truly non-integer `fs` rather than silently rounding
    it. No adapter method exists yet (Phase 1 of this plan adds
    `detect_ecg_peaks`); this test only pins down the upstream behavior so a
    regression is visible if it changes.
    """

    from resurfemg.preprocessing.ecg_removal import detect_ecg_peaks

    ecg, _reference_peaks = _boundary_hugging_fixture(sample_frequency=2048.0)

    with pytest.raises(ValueError, match="window must be an integer"):
        detect_ecg_peaks(ecg, 2048.0)

    # The identical call succeeds once fs is passed as a true int - this is
    # exactly the normalization a Stage 2 adapter method should apply for an
    # exact integer-valued float.
    detect_ecg_peaks(ecg, 2048)
