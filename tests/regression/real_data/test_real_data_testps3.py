"""Dataset 3 (TestPS3): quiet breathing, one minute with EIT running and one
minute without.

The windows (37-97 s with EIT on, 140-200 s with EIT off, on the Biopac
clock) and the files are in `datasets.yaml`. The FluxMed and EIT recordings
are lined up on the Biopac clock by cross-correlating the airway pressure
they all record. Three kinds of check:

1. Same as upstream: each m3resp step gives exactly what eitprocessing or
   ReSurfEMG gives when called directly with the same settings on the same
   data.
2. Same as last time: every collected number equals the stored value in
   `reference_values.json` (counts and sample indices exactly, other numbers
   to a relative difference of 1e-9). After an intended change, rewrite the
   file with `scripts/update_real_data_references.py` and review the diff.
3. Agrees with physiology: heart rate from EIT against heart rate from the
   ECG in the EMG, breathing rates against the flow, breath onsets against
   flow onsets, and the EMG filter chains against each other with EIT on.

Run with ``pytest -m real_data``. Skipped when the recordings are not found
(see `real_data_helpers.data_folder`).
"""

from __future__ import annotations

import numpy as np
import pytest
import real_data_helpers as helpers

pytestmark = pytest.mark.real_data

#: Relative difference allowed for "same as last time" numbers.
REFERENCE_RTOL = 1e-9

#: Pass limits of the physiology checks.
MAX_HEART_RATE_DIFFERENCE = 5.0  # beats/min, EIT against ECG
MAX_RESPIRATORY_RATE_DIFFERENCE = 1.0  # breaths/min, against flow
MIN_F1 = 0.90  # breath onsets against flow onsets, within 1.0 s
MAX_CHAIN_SPREAD = 2.0  # per minute, between EMG filter chains
MIN_SYNC_CORRELATION = 0.9  # airway pressure, after lining up

EMG_FS = helpers.EMG_SAMPLE_FREQUENCY
SETTINGS = helpers.EMG_SETTINGS
WINDOWS = ("eit_on", "eit_off")

#: Filter chains compared with each other, with EIT on. The chain without a
#: notch filter is left out: the EIT interference is then counted as
#: heartbeats (about 158 per minute) and no EMG breath matches the flow.
CHAINS_TO_COMPARE = (
    "notch_bandpass",
    "notch_bandpass_gating",
    "notch_bandpass_wavelet",
)


# --- Helpers -----------------------------------------------------------------


def _assert_same(actual, expected, where="results"):
    """Integers, text and lists of integers exactly; other numbers to
    `REFERENCE_RTOL`."""

    if isinstance(expected, dict):
        assert sorted(actual) == sorted(expected), f"{where}: keys differ"
        for key in expected:
            _assert_same(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list) and all(isinstance(v, int) for v in expected):
        assert actual == expected, f"{where} differs"
    elif isinstance(expected, (list, float)):
        np.testing.assert_allclose(
            actual, expected, rtol=REFERENCE_RTOL, atol=0, err_msg=where
        )
    else:
        assert actual == expected, f"{where}: {actual!r} != {expected!r}"


def _expected_eit_window(testps3_run, testps3_paths):
    """The EIT window of the EIT file, cut with eitprocessing itself.

    The kept frames run from the first frame at or after the start up to,
    not including, the first frame at or after the end (seconds from the
    file's first frame).
    """

    from eitprocessing.datahandling.loading import load_eit_data

    dataset = testps3_run["dataset"]
    eit_window = dataset["eit_window"]
    start, end = dataset["windows"][eit_window]
    shift = testps3_run["sync"][eit_window]["eit"]["offset"]
    sequence = load_eit_data(testps3_paths["eit"], vendor="draeger")
    time = np.asarray(sequence.eit_data["raw"].time)
    seconds = time - time[0]
    first = int(np.searchsorted(seconds, start - shift, side="left"))
    stop = int(np.searchsorted(seconds, end - shift, side="left"))
    return sequence.select_by_index(start=first, end=stop).eit_data["raw"]


def _main(results, window):
    return results["emg"][window][helpers.MAIN_EMG_CHAIN]


# --- 1. Same as upstream: EIT ------------------------------------------------


def test_eit_slice_gives_the_frames_eitprocessing_cuts(testps3_run, testps3_paths):
    expected = _expected_eit_window(testps3_run, testps3_paths)
    in_session = testps3_run["eit_session"].eit.raw
    passed_on = testps3_run["eit_context"].get("raw_eit")

    np.testing.assert_array_equal(in_session.pixel_impedance, expected.pixel_impedance)
    # The data the next workflow steps read.
    np.testing.assert_array_equal(passed_on.pixel_impedance, expected.pixel_impedance)


def test_eit_rates_match_eitprocessing(testps3_run, testps3_paths):
    from eitprocessing.features.rate_detection import RateDetection

    expected = _expected_eit_window(testps3_run, testps3_paths)
    respiratory_rate_hz, heart_rate_hz = RateDetection("adult").apply(expected)

    context = testps3_run["eit_context"]
    assert context.get("respiratory_rate_hz") == respiratory_rate_hz
    assert context.get("heart_rate_hz") == heart_rate_hz


def test_eit_mdn_filter_and_breaths_match_eitprocessing(testps3_run, testps3_paths):
    from eitprocessing.features.breath_detection import BreathDetection
    from eitprocessing.features.rate_detection import RateDetection
    from eitprocessing.filters.mdn import MDNFilter

    expected = _expected_eit_window(testps3_run, testps3_paths)
    respiratory_rate_hz, heart_rate_hz = RateDetection("adult").apply(expected)
    filtered = MDNFilter(
        respiratory_rate=respiratory_rate_hz, heart_rate=heart_rate_hz
    ).apply(expected, captures={}, label="filtered")
    breaths = BreathDetection(minimum_duration=2 / 3).find_breaths(
        filtered.get_summed_impedance()
    )

    context = testps3_run["eit_context"]
    np.testing.assert_array_equal(
        context.get("filtered_eit").pixel_impedance, filtered.pixel_impedance
    )
    found = [
        (b.start_time, b.middle_time, b.end_time)
        for b in context.get("breath_intervals").values
    ]
    assert found == [(b.start_time, b.middle_time, b.end_time) for b in breaths.values]


# --- 1. Same as upstream: EMG ------------------------------------------------


@pytest.mark.parametrize("window", WINDOWS)
def test_emg_ecg_peaks_match_resurfemg(testps3_run, window):
    from resurfemg.preprocessing.ecg_removal import detect_ecg_peaks

    context = testps3_run["emg"][window][helpers.MAIN_EMG_CHAIN]
    expected = detect_ecg_peaks(
        context.get("filtered_emg")["filtered"],
        int(EMG_FS),
        peak_fraction=SETTINGS["ecg_peak_fraction"],
        peak_width_s=None,
        peak_distance=None,
        bp_filter=True,
    )
    np.testing.assert_array_equal(context.get("ecg_peak_indices"), expected)


@pytest.mark.parametrize("window", WINDOWS)
def test_emg_wavelet_denoising_matches_resurfemg(testps3_run, window):
    from resurfemg.preprocessing.ecg_removal import wavelet_denoising

    context = testps3_run["emg"][window][helpers.MAIN_EMG_CHAIN]
    cleaned, _, _, _ = wavelet_denoising(
        context.get("filtered_emg")["filtered"],
        np.asarray(context.get("ecg_peak_indices")),
        int(EMG_FS),
        hard_thresholding=True,
        n=SETTINGS["wavelet_levels"],
        wavelet_type="db2",
        fixed_threshold=SETTINGS["wavelet_threshold"],
    )
    np.testing.assert_array_equal(context.get("processed_emg")["ecg_cleaned"], cleaned)


@pytest.mark.parametrize("window", WINDOWS)
def test_emg_baseline_and_breath_peaks_match_resurfemg(testps3_run, window):
    from resurfemg.postprocessing.baseline import moving_baseline
    from resurfemg.postprocessing.event_detection import detect_emg_breaths

    from m3resp.processing.peaks import merge_close_peaks

    context = testps3_run["emg"][window][helpers.MAIN_EMG_CHAIN]
    envelope = context.get("processed_emg")["envelope"]
    baseline = moving_baseline(
        envelope,
        int(SETTINGS["baseline_window_seconds"] * EMG_FS),
        int(SETTINGS["baseline_step_seconds"] * EMG_FS),
        set_percentile=SETTINGS["baseline_percentile"],
    )
    np.testing.assert_array_equal(context.get("baseline"), baseline)

    min_width = int(SETTINGS["min_breath_width_seconds"] * EMG_FS)
    peaks = detect_emg_breaths(
        envelope,
        baseline,
        threshold=0,
        prominence_factor=SETTINGS["prominence_factor"],
        min_peak_width_s=min_width,
    )
    peaks = merge_close_peaks(
        np.asarray(peaks), envelope, min_distance_samples=min_width
    )
    np.testing.assert_array_equal(context.get("peak_indices"), peaks)


# --- 2. Same as last time ----------------------------------------------------


def test_results_equal_the_reference_values(testps3_results):
    reference = helpers.read_reference_values().get("testps3")
    if reference is None:
        pytest.fail(
            "No reference values for testps3; write them with "
            "scripts/update_real_data_references.py"
        )
    _assert_same(testps3_results, reference)


# --- 3. Agrees with physiology -----------------------------------------------


def test_devices_line_up_on_the_airway_pressure(testps3_results):
    for window, devices in testps3_results["sync"].items():
        for device, sync in devices.items():
            assert sync["correlation"] >= MIN_SYNC_CORRELATION, (window, device, sync)


def test_eit_heart_rate_agrees_with_ecg(testps3_results):
    eit = testps3_results["eit"]["heart_rate"]
    ecg = _main(testps3_results, "eit_on")["heart_rate"]
    assert abs(eit - ecg) <= MAX_HEART_RATE_DIFFERENCE, (
        f"EIT {eit:.2f} vs ECG {ecg:.2f} beats/min"
    )


@pytest.mark.parametrize("window", WINDOWS)
def test_emg_respiratory_rate_agrees_with_flow(testps3_results, window):
    flow = testps3_results["flow"][window]["respiratory_rate"]
    emg = _main(testps3_results, window)["respiratory_rate"]
    assert abs(emg - flow) <= MAX_RESPIRATORY_RATE_DIFFERENCE, (
        f"EMG {emg:.2f} vs flow {flow:.2f} breaths/min"
    )


def test_eit_respiratory_rate_agrees_with_flow(testps3_results):
    flow = testps3_results["flow"]["eit_on"]["respiratory_rate"]
    eit = testps3_results["eit"]["respiratory_rate"]
    assert abs(eit - flow) <= MAX_RESPIRATORY_RATE_DIFFERENCE, (
        f"EIT {eit:.2f} vs flow {flow:.2f} breaths/min"
    )


@pytest.mark.parametrize("window", WINDOWS)
def test_emg_breath_onsets_match_flow_onsets(testps3_results, window):
    match = _main(testps3_results, window)["flow_match"]
    assert helpers.f1_score(**match) >= MIN_F1, match


def test_eit_breath_starts_match_flow_onsets(testps3_results):
    start, end = helpers.load_datasets()["testps3"]["windows"]["eit_on"]
    times = np.asarray(testps3_results["eit"]["breath_start_times"])
    times = times[(times >= start) & (times < end)]
    flow = np.asarray(testps3_results["flow"]["eit_on"]["onset_times"])
    match = helpers.match_events(times, flow, helpers.MATCH_TOLERANCE_SECONDS)
    assert helpers.f1_score(*match) >= MIN_F1, match


@pytest.mark.parametrize("quantity", ["heart_rate", "respiratory_rate"])
def test_emg_filter_chains_agree_with_eit_on(testps3_results, quantity):
    values = {
        chain: testps3_results["emg"]["eit_on"][chain][quantity]
        for chain in CHAINS_TO_COMPARE
    }
    spread = max(values.values()) - min(values.values())
    assert spread <= MAX_CHAIN_SPREAD, values
