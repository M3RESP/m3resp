"""The separate EMG preprocessing steps: channel, notch filter, band-pass filter, envelope."""

from __future__ import annotations

from pathlib import Path

import pytest

from m3resp.core.session import M3Session
from m3resp.processing.windows import rolling_envelope
from m3resp.workflows import run_workflow
from m3resp.workflows.steps.emg import (
    bandpass_filter,
    envelope,
    moving_baseline,
    notch_filter,
    preprocess,
    select_channel,
)

pytest.importorskip("resurfemg")
np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
EMG_PATH = REPO_ROOT / "tests" / "data" / "emg_data_synth_quiet_breathing.Poly5"
FS = 2048.0


@pytest.fixture
def loaded_session() -> M3Session:
    session = M3Session()
    session.load_emg(str(EMG_PATH), verbose=False)
    return session


def _bundle(values: np.ndarray, **extra) -> dict:
    return {
        "fs": FS,
        "channel": 0,
        "raw_channel": values,
        "filtered": values,
        "filter": {},
        **extra,
    }


@pytest.mark.parametrize("notch_before_bandpass", [True, False])
def test_separate_steps_give_the_same_trace_as_emg_preprocess(
    loaded_session, notch_before_bandpass
):
    settings = {"high_pass_hz": 20.0, "low_pass_hz": 500.0}
    notch = {"base_frequency": 50.0, "max_frequency": 950.0, "quality_factor": 30.0}

    combined = preprocess(
        loaded_session,
        channel=1,
        notch_base_frequency=50.0,
        notch_max_frequency=950.0,
        notch_quality_factor=30.0,
        notch_before_bandpass=notch_before_bandpass,
        **settings,
    )["processed_emg"]

    bundle = select_channel(loaded_session, channel=1)["processed_emg"]
    if notch_before_bandpass:
        bundle = notch_filter(loaded_session, bundle, **notch)[
            "processed_emg_after_notch"
        ]
    bundle = bandpass_filter(loaded_session, bundle, **settings)[
        "processed_emg_after_bandpass"
    ]
    if not notch_before_bandpass:
        bundle = notch_filter(loaded_session, bundle, **notch)[
            "processed_emg_after_notch"
        ]

    np.testing.assert_array_equal(bundle["filtered"], combined["filtered"])
    assert combined["envelope"] is None
    assert bundle["filter"]["notch_before_bandpass"] is notch_before_bandpass
    assert loaded_session.processed["emg"] is bundle


def test_envelope_follows_the_chosen_window_and_method():
    rng = np.random.default_rng(3)
    values = rng.normal(size=4096)
    session = M3Session()

    result = envelope(session, _bundle(values), window_seconds=0.3, method="median")

    expected = rolling_envelope(values, window_length=int(0.3 * FS), method="median")
    bundle = result["processed_emg_with_envelope"]
    np.testing.assert_array_equal(bundle["envelope"], expected)
    assert bundle["envelope_source"] == "filtered"
    assert result["envelope_signal"].metadata["envelope_window_samples"] == int(
        0.3 * FS
    )
    assert session.processed["emg"] is bundle


def test_rms_envelope_of_a_constant_is_that_constant():
    session = M3Session()
    values = np.full(2048, 3.0)

    result = envelope(session, _bundle(values), window_seconds=0.25)

    np.testing.assert_allclose(result["processed_emg_with_envelope"]["envelope"], 3.0)


def test_envelope_uses_the_ecg_cleaned_trace_when_present():
    session = M3Session()
    values = np.ones(2048)
    cleaned = np.full(2048, 2.0)

    result = envelope(session, _bundle(values, ecg_cleaned=cleaned))

    np.testing.assert_allclose(result["processed_emg_with_envelope"]["envelope"], 2.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_seconds": 0.0}, "window_seconds"),
        ({"window_seconds": -1.0}, "window_seconds"),
        ({"method": "mean"}, "method must be one of"),
    ],
)
def test_envelope_refuses_bad_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        envelope(M3Session(), _bundle(np.ones(100)), **kwargs)


def test_emg_preprocess_refuses_envelope_settings(loaded_session):
    with pytest.raises(ValueError, match="emg.envelope"):
        preprocess(loaded_session, channel=1, envelope_window_seconds=0.5)


def test_a_step_that_needs_an_envelope_names_the_envelope_step():
    bundle = _bundle(np.ones(4096), envelope=None)

    with pytest.raises(ValueError, match="emg.envelope"):
        moving_baseline(M3Session(), bundle)


def test_filters_refuse_a_trace_already_cleaned_of_ecg():
    bundle = _bundle(np.ones(4096), ecg_cleaned=np.ones(4096))

    with pytest.raises(ValueError, match="before the ECG-removal step"):
        bandpass_filter(M3Session(), bundle)
    with pytest.raises(ValueError, match="before the ECG-removal step"):
        notch_filter(M3Session(), bundle, base_frequency=50.0)


def test_notch_max_frequency_above_nyquist_warns_and_records_nyquist():
    with pytest.warns(UserWarning, match="above the Nyquist frequency"):
        after = notch_filter(
            M3Session(),
            _bundle(np.ones(4096)),
            base_frequency=50.0,
            max_frequency=2000.0,
        )["processed_emg_after_notch"]

    assert after["filter"]["notch_max_frequency"] == FS / 2


def test_emg_preprocess_records_nyquist_for_a_notch_max_frequency_above_it(
    loaded_session,
):
    with pytest.warns(UserWarning, match="above the Nyquist frequency"):
        processed = preprocess(
            loaded_session,
            channel=1,
            notch_base_frequency=50.0,
            notch_max_frequency=2000.0,
        )["processed_emg"]

    assert processed["filter"]["notch_max_frequency"] == FS / 2


def test_bandpass_refuses_a_lower_cutoff_above_the_upper_one():
    with pytest.raises(ValueError, match="must be below"):
        bandpass_filter(
            M3Session(), _bundle(np.ones(4096)), high_pass_hz=300.0, low_pass_hz=200.0
        )


def test_filters_start_from_the_raw_channel_after_select_channel(loaded_session):
    bundle = select_channel(loaded_session, channel=1)["processed_emg"]
    assert "filtered" not in bundle

    after = bandpass_filter(loaded_session, bundle)["processed_emg_after_bandpass"]

    assert after["filter"]["high_pass_hz"] == 20.0
    assert after["filter"]["low_pass_hz"] == 500.0
    assert after["raw_channel"] is bundle["raw_channel"]


def test_workflow_runs_the_separate_steps_in_order():
    spec = {
        "name": "emg-separate-preprocessing",
        "inputs": {"emg_file": str(EMG_PATH)},
        "steps": [
            {"uses": "emg.load", "with": {"file_path": "@emg_file"}},
            {
                "uses": "emg.select_channel",
                "with": {"channel": 1},
                "out": {"processed_emg": "selected_emg"},
            },
            {
                "uses": "emg.bandpass_filter",
                "in": {"processed_emg": "selected_emg"},
            },
            {
                "uses": "emg.envelope",
                "in": {"processed_emg": "processed_emg_after_bandpass"},
                "out": {"processed_emg_with_envelope": "processed_emg"},
            },
            {"uses": "emg.moving_baseline"},
            {"uses": "emg.detect_breaths"},
        ],
    }

    result = run_workflow(spec)

    session = result.context.session
    assert session.processed["emg"]["envelope"] is not None
    assert len(result.value("emg_breath_events")) > 0
    recorded = [entry.action for entry in session.provenance]
    for step in ("emg.select_channel", "emg.bandpass_filter", "emg.envelope"):
        assert recorded.count(step) == 1
