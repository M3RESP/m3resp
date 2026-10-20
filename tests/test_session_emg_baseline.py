"""`M3Session.emg_moving_baseline` and `emg_slopesum_baseline`: the EMG baseline from the session."""

from __future__ import annotations

from pathlib import Path

import pytest

from m3resp.core.exceptions import MissingModalityDataError
from m3resp.core.session import M3Session

pytest.importorskip("resurfemg")
np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
EMG_PATH = REPO_ROOT / "tests" / "data" / "m3resp_multimodal_1_emg.Poly5"


@pytest.fixture
def preprocessed_session() -> M3Session:
    session = M3Session()
    session.load_emg(str(EMG_PATH), verbose=False)
    session.preprocess_emg(channel=1)
    return session


def test_baseline_matches_the_adapter_with_the_same_settings(preprocessed_session):
    processed = preprocessed_session.processed["emg"]
    fs = processed["fs"]

    baseline = preprocessed_session.emg_moving_baseline(
        window_seconds=5.0, step_seconds=0.5, percentile=40.0
    )

    expected = preprocessed_session.emg_adapter.moving_baseline(
        np.asarray(processed["envelope"], dtype=float),
        window_samples=int(5.0 * fs),
        step_samples=int(0.5 * fs),
        percentile=40.0,
    )
    np.testing.assert_array_equal(baseline, expected)
    assert len(baseline) == len(processed["envelope"])


def _baseline_signals(session: M3Session) -> list:
    return [
        signal
        for signal in session.signals.for_modality("emg")
        if "_moving_baseline" in (signal.name or "")
    ]


def test_baseline_is_recorded_and_added_to_the_signals(preprocessed_session):
    preprocessed_session.emg_moving_baseline(window_seconds=5.0)

    entry = preprocessed_session.provenance[-1]
    assert entry.action == "emg_moving_baseline"
    assert entry.parameters["variant"] is None
    assert entry.parameters["window_seconds"] == 5.0
    [signal] = _baseline_signals(preprocessed_session)
    assert signal.name.endswith("_moving_baseline")
    assert "variant" not in signal.metadata


def test_a_second_call_replaces_the_baseline_signal(preprocessed_session):
    preprocessed_session.emg_moving_baseline(window_seconds=5.0)
    second = preprocessed_session.emg_moving_baseline(window_seconds=10.0)

    [signal] = _baseline_signals(preprocessed_session)
    np.testing.assert_array_equal(signal.values, second)
    assert signal.metadata["requested_window_seconds"] == 10.0


def test_baseline_reads_a_named_preprocessing_variant(preprocessed_session):
    preprocessed_session.preprocess_emg(
        channel=1, variant="wide", envelope_window_seconds=1.0
    )

    default = preprocessed_session.emg_moving_baseline()
    wide = preprocessed_session.emg_moving_baseline(variant="wide")

    expected = preprocessed_session.emg_adapter.moving_baseline(
        np.asarray(
            preprocessed_session.processed_variants["emg"]["wide"]["envelope"],
            dtype=float,
        ),
        window_samples=int(7.5 * preprocessed_session.processed["emg"]["fs"]),
        step_samples=int(0.2 * preprocessed_session.processed["emg"]["fs"]),
        percentile=33.0,
    )
    np.testing.assert_array_equal(wide, expected)
    assert not np.array_equal(default, wide)
    assert preprocessed_session.provenance[-1].parameters["variant"] == "wide"
    by_name = {s.name: s for s in _baseline_signals(preprocessed_session)}
    wide_signal = next(s for name, s in by_name.items() if name.endswith(":wide"))
    assert wide_signal.metadata["variant"] == "wide"
    assert len(by_name) == 2


def test_baseline_needs_preprocessed_emg():
    with pytest.raises(MissingModalityDataError, match="preprocess_emg"):
        M3Session().emg_moving_baseline()


def test_baseline_names_an_unknown_variant(preprocessed_session):
    with pytest.raises(MissingModalityDataError, match="'other'"):
        preprocessed_session.emg_moving_baseline(variant="other")


def test_baseline_needs_an_envelope():
    session = M3Session()
    session.load_emg(str(EMG_PATH), verbose=False)
    session.preprocess_emg(channel=1, compute_envelope=False)

    with pytest.raises(ValueError, match="emg.envelope"):
        session.emg_moving_baseline()


def test_preset_converts_baseline_settings_given_as_text(monkeypatch):
    session = M3Session()
    session.load_emg(str(EMG_PATH), verbose=False)
    seen: dict = {}
    original = session.emg_moving_baseline

    def spy(**kwargs):
        seen.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(session, "emg_moving_baseline", spy)

    session.run_preset(
        "emg",
        config={
            "preprocess": {"channel": 1},
            "baseline": {"window_seconds": "5", "percentile": "40"},
        },
    )

    assert seen == {"window_seconds": 5.0, "step_seconds": 0.2, "percentile": 40.0}


def test_preset_refuses_a_baseline_variant():
    session = M3Session()
    session.load_emg(str(EMG_PATH), verbose=False)

    with pytest.raises(TypeError, match="variant"):
        session.run_preset(
            "emg",
            config={"preprocess": {"channel": 1}, "baseline": {"variant": "wide"}},
        )


def _slopesum_signals(session: M3Session) -> list:
    return [
        signal
        for signal in session.signals.for_modality("emg")
        if "_slopesum_baseline" in (signal.name or "")
    ]


def test_slopesum_baseline_matches_the_adapter_with_the_same_settings(
    preprocessed_session,
):
    processed = preprocessed_session.processed["emg"]
    fs = processed["fs"]

    baseline = preprocessed_session.emg_slopesum_baseline(
        window_seconds=5.0, step_seconds=0.5, percentile=40.0
    )

    expected, _, _, _ = preprocessed_session.emg_adapter.slopesum_baseline(
        np.asarray(processed["envelope"], dtype=float),
        window_samples=int(5.0 * fs),
        step_samples=int(0.5 * fs),
        sample_frequency=fs,
        percentile=40.0,
        augmented_percentile=25.0,
        moving_average_samples=int(0.5 * fs),
        percentile_window_samples=int(1.0 * fs),
    )
    np.testing.assert_array_equal(baseline, expected)
    assert len(baseline) == len(processed["envelope"])


def test_slopesum_baseline_is_recorded_and_adds_three_signals(preprocessed_session):
    preprocessed_session.emg_slopesum_baseline(window_seconds=5.0)

    entry = preprocessed_session.provenance[-1]
    assert entry.action == "emg_slopesum_baseline"
    assert entry.parameters["variant"] is None
    assert entry.parameters["requested_window_seconds"] == 5.0
    names = sorted(
        s.name[s.name.index("slopesum") :]
        for s in _slopesum_signals(preprocessed_session)
    )
    assert names == [
        "slopesum_baseline",
        "slopesum_baseline_running_mean",
        "slopesum_baseline_running_std",
    ]


def test_a_second_slopesum_call_replaces_the_three_signals(preprocessed_session):
    preprocessed_session.emg_slopesum_baseline(window_seconds=5.0)
    second = preprocessed_session.emg_slopesum_baseline(window_seconds=10.0)

    signals = _slopesum_signals(preprocessed_session)
    assert len(signals) == 3
    [baseline_signal] = [s for s in signals if s.name.endswith("_slopesum_baseline")]
    np.testing.assert_array_equal(baseline_signal.values, second)
    assert baseline_signal.metadata["requested_window_seconds"] == 10.0


def test_slopesum_baseline_reads_a_named_preprocessing_variant(preprocessed_session):
    preprocessed_session.preprocess_emg(
        channel=1, variant="wide", envelope_window_seconds=1.0
    )

    default = preprocessed_session.emg_slopesum_baseline()
    wide = preprocessed_session.emg_slopesum_baseline(variant="wide")

    assert not np.array_equal(default, wide)
    assert preprocessed_session.provenance[-1].parameters["variant"] == "wide"
    wide_signals = [
        s for s in _slopesum_signals(preprocessed_session) if s.name.endswith(":wide")
    ]
    assert len(wide_signals) == 3
    assert all(s.metadata["variant"] == "wide" for s in wide_signals)
    assert len(_slopesum_signals(preprocessed_session)) == 6


def test_slopesum_baseline_needs_preprocessed_emg():
    with pytest.raises(MissingModalityDataError, match="preprocess_emg"):
        M3Session().emg_slopesum_baseline()
