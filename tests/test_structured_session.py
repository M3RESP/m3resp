"""Milestone 2.2 - structured M3Session (plan_stage2.md Sec 14).

Acceptance criterion under test: a session can store EIT and EMG data in the
same structure (``session.signals``/``parameter_results``/``quality``),
populated on the default preprocess/postprocess path. The EIT preprocessing
(``preprocess_eit_recording``) and the EMG adapter are given fixed, fake
outputs, so these tests check how the session stores them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from m3resp import M3Session
from m3resp.adapters import ReSurfEMGAdapter
from m3resp.core.exceptions import VariantAlreadyExistsError
from m3resp.data import (
    BreathEvent,
    IntervalData,
    ParameterResult,
    QualityFlag,
    Signal,
)


def _waveform(values: list[float], name: str) -> Signal:
    return Signal(
        values=values,
        time=[0.0, 1.0, 2.0],
        sample_frequency=1.0,
        unit="a.u.",
        name=name,
        modality="eit",
        category="impedance",
        channel="global_impedance",
    )


def _fake_eit_preprocessed() -> dict[str, Any]:
    return {
        "raw_global_impedance": _waveform([1.0, 2.0, 3.0], "raw gi"),
        "filtered_global_impedance": _waveform([1.1, 2.1, 3.1], "filtered gi"),
        "respiratory_rate_hz": 0.3,
        "heart_rate_hz": 1.1,
        "breath_intervals": IntervalData(
            name="eit_breaths", modality="eit", intervals=[]
        ),
        "continuous_tiv": None,
        "eeli": None,
        "pixel_tiv": None,
    }


def _use_fake_eit_preprocessing(monkeypatch, make_result) -> None:
    """Make ``session.preprocess_eit()`` return a new ``make_result()``."""

    monkeypatch.setattr(
        "m3resp.core.session.preprocess_eit_recording",
        lambda *args, **kwargs: make_result(),
    )


def _fake_emg_preprocessed() -> dict[str, Any]:
    return {
        "fs": 100.0,
        "channel": 0,
        "raw_channel": [0.0, 1.0],
        "filtered": [0.1, 0.2],
        "envelope": [0.05, 0.15],
    }


def test_preprocess_eit_populates_typed_collections_on_default_adapter_path(
    monkeypatch,
):
    _use_fake_eit_preprocessing(monkeypatch, _fake_eit_preprocessed)
    session = M3Session()
    session.raw["eit"] = SimpleNamespace(data=object(), path="subject.eit")

    session.preprocess_eit()

    assert len(session.signals) == 2
    assert all(isinstance(s, Signal) for s in session.signals)
    assert {s.modality for s in session.signals.for_modality("eit")} == {"eit"}
    parameter_names = {p.name for p in session.parameter_results}
    assert {"respiratory_rate", "heart_rate"} <= parameter_names
    assert len(session.quality) == 2
    assert all(isinstance(flag, QualityFlag) for flag in session.quality)


def test_preprocess_eit_with_custom_callable_does_not_populate_collections():
    session = M3Session()
    session.raw["eit"] = SimpleNamespace(data=object(), path="subject.eit")

    session.preprocess_eit(preprocess=lambda data, **kwargs: {"custom": True})

    assert len(session.signals) == 0
    assert len(session.parameter_results) == 0
    assert len(session.quality) == 0


def test_preprocess_emg_populates_signals_on_default_adapter_path():
    emg_adapter = ReSurfEMGAdapter()
    emg_adapter.preprocess = lambda *args, **kwargs: _fake_emg_preprocessed()  # type: ignore[method-assign]
    session = M3Session(emg_adapter=emg_adapter)
    session.raw["emg"] = SimpleNamespace(data=object())

    session.preprocess_emg()

    assert len(session.signals) == 3
    processing_states = {s.processing_state for s in session.signals}
    assert processing_states == {"raw", "intermediate", "processed"}
    assert all(s.modality == "emg" for s in session.signals)


def test_postprocess_emg_populates_parameter_results_and_quality():
    emg_adapter = ReSurfEMGAdapter()
    emg_adapter.preprocess = lambda *args, **kwargs: _fake_emg_preprocessed()  # type: ignore[method-assign]
    emg_adapter.postprocess = lambda *args, **kwargs: {  # type: ignore[method-assign]
        "computed": {
            "baseline": {},
            "event_detection": {},
            "features": {"amplitude": 1.5},
            "quality_assessment": {"snr_pseudo": 12.0},
        },
        "skipped": {"pocc_quality": "missing ventilator pressure"},
    }
    session = M3Session(emg_adapter=emg_adapter)
    session.raw["emg"] = SimpleNamespace(data=object())
    session.preprocess_emg()

    session.postprocess_emg()

    parameters = list(session.parameter_results)
    assert len(parameters) == 1
    assert isinstance(parameters[0], ParameterResult)
    assert parameters[0].name == "amplitude"

    quality_by_name = {flag.name: flag for flag in session.quality}
    # snr_pseudo is a measurement, not a criterion - no upstream threshold
    # was configured, so it is not evaluated as a pass, but it is also not a
    # failed clinical check (see quality_flag_from_result's docstring).
    assert quality_by_name["snr_pseudo"].passed is False
    assert quality_by_name["snr_pseudo"].metadata["measurement_only"] is True
    assert quality_by_name["pocc_quality"].passed is False
    assert "measurement_only" not in quality_by_name["pocc_quality"].metadata
    assert sorted(flag.name for flag in session.quality.failed()) == [
        "pocc_quality",
        "snr_pseudo",
    ]


def test_session_stores_eit_and_emg_signals_in_the_same_collection(monkeypatch):
    """The Milestone 2.2 acceptance criterion, verbatim."""

    _use_fake_eit_preprocessing(monkeypatch, _fake_eit_preprocessed)
    emg_adapter = ReSurfEMGAdapter()
    emg_adapter.preprocess = lambda *args, **kwargs: _fake_emg_preprocessed()  # type: ignore[method-assign]

    session = M3Session(emg_adapter=emg_adapter)
    session.raw["eit"] = SimpleNamespace(data=object(), path="subject.eit")
    session.raw["emg"] = SimpleNamespace(data=object())

    session.preprocess_eit()
    session.preprocess_emg()

    modalities = {s.modality for s in session.signals}
    assert modalities == {"eit", "emg"}


def test_preprocess_eit_raises_on_duplicate_variant(monkeypatch):
    """Second write to an already-populated variant must not silently overwrite it.

    Reproduces the ambiguity flagged in PR #23 review: a reference stashed
    from an earlier `preprocess_eit()` call must not change meaning
    underneath the caller just because something else preprocessed again.
    """

    _use_fake_eit_preprocessing(monkeypatch, _fake_eit_preprocessed)
    session = M3Session()
    session.raw["eit"] = SimpleNamespace(data=object(), path="subject.eit")

    session.preprocess_eit()

    with pytest.raises(VariantAlreadyExistsError):
        session.preprocess_eit()

    with pytest.raises(VariantAlreadyExistsError):
        session.preprocess_eit(variant="default")


def test_preprocess_eit_overwrite_true_replaces_existing_variant(monkeypatch):
    _use_fake_eit_preprocessing(monkeypatch, _fake_eit_preprocessed)
    session = M3Session()
    session.raw["eit"] = SimpleNamespace(data=object(), path="subject.eit")

    first = session.preprocess_eit()
    second = session.preprocess_eit(overwrite=True)

    assert first is not second
    assert session.processed["eit"] is second
    assert session.processed_variants["eit"]["default"] is second


def test_session_allow_overwrite_lets_repeated_preprocess_calls_through(monkeypatch):
    """`allow_overwrite` lets notebook code opt in once instead of passing
    `overwrite=True` on every call, without weakening the default guard for
    code that doesn't set it (e.g. once copied into a reusable workflow)."""

    _use_fake_eit_preprocessing(monkeypatch, _fake_eit_preprocessed)
    session = M3Session(allow_overwrite=True)
    session.raw["eit"] = SimpleNamespace(data=object(), path="subject.eit")

    first = session.preprocess_eit()
    second = session.preprocess_eit()

    assert first is not second
    assert session.processed["eit"] is second


def test_session_allow_overwrite_defaults_to_false(monkeypatch):
    _use_fake_eit_preprocessing(monkeypatch, _fake_eit_preprocessed)
    session = M3Session()
    session.raw["eit"] = SimpleNamespace(data=object(), path="subject.eit")

    session.preprocess_eit()

    with pytest.raises(VariantAlreadyExistsError):
        session.preprocess_eit()


def test_preprocess_eit_distinct_variants_coexist_without_touching_default(monkeypatch):
    _use_fake_eit_preprocessing(monkeypatch, _fake_eit_preprocessed)
    session = M3Session()
    session.raw["eit"] = SimpleNamespace(data=object(), path="subject.eit")

    session.preprocess_eit(variant="mdn")
    session.preprocess_eit(variant="lowpass")

    assert set(session.processed_variants["eit"]) == {"mdn", "lowpass"}
    assert "eit" not in session.processed

    # A later unnamed call still targets "default" and doesn't collide with
    # either named variant.
    session.preprocess_eit()
    assert set(session.processed_variants["eit"]) == {"mdn", "lowpass", "default"}


def test_preprocess_emg_raises_on_duplicate_variant():
    emg_adapter = ReSurfEMGAdapter()
    emg_adapter.preprocess = lambda *args, **kwargs: _fake_emg_preprocessed()  # type: ignore[method-assign]
    session = M3Session(emg_adapter=emg_adapter)
    session.raw["emg"] = SimpleNamespace(data=object())

    session.preprocess_emg(variant="mdn")

    with pytest.raises(VariantAlreadyExistsError):
        session.preprocess_emg(variant="mdn")


def _fake_eit_preprocessed_with_tiv() -> dict[str, Any]:
    preprocessed = _fake_eit_preprocessed()
    breaths = [
        BreathEvent("eit", 0.0, 1.0, extremum_time=0.5),
        BreathEvent("eit", 1.0, 2.0, extremum_time=1.5),
    ]
    preprocessed["breath_intervals"] = IntervalData(
        name="eit_breaths", modality="eit", intervals=list(breaths)
    )
    preprocessed["continuous_tiv"] = IntervalData(
        name="continuous_tivs",
        modality="eit",
        intervals=list(breaths),
        values=[1.0, 2.0],
        unit="a.u.",
    )
    return preprocessed


def test_breaths_detected_after_preprocessing_are_the_breaths_of_the_tiv_values(
    monkeypatch,
):
    _use_fake_eit_preprocessing(monkeypatch, _fake_eit_preprocessed_with_tiv)
    session = M3Session()
    session.raw["eit"] = SimpleNamespace(data=object(), path="subject.eit")

    session.preprocess_eit()
    [tiv] = session.interval_data
    breaths = session.detect_eit_breaths()

    assert all(stored is used for stored, used in zip(breaths, tiv.intervals))


def test_tiv_values_computed_after_breath_detection_use_the_stored_breaths(monkeypatch):
    _use_fake_eit_preprocessing(monkeypatch, _fake_eit_preprocessed_with_tiv)
    session = M3Session()
    session.raw["eit"] = SimpleNamespace(data=object(), path="subject.eit")
    stored = BreathEvent("eit", 0.0, 1.0, extremum_time=0.5)
    session.add_events("eit_breaths", [stored])

    session.preprocess_eit()
    [tiv] = session.interval_data

    assert tiv.intervals[0] is stored
