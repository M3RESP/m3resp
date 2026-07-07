"""Milestone 2.2 - structured M3Session (plan_stage2.md Sec 14).

Acceptance criterion under test: a session can store EIT and EMG data in the
same structure (``session.signals``/``parameter_results``/``quality``),
populated via each adapter's Milestone 2.3 conversion methods on the default
preprocess/postprocess path. Adapters are given fixed, fake preprocessing
outputs (rather than driving real ``eitprocessing``/``resurfemg`` numerics)
so these tests isolate the session <-> adapter wiring itself; the conversion
logic is already covered by ``tests/test_adapter_conversions.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from m3resp import M3Session
from m3resp.adapters import EITProcessingAdapter, ReSurfEMGAdapter
from m3resp.data import ParameterResult, QualityFlag, Signal


def _fake_eit_preprocessed() -> dict[str, Any]:
    return {
        "raw_global_impedance": SimpleNamespace(
            values=[1.0, 2.0, 3.0],
            time=[0.0, 1.0, 2.0],
            unit="a.u.",
            name="raw gi",
            sample_frequency=1.0,
        ),
        "filtered_global_impedance": SimpleNamespace(
            values=[1.1, 2.1, 3.1],
            time=[0.0, 1.0, 2.0],
            unit="a.u.",
            name="filtered gi",
            sample_frequency=1.0,
        ),
        "respiratory_rate_hz": 0.3,
        "heart_rate_hz": 1.1,
        "breath_intervals": object(),
        "continuous_tiv": None,
        "eeli": None,
        "pixel_tiv": None,
    }


def _fake_emg_preprocessed() -> dict[str, Any]:
    return {
        "fs": 100.0,
        "channel": 0,
        "raw_channel": [0.0, 1.0],
        "filtered": [0.1, 0.2],
        "envelope": [0.05, 0.15],
    }


def test_preprocess_eit_populates_typed_collections_on_default_adapter_path():
    eit_adapter = EITProcessingAdapter()
    eit_adapter.preprocess = lambda *args, **kwargs: _fake_eit_preprocessed()  # type: ignore[method-assign]
    session = M3Session(eit_adapter=eit_adapter)
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
    session = M3Session(eit_adapter=EITProcessingAdapter())
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
    assert processing_states == {"raw", "filtered", "processed"}
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
    assert quality_by_name["snr_pseudo"].passed is True
    assert quality_by_name["pocc_quality"].passed is False
    assert [flag.name for flag in session.quality.failed()] == ["pocc_quality"]


def test_session_stores_eit_and_emg_signals_in_the_same_collection():
    """The Milestone 2.2 acceptance criterion, verbatim."""

    eit_adapter = EITProcessingAdapter()
    eit_adapter.preprocess = lambda *args, **kwargs: _fake_eit_preprocessed()  # type: ignore[method-assign]
    emg_adapter = ReSurfEMGAdapter()
    emg_adapter.preprocess = lambda *args, **kwargs: _fake_emg_preprocessed()  # type: ignore[method-assign]

    session = M3Session(eit_adapter=eit_adapter, emg_adapter=emg_adapter)
    session.raw["eit"] = SimpleNamespace(data=object(), path="subject.eit")
    session.raw["emg"] = SimpleNamespace(data=object())

    session.preprocess_eit()
    session.preprocess_emg()

    modalities = {s.modality for s in session.signals}
    assert modalities == {"eit", "emg"}
