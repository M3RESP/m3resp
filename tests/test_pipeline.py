"""Tests for the declarative pipeline engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from m3resp.adapters import EITProcessingAdapter
from m3resp.core.exceptions import PipelineSpecError, UnknownStepError
from m3resp.core.session import M3Session
from m3resp.pipeline import load_spec, register_step, run_pipeline, validate_spec


# --------------------------------------------------------------------------- #
# Engine core (no upstream modality dependencies)                             #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _temp_steps():
    """Register throwaway steps for engine tests and clean them up after."""

    from m3resp.pipeline.registry import STEP_REGISTRY

    created: list[str] = []

    @register_step("t.make", writes=("a",))
    def _make(*, value: Any) -> dict[str, Any]:
        return {"a": value}

    @register_step("t.double", reads={"x": "a"}, writes=("result",))
    def _double(x: Any) -> dict[str, Any]:
        return {"result": x * 2}

    @register_step("t.bad_output", writes=("missing",))
    def _bad_output() -> dict[str, Any]:
        return {"something_else": 1}

    created.extend(["t.make", "t.double", "t.bad_output"])
    yield
    for name in created:
        STEP_REGISTRY.pop(name, None)


def test_engine_binds_inputs_outputs_and_at_references():
    spec = {
        "name": "smoke",
        "inputs": {"seed": 21},
        "steps": [
            {"uses": "t.make", "with": {"value": "@seed"}},
            {"uses": "t.double", "in": {"x": "a"}, "out": {"result": "doubled"}},
        ],
    }
    result = run_pipeline(spec)
    assert result.value("doubled") == 42
    assert result.outputs == {"a": 21, "doubled": 42}


def test_validation_rejects_unproduced_context_key():
    spec = {"name": "bad", "steps": [{"uses": "t.double", "in": {"x": "nope"}}]}
    with pytest.raises(PipelineSpecError, match="not produced"):
        validate_spec(load_spec(spec))


def test_engine_rejects_undeclared_output():
    spec = {"name": "bad", "steps": [{"uses": "t.bad_output"}]}
    with pytest.raises(PipelineSpecError, match="did not return it"):
        run_pipeline(spec)


def test_unknown_step_raises():
    with pytest.raises(UnknownStepError, match="no_such.step"):
        run_pipeline({"name": "x", "steps": [{"uses": "no_such.step"}]})


def test_unknown_input_reference_raises():
    spec = {"name": "x", "steps": [{"uses": "t.make", "with": {"value": "@absent"}}]}
    with pytest.raises(PipelineSpecError, match="unknown input"):
        run_pipeline(spec)


def test_yaml_and_json_parse_to_same_model():
    import yaml

    text = (
        "name: p\ninputs: {seed: 1}\nsteps:\n  - uses: t.make\n    with: {value: 2}\n"
    )
    yaml_model = load_spec(yaml.safe_load(text))
    json_model = load_spec(json.loads(json.dumps(yaml.safe_load(text))))
    assert yaml_model == json_model


def test_spec_requires_non_empty_steps():
    with pytest.raises(PipelineSpecError, match="non-empty 'steps'"):
        load_spec({"name": "p", "steps": []})


def test_public_api_exposes_engine_and_steps():
    import m3resp

    assert callable(m3resp.run_pipeline)
    steps = m3resp.available_steps()
    assert {
        "eit.load",
        "eit.detect_breaths",
        "emg.preprocess",
        "metric.interval_cv",
    } <= (set(steps))


# --------------------------------------------------------------------------- #
# ROTARC pipeline — fake eitprocessing primitives                             #
# --------------------------------------------------------------------------- #


class FakeCollection(dict):
    def add(self, value: Any, overwrite: bool = False) -> None:
        self[getattr(value, "label", "x")] = value


class FakeBreath:
    def __init__(self, start: float, end: float):
        self.start_time = start
        self.end_time = end
        self.middle_time = (start + end) / 2


class FakeIntervals:
    def __init__(self, pairs: list[tuple[float, float]]):
        self.intervals = pairs
        self.values = [FakeBreath(start, end) for start, end in pairs]


class FakeSignal:
    def __init__(self, label: str = "raw"):
        self.label = label

    def __getitem__(self, _slice: Any) -> "FakeSignal":
        return FakeSignal(self.label + "_sliced")

    def get_summed_impedance(self, *args: Any, **kwargs: Any) -> "FakeSignal":
        return FakeSignal("global_impedance_(filtered)")


class FakeSequence:
    def __init__(self) -> None:
        self.eit_data = FakeCollection(raw=FakeSignal("raw"))
        self.continuous_data = FakeCollection()


BREATH_PAIRS = [(0.0, 1.0), (1.0, 2.5), (2.5, 3.2), (3.2, 4.6)]


@pytest.fixture
def fake_eitprocessing(monkeypatch):
    """Patch the upstream classes the EIT steps import at call time."""

    import eitprocessing.features.breath_detection as bd
    import eitprocessing.features.rate_detection as rd
    import eitprocessing.filters.mdn as mdn

    class FakeRateDetection:
        def __init__(self, subject_type: str, **kwargs: Any) -> None:
            self.subject_type = subject_type

        def apply(self, signal: Any, **kwargs: Any) -> tuple[float, float]:
            return 0.25, 1.5

    class FakeMDNFilter:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def apply(self, signal: Any, label: str = "filtered", **kwargs: Any):
            return FakeSignal(label)

    class FakeBreathDetection:
        def __init__(self, minimum_duration: float = 0.0, **kwargs: Any) -> None:
            self.minimum_duration = minimum_duration

        def find_breaths(self, signal: Any, **kwargs: Any) -> FakeIntervals:
            return FakeIntervals(BREATH_PAIRS)

    monkeypatch.setattr(rd, "RateDetection", FakeRateDetection)
    monkeypatch.setattr(mdn, "MDNFilter", FakeMDNFilter)
    monkeypatch.setattr(bd, "BreathDetection", FakeBreathDetection)


def _fake_eit_session() -> M3Session:
    return M3Session(
        eit_adapter=EITProcessingAdapter(loader=lambda *a, **k: FakeSequence())
    )


def _expected_cv() -> tuple[float, float, float, int]:
    import numpy as np

    durations = np.asarray([end - start for start, end in BREATH_PAIRS], dtype=float)
    return (
        float(durations.std() / durations.mean()),
        float(durations.mean()),
        float(durations.std()),
        len(BREATH_PAIRS),
    )


# Inline ROTARC spec (selection="selected"): slice → detect_rates → mdn_filter →
# global_impedance → slice detection window → detect_breaths → normalize → cv.
_ROTARC_SPEC: dict[str, Any] = {
    "name": "rotarc-breath-duration",
    "steps": [
        {"uses": "eit.load", "with": {"file": "data/eit.bin", "vendor": "draeger"}},
        {
            "uses": "eit.slice",
            "in": {"signal": "raw_eit"},
            "with": {"start": 1, "end": 3, "mode": "index"},
            "out": {"result": "selected_eit"},
        },
        {
            "uses": "eit.detect_rates",
            "in": {"signal": "selected_eit"},
            "with": {"subject_type": "adult"},
        },
        {"uses": "eit.mdn_filter", "in": {"signal": "raw_eit"}},
        {"uses": "eit.global_impedance", "in": {"signal": "filtered_eit"}},
        {
            "uses": "eit.slice",
            "in": {"signal": "global_impedance"},
            "with": {"start": 1, "end": 3, "mode": "index"},
            "out": {"result": "detection_signal"},
        },
        {
            "uses": "eit.detect_breaths",
            "in": {"signal": "detection_signal"},
            "with": {"min_duration_s": 0.5},
        },
        {"uses": "eit.normalize_breaths"},
        {"uses": "metric.interval_cv", "in": {"intervals": "breath_intervals"}},
    ],
}


def test_rotarc_pipeline_runs_through_engine(fake_eitprocessing):
    result = run_pipeline(_ROTARC_SPEC, session=_fake_eit_session())

    expected_cv, expected_mean, expected_std, expected_n = _expected_cv()
    assert result.value("cv") == pytest.approx(expected_cv)
    assert result.value("mean") == pytest.approx(expected_mean)
    assert result.value("n") == expected_n
    assert len(result.session.events["eit_breaths"]) == expected_n


def test_rotarc_spec_validates():
    validate_spec(load_spec(_ROTARC_SPEC))


def test_rotarc_full_selection_spec_has_one_slice():
    """'full' selection: no second slice, detect_breaths reads global_impedance."""

    spec = {
        "name": "rotarc-breath-duration",
        "steps": [
            {"uses": "eit.load", "with": {"file": "data/eit.bin", "vendor": "draeger"}},
            {
                "uses": "eit.slice",
                "in": {"signal": "raw_eit"},
                "with": {"start": 1, "end": 3, "mode": "index"},
                "out": {"result": "selected_eit"},
            },
            {
                "uses": "eit.detect_rates",
                "in": {"signal": "selected_eit"},
                "with": {"subject_type": "adult"},
            },
            {"uses": "eit.mdn_filter", "in": {"signal": "raw_eit"}},
            {"uses": "eit.global_impedance", "in": {"signal": "filtered_eit"}},
            {
                "uses": "eit.detect_breaths",
                "in": {"signal": "global_impedance"},
                "with": {"min_duration_s": 0.5},
            },
            {"uses": "eit.normalize_breaths"},
            {"uses": "metric.interval_cv", "in": {"intervals": "breath_intervals"}},
        ],
    }
    slice_steps = [s for s in spec["steps"] if s["uses"] == "eit.slice"]
    assert len(slice_steps) == 1
    detect = next(s for s in spec["steps"] if s["uses"] == "eit.detect_breaths")
    assert detect["in"]["signal"] == "global_impedance"
    validate_spec(load_spec(spec))


# --------------------------------------------------------------------------- #
# export.rotarc_result step                                                   #
# --------------------------------------------------------------------------- #


def test_rotarc_result_step_writes_file_and_summary(tmp_path):
    from m3resp.pipeline.spec import SpecExperimentConfig, SpecOutputsConfig

    session = M3Session()
    outputs = SpecOutputsConfig(
        dir=tmp_path,
        timestamped=False,
        summary_json=False,
        event_csvs=False,
    )
    experiment = SpecExperimentConfig(
        subject_id="s01",
        mode="quiet",
        timepoint=None,
        run_identifier="run-1",
        selection="selected",
    )
    spec = {
        "name": "rotarc",
        "steps": [
            {
                "uses": "export.rotarc_result",
                "in": {"value": "cv"},
                "with": {"precision": 4},
            }
        ],
    }
    result = run_pipeline(
        spec,
        session=session,
        extra_context={
            "cv": 0.1234,
            "_spec_outputs": outputs,
            "_spec_experiment": experiment,
        },
    )
    result_path = Path(result.value("result_path"))
    assert result_path.exists()
    assert result_path.read_text(encoding="utf-8") == "0.1234"
    assert (tmp_path / "subject_results" / "run-1" / "rotarc_summary.json").exists()
