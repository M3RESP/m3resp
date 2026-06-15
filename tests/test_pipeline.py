"""Tests for the declarative pipeline engine and the ROTARC migration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from m3resp.adapters import EITProcessingAdapter
from m3resp.core.exceptions import PipelineSpecError, UnknownStepError
from m3resp.core.session import M3Session
from m3resp.pipeline import load_spec, register_step, run_pipeline, validate_spec
from m3resp.workflows.rotarc_breath_duration import (
    build_rotarc_spec,
    run_rotarc_breath_duration_workflow,
)
from m3resp import load_workflow_config


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
    # Built-in steps register on demand and are discoverable.
    assert {
        "eit.load",
        "eit.detect_breaths",
        "emg.preprocess",
        "metric.interval_cv",
    } <= (set(steps))


# --------------------------------------------------------------------------- #
# ROTARC pipeline equivalence (fake eitprocessing primitives)                 #
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
    """Mimics eitprocessing breath intervals: ``.intervals`` and ``.values``."""

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


def _write_rotarc_config(tmp_path: Path, selection: str = "selected") -> Path:
    config_path = Path(os.path.join(tmp_path, "config.yaml"))
    config_path.write_text(
        f"""
modules: {{eit: true, emg: false, vent: false}}
eit:
  file: {os.path.join("data", "eit.bin")}
  vendor: draeger
  processing:
    subject_type: adult
    breath_min_duration_seconds: 0.5
output:
  combined: {os.path.join("output", "rotarc")}
rotarc:
  subject_id: subj
  mode: quiet
  timepoint: t0
  start: 1
  end: 3
  slicing_mode: index
  selection: {selection}
  run_identifier: run-1
""",
        encoding="utf-8",
    )
    return config_path


def test_rotarc_pipeline_runs_through_engine(tmp_path, fake_eitprocessing):
    cfg = load_workflow_config(_write_rotarc_config(tmp_path), root=tmp_path)
    result = run_pipeline(build_rotarc_spec(cfg), session=_fake_eit_session())

    expected_cv, expected_mean, expected_std, expected_n = _expected_cv()
    assert result.value("cv") == pytest.approx(expected_cv)
    assert result.value("mean") == pytest.approx(expected_mean)
    assert result.value("n") == expected_n
    # Breath events were normalized onto the session for export.
    assert len(result.session.events["eit_breaths"]) == expected_n


def test_rotarc_workflow_end_to_end_matches_cv(tmp_path, fake_eitprocessing):
    cfg = load_workflow_config(_write_rotarc_config(tmp_path), root=tmp_path)
    fake_adapter = EITProcessingAdapter(loader=lambda *a, **k: FakeSequence())
    result = run_rotarc_breath_duration_workflow(cfg, eit_adapter=fake_adapter)

    expected_cv, expected_mean, expected_std, expected_n = _expected_cv()
    assert result.summary["breath_duration_cv"] == pytest.approx(expected_cv)
    assert result.summary["mean_breath_duration_seconds"] == pytest.approx(
        expected_mean
    )
    assert result.summary["std_breath_duration_seconds"] == pytest.approx(expected_std)
    assert result.summary["n_breaths"] == expected_n
    assert result.summary["respiratory_rate_hz"] == pytest.approx(0.25)
    assert result.summary["heart_rate_hz"] == pytest.approx(1.5)

    result_file = Path(result.summary["result_path"])
    assert result_file.read_text(encoding="utf-8") == f"{expected_cv:.8f}"
    assert os.path.exists(os.path.join(result.output_dir, "rotarc_summary.json"))


# --------------------------------------------------------------------------- #
# Multimodal pipeline parity tests                                            #
# --------------------------------------------------------------------------- #


from m3resp.adapters import ReSurfEMGAdapter  # noqa: E402
from m3resp.pipeline.compile_config import (  # noqa: E402
    build_multimodal_spec,
    build_eit_processing_plan,
)
from m3resp.workflows.configured.steps import assemble_eit_processed  # noqa: E402
from m3resp.workflows.configured.runner import run_configured_workflow  # noqa: E402
from m3resp.workflows.configured.summaries import summarize_multimodal  # noqa: E402


# ── shared fake for multimodal ────────────────────────────────────────────────


class FakeEITDataMM:
    """EIT data fake with the attrs the granular steps need."""

    label = "raw"
    sample_frequency = 20.0
    pixel_impedance = [[[1.0]], [[2.0]], [[3.0]]]

    def get_summed_impedance(self, *a: Any, **kw: Any) -> "FakeSignalMM":
        return FakeSignalMM("global_impedance_(mdn_filtered)")


class FakeSignalMM:
    def __init__(self, label: str = "raw") -> None:
        self.label = label

    def __getitem__(self, _: Any) -> "FakeSignalMM":
        return FakeSignalMM(self.label + "_sliced")

    def get_summed_impedance(self, *a: Any, **kw: Any) -> "FakeSignalMM":
        return FakeSignalMM("global_impedance_(mdn_filtered)")


class FakeSparseData:
    label = "sparse"
    time = [1.0]
    values = [2.0]

    def __len__(self) -> int:
        return 1


class FakePixelTIVMM:
    label = "pixel"
    values: list[Any] = []


class FakeBreathEventsMM:
    start_time = 1.0
    end_time = 2.0
    middle_time = 1.5


class FakeIntervalsMM:
    def __init__(self) -> None:
        self.intervals = [(1.0, 2.0), (2.0, 3.2)]
        self.values = [FakeBreathEventsMM(), FakeBreathEventsMM()]


class FakeCollectionMM(dict):
    def add(self, value: Any, overwrite: bool = False) -> None:
        self[getattr(value, "label", "x")] = value


class FakeSequenceMM:
    def __init__(self) -> None:
        self.eit_data = FakeCollectionMM(raw=FakeEITDataMM())
        self.continuous_data = FakeCollectionMM()
        self.sparse_data = FakeCollectionMM()
        self.interval_data = FakeCollectionMM()


@pytest.fixture
def fake_eit_upstream_mm(monkeypatch):
    """Patch all eitprocessing upstream classes used in a full EIT sub-pipeline."""

    import eitprocessing.features.breath_detection as bd
    import eitprocessing.features.rate_detection as rd
    import eitprocessing.filters.butterworth_filters as butter
    import eitprocessing.filters.mdn as mdn
    import eitprocessing.parameters.eeli as eeli_mod
    import eitprocessing.parameters.tidal_impedance_variation as tiv_mod

    class _RD:
        def __init__(self, *a: Any, **kw: Any) -> None: ...
        def apply(self, *a: Any, **kw: Any) -> tuple[float, float]:
            return 0.25, 1.0

    class _MDN:
        def __init__(self, **kw: Any) -> None: ...
        def apply(self, s: Any, label: str = "filtered", **kw: Any) -> FakeSignalMM:
            return FakeSignalMM(label)

    class _BW:
        def __init__(self, **kw: Any) -> None: ...
        def apply(self, px: Any, **kw: Any) -> Any:
            return px

    class _BD:
        def __init__(self, **kw: Any) -> None: ...
        def find_breaths(self, *a: Any, **kw: Any) -> FakeIntervalsMM:
            return FakeIntervalsMM()

    class _TIV:
        def __init__(self, **kw: Any) -> None: ...
        def compute_parameter(self, *a: Any, **kw: Any) -> Any:
            return FakePixelTIVMM() if kw.get("tiv_timing") else FakeSparseData()

    class _EELI:
        def __init__(self, **kw: Any) -> None: ...
        def compute_parameter(self, *a: Any, **kw: Any) -> FakeSparseData:
            return FakeSparseData()

    monkeypatch.setattr(rd, "RateDetection", _RD)
    monkeypatch.setattr(mdn, "MDNFilter", _MDN)
    monkeypatch.setattr(butter, "ButterworthFilter", _BW)
    monkeypatch.setattr(bd, "BreathDetection", _BD)
    monkeypatch.setattr(tiv_mod, "TIV", _TIV)
    monkeypatch.setattr(eeli_mod, "EELI", _EELI)


def _make_fake_emg_adapter() -> ReSurfEMGAdapter:
    from m3resp import BreathEvent

    class _FakeEMGAdapter(ReSurfEMGAdapter):
        def __init__(self) -> None:
            super().__init__(
                loader=lambda *a, **kw: {
                    "array": [[0.0, 1.0, 0.0]],
                    "dataframe": {},
                    "metadata": {"fs": 1000.0, "labels": ["EMG"], "units": ["uV"]},
                }
            )

        def preprocess(self, signal: Any, **kw: Any) -> dict[str, Any]:
            return {
                **signal,
                "channel": kw.get("channel", 0),
                "fs": 1000.0,
                "raw_channel": [0.0, 1.0, 0.0],
                "filtered": [0.0, 0.5, 0.0],
                "envelope": [0.0, 1.0, 0.0],
                "filter": {"high_pass_hz": kw.get("high_pass_hz", 80)},
            }

        def detect_breaths(self, signal: Any, **kw: Any) -> list[BreathEvent]:
            return [BreathEvent("emg", 0.0, 1.0, peak_time=0.5)]

        def postprocess(
            self, proc: Any, events: Any = None, **kw: Any
        ) -> dict[str, Any]:
            vent = kw.get("ventilator")
            vent_breaths = (
                [BreathEvent("vent", 0.0, 1.0, peak_time=0.5)] if vent else []
            )
            return {
                "available": {"event_detection": ["detect_ventilator_breath"]},
                "computed": {
                    "event_detection": {"detect_ventilator_breath": vent_breaths}
                },
                "skipped": {},
            }

    return _FakeEMGAdapter()


def _write_multimodal_config(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        f"""
modules: {{eit: true, emg: true, vent: true}}
eit:
  file: {os.path.join("data", "eit.bin")}
  vendor: draeger
  processing:
    subject_type: adult
    breath_min_duration_seconds: 0.5
    filter: {{enabled: true, mode: mdn}}
    outputs: {{rates: true, breath_intervals: true, continuous_tiv: true, eeli: true, pixel_tiv: true}}
emg:
  file: {os.path.join("data", "emg.Poly5")}
  processing:
    preprocess: {{enabled: true}}
    breath_detection: {{enabled: true}}
    postprocessing: {{enabled: true}}
vent:
  file: {os.path.join("data", "vent.Poly5")}
alignment:
  method: manual_offset
  manual_offset_seconds: 0.0
  reference_modality: vent
output:
  combined: {os.path.join("output", "multimodal")}
results:
  summary_json: true
  event_csvs: true
  parameters_csv: false
  postprocessing: false
  figures: false
""",
        encoding="utf-8",
    )
    return p


def test_multimodal_spec_validates(tmp_path):
    """build_multimodal_spec generates a spec that passes static validation."""

    cfg = load_workflow_config(_write_multimodal_config(tmp_path), root=tmp_path)
    spec = build_multimodal_spec(cfg)
    validate_spec(load_spec(spec))
    uses = [s["uses"] for s in spec["steps"]]
    assert "eit.load" in uses
    assert "emg.load" in uses
    assert "session.sync_raw" in uses
    assert "eit.mdn_filter" in uses
    assert "emg.preprocess" in uses


def test_multimodal_pipeline_produces_same_session_state_as_configured_runner(
    tmp_path, fake_eit_upstream_mm
):
    """Pipeline spec and configured runner must produce equivalent session state."""

    cfg = load_workflow_config(_write_multimodal_config(tmp_path), root=tmp_path)
    eit_loader = lambda *a, **kw: FakeSequenceMM()  # noqa: E731
    emg_adapter = _make_fake_emg_adapter()

    # ── Old path: configured runner ────────────────────────────────────────
    old_result = run_configured_workflow(
        cfg,
        eit_adapter=EITProcessingAdapter(loader=eit_loader),
        emg_adapter=_make_fake_emg_adapter(),
        export=False,
        save_figures=False,
    )
    old_summary = old_result.summary
    old_session = old_result.session

    # ── New path: build_multimodal_spec + run_pipeline ─────────────────────
    from m3resp.core.session import M3Session

    new_session = M3Session(
        eit_adapter=EITProcessingAdapter(loader=eit_loader),
        emg_adapter=emg_adapter,
    )
    spec = build_multimodal_spec(cfg)
    new_result = run_pipeline(spec, session=new_session)

    # Assemble session.processed["eit"] so summarize_eit works.
    plan = build_eit_processing_plan(cfg)
    seed = {
        "eit_sequence": new_result.value("eit_sequence"),
        "raw_eit": new_result.value("raw_eit"),
        "raw_global_impedance": new_result.value("raw_global_impedance"),
    }
    new_session.processed["eit"] = assemble_eit_processed(
        plan, seed, new_result.context.values
    )

    new_summary = summarize_multimodal(new_session, include_eit=True, include_emg=True)

    # ── Compare key outputs ────────────────────────────────────────────────
    # Breath counts must match exactly.
    assert new_summary["n_eit_breaths"] == old_summary["n_eit_breaths"]
    assert new_summary["n_emg_breaths"] == old_summary["n_emg_breaths"]
    assert new_summary["n_ventilator_breaths"] == old_summary["n_ventilator_breaths"]

    # EIT rates match.
    assert new_summary.get("respiratory_rate_bpm") == pytest.approx(
        old_summary.get("respiratory_rate_bpm")
    )

    # Both sessions have the same raw synchronization reference frame.
    assert (
        new_session.parameters["raw_alignment"]["reference_modality"]
        == old_session.parameters["raw_alignment"]["reference_modality"]
    )
    assert (
        new_session.parameters["raw_alignment"]["offset_seconds"]
        == old_session.parameters["raw_alignment"]["offset_seconds"]
    )

    # EIT output dict shape is identical.
    old_eit = old_session.processed["eit"]
    new_eit = new_session.processed["eit"]
    assert new_eit["filter_mode"] == old_eit["filter_mode"]
    assert (new_eit["respiratory_rate_hz"] is None) == (
        old_eit["respiratory_rate_hz"] is None
    )
    assert (new_eit["breath_intervals"] is not None) == (
        old_eit["breath_intervals"] is not None
    )
    assert (new_eit["continuous_tiv"] is not None) == (
        old_eit["continuous_tiv"] is not None
    )
    assert (new_eit["eeli"] is not None) == (old_eit["eeli"] is not None)


def test_rotarc_full_selection_skips_second_slice(tmp_path):
    cfg = load_workflow_config(
        _write_rotarc_config(tmp_path, selection="full"), root=tmp_path
    )
    spec = build_rotarc_spec(cfg)
    slice_steps = [s for s in spec["steps"] if s["uses"] == "eit.slice"]
    assert len(slice_steps) == 1
    detect = next(s for s in spec["steps"] if s["uses"] == "eit.detect_breaths")
    assert detect["in"]["signal"] == "global_impedance"
    validate_spec(load_spec(spec))
