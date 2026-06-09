from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from m3resp import BreathEvent, M3Session, load_workflow_config
from m3resp.adapters import EITProcessingAdapter, ReSurfEMGAdapter
from m3resp.workflows import auto as auto_workflows
from m3resp.workflows import (
    WorkflowResult,
    run_eit_workflow,
    run_emg_workflow,
    run_multimodal_workflow,
    run_workflow,
    select_workflow,
)


class FakeCollection(dict):
    def add(self, value: Any, overwrite: bool = False) -> None:
        if not overwrite and value.label in self:
            raise KeyError(value.label)
        self[value.label] = value


class FakeEITData:
    label = "raw"

    def get_summed_impedance(self, return_label: str | None = None, **kwargs: Any):
        return FakeContinuousData(return_label or "global_impedance_(raw)")


class FakeContinuousData:
    def __init__(self, label: str):
        self.label = label
        self.time = [0.0, 1.0, 2.0]
        self.values = [1.0, 2.0, 3.0]


class FakeSequence:
    def __init__(self):
        self.eit_data = FakeCollection(raw=FakeEITData())
        self.continuous_data = FakeCollection()


class FakeBreath:
    start_time = 1.0
    middle_time = 1.5
    end_time = 2.0


class FakeIntervals:
    values = [FakeBreath()]


class FakeSparse:
    label = "sparse"
    time = [1.0]
    values = [2.0]

    def __len__(self) -> int:
        return len(self.values)


class FakePixelTIV:
    values: list[Any] = []


class FakeEITAdapter(EITProcessingAdapter):
    def __init__(self):
        super().__init__(loader=lambda *args, **kwargs: FakeSequence())
        self.preprocess_kwargs: dict[str, Any] = {}

    def preprocess(self, sequence: Any, **kwargs: Any) -> dict[str, Any]:
        self.preprocess_kwargs = kwargs
        return {
            "sequence": sequence,
            "filter_mode": kwargs.get("filter_mode", "none"),
            "respiratory_rate_hz": (
                0.25 if kwargs.get("compute_rates", True) else None
            ),
            "heart_rate_hz": 1.0 if kwargs.get("compute_rates", True) else None,
            "breath_intervals": (
                FakeIntervals()
                if kwargs.get("compute_breath_intervals", True)
                else None
            ),
            "continuous_tiv": (
                FakeSparse() if kwargs.get("compute_continuous_tiv", True) else None
            ),
            "eeli": FakeSparse() if kwargs.get("compute_eeli", True) else None,
            "pixel_tiv": (
                FakePixelTIV() if kwargs.get("compute_pixel_tiv", True) else None
            ),
        }

    def detect_breaths(self, data: Any, **kwargs: Any) -> list[BreathEvent]:
        if data.get("breath_intervals") is None:
            return []
        return super().detect_breaths(data, **kwargs)


class FakeEMGAdapter(ReSurfEMGAdapter):
    def __init__(self):
        super().__init__(loader=lambda *args, **kwargs: fake_emg_recording())
        self.preprocess_kwargs: dict[str, Any] = {}
        self.detection_kwargs: dict[str, Any] = {}
        self.postprocess_kwargs: dict[str, Any] = {}

    def preprocess(self, signal: Any, **kwargs: Any) -> dict[str, Any]:
        self.preprocess_kwargs = kwargs
        return {
            **signal,
            "channel": kwargs.get("channel", 0),
            "fs": 1000.0,
            "raw_channel": [0.0, 1.0, 0.0],
            "filtered": [0.0, 0.5, 0.0],
            "envelope": [0.0, 1.0, 0.0],
            "filter": {"high_pass_hz": kwargs.get("high_pass_hz", 80)},
        }

    def detect_breaths(self, signal: Any, **kwargs: Any) -> list[BreathEvent]:
        self.detection_kwargs = kwargs
        return [BreathEvent("emg", 0.0, 1.0, peak_time=0.5)]

    def postprocess(
        self,
        processed_emg: Any,
        events: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.postprocess_kwargs = kwargs
        ventilator_breaths = [BreathEvent("vent", 0.0, 1.0, peak_time=0.5)]
        if kwargs.get("ventilator") is None:
            ventilator_breaths = []
        return {
            "available": {"event_detection": ["detect_ventilator_breath"]},
            "computed": {
                "event_detection": {
                    "detect_ventilator_breath": ventilator_breaths,
                }
            },
            "skipped": {},
        }


def fake_emg_recording() -> dict[str, Any]:
    return {
        "array": [[0.0, 1.0, 0.0]],
        "dataframe": {"kind": "fake"},
        "metadata": {"fs": 1000.0, "labels": ["EMG"], "units": ["uV"]},
    }


def write_config(
    tmp_path: Path,
    *,
    eit: bool = True,
    emg: bool = True,
    vent: bool = True,
    extra: str = "",
) -> Path:
    config_path = Path(os.path.join(tmp_path, "config.yaml"))
    eit_file = os.path.join("data", "eit.bin")
    emg_file = os.path.join("data", "emg.Poly5")
    vent_file = os.path.join("data", "vent.Poly5")
    combined_output = os.path.join("output", "combined")
    eit_output = os.path.join("output", "eit")
    emg_output = os.path.join("output", "emg")
    config_path.write_text(
        f"""
modules:
  eit: {str(eit).lower()}
  emg: {str(emg).lower()}
  vent: {str(vent).lower()}
eit:
  file: {eit_file}
  vendor: sentec
emg:
  file: {emg_file}
vent:
  file: {vent_file}
alignment:
  method: manual_offset
  manual_offset_seconds: 0.25
output:
  combined: {combined_output}
  eit_only: {eit_output}
  emg_only: {emg_output}
{extra}
""",
        encoding="utf-8",
    )
    return config_path


def test_load_workflow_config_resolves_paths_against_root(tmp_path):
    config_path = write_config(tmp_path)

    cfg = load_workflow_config(config_path, root=tmp_path)

    assert cfg.eit.file == Path(os.path.join(tmp_path, "data", "eit.bin"))
    assert cfg.output.combined == Path(os.path.join(tmp_path, "output", "combined"))
    assert cfg.eit.processing.filter.mode == "mdn"
    assert cfg.emg.processing.preprocess.channel == 0
    assert cfg.results.summary_json is True


def test_load_workflow_config_reads_processing_and_result_switches(tmp_path):
    config_path = write_config(
        tmp_path,
        extra=f"""
eit:
  file: {os.path.join("data", "eit.bin")}
  vendor: sentec
  processing:
    filter:
      enabled: true
      mode: bandpass
      lowpass_hz: 0.8
      highpass_hz: 0.1
      order: 2
    outputs:
      pixel_tiv: false
emg:
  file: {os.path.join("data", "emg.Poly5")}
  processing:
    preprocess:
      channel: 2
      high_pass_hz: 90
    breath_detection:
      min_breath_width_seconds: 1.5
    postprocessing:
      functions:
        features:
          amplitude: false
results:
  figures: false
""",
    )

    cfg = load_workflow_config(config_path, root=tmp_path)

    assert cfg.eit.processing.filter.mode == "bandpass"
    assert cfg.eit.processing.filter.order == 2
    assert cfg.eit.processing.outputs.pixel_tiv is False
    assert cfg.emg.processing.preprocess.channel == 2
    assert cfg.emg.processing.breath_detection.min_breath_width_seconds == 1.5
    assert cfg.emg.processing.postprocessing.functions.features["amplitude"] is False
    assert cfg.results.figures is False


def test_configured_eit_workflow_exports_summary(tmp_path):
    adapter = FakeEITAdapter()
    result = run_eit_workflow(
        config=write_config(tmp_path, emg=False, vent=False),
        root=tmp_path,
        eit_adapter=adapter,
        save_figures=False,
    )

    assert isinstance(result, WorkflowResult)
    assert result.summary["n_eit_breaths"] == 1
    assert adapter.preprocess_kwargs["filter_mode"] == "mdn"
    assert os.path.exists(os.path.join(result.output_dir, "summary.json"))
    assert os.path.exists(os.path.join(result.output_dir, "eit_breaths.csv"))


def test_configured_eit_workflow_respects_processing_switches(tmp_path):
    adapter = FakeEITAdapter()
    result = run_eit_workflow(
        config=write_config(
            tmp_path,
            emg=False,
            vent=False,
            extra=f"""
eit:
  file: {os.path.join("data", "eit.bin")}
  vendor: sentec
  processing:
    filter:
      enabled: true
      mode: lowpass
      lowpass_hz: 0.9
    outputs:
      rates: false
      continuous_tiv: false
      eeli: false
      pixel_tiv: false
""",
        ),
        root=tmp_path,
        eit_adapter=adapter,
        save_figures=False,
    )

    assert adapter.preprocess_kwargs["filter_mode"] == "lowpass"
    assert adapter.preprocess_kwargs["lowpass_hz"] == 0.9
    assert adapter.preprocess_kwargs["compute_rates"] is False
    assert adapter.preprocess_kwargs["compute_pixel_tiv"] is False
    assert "respiratory_rate_bpm" not in result.summary
    assert "n_continuous_tiv_values" not in result.summary


def test_configured_emg_workflow_handles_ventilator_toggle(tmp_path):
    adapter = FakeEMGAdapter()
    with_vent = run_emg_workflow(
        config=write_config(tmp_path, eit=False, vent=True),
        root=tmp_path,
        emg_adapter=adapter,
        save_figures=False,
    )
    without_vent = run_emg_workflow(
        config=write_config(tmp_path, eit=False, vent=False),
        root=tmp_path,
        emg_adapter=FakeEMGAdapter(),
        save_figures=False,
    )

    assert with_vent.summary["n_ventilator_breaths"] == 1
    assert without_vent.summary["n_ventilator_breaths"] == 0
    assert adapter.preprocess_kwargs["channel"] == 0


def test_configured_emg_workflow_respects_processing_switches(tmp_path):
    adapter = FakeEMGAdapter()
    run_emg_workflow(
        config=write_config(
            tmp_path,
            eit=False,
            vent=False,
            extra=f"""
emg:
  file: {os.path.join("data", "emg.Poly5")}
  processing:
    preprocess:
      channel: 3
      high_pass_hz: 95
    breath_detection:
      min_breath_width_seconds: 2.0
      half_window_seconds: 0.25
    postprocessing:
      functions:
        baseline:
          moving_baseline: false
        features:
          amplitude: false
""",
        ),
        root=tmp_path,
        emg_adapter=adapter,
        save_figures=False,
    )

    assert adapter.preprocess_kwargs["channel"] == 3
    assert adapter.preprocess_kwargs["high_pass_hz"] == 95
    assert adapter.detection_kwargs["min_breath_width_seconds"] == 2.0
    assert adapter.detection_kwargs["half_window_seconds"] == 0.25
    selected = adapter.postprocess_kwargs["selected_functions"]
    assert selected["baseline"]["moving_baseline"] is False
    assert selected["features"]["amplitude"] is False


def test_configured_multimodal_workflow_aligns_and_exports(tmp_path):
    result = run_multimodal_workflow(
        config=write_config(tmp_path),
        root=tmp_path,
        eit_adapter=FakeEITAdapter(),
        emg_adapter=FakeEMGAdapter(),
        save_figures=False,
    )

    assert isinstance(result, WorkflowResult)
    assert "synchronized" in result.session.processed
    assert result.session.processed["synchronized"]["eit_breaths"][0].start_time == 1.25
    assert result.summary["n_eit_breaths"] == 1
    assert result.summary["n_emg_breaths"] == 1
    assert result.output_dir == Path(os.path.join(tmp_path, "output", "combined"))


def test_configured_multimodal_workflow_respects_module_toggles(tmp_path):
    result = run_multimodal_workflow(
        config=write_config(tmp_path, eit=False, emg=True, vent=False),
        root=tmp_path,
        eit_adapter=FakeEITAdapter(),
        emg_adapter=FakeEMGAdapter(),
        save_figures=False,
    )

    assert "eit" not in result.session.raw
    assert "emg" in result.session.raw
    assert "synchronized" not in result.session.processed
    assert "n_eit_breaths" not in result.summary
    assert result.summary["n_emg_breaths"] == 1


def test_configured_workflow_respects_export_result_switches(tmp_path):
    result = run_emg_workflow(
        config=write_config(
            tmp_path,
            eit=False,
            vent=False,
            extra="""
results:
  summary_json: false
  parameters_csv: false
  event_csvs: false
  postprocessing: false
  figures: false
""",
        ),
        root=tmp_path,
        emg_adapter=FakeEMGAdapter(),
        save_figures=True,
    )

    assert not os.path.exists(os.path.join(result.output_dir, "summary.json"))
    assert not os.path.exists(os.path.join(result.output_dir, "parameters.csv"))
    assert not os.path.exists(os.path.join(result.output_dir, "emg_breaths.csv"))
    assert result.figures == {}


def test_configured_emg_postprocessing_rejects_missing_dependencies(tmp_path):
    config_path = write_config(
        tmp_path,
        eit=False,
        vent=False,
        extra=f"""
emg:
  file: {os.path.join("data", "emg.Poly5")}
  processing:
    postprocessing:
      functions:
        baseline:
          moving_baseline: false
          slopesum_baseline: false
""",
    )

    try:
        run_emg_workflow(
            config=config_path,
            root=tmp_path,
            emg_adapter=FakeEMGAdapter(),
            save_figures=False,
        )
    except ValueError as exc:
        assert "require at least one baseline" in str(exc)
    else:
        raise AssertionError("Expected invalid EMG postprocessing config to fail.")


def test_auto_workflow_selects_multimodal_when_eit_and_emg_enabled(tmp_path):
    config_path = write_config(tmp_path, eit=True, emg=True, vent=True)

    result = run_workflow(
        config_path,
        root=tmp_path,
        eit_adapter=FakeEITAdapter(),
        emg_adapter=FakeEMGAdapter(),
        save_figures=False,
    )

    assert select_workflow(config_path, root=tmp_path) == "multimodal"
    assert result.output_dir == Path(os.path.join(tmp_path, "output", "combined"))
    assert "synchronized" in result.session.processed


def test_auto_run_delegates_to_config_path(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    sentinel = object()
    calls: dict[str, Any] = {}

    def fake_run_workflow(*, config: Path) -> object:
        calls["config"] = config
        return sentinel

    monkeypatch.setattr(auto_workflows, "CONFIG_PATH", config_path)
    monkeypatch.setattr(auto_workflows, "run_workflow", fake_run_workflow)

    result = auto_workflows.run()

    assert result is sentinel
    assert calls["config"] == config_path


def test_auto_workflow_selects_eit_when_only_eit_primary_enabled(tmp_path):
    config_path = write_config(tmp_path, eit=True, emg=False, vent=True)

    result = run_workflow(
        config_path,
        root=tmp_path,
        eit_adapter=FakeEITAdapter(),
        emg_adapter=FakeEMGAdapter(),
        save_figures=False,
    )

    assert select_workflow(config_path, root=tmp_path) == "eit"
    assert result.output_dir == Path(os.path.join(tmp_path, "output", "eit"))
    assert "eit" in result.session.raw
    assert "emg" not in result.session.raw


def test_auto_workflow_selects_emg_when_only_emg_primary_enabled(tmp_path):
    config_path = write_config(tmp_path, eit=False, emg=True, vent=True)

    result = run_workflow(
        config_path,
        root=tmp_path,
        eit_adapter=FakeEITAdapter(),
        emg_adapter=FakeEMGAdapter(),
        save_figures=False,
    )

    assert select_workflow(config_path, root=tmp_path) == "emg"
    assert result.output_dir == Path(os.path.join(tmp_path, "output", "emg"))
    assert "eit" not in result.session.raw
    assert "emg" in result.session.raw
    assert result.summary["n_ventilator_breaths"] == 1


def test_auto_workflow_rejects_ventilator_without_primary_modality(tmp_path):
    config_path = write_config(tmp_path, eit=False, emg=False, vent=True)

    try:
        select_workflow(config_path, root=tmp_path)
    except ValueError as exc:
        assert "modules.eit or modules.emg" in str(exc)
    else:
        raise AssertionError("select_workflow should reject vent-only configs")


def test_direct_workflow_calls_still_return_sessions():
    eit_session = run_eit_workflow("subject.eit", eit_adapter=FakeEITAdapter())
    emg_session = run_emg_workflow("subject.Poly5", emg_adapter=FakeEMGAdapter())
    multimodal_session = run_multimodal_workflow(
        "subject.eit",
        "subject.Poly5",
        "sentec",
        eit_adapter=FakeEITAdapter(),
        emg_adapter=FakeEMGAdapter(),
    )

    assert isinstance(eit_session, M3Session)
    assert isinstance(emg_session, M3Session)
    assert isinstance(multimodal_session, M3Session)
