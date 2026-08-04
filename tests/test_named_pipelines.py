"""Milestone 2.4 - named `Pipeline` presets (plan_stage2.md Sec 18-19).

Each `Pipeline.run` is just a fixed sequence of calls to `M3Session`'s own
methods (already covered by `tests/test_session.py`, `tests/test_eit.py`,
`tests/test_emg.py`), so these tests verify the *contract* - which methods
get called, in what order, with the `config` kwargs passed through - by
spying on the session's methods rather than re-deriving domain correctness.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest

from m3resp import M3Session
from m3resp.core.exceptions import UnknownPipelineError
from m3resp.presets import (
    EITPipeline,
    EMGPipeline,
    MultimodalPipeline,
    available_pipelines,
    get_pipeline,
)
from m3resp.workflows import register_step
from m3resp.workflows.registry import STEP_REGISTRY


def _spy(calls: list[tuple[str, dict[str, Any]]], name: str, result: Any = None):
    def _record(**kwargs: Any) -> Any:
        calls.append((name, kwargs))
        return result

    return _record


def _step_spy(calls: list[tuple[str, dict[str, Any]]], name: str, result: Any = None):
    """Spy for a registered step, which takes positional arguments."""

    def _record(*args: Any, **kwargs: Any) -> Any:
        calls.append((name, kwargs))
        return result

    return _record


def _patch_ecg_removal_steps(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, dict[str, Any]]],
) -> None:
    """Spy on the two ECG-removal steps `EMGPipeline` calls.

    `EMGPipeline._remove_ecg` imports them lazily from their own modules, so
    patching the module attributes is what the pipeline actually looks up.
    """

    # `import_module`, not `from ... import ecg_gating`: the package's
    # `__init__` re-exports the step *function* under the same name as its
    # module, which would shadow the module object here.
    detection_module = import_module("m3resp.workflows.steps.emg.ecg_detection")
    gating_module = import_module("m3resp.workflows.steps.emg.ecg_gating")

    monkeypatch.setattr(
        detection_module,
        "ecg_detect_peaks",
        _step_spy(calls, "ecg_detect_peaks", result={"ecg_peak_indices": []}),
    )
    monkeypatch.setattr(
        gating_module, "ecg_gating", _step_spy(calls, "ecg_gating", result={})
    )


class TestPipelineRegistry:
    def test_built_in_pipelines_are_registered(self):
        assert available_pipelines() == ["eit", "emg", "multimodal"]
        assert get_pipeline("eit") is EITPipeline
        assert get_pipeline("emg") is EMGPipeline
        assert get_pipeline("multimodal") is MultimodalPipeline

    def test_get_pipeline_raises_for_unknown_name(self):
        with pytest.raises(UnknownPipelineError, match="unknown_pipeline"):
            get_pipeline("unknown_pipeline")


class TestEITPipeline:
    def test_run_calls_preprocess_then_detect_breaths_with_config(self):
        session = M3Session()
        calls: list[tuple[str, dict[str, Any]]] = []
        session.preprocess_eit = _spy(calls, "preprocess_eit")
        session.detect_eit_breaths = _spy(calls, "detect_eit_breaths")

        result = session.run_pipeline(
            "eit",
            config={
                "preprocess": {"filter_mode": "none"},
                "detect_breaths": {"breath_min_duration_seconds": 1.0},
            },
        )

        assert result is session
        assert calls == [
            ("preprocess_eit", {"filter_mode": "none"}),
            ("detect_eit_breaths", {"breath_min_duration_seconds": 1.0}),
        ]

    def test_run_with_no_config_calls_methods_with_no_kwargs(self):
        session = M3Session()
        calls: list[tuple[str, dict[str, Any]]] = []
        session.preprocess_eit = _spy(calls, "preprocess_eit")
        session.detect_eit_breaths = _spy(calls, "detect_eit_breaths")

        session.run_pipeline("eit")

        assert calls == [("preprocess_eit", {}), ("detect_eit_breaths", {})]


class TestEMGPipeline:
    def _session_with_spies(self, calls: list[tuple[str, dict[str, Any]]]) -> M3Session:
        session = M3Session()
        session.preprocess_emg = _spy(
            calls, "preprocess_emg", result={"fs": 1000.0, "channel": 0}
        )
        session.detect_emg_breaths = _spy(calls, "detect_emg_breaths")
        session.postprocess_emg = _spy(calls, "postprocess_emg")
        return session

    def test_ecg_removal_runs_between_preprocess_and_breath_detection(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The standard EMG chain gates ECG *before* the envelope feeds
        breath detection, so the two ECG steps must sit between
        `preprocess_emg` and `detect_emg_breaths`."""

        calls: list[tuple[str, dict[str, Any]]] = []
        session = self._session_with_spies(calls)
        _patch_ecg_removal_steps(monkeypatch, calls)

        session.run_pipeline("emg", config={"postprocess": {"peep": 5.0}})

        assert [name for name, _ in calls] == [
            "preprocess_emg",
            "ecg_detect_peaks",
            "ecg_gating",
            "detect_emg_breaths",
            "postprocess_emg",
        ]
        assert calls[-1] == ("postprocess_emg", {"peep": 5.0})

    def test_ecg_step_kwargs_are_passed_through(self, monkeypatch: pytest.MonkeyPatch):
        calls: list[tuple[str, dict[str, Any]]] = []
        session = self._session_with_spies(calls)
        _patch_ecg_removal_steps(monkeypatch, calls)

        session.run_pipeline(
            "emg",
            config={
                "ecg_detect_peaks": {"ecg_channel": 0},
                "ecg_gating": {"fill_method": 1},
            },
        )

        assert ("ecg_detect_peaks", {"ecg_channel": 0}) in calls
        assert ("ecg_gating", {"fill_method": 1}) in calls

    def test_ecg_removal_can_be_disabled(self, monkeypatch: pytest.MonkeyPatch):
        calls: list[tuple[str, dict[str, Any]]] = []
        session = self._session_with_spies(calls)
        _patch_ecg_removal_steps(monkeypatch, calls)

        session.run_pipeline("emg", config={"ecg_removal": {"enabled": False}})

        assert [name for name, _ in calls] == [
            "preprocess_emg",
            "detect_emg_breaths",
            "postprocess_emg",
        ]

    def test_supplied_ecg_peaks_skip_detection_and_are_gated(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """ "ECG peaks detection (if ECG peaks aren't already provided)" - peaks
        established elsewhere (separate ECG recording, annotations, an earlier
        run) go straight to gating."""

        calls: list[tuple[str, dict[str, Any]]] = []
        session = self._session_with_spies(calls)
        gated: list[Any] = []

        gating_module = import_module("m3resp.workflows.steps.emg.ecg_gating")
        detection_module = import_module("m3resp.workflows.steps.emg.ecg_detection")
        monkeypatch.setattr(
            detection_module,
            "ecg_detect_peaks",
            _step_spy(calls, "ecg_detect_peaks", result={"ecg_peak_indices": []}),
        )

        def _capture_gating(_session: Any, _processed: Any, peaks: Any, **kwargs: Any):
            calls.append(("ecg_gating", kwargs))
            gated.append(peaks)
            return {}

        monkeypatch.setattr(gating_module, "ecg_gating", _capture_gating)

        session.run_pipeline(
            "emg", config={"ecg_removal": {"ecg_peak_indices": [10, 20, 30]}}
        )

        assert [name for name, _ in calls] == [
            "preprocess_emg",
            "ecg_gating",
            "detect_emg_breaths",
            "postprocess_emg",
        ]
        assert gated == [[10, 20, 30]]

    def test_supplied_peaks_together_with_detection_kwargs_is_rejected(self):
        """Detection kwargs alongside supplied peaks would silently configure a
        pass that never runs."""

        calls: list[tuple[str, dict[str, Any]]] = []
        session = self._session_with_spies(calls)

        with pytest.raises(TypeError, match="would have no effect"):
            session.run_pipeline(
                "emg",
                config={
                    "ecg_removal": {"ecg_peak_indices": [10]},
                    "ecg_detect_peaks": {"ecg_channel": 0},
                },
            )

    def test_unknown_ecg_removal_option_is_rejected(self):
        calls: list[tuple[str, dict[str, Any]]] = []
        session = self._session_with_spies(calls)

        with pytest.raises(TypeError, match="only accepts 'enabled' and"):
            session.run_pipeline("emg", config={"ecg_removal": {"fill_method": 1}})


class TestMultimodalPipeline:
    def test_run_calls_synchronize_then_align_in_order(self):
        session = M3Session()
        calls: list[tuple[str, dict[str, Any]]] = []
        session.synchronize_raw_modalities = _spy(calls, "synchronize_raw_modalities")
        session.synchronize_multimodal_breaths = _spy(
            calls, "synchronize_multimodal_breaths"
        )

        session.run_pipeline(
            "multimodal",
            config={"align": {"offset_seconds": 0.5}},
        )

        assert [name for name, _ in calls] == [
            "synchronize_raw_modalities",
            "synchronize_multimodal_breaths",
        ]
        assert calls[-1] == ("synchronize_multimodal_breaths", {"offset_seconds": 0.5})


def test_session_run_pipeline_is_distinct_from_module_level_run_pipeline():
    """`session.run_pipeline(name)` and `m3resp.run_pipeline(spec, ...)` are two
    different mechanisms (Milestone 2.4 named presets vs. the Stage 1
    declarative engine) that happen to share a name; this pins that both
    remain independently usable.
    """

    import m3resp

    @register_step("t.named_pipeline_smoke", writes=("value",))
    def _noop(**kwargs: Any) -> dict[str, Any]:
        return {"value": 1}

    try:
        session = M3Session()
        session.preprocess_eit = lambda **kwargs: None
        session.detect_eit_breaths = lambda **kwargs: None

        assert session.run_pipeline("eit") is session

        result = m3resp.run_pipeline(
            {"name": "noop", "steps": [{"uses": "t.named_pipeline_smoke"}]}
        )
        assert result.name == "noop"
        assert result.outputs["value"] == 1
    finally:
        STEP_REGISTRY.pop("t.named_pipeline_smoke", None)
