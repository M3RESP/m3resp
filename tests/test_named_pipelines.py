"""Milestone 2.4 - named `Pipeline` presets (plan_stage2.md Sec 18-19).

Each `Pipeline.run` is just a fixed sequence of calls to `M3Session`'s own
methods (already covered by `tests/test_session.py`, `tests/test_eit.py`,
`tests/test_emg.py`), so these tests verify the *contract* - which methods
get called, in what order, with the `config` kwargs passed through - by
spying on the session's methods rather than re-deriving domain correctness.
"""

from __future__ import annotations

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


def _spy(calls: list[tuple[str, dict[str, Any]]], name: str):
    def _record(**kwargs: Any) -> None:
        calls.append((name, kwargs))

    return _record


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
    def test_run_calls_preprocess_detect_and_postprocess_in_order(self):
        session = M3Session()
        calls: list[tuple[str, dict[str, Any]]] = []
        session.preprocess_emg = _spy(calls, "preprocess_emg")
        session.detect_emg_breaths = _spy(calls, "detect_emg_breaths")
        session.postprocess_emg = _spy(calls, "postprocess_emg")

        session.run_pipeline("emg", config={"postprocess": {"peep": 5.0}})

        assert [name for name, _ in calls] == [
            "preprocess_emg",
            "detect_emg_breaths",
            "postprocess_emg",
        ]
        assert calls[-1] == ("postprocess_emg", {"peep": 5.0})


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
