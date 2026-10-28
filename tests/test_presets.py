"""Named, built-in presets (`session.run_preset("eit" | "emg" | "multimodal")`).

Each `Preset.run` is just a fixed sequence of calls to `M3Session`'s own
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
from m3resp.core.exceptions import UnknownPresetError
from m3resp.presets import (
    EITPreset,
    EMGPreset,
    MultimodalPreset,
    available_presets,
    get_preset,
)
from m3resp.workflows import register_step
from m3resp.workflows.registry import STEP_REGISTRY

Calls = list[tuple[str, dict[str, Any]]]

#: Moving-baseline settings `EMGPreset` uses when config has no ``baseline``.
_BASELINE_DEFAULTS = {"window_seconds": 7.5, "step_seconds": 0.2, "percentile": 33.0}


def _spy(calls: Calls, name: str, result: Any = None):
    def _record(**kwargs: Any) -> Any:
        calls.append((name, kwargs))
        return result

    return _record


def _step_spy(calls: Calls, name: str, result: Any = None):
    """Spy for a registered step, which takes positional arguments."""

    def _record(*args: Any, **kwargs: Any) -> Any:
        calls.append((name, kwargs))
        return result

    return _record


class TestPresetRegistry:
    def test_built_in_presets_are_registered(self):
        assert available_presets() == ["eit", "emg", "multimodal"]
        assert get_preset("eit") is EITPreset
        assert get_preset("emg") is EMGPreset
        assert get_preset("multimodal") is MultimodalPreset

    def test_get_preset_raises_for_unknown_name(self):
        with pytest.raises(UnknownPresetError, match="Unknown preset 'unknown'"):
            get_preset("unknown")


class TestEITPreset:
    def test_run_calls_preprocess_then_detect_breaths_with_config(self):
        session = M3Session()
        calls: Calls = []
        session.preprocess_eit = _spy(calls, "preprocess_eit")
        session.detect_eit_breaths = _spy(calls, "detect_eit_breaths")

        result = session.run_preset(
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
        calls: Calls = []
        session.preprocess_eit = _spy(calls, "preprocess_eit")
        session.detect_eit_breaths = _spy(calls, "detect_eit_breaths")

        session.run_preset("eit")

        assert calls == [("preprocess_eit", {}), ("detect_eit_breaths", {})]


class TestEMGPreset:
    """`EMGPreset` = preprocess (filter + ECG removal + envelope) -> moving
    baseline -> breath detection -> postprocessing.

    ECG peak detection, ECG removal and the envelope are part of the default
    preprocessing function, so their options travel in ``config["preprocess"]``
    and the preset never calls the standalone workflow steps.
    """

    @staticmethod
    def _session_with_spies(
        calls: Calls,
        *,
        envelope: Any = None,
        baseline: Any = "baseline-trace",
    ) -> M3Session:
        """Session whose EMG methods record their calls.

        The `preprocess_emg` spy stores its result on `session.processed["emg"]`
        like the real method, so `envelope` controls whether the preset
        estimates a moving baseline.
        """

        session = M3Session()

        def _preprocess(**kwargs: Any) -> dict[str, Any]:
            calls.append(("preprocess_emg", kwargs))
            result = {"fs": 1000.0, "channel": 0, "envelope": envelope}
            session.processed["emg"] = result
            return result

        session.preprocess_emg = _preprocess
        session.emg_moving_baseline = _spy(calls, "emg_moving_baseline", baseline)
        session.detect_emg_breaths = _spy(calls, "detect_emg_breaths")
        session.postprocess_emg = _spy(calls, "postprocess_emg")
        return session

    @staticmethod
    def _patch_standalone_steps(monkeypatch: pytest.MonkeyPatch, calls: Calls) -> None:
        """Record any call to the standalone ECG/envelope steps.

        `import_module`, not `from ... import ecg_gating`: the package's
        `__init__` re-exports the step *function* under the same name as its
        module, which would shadow the module object here.
        """

        for module_name, attribute in (
            ("m3resp.workflows.steps.emg.ecg_detection", "ecg_detect_peaks"),
            ("m3resp.workflows.steps.emg.ecg_gating", "ecg_gating"),
            ("m3resp.workflows.steps.emg.preprocessing", "envelope"),
        ):
            monkeypatch.setattr(
                import_module(module_name), attribute, _step_spy(calls, attribute)
            )

    def test_run_calls_methods_in_order_and_returns_session(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        calls: Calls = []
        session = self._session_with_spies(calls, envelope=[1.0, 2.0])
        self._patch_standalone_steps(monkeypatch, calls)

        result = session.run_preset("emg", config={"postprocess": {"peep": 5.0}})

        assert result is session
        assert [name for name, _ in calls] == [
            "preprocess_emg",
            "emg_moving_baseline",
            "detect_emg_breaths",
            "postprocess_emg",
        ]
        assert calls[-1] == ("postprocess_emg", {"peep": 5.0})

    def test_ecg_and_envelope_options_are_passed_to_preprocessing(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        calls: Calls = []
        session = self._session_with_spies(calls)
        self._patch_standalone_steps(monkeypatch, calls)
        preprocess_options = {
            "high_pass_hz": 20.0,
            "remove_ecg": True,
            "ecg_removal_method": "wavelet",
            "ecg_gate_fill_method": 2,
            "ecg_channel": [1],
            "compute_envelope": True,
            "envelope_method": "arv",
            "envelope_window_seconds": 0.5,
        }

        session.run_preset("emg", config={"preprocess": preprocess_options})

        assert calls[0] == ("preprocess_emg", preprocess_options)
        called = {name for name, _ in calls}
        assert called.isdisjoint({"ecg_detect_peaks", "ecg_gating", "envelope"})

    def test_ecg_removal_can_be_disabled_through_preprocessing(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        calls: Calls = []
        session = self._session_with_spies(calls, envelope=[1.0])
        self._patch_standalone_steps(monkeypatch, calls)

        session.run_preset("emg", config={"preprocess": {"remove_ecg": False}})

        assert calls[0] == ("preprocess_emg", {"remove_ecg": False})
        assert [name for name, _ in calls] == [
            "preprocess_emg",
            "emg_moving_baseline",
            "detect_emg_breaths",
            "postprocess_emg",
        ]

    def test_moving_baseline_uses_defaults_and_feeds_breath_detection(self):
        calls: Calls = []
        session = self._session_with_spies(
            calls, envelope=[1.0, 2.0], baseline="baseline-trace"
        )

        session.run_preset("emg", config={"detect_breaths": {"min_peak_height": 0.1}})

        assert ("emg_moving_baseline", _BASELINE_DEFAULTS) in calls
        assert (
            "detect_emg_breaths",
            {"baseline": "baseline-trace", "min_peak_height": 0.1},
        ) in calls

    def test_baseline_options_override_defaults(self):
        calls: Calls = []
        session = self._session_with_spies(calls, envelope=[1.0, 2.0])

        session.run_preset(
            "emg", config={"baseline": {"window_seconds": 5, "percentile": "50"}}
        )

        assert (
            "emg_moving_baseline",
            {"window_seconds": 5.0, "step_seconds": 0.2, "percentile": 50.0},
        ) in calls

    def test_no_envelope_skips_baseline(self):
        """With ``compute_envelope=False`` there is nothing to estimate a
        baseline from, so breath detection gets ``baseline=None``."""

        calls: Calls = []
        session = self._session_with_spies(calls, envelope=None)

        session.run_preset("emg", config={"preprocess": {"compute_envelope": False}})

        assert [name for name, _ in calls] == [
            "preprocess_emg",
            "detect_emg_breaths",
            "postprocess_emg",
        ]
        assert calls[1] == ("detect_emg_breaths", {"baseline": None})

    def test_unknown_baseline_option_is_rejected(self):
        calls: Calls = []
        session = self._session_with_spies(calls, envelope=[1.0])

        with pytest.raises(TypeError, match=r"config\['baseline'\] accepts"):
            session.run_preset("emg", config={"baseline": {"method": "median"}})

    @pytest.mark.parametrize(
        "group", ["ecg_removal", "ecg_detect_peaks", "ecg_gating", "envelope"]
    )
    def test_moved_config_groups_are_rejected(self, group: str):
        calls: Calls = []
        session = self._session_with_spies(calls)

        with pytest.raises(TypeError, match=r"pass these under config\['preprocess'\]"):
            session.run_preset("emg", config={group: {}})
        assert calls == []

    def test_unknown_config_group_is_rejected(self):
        calls: Calls = []
        session = self._session_with_spies(calls)

        with pytest.raises(TypeError, match="EMGPreset config accepts"):
            session.run_preset("emg", config={"detect": {}})
        assert calls == []


class TestMultimodalPreset:
    def test_run_calls_synchronize_then_align_in_order(self):
        session = M3Session()
        calls: Calls = []
        session.synchronize_raw_modalities = _spy(calls, "synchronize_raw_modalities")
        session.synchronize_multimodal_breaths = _spy(
            calls, "synchronize_multimodal_breaths"
        )

        session.run_preset(
            "multimodal",
            config={"align": {"offset_seconds": 0.5}},
        )

        assert [name for name, _ in calls] == [
            "synchronize_raw_modalities",
            "synchronize_multimodal_breaths",
        ]
        assert calls[-1] == ("synchronize_multimodal_breaths", {"offset_seconds": 0.5})


def test_session_run_preset_is_distinct_from_module_level_run_workflow():
    """`session.run_preset(name)` and `m3resp.run_workflow(spec, ...)` are two
    different mechanisms (named presets vs. the declarative workflow
    engine); this pins that both remain independently usable.
    """

    import m3resp

    @register_step("t.named_workflow_smoke", writes=("value",))
    def _noop(**kwargs: Any) -> dict[str, Any]:
        return {"value": 1}

    try:
        session = M3Session()
        session.preprocess_eit = lambda **kwargs: None
        session.detect_eit_breaths = lambda **kwargs: None

        assert session.run_preset("eit") is session

        result = m3resp.run_workflow(
            {"name": "noop", "steps": [{"uses": "t.named_workflow_smoke"}]}
        )
        assert result.name == "noop"
        assert result.outputs["value"] == 1
    finally:
        STEP_REGISTRY.pop("t.named_workflow_smoke", None)
