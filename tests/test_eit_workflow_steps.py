"""Phase 7 (plan/stage2/1_eit_gap_migration_implementation_plan.md) - workflow
contract tests for the Stage 2 EIT gap migration steps.

These tests do not need the optional `eitprocessing` dependency: every
migrated/added `eit.*` step calls `session.eit_adapter.<method>()` rather than
importing `eitprocessing` itself, so a fake adapter is enough to prove the
granular steps work standalone (Phase 1's "no operation is implemented twice"
/ Phase 2-4's "granular steps use injected adapter methods" acceptance
criteria). The one exception - running the full example end to end - needs
the real package and skips cleanly without it.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from m3resp.core.exceptions import PipelineSpecError
from m3resp.core.session import M3Session
from m3resp.data import BreathEvent, Event, IntervalData, PixelMap, PixelMask
from m3resp.workflows import available_steps, run_pipeline
from m3resp.workflows.registry import get_step
from m3resp.workflows.spec import load_spec
from m3resp.workflows.steps.eit import (
    butterworth_filter,
    detect_rates,
    eeli,
    load,
    mdn_filter,
    pixel_breaths,
    pixel_tiv,
    roi_amplitude_lungspace,
    roi_filter_by_size,
    roi_tiv_lungspace,
    roi_watershed,
)

_NEW_EIT_STEPS = (
    "eit.load",
    "eit.detect_rates",
    "eit.mdn_filter",
    "eit.eeli",
    "eit.pixel_tiv",
    "eit.pixel_breaths",
    "eit.roi_tiv_lungspace",
    "eit.roi_amplitude_lungspace",
    "eit.roi_watershed",
    "eit.roi_filter_by_size",
)


def test_every_migrated_step_is_registered_and_listed():
    steps = available_steps()
    for name in _NEW_EIT_STEPS:
        assert name in steps, f"{name} missing from available_steps()"
        assert steps[name], f"{name} has an empty summary"


# -- fakes: shaped like the real eitprocessing/upstream objects, but with no
# import of eitprocessing anywhere in this module ---------------------------


class _FakeCollection(dict):
    def add(self, value: Any, overwrite: bool = False) -> None:
        if not overwrite and getattr(value, "label", None) in self:
            raise KeyError(value.label)
        self[getattr(value, "label", None)] = value


class _FakeContinuousData:
    def __init__(self, label: str, values: Any, time: Any):
        self.label = label
        self.name = label
        self.values = np.asarray(values, dtype=float)
        self.time = np.asarray(time, dtype=float)
        self.sample_frequency = 1.0
        self.unit = "a.u."


class _FakeEITData:
    def __init__(self, pixel_impedance: Any, time: Any, label: str = "raw"):
        self.label = label
        self.name = label
        self.pixel_impedance = np.asarray(pixel_impedance, dtype=float)
        self.time = np.asarray(time, dtype=float)
        self.sample_frequency = 1.0
        self.unit = "a.u."

    def get_summed_impedance(self, return_label: str | None = None, **_: Any):
        return _FakeContinuousData(
            return_label or f"summed_{self.label}",
            self.pixel_impedance.sum(axis=(1, 2)),
            self.time,
        )


class _FakeSequence:
    def __init__(self, raw: _FakeEITData):
        self.eit_data = _FakeCollection(raw=raw)
        self.continuous_data = _FakeCollection()
        self.sparse_data = _FakeCollection()
        self.interval_data = _FakeCollection()


class _FakeBreath:
    def __init__(self, start: float, middle: float, end: float):
        self.start_time = start
        self.middle_time = middle
        self.end_time = end


class _FakeSparseData:
    def __init__(self, values: Any, time: Any, *, unit: str = "a.u.", label: str = "r"):
        self.values = values
        self.time = time
        self.unit = unit
        self.label = label
        self.name = label


class _FakeBreathDetector:
    """Finds `count` one-second breaths, like eitprocessing's BreathDetection
    returns them (an object whose `values` are the breaths)."""

    def __init__(self, count: int):
        self.count = count
        self.calls = 0

    def find_breaths(self, signal: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(
            values=[
                _FakeBreath(float(i), i + 0.5, float(i + 1)) for i in range(self.count)
            ]
        )


class _FakePixelMask:
    def __init__(self, mask: np.ndarray):
        self.mask = mask


class _FakeAdapter:
    """Duck-types every `EITProcessingAdapter` method the migrated EIT steps
    call - no `eitprocessing` import anywhere in this class."""

    def __init__(self) -> None:
        self._sequence: _FakeSequence | None = None

    def load(self, path: str, vendor: str | None = None, **kwargs: Any) -> Any:
        self.load_kwargs = dict(kwargs)
        pixel_impedance = np.ones((6, 2, 2))
        raw = _FakeEITData(pixel_impedance, time=np.arange(6, dtype=float))
        self._sequence = _FakeSequence(raw)
        return self._sequence

    def get_raw_eit(self, sequence: Any, label: str = "raw") -> Any:
        return sequence.eit_data[label]

    def get_global_impedance(self, sequence: Any, label: str = "raw") -> Any:
        return sequence.eit_data[label].get_summed_impedance(
            return_label=f"global_impedance_({label})"
        )

    def detect_rates(self, signal: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "respiratory_rate_hz": 0.3,
            "heart_rate_hz": 1.2,
            "rate_detector": "fake-rate-detector",
            "rate_captures": {},
        }

    def apply_mdn(
        self,
        signal: Any,
        *,
        respiratory_rate_hz: float,
        heart_rate_hz: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        filtered = _FakeEITData(signal.pixel_impedance, signal.time, label="filtered")
        return {"filtered_eit": filtered, "filter_captures": {}}

    def compute_eeli(
        self, timing_data: Any, *, sequence: Any, breath_detector: Any, **kwargs: Any
    ) -> Any:
        return _FakeSparseData(
            [1.0, 2.0, np.nan], [0.0, 1.0, 2.0], label="continuous_eelis"
        )

    def compute_pixel_tiv(
        self,
        eit_data: Any,
        timing_data: Any,
        *,
        sequence: Any,
        breath_detector: Any,
        **kwargs: Any,
    ) -> Any:
        values = np.full((2, 2, 2), None, dtype=object)
        values[0, 0, 0] = 0.5
        time = np.full((2, 2, 2), None, dtype=object)
        time[0, 0, 0] = 1.0
        return _FakeSparseData(values, time, label="pixel_tivs")

    def find_pixel_breaths(
        self,
        eit_data: Any,
        timing_data: Any,
        *,
        sequence: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        values = np.full((3, 2, 2), None, dtype=object)
        values[1, 0, 0] = _FakeBreath(0.0, 0.5, 1.0)
        result = _FakeSparseData(values, None, label="pixel_breaths")
        result.intervals = [(-1.0, 0.0), (0.0, 1.0), (1.0, 2.0)]
        return result

    def find_breaths(self, timing_data: Any, **kwargs: Any) -> Any:
        # The same breaths as the intervals `find_pixel_breaths` returns.
        return SimpleNamespace(
            values=[
                _FakeBreath(-1.0, -0.5, 0.0),
                _FakeBreath(0.0, 0.5, 1.0),
                _FakeBreath(1.0, 1.5, 2.0),
            ]
        )

    def compute_tiv_lungspace(self, eit_data: Any, **kwargs: Any) -> dict[str, Any]:
        mask = np.array([[1.0, np.nan], [np.nan, 1.0]])
        return {"mask": _FakePixelMask(mask), "captures": {"mean TIV": "fake"}}

    def compute_amplitude_lungspace(
        self, eit_data: Any, **kwargs: Any
    ) -> dict[str, Any]:
        mask = np.array([[1.0, np.nan], [1.0, np.nan]])
        return {"mask": _FakePixelMask(mask), "captures": {"mean amplitude": "fake"}}

    def compute_watershed_lungspace(
        self, eit_data: Any, **kwargs: Any
    ) -> dict[str, Any]:
        mask = np.array([[1.0, 1.0], [np.nan, np.nan]])
        return {"mask": _FakePixelMask(mask), "captures": {"included region": "fake"}}

    def filter_roi_by_size(self, mask: Any, **kwargs: Any) -> Any:
        return _FakePixelMask(mask.mask)


def _session_with_fake_adapter() -> M3Session:
    session = M3Session()
    session.eit_adapter = _FakeAdapter()  # type: ignore[assignment]
    return session


def test_load_step_works_without_eitprocessing_and_matches_declared_writes():
    session = _session_with_fake_adapter()

    result = load(session, file_path="fake.bin", vendor="draeger")

    assert set(get_step("eit.load").writes) <= set(result)
    assert result["raw_global_impedance_signal"].modality == "eit"
    assert session.signals.for_modality("eit")

    # Both impedances are signals from the moment the file is read, and the
    # channel is what tells them apart.
    pixel_signal = result["raw_pixel_impedance_signal"]
    assert pixel_signal.modality == "eit"
    assert pixel_signal.category == "impedance"
    assert pixel_signal.channel == "pixel_impedance"
    assert pixel_signal.processing_state == "raw"
    assert {signal.channel for signal in session.signals.for_modality("eit")} == {
        "global_impedance",
        "pixel_impedance",
    }


def test_load_step_forwards_the_declared_reading_parameters():
    """sample_frequency/first_frame/max_frames are real parameters, not bag keys."""

    session = _session_with_fake_adapter()

    load(
        session,
        file_path="fake.bin",
        vendor="draeger",
        sample_frequency=20.0,
        first_frame=5,
        max_frames=100,
    )

    forwarded = session.eit_adapter.load_kwargs
    assert forwarded["sample_frequency"] == 20.0
    assert forwarded["first_frame"] == 5
    assert forwarded["max_frames"] == 100


def test_load_step_omits_unset_reading_parameters():
    """Unset means 'let the vendor loader decide', not 'pass None'."""

    session = _session_with_fake_adapter()

    load(session, file_path="fake.bin", vendor="draeger")

    forwarded = session.eit_adapter.load_kwargs
    assert "sample_frequency" not in forwarded
    assert "max_frames" not in forwarded
    assert forwarded["first_frame"] == 0


def test_load_step_rejects_reading_parameters_hidden_in_loader_options():
    session = _session_with_fake_adapter()

    with pytest.raises(ValueError, match="must be set directly"):
        load(
            session,
            file_path="fake.bin",
            vendor="draeger",
            first_frame=0,
            loader_options={"first_frame": 99},
        )


def test_detect_rates_step_works_without_eitprocessing_and_matches_declared_writes():
    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((4, 1, 1)), time=np.arange(4, dtype=float))

    result = detect_rates(raw, session=session, subject_type="adult")

    assert set(get_step("eit.detect_rates").writes) <= set(result)
    assert result["respiratory_rate_hz"] == pytest.approx(0.3)
    names = {p.name for p in session.parameter_results}
    assert {"respiratory_rate", "heart_rate"} <= names


def _detect_rates_with(rate_name, rate):
    """Run `eit.detect_rates` against a detector returning one bad rate."""

    session = _session_with_fake_adapter()
    rates = {
        "respiratory_rate_hz": 0.3,
        "heart_rate_hz": 1.0,
        "rate_detector": None,
        "rate_captures": {},
    }
    rates[rate_name] = rate
    session.eit_adapter.detect_rates = lambda *a, **k: rates  # type: ignore[method-assign]
    raw = _FakeEITData(np.ones((4, 1, 1)), time=np.arange(4, dtype=float))
    return detect_rates(raw, session=session)


@pytest.mark.parametrize("rate_name", ["respiratory_rate_hz", "heart_rate_hz"])
def test_detect_rates_reports_a_missing_rate_as_such(rate_name):
    """NaN means no rate could be estimated, which is the only failure
    eitprocessing's own detector can produce. It is reported in those terms
    rather than lumped in with an implausible value."""

    with pytest.raises(ValueError, match="could not estimate"):
        _detect_rates_with(rate_name, float("nan"))


@pytest.mark.parametrize(
    ("rate_name", "rate"),
    [
        ("respiratory_rate_hz", -0.1),
        ("heart_rate_hz", 0.0),
        ("heart_rate_hz", float("inf")),
    ],
)
def test_detect_rates_rejects_an_implausible_rate(rate_name, rate):
    """Zero, negative and infinite rates cannot come from eitprocessing, but a
    substituted detector is under no such constraint."""

    with pytest.raises(ValueError, match="implausible"):
        _detect_rates_with(rate_name, rate)


def test_mdn_filter_step_works_without_eitprocessing_and_matches_declared_writes():
    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((4, 2, 2)), time=np.arange(4, dtype=float))
    sequence = _FakeSequence(raw)

    result = mdn_filter(
        raw,
        respiratory_rate_hz=0.3,
        heart_rate_hz=1.2,
        eit_sequence=sequence,
        session=session,
        label="mdn_filtered",
    )

    assert set(get_step("eit.mdn_filter").writes) <= set(result)
    assert result["filtered_eit_signal"].channel == "pixel_impedance"
    assert sequence.eit_data["filtered"] is result["filtered_eit"]


@pytest.mark.parametrize("mode", ["lowpass", "highpass", "bandpass", "bandstop"])
def test_butterworth_filter_step_supports_every_mode_and_matches_declared_writes(mode):
    pytest.importorskip("eitprocessing")
    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((40, 2, 2)), time=np.arange(40, dtype=float))
    raw.sample_frequency = 20.0
    sequence = _FakeSequence(raw)

    result = butterworth_filter(
        raw,
        eit_sequence=sequence,
        session=session,
        mode=mode,
        lowpass_hz=1.0,
        highpass_hz=0.05,
    )

    assert set(get_step("eit.butterworth_filter").writes) <= set(result)
    # Same native signal and session bookkeeping the MDN filter produces.
    assert result["filtered_eit_signal"].channel == "pixel_impedance"
    assert session.signals.for_modality("eit")
    assert sequence.eit_data["filtered"] is result["filtered_eit"]


def test_butterworth_filter_step_accepts_an_already_filtered_signal():
    """The signal to filter is bound explicitly, not pinned to raw_eit."""

    pytest.importorskip("eitprocessing")
    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((40, 2, 2)), time=np.arange(40, dtype=float))
    raw.sample_frequency = 20.0
    sequence = _FakeSequence(raw)

    once = butterworth_filter(
        raw, eit_sequence=sequence, session=session, label="pass_one"
    )
    twice = butterworth_filter(
        once["filtered_eit"], eit_sequence=sequence, session=session, label="pass_two"
    )

    assert twice["filtered_eit"].label == "pass_two"
    assert get_step("eit.butterworth_filter").reads["signal"] is None


def test_eeli_step_stores_one_value_per_breath():
    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((3, 1, 1)), time=np.arange(3, dtype=float))
    sequence = _FakeSequence(raw)

    result = eeli(
        raw,
        eit_sequence=sequence,
        breath_detector=_FakeBreathDetector(3),
        session=session,
    )

    assert set(get_step("eit.eeli").writes) <= set(result)
    eeli_result = result["eeli_result"]
    assert isinstance(eeli_result, IntervalData)
    np.testing.assert_array_equal(eeli_result.values, [1.0, 2.0, np.nan])
    assert [b.start_time for b in eeli_result.intervals] == [0.0, 1.0, 2.0]
    assert eeli_result.method == "eitprocessing.EELI"
    assert list(session.interval_data) == [eeli_result]
    assert len(session.parameter_results) == 0


def test_eeli_step_refuses_breaths_that_do_not_match_the_values():
    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((3, 1, 1)), time=np.arange(3, dtype=float))
    sequence = _FakeSequence(raw)

    with pytest.raises(ValueError, match="per-breath values"):
        eeli(
            raw,
            eit_sequence=sequence,
            breath_detector=_FakeBreathDetector(2),
            session=session,
        )


def test_pixel_tiv_step_preserves_shape_and_valid_breath_metadata():
    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((2, 2, 2)), time=np.arange(2, dtype=float))
    sequence = _FakeSequence(raw)

    result = pixel_tiv(
        eit_data=raw,
        signal=raw,
        eit_sequence=sequence,
        breath_detector=_FakeBreathDetector(2),
        session=session,
    )

    assert set(get_step("eit.pixel_tiv").writes) <= set(result)
    pixel_tiv_result = result["pixel_tiv_result"]
    assert isinstance(pixel_tiv_result, IntervalData)
    assert len(pixel_tiv_result) == 2
    assert all(isinstance(m, PixelMap) for m in pixel_tiv_result.values)
    assert pixel_tiv_result.values[0].shape == (2, 2)
    assert pixel_tiv_result.values[0].values[0, 0] == 0.5
    assert pixel_tiv_result.metadata["valid_breath_indices"] == [0]
    assert pixel_tiv_result.metadata["axes"] == ["row", "column"]
    assert list(session.interval_data) == [pixel_tiv_result]


def test_pixel_tiv_accepts_unfiltered_pixel_data():
    """The input is any pixel signal; filtering is the default, not a demand."""

    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((2, 2, 2)), time=np.arange(2, dtype=float), label="raw")
    sequence = _FakeSequence(raw)

    result = pixel_tiv(
        eit_data=raw,
        signal=raw,
        eit_sequence=sequence,
        breath_detector=_FakeBreathDetector(2),
        session=session,
    )

    assert len(result["pixel_tiv_result"]) == 2
    definition = get_step("eit.pixel_tiv")
    assert "eit_data" in definition.reads
    assert "filtered_eit" not in definition.reads


def test_pixel_breaths_accepts_the_empty_phase_correction_mode():
    """`null` in a spec / `None` from Python is a real option, not a mistake."""

    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((3, 2, 2)), time=np.arange(3, dtype=float))
    sequence = _FakeSequence(raw)

    result = pixel_breaths(
        eit_data=raw,
        timing_data=raw,
        eit_sequence=sequence,
        session=session,
        phase_correction_mode=None,
    )

    assert set(get_step("eit.pixel_breaths").writes) <= set(result)


def test_pixel_breaths_rejects_an_unknown_phase_correction_mode():
    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((3, 2, 2)), time=np.arange(3, dtype=float))
    sequence = _FakeSequence(raw)

    with pytest.raises(ValueError, match="phase_correction_mode"):
        pixel_breaths(
            eit_data=raw,
            timing_data=raw,
            eit_sequence=sequence,
            session=session,
            phase_correction_mode="nope",
        )


def test_pixel_breaths_step_converts_object_array_to_landmark_array():
    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((3, 2, 2)), time=np.arange(3, dtype=float))
    sequence = _FakeSequence(raw)

    result = pixel_breaths(
        eit_data=raw, timing_data=raw, eit_sequence=sequence, session=session
    )

    assert set(get_step("eit.pixel_breaths").writes) <= set(result)
    timing = result["pixel_breath_timing_result"]
    assert isinstance(timing, IntervalData)
    assert [b.start_time for b in timing.intervals] == [-1.0, 0.0, 1.0]
    # The breaths keep their turning point, like the breaths of TIV and EELI.
    assert [b.peak_time for b in timing.intervals] == [-0.5, 0.5, 1.5]
    assert timing.unit == "s"
    value = np.stack(timing.values)
    assert value.shape == (3, 2, 2, 3)
    assert np.array_equal(value[1, 0, 0], [0.0, 0.5, 1.0])
    assert np.isnan(value[0]).all()  # unresolved pixel breaths stay NaN
    assert list(session.interval_data) == [timing]


def test_pixel_breaths_refuses_breaths_that_do_not_match_pixelbreath():
    session = _session_with_fake_adapter()
    session.eit_adapter.find_breaths = lambda timing_data, **kwargs: SimpleNamespace(  # type: ignore[attr-defined]
        values=[_FakeBreath(0.0, 0.5, 1.0)]
    )
    raw = _FakeEITData(np.ones((3, 2, 2)), time=np.arange(3, dtype=float))

    with pytest.raises(ValueError, match="do not match"):
        pixel_breaths(
            eit_data=raw,
            timing_data=raw,
            eit_sequence=_FakeSequence(raw),
            session=session,
        )


def test_steps_on_the_same_breaths_share_one_detection_and_the_same_breaths():
    """EELI and pixel TIV over the same detector and signal find the breaths
    once and store the very same breath objects; a breath already stored in
    session.events is reused, so values can be traced to it."""

    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((2, 2, 2)), time=np.arange(2, dtype=float))
    sequence = _FakeSequence(raw)
    detector = _FakeBreathDetector(2)
    stored_first_breath = BreathEvent("eit", 0.0, 1.0, peak_time=0.5)
    session.add_events("eit_breaths", [stored_first_breath])
    session.eit_adapter.compute_eeli = lambda *args, **kwargs: _FakeSparseData(  # type: ignore[attr-defined]
        [1.0, 2.0], [1.0, 2.0], label="continuous_eelis"
    )

    eeli_result = eeli(
        raw, eit_sequence=sequence, breath_detector=detector, session=session
    )["eeli_result"]
    tiv_result = pixel_tiv(
        eit_data=raw,
        signal=raw,
        eit_sequence=sequence,
        breath_detector=detector,
        session=session,
    )["pixel_tiv_result"]

    assert detector.calls == 1
    assert all(
        first is second
        for first, second in zip(eeli_result.intervals, tiv_result.intervals)
    )
    assert eeli_result.intervals[0] is stored_first_breath


def test_steps_ignore_stored_eit_breaths_that_are_not_breaths():
    """`add_events` accepts anything; a marker stored under "eit_breaths"
    must not break the steps that reuse stored breaths."""

    session = _session_with_fake_adapter()
    session.add_events("eit_breaths", [Event(name="marker", modality="eit", time=0.0)])
    raw = _FakeEITData(np.ones((3, 1, 1)), time=np.arange(3, dtype=float))

    result = eeli(
        raw,
        eit_sequence=_FakeSequence(raw),
        breath_detector=_FakeBreathDetector(3),
        session=session,
    )

    assert len(result["eeli_result"]) == 3


def test_pixel_breaths_rejects_unknown_phase_correction_mode():
    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((3, 2, 2)), time=np.arange(3, dtype=float))
    sequence = _FakeSequence(raw)

    with pytest.raises(ValueError):
        pixel_breaths(
            eit_data=raw,
            timing_data=raw,
            eit_sequence=sequence,
            session=session,
            phase_correction_mode="sideways",
        )


@pytest.mark.parametrize(
    "step_func,step_name",
    [
        (roi_tiv_lungspace, "eit.roi_tiv_lungspace"),
        (roi_amplitude_lungspace, "eit.roi_amplitude_lungspace"),
        (roi_watershed, "eit.roi_watershed"),
    ],
)
def test_roi_lungspace_steps_preserve_nan_as_excluded_pixels(step_func, step_name):
    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((2, 2, 2)), time=np.arange(2, dtype=float))

    kwargs = (
        {"threshold_fraction": 0.15}
        if step_func is roi_watershed
        else {"threshold": 0.15}
    )
    result = step_func(eit_data=raw, timing_data=raw, session=session, **kwargs)

    assert set(get_step(step_name).writes) <= set(result)
    mask_key = next(k for k in result if k.endswith("_mask"))
    assert np.isnan(result[mask_key].mask).any()
    result_key = next(k for k in result if k.endswith("_result"))
    assert isinstance(result[result_key], PixelMask)
    np.testing.assert_array_equal(result[result_key].values, result[mask_key].mask)
    assert list(session.pixel_masks) == [result[result_key]]
    assert len(session.parameter_results) == 0


@pytest.mark.parametrize("bad_threshold", [0.0, 1.0, -0.1, 1.5])
def test_roi_lungspace_steps_reject_out_of_range_threshold(bad_threshold):
    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((2, 2, 2)), time=np.arange(2, dtype=float))

    with pytest.raises(ValueError):
        roi_tiv_lungspace(
            eit_data=raw, timing_data=raw, session=session, threshold=bad_threshold
        )


def test_roi_filter_by_size_accepts_the_native_mask_result():
    """Either form of a mask can be bound: upstream object or m3resp mask."""

    pytest.importorskip("eitprocessing")
    import numpy as np
    from eitprocessing.roi import PixelMask as UpstreamPixelMask

    from m3resp.adapters import EITProcessingAdapter

    mask = np.full((4, 4), np.nan)
    mask[1:3, 1:3] = 1.0
    mask[0, 0] = 1.0

    adapter = EITProcessingAdapter()
    native = PixelMask(name="watershed_lungspace_mask", values=mask, method="test")

    from_native = adapter.filter_roi_by_size(native, min_region_size=2)
    from_upstream = adapter.filter_roi_by_size(
        UpstreamPixelMask(mask), min_region_size=2
    )

    np.testing.assert_array_equal(
        np.nan_to_num(from_native.mask, nan=-1),
        np.nan_to_num(from_upstream.mask, nan=-1),
    )
    assert np.isnan(from_native.mask[0, 0]), "isolated pixel should be dropped"


def test_pixel_breath_needs_all_three_timings_to_count_as_valid():
    """A breath missing its middle or end time is not a determined breath."""

    import numpy as np

    from m3resp.workflows.steps.eit.pixel import _pixel_breaths_to_landmark_array

    class _PartialBreath:
        start_time, middle_time, end_time = 0.0, float("nan"), 1.0

    class _WholeBreath:
        start_time, middle_time, end_time = 0.0, 0.5, 1.0

    values = np.empty((1, 1, 2), dtype=object)
    values[0, 0, 0] = _WholeBreath()
    values[0, 0, 1] = _PartialBreath()

    landmarks = _pixel_breaths_to_landmark_array(values)
    valid = ~np.isnan(landmarks).any(axis=-1)

    assert valid[0, 0, 0], "fully timed breath is valid"
    assert not valid[0, 0, 1], "breath with a missing middle time is not valid"


def test_roi_filter_by_size_rejects_non_positive_min_region_size():
    session = _session_with_fake_adapter()
    mask = _FakePixelMask(np.array([[1.0, np.nan], [np.nan, 1.0]]))

    with pytest.raises(ValueError):
        roi_filter_by_size(mask, session=session, min_region_size=0)


def test_roi_filter_by_size_step_matches_declared_writes():
    session = _session_with_fake_adapter()
    mask = _FakePixelMask(np.array([[1.0, np.nan], [np.nan, 1.0]]))

    result = roi_filter_by_size(mask, session=session, min_region_size=1)

    assert set(get_step("eit.roi_filter_by_size").writes) <= set(result)


# -- YAML/spec-level validation (no execution, so no eitprocessing needed) --


def test_validation_rejects_mdn_filter_without_explicit_signal_binding():
    spec = {
        "name": "bad-spec",
        "steps": [
            {"uses": "eit.load", "with": {"file_path": "x.bin", "vendor": "draeger"}},
            {
                "uses": "eit.mdn_filter",
                # 'signal' has no default binding and is not bound here.
            },
        ],
    }

    with pytest.raises(PipelineSpecError, match="explicit 'in:"):
        from m3resp.workflows.engine import validate_spec

        validate_spec(load_spec(spec))


def test_validation_rejects_duplicate_context_writes():
    spec = {
        "name": "bad-spec",
        "steps": [
            {"uses": "eit.load", "with": {"file_path": "x.bin", "vendor": "draeger"}},
            {
                "uses": "eit.mdn_filter",
                "in": {"signal": "raw_eit"},
            },
            {
                "uses": "eit.mdn_filter",
                "in": {"signal": "raw_eit"},
                # writes 'filtered_eit' again without renaming via 'out:'.
            },
        ],
    }

    with pytest.raises(PipelineSpecError, match="already produced"):
        from m3resp.workflows.engine import validate_spec

        # respiratory_rate_hz/heart_rate_hz are pre-seeded here purely to
        # isolate the duplicate-write check from eit.mdn_filter's other
        # required bindings.
        validate_spec(
            load_spec(spec),
            available={"respiratory_rate_hz", "heart_rate_hz"},
        )


def test_output_renaming_lets_the_same_roi_step_run_twice():
    session = _session_with_fake_adapter()
    spec = {
        "name": "repeated-roi",
        "steps": [
            {
                "uses": "eit.roi_tiv_lungspace",
                "in": {"eit_data": "eit_data", "timing_data": "eit_data"},
                "with": {"threshold": 0.1},
                "out": {
                    "tiv_lungspace_mask": "mask_a",
                    "tiv_lungspace_captures": "captures_a",
                    "tiv_lungspace_result": "result_a",
                },
            },
            {
                "uses": "eit.roi_tiv_lungspace",
                "in": {"eit_data": "eit_data", "timing_data": "eit_data"},
                "with": {"threshold": 0.2},
                "out": {
                    "tiv_lungspace_mask": "mask_b",
                    "tiv_lungspace_captures": "captures_b",
                    "tiv_lungspace_result": "result_b",
                },
            },
        ],
    }
    raw = _FakeEITData(np.ones((2, 2, 2)), time=np.arange(2, dtype=float))

    result = run_pipeline(spec, session=session, extra_context={"eit_data": raw})

    assert result.value("mask_a") is not result.value("mask_b")
    assert result.value("result_a").metadata["parameters"]["threshold"] == 0.1
    assert result.value("result_b").metadata["parameters"]["threshold"] == 0.2


# -- full example end to end (needs the real optional dependency) ----------


def test_full_eit_example_pipeline_runs_end_to_end(tmp_path):
    pytest.importorskip("eitprocessing")

    repo_root = Path(__file__).resolve().parents[1]
    fixture = os.path.join(
        repo_root,
        "tests",
        "data",
        "draeger_synthetic_draeger_20Hz.bin",
    )
    assert os.path.exists(fixture), f"missing committed EIT fixture: {fixture}"

    spec_path = os.path.join(
        repo_root, "examples", "eit_full_preprocessing", "eit-full.pipeline.yaml"
    )
    # run_pipeline (unlike run_spec) does not touch the spec's `outputs:`
    # section, so this exercises the example without writing into the
    # project's real output/ directory.
    result = run_pipeline(spec_path, session=M3Session())

    assert result.value("pixel_tiv_result").values[0].shape == (32, 32)
    assert result.value("size_filtered_roi_result").shape == (32, 32)

    output_path = result.session.export_summary(tmp_path)
    archive_path = output_path / "interval_data_arrays.npz"
    assert archive_path.exists()
    with np.load(archive_path) as archive:
        assert "pixel_tivs_0" in archive
        assert archive["pixel_tivs_0"].shape == (12, 32, 32)
    assert (output_path / "interval_data.csv").exists()
    with np.load(output_path / "pixel_masks.npz") as masks:
        assert masks["size_filtered_roi_mask_0"].shape == (32, 32)
