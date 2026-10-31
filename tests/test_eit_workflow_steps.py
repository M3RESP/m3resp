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

import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from m3resp.adapters.eitprocessing_adapter import (
    EITPROCESSING_FIELDS,
    as_signal,
    eitprocessing_fields,
)
from m3resp.core.exceptions import WorkflowSpecError
from m3resp.core.session import M3Session
from m3resp.data import (
    BreathEvent,
    Event,
    IntervalData,
    PixelBreaths,
    PixelMap,
    PixelMask,
    Signal,
    pixel_breath_events,
)
from m3resp.eit.processing import (
    BreathDetectionSettings,
    find_breaths,
    find_pixel_breaths,
    pixel_end_expiratory_lung_impedance,
    pixel_tidal_impedance_variation,
)
from m3resp.workflows import available_steps, run_workflow
from m3resp.workflows.engine.spec_runner import _infer_filter_mode
from m3resp.workflows.registry import get_step
from m3resp.workflows.spec import load_spec
from m3resp.workflows.steps.eit import (
    butterworth_filter,
    continuous_tiv,
    detect_rates,
    eeli,
    load,
    mdn_filter,
    pixel_breaths,
    pixel_eeli,
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
    "eit.pixel_eeli",
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
        self.vendor = "draeger"

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

    def load_with_eitprocessing(
        self, path: str, vendor: str | None = None, **kwargs: Any
    ) -> Any:
        return self.load(path, vendor=vendor, **kwargs)

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
        label: str = "filtered",
        name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        images = as_signal(signal)
        filtered = Signal(
            values=images.values,
            time=images.time,
            sample_frequency=images.sample_frequency,
            name=name or images.name,
            metadata={
                EITPROCESSING_FIELDS: {**eitprocessing_fields(images), "label": label}
            },
            modality="eit",
            category="impedance",
            channel="pixel_impedance",
            source="draeger",
            processing_state="intermediate",
        )
        return {"filtered_eit": filtered, "filter_captures": {}}


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


def test_mdn_filter_step_matches_declared_writes():
    pytest.importorskip("eitprocessing")
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
    assert result["filtered_eit"] is result["filtered_eit_signal"]
    np.testing.assert_array_equal(
        sequence.eit_data["mdn_filtered"].pixel_impedance,
        result["filtered_eit"].values,
    )
    assert result["filtered_eit"].metadata["operation"] == "eit.mdn_filter"
    assert result["filtered_eit"].metadata["source_package"] == "m3resp"
    assert _infer_filter_mode(result["filtered_eit"]) == "mdn"


@pytest.mark.parametrize(
    ("metadata", "label", "mode"),
    [
        ({"operation": "eit.mdn_filter"}, None, "mdn"),
        (
            {"operation": "eit.butterworth_filter", "parameters": {"mode": "bandstop"}},
            None,
            "bandstop",
        ),
        ({}, "lowpass_filtered", "lowpass"),
        ({}, None, "none"),
    ],
)
def test_the_filter_mode_of_a_workflow_comes_from_the_filtered_images(
    metadata: dict, label: str | None, mode: str
):
    filtered = SimpleNamespace(metadata=metadata, label=label)

    assert _infer_filter_mode(filtered) == mode
    assert _infer_filter_mode(None) == "none"


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
    assert result["filtered_eit"] is result["filtered_eit_signal"]
    np.testing.assert_array_equal(
        sequence.eit_data["filtered"].pixel_impedance, result["filtered_eit"].values
    )


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

    assert twice["filtered_eit"].metadata["eitprocessing_fields"]["label"] == "pass_two"
    assert sequence.eit_data["pass_two"].label == "pass_two"
    assert get_step("eit.butterworth_filter").reads["signal"] is None


def _waveform(values: list[float]) -> _FakeContinuousData:
    """A global impedance waveform, one sample per second from 0 s."""

    return _FakeContinuousData(
        "global_impedance", values, np.arange(len(values), dtype=float)
    )


def test_eeli_step_stores_one_value_per_breath():
    session = _session_with_fake_adapter()
    waveform = _waveform([10.0, 11.0, 12.0, 13.0])

    result = eeli(waveform, breath_detector=_FakeBreathDetector(3), session=session)

    assert set(get_step("eit.eeli").writes) == set(result) - {"_provenance"}
    eeli_result = result["eeli_result"]
    assert isinstance(eeli_result, IntervalData)
    # The EELI of each breath is the impedance at its end (1, 2 and 3 s).
    np.testing.assert_array_equal(eeli_result.values, [11.0, 12.0, 13.0])
    assert [b.start_time for b in eeli_result.intervals] == [0.0, 1.0, 2.0]
    assert eeli_result.name == "continuous_eelis"
    assert eeli_result.method == "m3resp.eit.processing.end_expiratory_lung_impedance"
    assert eeli_result.metadata["source_package"] == "m3resp"
    assert eeli_result.metadata["operation"] == "eit.eeli"
    assert list(session.interval_data) == [eeli_result]
    assert len(session.parameter_results) == 0


def test_continuous_tiv_step_uses_the_chosen_method():
    session = _session_with_fake_adapter()
    waveform = _waveform([1.0, 3.0, 2.0, 6.0, 4.0])
    detector = _FakeBreathDetector(2)  # breaths (0, 0.5, 1) and (1, 1.5, 2)

    inspiratory = continuous_tiv(waveform, breath_detector=detector, session=session)[
        "continuous_tiv_result"
    ]
    expiratory = continuous_tiv(
        waveform, breath_detector=detector, session=session, tiv_method="expiratory"
    )["continuous_tiv_result"]

    # The middle (0.5 s, 1.5 s) lies halfway, so the later sample is read.
    np.testing.assert_array_equal(inspiratory.values, [2.0, -1.0])
    np.testing.assert_array_equal(expiratory.values, [0.0, 0.0])
    assert expiratory.metadata["tiv_method"] == "expiratory"
    assert expiratory.metadata["parameters"] == {
        "tiv_method": "expiratory",
        "breaths": "breath_detector",
    }
    assert inspiratory.intervals[0] is expiratory.intervals[0]


def test_eeli_step_gives_nan_for_breaths_outside_the_data():
    session = _session_with_fake_adapter()
    waveform = _waveform([10.0, 11.0, 12.0, 13.0])

    with pytest.warns(UserWarning, match="1 of 4 breaths lie partly outside"):
        result = eeli(waveform, breath_detector=_FakeBreathDetector(4), session=session)

    np.testing.assert_array_equal(
        result["eeli_result"].values, [11.0, 12.0, 13.0, np.nan]
    )


def _breathing_images() -> tuple[Signal, Signal]:
    """EIT images of 2 x 2 pixels and their waveform: 40 s at 20 Hz, one
    breath every 4 s. Pixel (1, 0) goes down during inspiration, pixel (1, 1)
    is NaN."""

    fs = 20.0
    time = np.arange(int(40 * fs)) / fs
    breathing = np.sin(2 * np.pi * 0.25 * time)
    values = np.empty((len(time), 2, 2))
    values[:, 0, 0] = 10.0 + breathing
    values[:, 0, 1] = 10.0 + 2.0 * breathing
    values[:, 1, 0] = 10.0 - breathing
    values[:, 1, 1] = np.nan
    pixels = Signal(
        values=values,
        time=time,
        sample_frequency=fs,
        name="images",
        modality="eit",
        category="impedance",
        channel="pixel_impedance",
        unit="a.u.",
    )
    waveform = Signal(
        values=30.0 + 2.0 * breathing,
        time=time,
        sample_frequency=fs,
        name="global_impedance",
        modality="eit",
        category="impedance",
        unit="a.u.",
    )
    return pixels, waveform


def _default_breaths(waveform: Signal) -> list[BreathEvent]:
    return find_breaths(waveform, BreathDetectionSettings())


def _assert_same_pixel_times(result: PixelBreaths, expected: PixelBreaths) -> None:
    """The global breaths and every pixel breath time are the same."""

    assert [(b.start_time, b.extremum_time, b.end_time) for b in result.breaths] == [
        (b.start_time, b.extremum_time, b.end_time) for b in expected.breaths
    ]
    for point in ("start_time", "extremum_time", "end_time"):
        np.testing.assert_array_equal(getattr(result, point), getattr(expected, point))


def test_pixel_tiv_step_gives_one_pixel_map_per_global_breath():
    session = M3Session()
    pixels, waveform = _breathing_images()

    result = pixel_tiv(
        eit_data=pixels,
        signal=waveform,
        breath_detector=BreathDetectionSettings(),
        session=session,
        tiv_method="mean",
    )

    assert set(get_step("eit.pixel_tiv").writes) == set(result)
    pixel_tiv_result = result["pixel_tiv_result"]
    breaths = _default_breaths(waveform)
    expected = pixel_tidal_impedance_variation(pixels, breaths, method="mean")
    assert len(pixel_tiv_result) == len(breaths) > 5
    assert all(isinstance(m, PixelMap) for m in pixel_tiv_result.values)
    assert pixel_tiv_result.values == expected.values
    assert pixel_tiv_result.unit == "a.u."
    assert pixel_tiv_result.method == expected.method
    metadata = pixel_tiv_result.metadata
    assert metadata["axes"] == ["row", "column"]
    assert metadata["tiv_timing"] == "continuous"
    assert metadata["tiv_method"] == "mean"
    assert metadata["valid_breath_indices"] == list(range(len(breaths)))
    assert metadata["source_package"] == "m3resp"
    assert metadata["parameters"]["breaths"] == "breath_detector"
    assert list(session.interval_data) == [pixel_tiv_result]


def test_pixel_tiv_accepts_unfiltered_pixel_data():
    """The input is any pixel signal; filtering is the default, not a demand."""

    definition = get_step("eit.pixel_tiv")
    assert definition.reads["eit_data"] == "filtered_eit"
    assert "filtered_eit" not in definition.reads


def test_pixel_tiv_keeps_the_given_global_breaths():
    session = M3Session()
    pixels, waveform = _breathing_images()
    breaths = _default_breaths(waveform)

    result = pixel_tiv(
        eit_data=pixels, signal=waveform, breaths=breaths, session=session
    )["pixel_tiv_result"]

    assert all(a is b for a, b in zip(result.intervals, breaths, strict=True))
    assert result.values == pixel_tidal_impedance_variation(pixels, breaths).values


def test_pixel_timing_with_and_without_pixel_breaths_gives_the_same_values():
    pixels, waveform = _breathing_images()
    session = M3Session()
    found = pixel_breaths(eit_data=pixels, timing_data=waveform, session=session)

    connected = pixel_tiv(
        eit_data=pixels,
        signal=waveform,
        pixel_breaths=found["pixel_breaths"],
        session=session,
        tiv_timing="pixel",
    )["pixel_tiv_result"]
    alone = pixel_tiv(
        eit_data=pixels,
        signal=waveform,
        breath_detector=BreathDetectionSettings(),
        session=M3Session(),
        tiv_timing="pixel",
    )["pixel_tiv_result"]

    assert connected.values == alone.values
    assert connected.metadata["tiv_timing"] == "pixel"
    # The first and the last global breath never have pixel breaths.
    assert connected.metadata["valid_breath_indices"] == list(
        range(1, len(connected) - 1)
    )


def test_the_continuous_timing_leaves_pixel_breaths_unused():
    pixels, waveform = _breathing_images()
    session = M3Session()
    found = pixel_breaths(eit_data=pixels, timing_data=waveform, session=session)
    breaths = _default_breaths(waveform)

    with_pixel_breaths = pixel_tiv(
        eit_data=pixels,
        signal=waveform,
        breaths=breaths,
        pixel_breaths=found["pixel_breaths"],
        session=session,
    )["pixel_tiv_result"]

    assert (
        with_pixel_breaths.values
        == pixel_tidal_impedance_variation(pixels, breaths).values
    )
    assert with_pixel_breaths.metadata["tiv_timing"] == "continuous"
    assert with_pixel_breaths.metadata["parameters"]["breaths"] == "input"


def test_pixel_tiv_refuses_pixel_breaths_of_another_type():
    pixels, waveform = _breathing_images()
    session = M3Session()
    found = pixel_breaths(eit_data=pixels, timing_data=waveform, session=session)

    with pytest.raises(TypeError, match="must be the PixelBreaths"):
        pixel_tiv(
            eit_data=pixels,
            signal=waveform,
            pixel_breaths=found["pixel_breath_timing_result"],
            session=session,
            tiv_timing="pixel",
        )


def test_pixel_timing_finds_pixel_breaths_from_eit_breaths_only():
    pixels, waveform = _breathing_images()
    emg_breaths = [
        dataclasses.replace(breath, modality="emg")
        for breath in _default_breaths(waveform)
    ]

    with pytest.raises(ValueError, match="EIT breaths only; got breaths of emg"):
        pixel_tiv(
            eit_data=pixels,
            signal=waveform,
            breaths=emg_breaths,
            session=M3Session(),
            tiv_timing="pixel",
        )


@pytest.mark.parametrize("step", [pixel_tiv, pixel_eeli])
def test_pixel_steps_need_breaths(step):
    pixels, waveform = _breathing_images()

    with pytest.raises(ValueError, match="needs the breaths"):
        step(eit_data=pixels, signal=waveform, session=M3Session())


@pytest.mark.parametrize(
    "step,timing", [(pixel_tiv, "tiv_timing"), (pixel_eeli, "eeli_timing")]
)
def test_pixel_steps_refuse_an_unknown_timing(step, timing):
    pixels, waveform = _breathing_images()

    with pytest.raises(ValueError, match=f"'{timing}' must be one of"):
        step(
            eit_data=pixels,
            signal=waveform,
            breath_detector=BreathDetectionSettings(),
            session=M3Session(),
            **{timing: "breath"},
        )


@pytest.mark.parametrize("timing", ["continuous", "pixel"])
def test_pixel_eeli_step_gives_the_impedance_of_each_pixel_at_the_breath_end(timing):
    session = M3Session()
    pixels, waveform = _breathing_images()

    result = pixel_eeli(
        eit_data=pixels,
        signal=waveform,
        breath_detector=BreathDetectionSettings(),
        session=session,
        eeli_timing=timing,
    )

    assert set(get_step("eit.pixel_eeli").writes) == set(result)
    pixel_eeli_result = result["pixel_eeli_result"]
    breaths = _default_breaths(waveform)
    timing_breaths = (
        breaths
        if timing == "continuous"
        else find_pixel_breaths(pixels, waveform, breaths)
    )
    expected = pixel_end_expiratory_lung_impedance(pixels, timing_breaths)
    assert pixel_eeli_result.values == expected.values
    assert pixel_eeli_result.name == "pixel_eelis"
    assert pixel_eeli_result.metadata["eeli_timing"] == timing
    assert pixel_eeli_result.metadata["operation"] == "eit.pixel_eeli"
    assert list(session.interval_data) == [pixel_eeli_result]


def test_pixel_breaths_accepts_the_empty_phase_correction_mode():
    """`null` in a spec / `None` from Python is a real option, not a mistake."""

    pixels, waveform = _breathing_images()

    result = pixel_breaths(
        eit_data=pixels,
        timing_data=waveform,
        session=M3Session(),
        phase_correction_mode=None,
    )

    assert set(get_step("eit.pixel_breaths").writes) == set(result)
    assert result["pixel_breaths"].metadata["phase_correction_mode"] is None


def test_pixel_breaths_rejects_an_unknown_phase_correction_mode():
    pixels, waveform = _breathing_images()

    with pytest.raises(ValueError, match="Unknown phase correction mode"):
        pixel_breaths(
            eit_data=pixels,
            timing_data=waveform,
            session=M3Session(),
            phase_correction_mode="nope",
        )


def test_pixel_breaths_step_gives_pixel_breaths_and_their_time_grids():
    session = M3Session()
    pixels, waveform = _breathing_images()
    stored_first_breath = _default_breaths(waveform)[0]
    session.add_events("eit_breaths", [stored_first_breath])

    result = pixel_breaths(eit_data=pixels, timing_data=waveform, session=session)

    found = result["pixel_breaths"]
    assert isinstance(found, PixelBreaths)
    _assert_same_pixel_times(
        found, find_pixel_breaths(pixels, waveform, _default_breaths(waveform))
    )
    assert found.metadata["operation"] == "eit.pixel_breaths"
    # A stored EIT breath with the same times is reused.
    assert found.breaths[0] is stored_first_breath
    timing = result["pixel_breath_timing_result"]
    assert isinstance(timing, IntervalData)
    assert timing.intervals == found.breaths
    assert timing.unit == "s"
    grids = np.stack(timing.values)
    assert grids.shape == (len(found.breaths), 2, 2, 3)
    np.testing.assert_array_equal(grids[..., 1], found.extremum_time)
    assert np.isnan(grids[0]).all()
    assert np.isnan(grids[:, 1, 1]).all()  # the NaN pixel has no breaths
    assert found.metadata["skipped_pixels"]["no_tiv"] == [[1, 1]]
    assert timing.metadata["landmarks"] == ["start_time", "extremum_time", "end_time"]
    assert timing.metadata["valid_pixel_breath_count"] == int(found.has_breath.sum())
    assert list(session.interval_data) == [timing]


def test_pixel_breaths_step_output_gives_breath_events_per_pixel():
    pixels, waveform = _breathing_images()
    timing = pixel_breaths(eit_data=pixels, timing_data=waveform, session=M3Session())[
        "pixel_breath_timing_result"
    ]

    events = pixel_breath_events(timing, row=1, column=0)

    # Pixel (1, 0) goes down during inspiration, so its breath starts at a
    # maximum of the impedance, one breath period after the global start.
    assert len(events) == len(timing) - 2
    assert events[0].metadata["global_breath_id"] == timing.intervals[1].id
    assert pixels.values[
        np.searchsorted(pixels.time, events[0].start_time), 1, 0
    ] == pytest.approx(11.0, abs=1e-3)


def test_pixel_breaths_step_uses_the_given_eit_breaths():
    pixels, waveform = _breathing_images()
    breaths = _default_breaths(waveform)[:-2]

    result = pixel_breaths(
        eit_data=pixels, timing_data=waveform, breaths=breaths, session=M3Session()
    )

    assert result["pixel_breaths"].breaths == breaths
    assert result["pixel_breaths"].metadata["parameters"]["breaths"] == "input"


def test_pixel_breaths_step_refuses_breaths_of_another_modality():
    pixels, waveform = _breathing_images()
    breaths = [
        dataclasses.replace(breath, modality="ventilator")
        for breath in _default_breaths(waveform)
    ]

    with pytest.raises(ValueError, match="EIT breaths only; got breaths of ventilator"):
        pixel_breaths(
            eit_data=pixels, timing_data=waveform, breaths=breaths, session=M3Session()
        )


def test_pixel_breaths_step_takes_the_sign_from_the_default_breaths():
    """eitprocessing takes the sign of each pixel from the breaths found with
    the default settings, also when 'minimum_duration_seconds' differs."""

    pixels, waveform = _breathing_images()

    result = pixel_breaths(
        eit_data=pixels,
        timing_data=waveform,
        session=M3Session(),
        minimum_duration_seconds=1.0,
    )

    _assert_same_pixel_times(
        result["pixel_breaths"],
        find_pixel_breaths(
            pixels,
            waveform,
            find_breaths(waveform, BreathDetectionSettings(minimum_duration=1.0)),
            sign_breaths=_default_breaths(waveform),
        ),
    )


def test_steps_on_the_same_breaths_share_one_detection_and_the_same_breaths():
    """EELI and pixel TIV over the same detector and signal find the breaths
    once and store the very same breath objects; a breath already stored in
    session.events is reused, so values can be traced to it."""

    session = _session_with_fake_adapter()
    raw = _FakeEITData(np.ones((3, 2, 2)), time=np.arange(3, dtype=float))
    waveform = _waveform([10.0, 11.0, 12.0])
    detector = _FakeBreathDetector(2)
    stored_first_breath = BreathEvent("eit", 0.0, 1.0, extremum_time=0.5)
    session.add_events("eit_breaths", [stored_first_breath])

    eeli_result = eeli(waveform, breath_detector=detector, session=session)[
        "eeli_result"
    ]
    tiv_result = pixel_tiv(
        eit_data=raw,
        signal=waveform,
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

    result = eeli(
        _waveform([10.0, 11.0, 12.0, 13.0]),
        breath_detector=_FakeBreathDetector(3),
        session=session,
    )

    assert len(result["eeli_result"]) == 3


def _lungspace_images() -> tuple[Signal, Signal]:
    """8 x 8 EIT images, 60 s at 20 Hz, one breath every 4 s, with the
    largest breathing amplitude at the centre and a NaN corner pixel; and
    their global impedance."""

    time = np.arange(1200) / 20.0
    rows, columns = np.mgrid[0:8, 0:8]
    amplitude = np.exp(-((rows - 3.5) ** 2 + (columns - 3.5) ** 2) / 6.0)
    breathing = 0.5 - 0.5 * np.cos(2 * np.pi * 0.25 * time)
    values = 10.0 + breathing[:, None, None] * amplitude
    values[:, 0, 0] = np.nan
    pixels = Signal(
        values=values,
        time=time,
        sample_frequency=20.0,
        name="images",
        modality="eit",
        category="impedance",
    )
    waveform = Signal(
        values=np.nansum(values, axis=(1, 2)),
        time=time,
        sample_frequency=20.0,
        name="global_impedance",
        modality="eit",
        category="impedance",
    )
    return pixels, waveform


_LUNGSPACE_STEPS = [
    (roi_tiv_lungspace, "eit.roi_tiv_lungspace", "threshold"),
    (roi_amplitude_lungspace, "eit.roi_amplitude_lungspace", "threshold"),
    (roi_watershed, "eit.roi_watershed", "threshold_fraction"),
]


@pytest.mark.parametrize("step_func,step_name,threshold_name", _LUNGSPACE_STEPS)
def test_roi_lungspace_steps_give_an_m3resp_mask(step_func, step_name, threshold_name):
    session = M3Session()
    pixels, waveform = _lungspace_images()

    result = step_func(
        eit_data=pixels, timing_data=waveform, session=session, **{threshold_name: 0.15}
    )

    assert set(get_step(step_name).writes) == set(result) - {"_provenance"}
    [result_key] = [k for k in result if k.endswith("_result")]
    mask = result[result_key]
    assert isinstance(mask, PixelMask)
    assert mask.shape == (8, 8)
    assert np.isnan(mask.values[0, 0])
    assert mask.values[3, 3] == 1.0
    assert mask.metadata["source_package"] == "m3resp"
    assert mask.metadata["parameters"] == {
        threshold_name: 0.15,
        "breaths": "timing_data",
    }
    assert list(session.pixel_masks) == [mask]
    assert len(session.parameter_results) == 0
    assert get_step(step_name).optional_packages == ()


@pytest.mark.parametrize("step_func,step_name,threshold_name", _LUNGSPACE_STEPS)
def test_roi_lungspace_steps_with_eit_breaths_equal_the_steps_without(
    step_func, step_name, threshold_name
):
    """Breaths found with the default settings and given through 'breaths'
    give the same mask as the breaths the step finds itself."""

    pixels, waveform = _lungspace_images()
    breaths = find_breaths(waveform, BreathDetectionSettings())

    found = step_func(eit_data=pixels, timing_data=waveform, session=M3Session())
    given = step_func(
        eit_data=pixels, timing_data=waveform, session=M3Session(), breaths=breaths
    )

    [result_key] = [k for k in found if k.endswith("_result")]
    np.testing.assert_array_equal(given[result_key].values, found[result_key].values)
    assert given[result_key].metadata["parameters"]["breaths"] == "input"


@pytest.mark.parametrize(
    "step_func",
    [roi_amplitude_lungspace, roi_watershed],
    ids=["amplitude", "watershed"],
)
def test_amplitude_and_watershed_steps_refuse_breaths_of_another_modality(step_func):
    pixels, waveform = _lungspace_images()
    breaths = [
        dataclasses.replace(breath, modality="ventilator")
        for breath in find_breaths(waveform, BreathDetectionSettings())
    ]

    with pytest.raises(ValueError, match="EIT breaths only; got breaths of ventilator"):
        step_func(
            eit_data=pixels, timing_data=waveform, session=M3Session(), breaths=breaths
        )


def test_tiv_lungspace_step_takes_breaths_of_another_modality():
    """Breaths of another modality are moved to the EIT clock."""

    session = M3Session()
    pixels, waveform = _lungspace_images()
    breaths = find_breaths(waveform, BreathDetectionSettings())
    session.start_times["eit"] = 0.0
    session.start_times["ventilator"] = 1.0
    ventilator = [
        dataclasses.replace(
            breath,
            modality="ventilator",
            start_time=breath.start_time - 1.0,
            end_time=breath.end_time - 1.0,
            extremum_time=breath.extremum_time - 1.0,
        )
        for breath in breaths
    ]

    with pytest.warns(UserWarning):
        moved = roi_tiv_lungspace(
            eit_data=pixels, timing_data=waveform, session=session, breaths=ventilator
        )
    found = roi_tiv_lungspace(
        eit_data=pixels, timing_data=waveform, session=M3Session()
    )

    np.testing.assert_array_equal(
        moved["tiv_lungspace_result"].values, found["tiv_lungspace_result"].values
    )


@pytest.mark.parametrize("bad_threshold", [0.0, 1.0, -0.1, 1.5])
def test_roi_lungspace_steps_reject_out_of_range_threshold(bad_threshold):
    pixels, waveform = _lungspace_images()

    with pytest.raises(ValueError, match="between 0 and 1"):
        roi_tiv_lungspace(
            eit_data=pixels,
            timing_data=waveform,
            session=M3Session(),
            threshold=bad_threshold,
        )


def test_roi_filter_by_size_takes_a_mask_or_a_grid():
    mask = np.full((4, 4), np.nan)
    mask[1:3, 1:3] = 1.0
    mask[0, 0] = 1.0

    from_mask = roi_filter_by_size(
        PixelMask(name="watershed_lungspace_mask", values=mask),
        session=M3Session(),
        min_region_size=2,
    )["size_filtered_roi_result"]
    from_grid = roi_filter_by_size(mask, session=M3Session(), min_region_size=2)[
        "size_filtered_roi_result"
    ]

    np.testing.assert_array_equal(from_mask.values, from_grid.values)
    assert np.isnan(from_mask.values[0, 0]), "isolated pixel should be dropped"
    assert from_mask.name == "size_filtered_roi_mask"
    assert from_mask.metadata["source_mask"] == "watershed_lungspace_mask"


def test_roi_filter_by_size_refuses_other_types():
    with pytest.raises(TypeError, match="PixelMask or a 2D grid"):
        roi_filter_by_size(SimpleNamespace(mask=np.ones((2, 2))), session=M3Session())


def test_adapter_filter_roi_by_size_takes_every_mask_form():
    """The adapter also takes an eitprocessing mask, and gives an m3resp mask."""

    pytest.importorskip("eitprocessing")
    from eitprocessing.roi import PixelMask as UpstreamPixelMask

    from m3resp.adapters import EITProcessingAdapter

    mask = np.full((4, 4), np.nan)
    mask[1:3, 1:3] = 1.0
    mask[0, 0] = 1.0
    adapter = EITProcessingAdapter()

    results = [
        adapter.filter_roi_by_size(form, min_region_size=2)
        for form in (PixelMask(name="m", values=mask), UpstreamPixelMask(mask), mask)
    ]

    for result in results:
        assert isinstance(result, PixelMask)
        np.testing.assert_array_equal(result.values, results[0].values)
    assert np.isnan(results[0].values[0, 0])


def test_roi_filter_by_size_rejects_non_positive_min_region_size():
    mask = PixelMask(name="m", values=np.array([[1.0, np.nan], [np.nan, 1.0]]))

    with pytest.raises(ValueError, match="must be positive"):
        roi_filter_by_size(mask, session=M3Session(), min_region_size=0)


def test_roi_filter_by_size_step_matches_declared_writes():
    mask = PixelMask(name="m", values=np.array([[1.0, np.nan], [np.nan, 1.0]]))

    result = roi_filter_by_size(mask, session=M3Session(), min_region_size=1)

    assert set(get_step("eit.roi_filter_by_size").writes) == set(result) - {
        "_provenance"
    }


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

    with pytest.raises(WorkflowSpecError, match="explicit 'in:"):
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

    with pytest.raises(WorkflowSpecError, match="already produced"):
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
                "in": {"eit_data": "eit_data", "timing_data": "timing_data"},
                "with": {"threshold": 0.1},
                "out": {
                    "tiv_lungspace_captures": "captures_a",
                    "tiv_lungspace_result": "result_a",
                },
            },
            {
                "uses": "eit.roi_tiv_lungspace",
                "in": {"eit_data": "eit_data", "timing_data": "timing_data"},
                "with": {"threshold": 0.2},
                "out": {
                    "tiv_lungspace_captures": "captures_b",
                    "tiv_lungspace_result": "result_b",
                },
            },
        ],
    }
    pixels, waveform = _lungspace_images()

    result = run_workflow(
        spec,
        session=session,
        extra_context={"eit_data": pixels, "timing_data": waveform},
    )

    assert result.value("result_a") is not result.value("result_b")
    assert result.value("result_a").metadata["parameters"]["threshold"] == 0.1
    assert result.value("result_b").metadata["parameters"]["threshold"] == 0.2


# -- full example end to end (needs the real optional dependency) ----------


def test_full_eit_example_workflow_runs_end_to_end(tmp_path):
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
        repo_root, "examples", "eit_full_preprocessing", "eit-full.workflow.yaml"
    )
    # run_workflow (unlike run_spec) does not touch the spec's `outputs:`
    # section, so this exercises the example without writing into the
    # project's real output/ directory.
    result = run_workflow(spec_path, session=M3Session())

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
