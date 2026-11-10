"""Workflow tests of the EIT steps: their outputs, the declared writes and
the step inputs. The steps run m3resp code only, on small m3resp Signals.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

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


def _images(frames: int, *, rows: int = 2, columns: int = 2, fs: float = 1.0) -> Signal:
    """EIT images with every pixel 1.0, ``fs`` frames per second from 0 s."""

    return Signal(
        values=np.ones((frames, rows, columns)),
        time=np.arange(frames) / fs,
        sample_frequency=fs,
        name="raw",
        modality="eit",
        category="impedance",
        channel="pixel_impedance",
        processing_state="raw",
        source="draeger",
    )


@pytest.fixture
def fixed_breaths(monkeypatch) -> dict[str, Any]:
    """Make the breath search of the TIV and EELI steps give fixed breaths.

    The breaths are ``(i, i + 0.5, i + 1)`` s for ``i`` below ``count``; set
    ``state["count"]`` before the step runs. ``state["calls"]`` counts the
    searches.
    """

    state: dict[str, Any] = {"count": 0, "calls": 0}

    def find(signal, settings):
        state["calls"] += 1
        return [
            BreathEvent("eit", float(i), float(i + 1), extremum_time=i + 0.5)
            for i in range(state["count"])
        ]

    monkeypatch.setattr("m3resp.workflows.steps.eit._shared.find_breaths", find)
    return state


@pytest.fixture
def fake_reader(monkeypatch) -> dict[str, Any]:
    """Replace the vendor reader of ``eit.load`` with one that reads no file.

    Returns:
        dict[str, Any]: The keyword arguments of the last read.
    """

    from m3resp.eit.io import EITFileData

    calls: dict[str, Any] = {}

    def read(path, vendor, **kwargs):
        calls.clear()
        calls.update(kwargs)
        time = np.arange(6, dtype=float)
        return EITFileData(
            vendor=vendor,
            paths=(Path(path),),
            sample_frequency=1.0,
            pixel_impedance=Signal(
                values=np.ones((6, 2, 2)),
                time=time,
                sample_frequency=1.0,
                name="raw",
                modality="eit",
                category="impedance",
                channel="pixel_impedance",
                processing_state="raw",
                source=vendor,
            ),
        )

    monkeypatch.setattr("m3resp.modalities.eit.read_eit_file", read)
    return calls


def test_load_step_works_without_eitprocessing_and_matches_declared_writes(
    fake_reader,
):
    session = M3Session()

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


def test_load_step_forwards_the_declared_reading_parameters(fake_reader):
    """sample_frequency, first_frame and max_frames reach the vendor reader."""

    session = M3Session()

    load(
        session,
        file_path="fake.bin",
        vendor="draeger",
        sample_frequency=20.0,
        first_frame=5,
        max_frames=100,
    )

    forwarded = fake_reader
    assert forwarded["sample_frequency"] == 20.0
    assert forwarded["first_frame"] == 5
    assert forwarded["max_frames"] == 100


def test_load_step_omits_unset_reading_parameters(fake_reader):
    """Unset means 'let the vendor loader decide', not 'pass None'."""

    session = M3Session()

    load(session, file_path="fake.bin", vendor="draeger")

    forwarded = fake_reader
    assert "sample_frequency" not in forwarded
    assert "max_frames" not in forwarded
    assert forwarded["first_frame"] == 0


def _fixed_rates(monkeypatch, respiratory_rate_hz: float, heart_rate_hz: float) -> None:
    """Make ``eit.detect_rates`` find the given rates."""

    monkeypatch.setattr(
        "m3resp.workflows.steps.eit.rates.detect_rates_of",
        lambda *args, **kwargs: (respiratory_rate_hz, heart_rate_hz),
    )


def test_detect_rates_step_works_without_eitprocessing_and_matches_declared_writes(
    monkeypatch,
):
    _fixed_rates(monkeypatch, 0.3, 1.2)
    session = M3Session()

    result = detect_rates(_images(4, rows=1, columns=1), session=session)

    assert set(get_step("eit.detect_rates").writes) <= set(result)
    assert result["respiratory_rate_hz"] == pytest.approx(0.3)
    names = {p.name for p in session.parameter_results}
    assert {"respiratory_rate", "heart_rate"} <= names


@pytest.mark.parametrize("rate_name", ["respiratory_rate_hz", "heart_rate_hz"])
def test_detect_rates_reports_a_missing_rate_as_such(rate_name, monkeypatch):
    """NaN means no rate could be estimated; the message says so."""

    rates = {"respiratory_rate_hz": 0.3, "heart_rate_hz": 1.0, rate_name: float("nan")}
    _fixed_rates(monkeypatch, rates["respiratory_rate_hz"], rates["heart_rate_hz"])

    with pytest.raises(ValueError, match="could not estimate"):
        detect_rates(_images(4, rows=1, columns=1), session=M3Session())


@pytest.mark.parametrize(
    "step",
    [
        lambda signal, session: detect_rates(signal, session=session),
        lambda signal, session: mdn_filter(
            signal, respiratory_rate_hz=0.3, heart_rate_hz=1.2, session=session
        ),
        lambda signal, session: butterworth_filter(signal, session=session),
    ],
    ids=["eit.detect_rates", "eit.mdn_filter", "eit.butterworth_filter"],
)
def test_image_steps_refuse_images_that_are_not_a_signal(step):
    images = SimpleNamespace(pixel_impedance=np.ones((4, 2, 2)), time=np.arange(4.0))

    with pytest.raises(TypeError, match="'signal' must be an m3resp Signal"):
        step(images, M3Session())


def test_mdn_filter_step_matches_declared_writes():
    session = M3Session()
    raw = _images(400, fs=20.0)

    result = mdn_filter(
        raw,
        respiratory_rate_hz=0.3,
        heart_rate_hz=1.2,
        session=session,
        label="mdn_filtered",
    )

    assert set(get_step("eit.mdn_filter").writes) <= set(result)
    assert result["filtered_eit_signal"].channel == "pixel_impedance"
    assert result["filtered_eit"] is result["filtered_eit_signal"]
    assert result["filtered_eit"].metadata["operation"] == "eit.mdn_filter"
    assert result["filtered_eit"].metadata["source_package"] == "m3resp"
    assert result["filtered_eit"].metadata["label"] == "mdn_filtered"
    assert result["filtered_eit"].name == "MDN-filtered EIT data (mdn_filtered)"
    assert "eitprocessing_fields" not in result["filtered_eit"].metadata
    assert _infer_filter_mode(result["filtered_eit"]) == "mdn"


@pytest.mark.parametrize("rate", [0.0, -0.1, float("inf"), float("nan")])
def test_mdn_filter_step_refuses_a_rate_that_is_not_above_zero(rate):
    with pytest.raises(ValueError, match="rate"):
        mdn_filter(
            _images(400, fs=20.0),
            respiratory_rate_hz=rate,
            heart_rate_hz=1.2,
            session=M3Session(),
        )


@pytest.mark.parametrize(
    ("metadata", "mode"),
    [
        ({"operation": "eit.mdn_filter"}, "mdn"),
        (
            {"operation": "eit.butterworth_filter", "parameters": {"mode": "bandstop"}},
            "bandstop",
        ),
        ({}, "none"),
    ],
)
def test_the_filter_mode_of_a_workflow_comes_from_the_filtered_images(
    metadata: dict, mode: str
):
    filtered = SimpleNamespace(metadata=metadata)

    assert _infer_filter_mode(filtered) == mode
    assert _infer_filter_mode(None) == "none"


@pytest.mark.parametrize("mode", ["lowpass", "highpass", "bandpass", "bandstop"])
def test_butterworth_filter_step_supports_every_mode_and_matches_declared_writes(mode):
    session = M3Session()
    raw = _images(40, fs=20.0)

    result = butterworth_filter(
        raw,
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


def test_butterworth_filter_step_accepts_an_already_filtered_signal():
    """The signal to filter is bound explicitly, not pinned to raw_eit."""

    session = M3Session()
    raw = _images(40, fs=20.0)

    once = butterworth_filter(raw, session=session, label="pass_one")
    twice = butterworth_filter(once["filtered_eit"], session=session, label="pass_two")

    assert twice["filtered_eit"].metadata["label"] == "pass_two"
    assert twice["filter_captures"]["filtered_data"] is twice["filtered_eit"]
    assert get_step("eit.butterworth_filter").reads["signal"] is None


def _waveform(values: list[float]) -> Signal:
    """A global impedance waveform, one sample per second from 0 s."""

    return Signal(
        values=np.asarray(values, dtype=float),
        time=np.arange(len(values), dtype=float),
        sample_frequency=1.0,
        name="global_impedance",
        unit="a.u.",
        modality="eit",
        category="impedance",
        channel="global_impedance",
    )


def test_eeli_step_stores_one_value_per_breath(fixed_breaths):
    fixed_breaths["count"] = 3
    session = M3Session()
    waveform = _waveform([10.0, 11.0, 12.0, 13.0])

    result = eeli(waveform, breath_detector=BreathDetectionSettings(), session=session)

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


def test_continuous_tiv_step_uses_the_chosen_method(fixed_breaths):
    fixed_breaths["count"] = 2  # breaths (0, 0.5, 1) and (1, 1.5, 2)
    session = M3Session()
    waveform = _waveform([1.0, 3.0, 2.0, 6.0, 4.0])
    detector = BreathDetectionSettings()

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


def test_eeli_step_gives_nan_for_breaths_outside_the_data(fixed_breaths):
    fixed_breaths["count"] = 4
    session = M3Session()
    waveform = _waveform([10.0, 11.0, 12.0, 13.0])

    with pytest.warns(UserWarning, match="1 of 4 breaths lie partly outside"):
        result = eeli(
            waveform, breath_detector=BreathDetectionSettings(), session=session
        )

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


def test_steps_on_the_same_breaths_share_one_detection_and_the_same_breaths(
    fixed_breaths,
):
    """EELI and pixel TIV over the same detector and signal find the breaths
    once and store the very same breath objects; a breath already stored in
    session.events is reused, so values can be traced to it."""

    fixed_breaths["count"] = 2
    session = M3Session()
    raw = _images(3)
    waveform = _waveform([10.0, 11.0, 12.0])
    detector = BreathDetectionSettings()
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

    assert fixed_breaths["calls"] == 1
    assert all(
        first is second
        for first, second in zip(eeli_result.intervals, tiv_result.intervals)
    )
    assert eeli_result.intervals[0] is stored_first_breath


def test_steps_ignore_stored_eit_breaths_that_are_not_breaths(fixed_breaths):
    """`add_events` accepts anything; a marker stored under "eit_breaths"
    must not break the steps that reuse stored breaths."""

    fixed_breaths["count"] = 3
    session = M3Session()
    session.add_events("eit_breaths", [Event(name="marker", modality="eit", time=0.0)])

    result = eeli(
        _waveform([10.0, 11.0, 12.0, 13.0]),
        breath_detector=BreathDetectionSettings(),
        session=session,
    )

    assert len(result["eeli_result"]) == 3


def test_tiv_and_eeli_steps_refuse_a_breath_detector_of_another_type():
    with pytest.raises(TypeError, match="BreathDetectionSettings"):
        eeli(
            _waveform([10.0, 11.0, 12.0, 13.0]),
            breath_detector=SimpleNamespace(find_breaths=lambda signal: []),
            session=M3Session(),
        )


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
    from reference.eitprocessing import EITProcessingAdapter

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
    session = M3Session()
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


# -- full example end to end ------------------------------------------------


def test_full_eit_example_workflow_runs_end_to_end(tmp_path):
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
