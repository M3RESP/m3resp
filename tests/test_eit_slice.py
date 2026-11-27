"""Keeping only a time window of the loaded EIT recording (`eit.slice_recording`).

The images, waveforms, device marks and breaths are cut together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from m3resp import M3Session
from m3resp.core.exceptions import MissingModalityDataError
from m3resp.data import Signal
from m3resp.eit.io import EITFileData
from m3resp.synchronization.raw_traces import raw_synchronization_traces
from m3resp.synchronization.start_times import shared_clock_shift
from m3resp.workflows.steps.eit.slicing import slice_recording

FS = 20.0
N_FRAMES = 200
#: A Draeger file's time axis is the time of day, not 0.
FIRST_TIME = 36528.5


def _file_data() -> EITFileData:
    """Ten seconds of EIT at 20 Hz with airway pressure, flow and volume.

    Every pixel of frame `i` holds `i`, and the ventilator channels hold the
    frame's time, so what is kept is easy to check.
    """

    time = FIRST_TIME + np.arange(N_FRAMES) / FS
    pixels = np.broadcast_to(
        np.arange(N_FRAMES, dtype=float)[:, None, None], (N_FRAMES, 32, 32)
    ).copy()
    return EITFileData(
        vendor="draeger",
        paths=(Path("synthetic.bin"),),
        sample_frequency=FS,
        pixel_impedance=Signal(
            values=pixels,
            time=time,
            sample_frequency=FS,
            name="raw",
            modality="eit",
            channel="pixel_impedance",
            processing_state="raw",
            source="draeger",
        ),
        signals={
            label: Signal(
                values=time.copy(),
                time=time,
                sample_frequency=FS,
                unit=unit,
                name=label,
                modality="eit",
                source="draeger",
            )
            for label, unit in (
                ("airway pressure", "cmH2O"),
                ("flow", "L/min"),
                ("volume", "mL"),
            )
        },
    )


def _session() -> M3Session:
    # Every load reads the file again, as the real loader does.
    session = M3Session(eit_loader=lambda *args, **kwargs: _file_data())
    session.load_eit("study.bin")
    return session


def test_slice_keeps_only_the_window():
    session = _session()

    summary = session.slice_eit(2.0, 5.0)

    recording = session.eit
    assert recording.pixel_impedance.values.shape == (60, 32, 32)
    assert recording.pixel_impedance.values[0, 0, 0] == 40.0
    assert recording.pixel_impedance.time[0] == pytest.approx(FIRST_TIME + 2.0)
    assert len(recording.raw_global_impedance.time) == 60
    assert len(recording.signals["airway pressure"].values) == 60
    assert summary == {
        "start_seconds": 2.0,
        "end_seconds": 5.0,
        "removed_frames_start": 40,
        "removed_frames_end": 100,
        "kept_frames": 60,
        "ventilator_recordings_cut": 0,
    }
    assert session.parameters["eit_slice"] == summary
    assert session.provenance[-1].action == "slice_eit"


def test_without_an_end_the_rest_of_the_recording_is_kept():
    session = _session()

    session.slice_eit(8.0)

    assert session.eit.pixel_impedance.values.shape[0] == 40
    assert session.eit.pixel_impedance.time[-1] == pytest.approx(FIRST_TIME + 9.95)


def test_the_eit_stays_lined_up_with_the_other_recordings():
    # Frame times keep their time-of-day values, and the EIT start time moves
    # later by the part cut off, so an EIT breath lands on the same place on
    # the shared clock before and after slicing.
    session = _session()
    shift_before = shared_clock_shift(session, "eit")

    session.slice_eit(2.0, 5.0)

    assert session.start_times["eit"] == pytest.approx(2.0)
    assert shared_clock_shift(session, "eit") == pytest.approx(shift_before)


def test_a_ventilator_from_the_same_eit_file_is_cut_at_the_same_frames():
    session = _session()
    session.load_ventilator("study.bin", source="eit")

    summary = session.slice_eit(2.0, 5.0)

    payload = session.ventilator.data
    assert payload["array"].shape == (3, 60)
    assert payload["array"][0, 0] == pytest.approx(FIRST_TIME + 2.0)
    assert payload["metadata"]["time"][0] == pytest.approx(FIRST_TIME + 2.0)
    assert session.raw["vent"] is session.raw["ventilator"]
    assert summary["ventilator_recordings_cut"] == 1


def test_frames_are_chosen_by_their_time_stamps():
    # Draeger time stamps are not perfectly even, so counting frames from the
    # sampling rate would drift. The first kept frame is the first one at or
    # after the start time.
    session = _session()
    uneven = np.asarray(session.eit.pixel_impedance.time, dtype=float).copy()
    uneven[40] += 0.01  # frame 40 now lies just after 2.0 s + 0.01 s

    session.eit.pixel_impedance.time = uneven
    session.slice_eit(2.005)

    assert session.eit.pixel_impedance.values[0, 0, 0] == 40.0


def test_slice_must_run_before_preprocessing():
    session = _session()
    session.processed_variants["eit"]["default"] = {}

    with pytest.raises(ValueError, match="before preprocess_eit"):
        session.slice_eit(2.0, 5.0)


@pytest.mark.parametrize(("start", "end"), [(-1.0, 5.0), (5.0, 2.0), (2.0, 11.0)])
def test_a_window_outside_the_recording_is_refused(start, end):
    session = _session()

    with pytest.raises(ValueError, match="slice_eit"):
        session.slice_eit(start, end)


def test_slice_needs_a_loaded_recording():
    with pytest.raises(MissingModalityDataError):
        M3Session().slice_eit(1.0)


def test_the_image_sum_is_computed_once_and_follows_a_slice():
    session = _session()
    recording = session.eit

    summed = recording.raw_global_impedance

    assert recording.raw_global_impedance is summed
    np.testing.assert_array_equal(
        summed.values, np.nansum(recording.pixel_impedance.values, axis=(1, 2))
    )
    session.slice_eit(2.0, 5.0)
    assert recording.raw_global_impedance is not summed
    assert len(recording.raw_global_impedance.values) == 60
    assert recording.raw_global_impedance.values[0] == 40.0 * 32 * 32


def test_the_raw_sync_trace_has_the_eitprocessing_label():
    traces = raw_synchronization_traces(_session(), "eit")

    assert traces["eit"]["ylabel"] == "global_impedance_(raw)"
    assert len(traces["eit"]["values"]) == N_FRAMES


def test_step_delegates_to_the_session():
    session = _session()

    result = slice_recording(session, start_seconds=2.0, end_seconds=5.0)

    assert result["eit_slice"] == session.parameters["eit_slice"]


def test_steps_after_the_slice_step_read_the_cut_recording(tmp_path):
    from m3resp.workflows import load_spec, run_workflow

    (tmp_path / "study.bin").write_bytes(b"")
    session = M3Session(eit_loader=lambda *args, **kwargs: _file_data())
    spec = {
        "name": "cut_then_sum",
        "steps": [
            {"uses": "eit.load_custom", "with": {"file_path": "study.bin"}},
            {
                "uses": "eit.slice_recording",
                "with": {"start_seconds": 2.0, "end_seconds": 5.0},
            },
            {"uses": "eit.global_impedance", "in": {"signal": "raw_eit"}},
        ],
    }

    context = run_workflow(load_spec(spec, root=tmp_path), session=session).context

    # Frame i holds i in every pixel, so the cut part starts at frame 40.
    raw = context.get("raw_eit")
    assert raw.values.shape[0] == 60
    assert raw.values[0, 0, 0] == 40.0
    assert len(context.get("global_impedance").values) == 60
    raw_signals = [
        signal
        for signal in session.signals.items
        if signal.modality == "eit" and signal.processing_state == "raw"
    ]
    assert sorted(signal.channel for signal in raw_signals) == [
        "global_impedance",
        "pixel_impedance",
    ]
    assert all(len(signal.values) == 60 for signal in raw_signals)
