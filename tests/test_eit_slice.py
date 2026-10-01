"""Keeping only a time window of the loaded EIT recording (`eit.slice_recording`).

The cutting is done by `eitprocessing`, so these tests build a small real
`Sequence` rather than a stand-in.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("eitprocessing")

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.sequence import Sequence

from m3resp import M3Session
from m3resp.adapters import EITProcessingAdapter
from m3resp.core.exceptions import MissingModalityDataError
from m3resp.synchronization.start_times import shared_clock_shift
from m3resp.workflows.steps.eit.slicing import slice_recording

FS = 20.0
N_FRAMES = 200
#: A Draeger file's time axis is the time of day, not 0.
FIRST_TIME = 36528.5


def _sequence() -> Sequence:
    """Ten seconds of EIT at 20 Hz with airway pressure, flow and volume.

    Every pixel of frame `i` holds `i`, and the ventilator channels hold the
    frame's time, so what is kept is easy to check.
    """

    time = FIRST_TIME + np.arange(N_FRAMES) / FS
    pixels = np.broadcast_to(
        np.arange(N_FRAMES, dtype=float)[:, None, None], (N_FRAMES, 32, 32)
    ).copy()
    eit_data = DataCollection(EITData)
    eit_data.add(
        EITData(
            path="synthetic.bin",
            nframes=N_FRAMES,
            time=time,
            sample_frequency=FS,
            vendor=Vendor.DRAEGER,
            label="raw",
            pixel_impedance=pixels,
        )
    )
    continuous = DataCollection(ContinuousData)
    for label, unit, category in (
        ("airway pressure", "cmH2O", "pressure"),
        ("flow", "L/min", "flow"),
        ("volume", "mL", "volume"),
    ):
        continuous.add(
            ContinuousData(
                label=label,
                name=label,
                unit=unit,
                category=category,
                time=time,
                values=time.copy(),
                sample_frequency=FS,
            )
        )
    return Sequence(label="synthetic", eit_data=eit_data, continuous_data=continuous)


def _session() -> M3Session:
    # Every load reads the file again, as the real loader does.
    session = M3Session(
        eit_adapter=EITProcessingAdapter(loader=lambda *args, **kwargs: _sequence())
    )
    session.load_eit("study.bin")
    return session


def test_slice_keeps_only_the_window():
    session = _session()

    summary = session.slice_eit(2.0, 5.0)

    recording = session.eit
    assert recording.raw.pixel_impedance.shape == (60, 32, 32)
    assert recording.raw.pixel_impedance[0, 0, 0] == 40.0
    assert recording.raw.time[0] == pytest.approx(FIRST_TIME + 2.0)
    assert len(recording.global_impedance.time) == 60
    assert len(recording.data.continuous_data["airway pressure"].values) == 60
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

    assert session.eit.raw.pixel_impedance.shape[0] == 40
    assert session.eit.raw.time[-1] == pytest.approx(FIRST_TIME + 9.95)


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
    uneven = np.asarray(session.eit.data.time, dtype=float).copy()
    uneven[40] += 0.01  # frame 40 now lies just after 2.0 s + 0.01 s

    session.eit.data.eit_data["raw"].time = uneven
    session.slice_eit(2.005)

    assert session.eit.raw.pixel_impedance[0, 0, 0] == 40.0


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


def test_step_delegates_to_the_session():
    session = _session()

    result = slice_recording(session, start_seconds=2.0, end_seconds=5.0)

    assert result["eit_slice"] == session.parameters["eit_slice"]
