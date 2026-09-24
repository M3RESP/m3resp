"""Keeping only a time window of the loaded EMG recording (`emg.slice`)."""

from __future__ import annotations

import numpy as np
import pytest

from m3resp import M3Session
from m3resp.adapters import ReSurfEMGAdapter
from m3resp.core.exceptions import MissingModalityDataError
from m3resp.workflows.steps.emg.loading import slice_recording

FS = 100.0
DURATION_SECONDS = 60.0


def _recording() -> dict:
    """A Biopac-like export: channel 0 airway pressure, channel 1 EMG.

    Each sample holds its own time in seconds, so what is kept is easy to check.
    """

    time = np.arange(int(DURATION_SECONDS * FS)) / FS
    return {
        "array": np.vstack([time, time + 1000.0]),
        "dataframe": None,
        "metadata": {"fs": FS, "labels": ["Paw", "EMGdi"], "units": ["cmH2O", "uV"]},
    }


def _session() -> M3Session:
    # Every load reads the file again, as the real loader does.
    session = M3Session(
        emg_adapter=ReSurfEMGAdapter(loader=lambda *args, **kwargs: _recording())
    )
    session.load_emg("study.txt")
    return session


def test_slice_keeps_only_the_window():
    session = _session()

    summary = session.slice_emg(10.0, 25.0)

    array = session.emg.data["array"]
    assert array.shape[1] == int(15.0 * FS)
    assert array[0, 0] == pytest.approx(10.0)
    assert array[0, -1] == pytest.approx(25.0 - 1 / FS)
    assert summary["kept_samples"] == int(15.0 * FS)
    assert session.parameters["emg_slice"] == summary


def test_without_an_end_the_rest_of_the_recording_is_kept():
    session = _session()

    session.slice_emg(50.0)

    array = session.emg.data["array"]
    assert array.shape[1] == int(10.0 * FS)
    assert array[0, -1] == pytest.approx(DURATION_SECONDS - 1 / FS)


def test_airway_pressure_from_the_same_file_is_cut_the_same_way():
    """Paw and EMG share one clock, so they must stay lined up."""

    session = _session()
    session.load_ventilator("study.txt")

    session.slice_emg(10.0, 25.0)

    emg = session.emg.data["array"]
    ventilator = session.ventilator.data["array"]
    np.testing.assert_array_equal(ventilator, emg)


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1.0, 10.0), (20.0, 10.0), (10.0, 10.0), (0.0, DURATION_SECONDS + 1.0)],
)
def test_a_window_outside_the_recording_is_refused(start, end):
    session = _session()

    with pytest.raises(ValueError, match="start < end"):
        session.slice_emg(start, end)

    assert session.emg.data["array"].shape[1] == int(DURATION_SECONDS * FS)


def test_slicing_needs_a_loaded_recording():
    with pytest.raises(MissingModalityDataError):
        M3Session().slice_emg(0.0, 1.0)


def test_step_calls_the_session_method():
    session = _session()

    result = slice_recording(session, start_seconds=5.0)

    assert result["emg_slice"]["start_seconds"] == 5.0
    assert session.emg.data["array"].shape[1] == int(55.0 * FS)
