"""Keeping only a time window of standalone ventilator recordings
(`ventilator.slice_recording`)."""

from __future__ import annotations

import numpy as np
import pytest

from m3resp import M3Session
from m3resp.adapters import ReSurfEMGAdapter
from m3resp.core.exceptions import MissingModalityDataError
from m3resp.synchronization.start_times import shared_clock_shift
from m3resp.workflows.steps.ventilator.loading import load
from m3resp.workflows.steps.ventilator.slicing import slice_recording

FS = 10.0
DURATION_SECONDS = 10.0


def _recording() -> dict:
    """Pressure, flow and volume that each hold their own time in seconds, so
    what is kept is easy to check."""

    time = np.arange(int(DURATION_SECONDS * FS)) / FS
    return {
        "array": np.vstack([time, time, time]),
        "metadata": {"fs": FS, "labels": ["pressure", "flow", "volume"]},
    }


def _session() -> M3Session:
    # Every load reads the file again, as the real loader does.
    return M3Session(
        emg_adapter=ReSurfEMGAdapter(loader=lambda *args, **kwargs: _recording())
    )


def _standalone_session() -> M3Session:
    session = _session()
    session.load_ventilator("ventilator.txt", source="ventilator")
    return session


def test_slice_keeps_only_the_window():
    session = _standalone_session()

    summary = session.slice_ventilator(2.0, 5.0)

    array = session.ventilator.data["array"]
    assert array.shape == (3, 30)
    assert array[0, 0] == pytest.approx(2.0)
    assert array[0, -1] == pytest.approx(4.9)
    assert session.ventilator.raw is array
    assert summary == {
        "start_seconds": 2.0,
        "end_seconds": 5.0,
        "recordings": {
            "default": {
                "removed_samples_start": 20,
                "removed_samples_end": 50,
                "kept_samples": 30,
            }
        },
    }
    assert session.parameters["ventilator_slice"] == summary
    assert session.provenance[-1].action == "slice_ventilator"


def test_without_an_end_the_rest_of_the_recording_is_kept():
    session = _standalone_session()

    session.slice_ventilator(8.0)

    assert session.ventilator.data["array"][0, -1] == pytest.approx(9.9)
    assert session.ventilator.data["array"].shape[1] == 20


def test_the_ventilator_stays_lined_up_with_the_other_recordings():
    # A ventilator breath at 4 s before slicing is at 2 s after slicing from
    # 2 s. The recording's start time moves later by 2 s, so the breath stays
    # at the same place on the shared clock. The new start time is stored
    # under the recording's own entry; the shared entry is left alone.
    session = _standalone_session()
    session.synchronize_raw_modalities(
        offset_seconds={"ventilator": 1.0}, reference_modality="eit"
    )
    shift_before = shared_clock_shift(session, "ventilator")

    session.slice_ventilator(2.0)

    assert session.start_times["ventilator:default"] == pytest.approx(3.0)
    assert session.start_times["ventilator"] == pytest.approx(1.0)
    assert shared_clock_shift(session, "ventilator") == pytest.approx(
        shift_before + 2.0
    )


def test_every_standalone_recording_is_cut():
    session = _standalone_session()
    session.load_ventilator("monitor.txt", source="ventilator", name="monitor")

    summary = session.slice_ventilator(2.0, 5.0)

    assert session.ventilators["default"].data["array"].shape == (3, 30)
    assert session.ventilators["monitor"].data["array"].shape == (3, 30)
    assert set(summary["recordings"]) == {"default", "monitor"}


def test_a_name_cuts_only_that_recording():
    session = _standalone_session()
    session.load_ventilator("monitor.txt", source="ventilator", name="monitor")

    summary = session.slice_ventilator(2.0, 5.0, name="monitor")

    assert set(summary["recordings"]) == {"monitor"}
    assert session.ventilators["monitor"].data["array"].shape == (3, 30)
    assert session.ventilators["default"].data["array"].shape == (3, 100)
    # Only the cut recording's start time moves.
    assert session.start_times == {"ventilator:monitor": pytest.approx(2.0)}
    assert shared_clock_shift(session, "ventilator", "default") == 0.0


def test_each_cut_recording_keeps_its_own_start_time():
    session = _standalone_session()
    session.load_ventilator("monitor.txt", source="ventilator", name="monitor")
    session.synchronize_raw_modalities(
        offset_seconds={"ventilator": 1.0, "ventilator:monitor": 4.0},
        reference_modality="eit",
    )

    session.slice_ventilator(2.0)

    assert session.start_times["ventilator:default"] == pytest.approx(3.0)
    assert session.start_times["ventilator:monitor"] == pytest.approx(6.0)


def test_an_unknown_name_is_refused():
    session = _standalone_session()

    with pytest.raises(MissingModalityDataError, match="monitor"):
        session.slice_ventilator(2.0, 5.0, name="monitor")


def test_ventilator_data_from_the_emg_file_is_left_to_slice_emg():
    session = _session()
    session.load_ventilator("biopac_export.txt")  # on the EMG clock

    with pytest.raises(ValueError, match="slice_emg"):
        session.slice_ventilator(2.0, 5.0)
    assert session.ventilator.data["array"].shape == (3, 100)


def test_only_the_standalone_recording_is_cut_when_both_kinds_are_loaded():
    session = _standalone_session()
    session.load_ventilator("biopac_export.txt", name="biopac")  # on the EMG clock

    summary = session.slice_ventilator(2.0, 5.0)

    assert set(summary["recordings"]) == {"default"}
    assert session.ventilators["biopac"].data["array"].shape == (3, 100)


def test_a_legacy_bare_dict_is_cut():
    session = M3Session()
    payload = _recording()
    session.raw["vent"] = payload

    session.slice_ventilator(2.0, 5.0)

    assert payload["array"].shape == (3, 30)


def test_slice_must_run_before_preprocessing():
    session = _standalone_session()
    session.processed_variants["ventilator"]["default"] = {}

    with pytest.raises(ValueError, match="before preprocess_ventilator"):
        session.slice_ventilator(2.0, 5.0)


@pytest.mark.parametrize(("start", "end"), [(-1.0, 5.0), (5.0, 2.0), (2.0, 11.0)])
def test_a_window_outside_the_recording_is_refused(start, end):
    session = _standalone_session()

    with pytest.raises(ValueError, match="slice_ventilator"):
        session.slice_ventilator(start, end)


def test_slice_needs_a_loaded_recording():
    with pytest.raises(MissingModalityDataError):
        M3Session().slice_ventilator(1.0)


def test_the_steps_load_and_cut_a_named_recording():
    session = _standalone_session()

    load(session, file_path="monitor.txt", source="ventilator", name="monitor")
    slice_recording(session, start_seconds=2.0, name="monitor")

    assert session.ventilators["monitor"].data["array"].shape == (3, 80)
    assert session.ventilators["default"].data["array"].shape == (3, 100)


def test_the_steps_mark_a_file_standalone_and_cut_it():
    session = _session()

    load(session, file_path="ventilator.txt", source="ventilator")
    result = slice_recording(session, start_seconds=2.0, end_seconds=5.0)

    assert session.ventilator.source_modality == "ventilator"
    assert result["ventilator_slice"] == session.parameters["ventilator_slice"]
    assert session.ventilator.data["array"].shape == (3, 30)
