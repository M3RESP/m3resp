"""Tests for placing recordings on one shared clock by start time (#118).

`M3Session.synchronize_raw_modalities` sets a start time per recording and
keeps every sample. The start times are applied only when modalities are
compared: `synchronize_multimodal_breaths`, `link_breaths` and the
before/after synchronization plot.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from m3resp.adapters import ReSurfEMGAdapter
from m3resp.core.events import BreathEvent
from m3resp.core.session import M3Session
from m3resp.modalities.eit import EITRecording
from m3resp.modalities.emg import EMGRecording
from m3resp.synchronization.start_times import own_first_time, shared_clock_shift


def _emg(n_samples: int = 100, fs: float = 10.0) -> EMGRecording:
    return EMGRecording(
        data={"array": np.zeros((1, n_samples)), "metadata": {"fs": fs}},
        path=Path("synthetic_emg.npy"),
    )


def _eit(
    first_time: float = 0.0, n_samples: int = 200, fs: float = 20.0
) -> EITRecording:
    time = first_time + np.arange(n_samples) / fs
    return EITRecording(
        data=None,
        path=Path("synthetic_eit.bin"),
        global_impedance=SimpleNamespace(
            time=time, values=np.zeros(n_samples), label="global_impedance"
        ),
    )


def _session_with_ventilator(
    n_samples: int = 100, fs: float = 10.0, **load_kwargs
) -> M3Session:
    payload = {
        "array": np.zeros((3, n_samples)),
        "metadata": {"fs": fs, "labels": ["pressure", "flow", "volume"]},
    }
    session = M3Session(
        emg_adapter=ReSurfEMGAdapter(loader=lambda path, **kwargs: payload)
    )
    session.load_ventilator("vent.txt", **load_kwargs)
    return session


def _breath(modality: str, peak: float) -> BreathEvent:
    return BreathEvent(modality, peak - 0.5, peak + 0.5, peak_time=peak)


def _linked_modalities(session: M3Session) -> list[set[str]]:
    return [set(link.modalities) for link in session.link_breaths(time_tolerance=0.2)]


class TestSynchronizeRawModalitiesKeepsEverySample:
    def test_arrays_are_unchanged_and_start_times_are_stored(self):
        session = _session_with_ventilator(source="ventilator")
        session.emg = _emg()

        summary = session.synchronize_raw_modalities(
            offset_seconds={"emg": -2.0, "ventilator": 1.0}, reference_modality="eit"
        )

        assert session.emg.data["array"].shape == (1, 100)
        assert session.ventilator.data["array"].shape == (3, 100)
        assert session.start_times == {"eit": 0.0, "emg": -2.0, "ventilator": 1.0}
        assert summary == {
            "emg": {"start_time_seconds": -2.0},
            "ventilator": {"start_time_seconds": 1.0},
        }

    def test_a_short_recording_starting_late_in_a_long_one_keeps_both_whole(self):
        # The case from the #118 review: three hours of pressure, and an EMG
        # recording that starts 90 minutes in. Cutting used to discard the
        # first 90 minutes of pressure, baseline included.
        session = _session_with_ventilator(
            n_samples=3 * 3600, fs=1.0, source="ventilator"
        )
        session.emg = _emg(n_samples=30 * 60, fs=1.0)

        session.synchronize_raw_modalities(
            offset_seconds={"ventilator": 0.0, "emg": 90 * 60.0},
            reference_modality="ventilator",
        )

        assert session.ventilator.data["array"].shape == (3, 3 * 3600)
        assert session.emg.data["array"].shape == (1, 30 * 60)
        # An EMG breath 10 s into the EMG recording is the ventilator breath
        # 90 min + 10 s into the ventilator recording.
        session.add_events("emg_breaths", [_breath("emg", 10.0)])
        session.add_events("ventilator_breaths", [_breath("ventilator", 5410.0)])
        assert _linked_modalities(session) == [{"emg", "ventilator"}]

    def test_calling_again_replaces_the_start_times(self):
        session = M3Session()
        session.emg = _emg()

        session.synchronize_raw_modalities(offset_seconds={"emg": 5.0})
        session.synchronize_raw_modalities(offset_seconds={"emg": 1.0})

        assert session.start_times["emg"] == 1.0

    def test_the_after_trace_is_moved_not_cut(self):
        session = M3Session()
        session.emg = _emg(n_samples=4, fs=2.0)

        session.synchronize_raw_modalities(
            offset_seconds={"emg": -1.0}, reference_modality="eit"
        )

        trace = session.processed["raw_synchronization"]["emg"]
        assert trace["before"]["time"] == [0.0, 0.5, 1.0, 1.5]
        assert trace["after"]["time"] == [-1.0, -0.5, 0.0, 0.5]
        assert trace["after"]["values"] == trace["before"]["values"]


class TestStartTimesMoveBreathsWhenLinking:
    def test_positive_start_time_moves_breaths_later(self):
        # Cutting samples off the end, as a positive offset used to do,
        # did not move the recording in time at all.
        session = M3Session()
        session.eit = _eit()
        session.emg = _emg()
        session.synchronize_raw_modalities(
            offset_seconds={"emg": 2.0}, reference_modality="eit"
        )
        session.add_events("eit_breaths", [_breath("eit", 3.0)])
        session.add_events("emg_breaths", [_breath("emg", 1.0)])

        assert _linked_modalities(session) == [{"eit", "emg"}]

    def test_negative_start_time_gives_the_same_times_as_cutting_did(self):
        session = M3Session()
        session.eit = _eit()
        session.emg = _emg()
        session.synchronize_raw_modalities(
            offset_seconds={"emg": -2.0}, reference_modality="eit"
        )
        session.add_events("eit_breaths", [_breath("eit", 1.0)])
        session.add_events("emg_breaths", [_breath("emg", 3.0)])

        assert _linked_modalities(session) == [{"eit", "emg"}]
        # Each modality's own results stay on its own clock.
        assert session.events["emg_breaths"][0].peak_time == 3.0

    def test_eit_time_of_day_is_counted_from_the_first_sample(self):
        # A Draeger file's time axis is the time of day, while EMG time
        # starts at 0. Both recordings started together here.
        session = M3Session()
        session.eit = _eit(first_time=36528.5)
        session.emg = _emg()
        session.add_events("eit_breaths", [_breath("eit", 36530.5)])
        session.add_events("emg_breaths", [_breath("emg", 2.0)])
        session.skip_synchronization()

        assert own_first_time(session, "eit") == 36528.5
        assert _linked_modalities(session) == [{"eit", "emg"}]

    def test_a_ventilator_inside_the_emg_file_uses_the_emg_start_time(self):
        # ".txt" without `source=` is the multi-channel export shared with the
        # sEMG, so its channels were recorded on the EMG clock.
        session = _session_with_ventilator()
        session.emg = _emg()

        session.synchronize_raw_modalities(
            offset_seconds={"emg": -2.0, "ventilator": 7.0}, reference_modality="eit"
        )

        assert shared_clock_shift(session, "ventilator") == -2.0

    def test_a_breath_offset_adds_to_the_start_time(self):
        session = M3Session()
        session.emg = _emg()
        session.synchronize_raw_modalities(
            offset_seconds={"emg": -2.0}, reference_modality="eit"
        )
        session.add_events("emg_breaths", [_breath("emg", 3.0)])

        synchronized = session.synchronize_multimodal_breaths(
            offset_seconds={"emg": 0.5}, reference_modality="eit"
        )

        assert synchronized["emg_breaths"][0].peak_time == pytest.approx(1.5)
        assert (
            session.parameters["alignment"]["start_time_shift_seconds"]["emg"] == -2.0
        )


class TestSliceEmgKeepsTheEmgInLine:
    def test_slicing_moves_the_emg_start_time_later(self):
        session = M3Session()
        session.eit = _eit()
        session.emg = _emg()
        session.synchronize_raw_modalities(
            offset_seconds={"emg": -2.0}, reference_modality="eit"
        )

        session.slice_emg(1.0)

        assert session.start_times["emg"] == pytest.approx(-1.0)
        # A breath at 3.0 s in the original EMG is at 2.0 s after slicing,
        # and still lines up with the EIT breath at 1.0 s.
        session.add_events("eit_breaths", [_breath("eit", 1.0)])
        session.add_events("emg_breaths", [_breath("emg", 2.0)])
        assert _linked_modalities(session) == [{"eit", "emg"}]


class TestStandaloneVentilatorRecordingsCanStartAtDifferentTimes:
    """Two standalone ventilator exports (say a ventilator and a separate
    monitor) each have a clock of their own, so each can get its own start
    time under ``"ventilator:<name>"``."""

    def _session(self) -> M3Session:
        session = _session_with_ventilator(source="ventilator")
        session.load_ventilator("monitor.txt", source="ventilator", name="Monitor")
        return session

    def test_each_recording_gets_its_own_start_time(self):
        session = self._session()

        summary = session.synchronize_raw_modalities(
            offset_seconds={"ventilator": 1.0, "ventilator:Monitor": 12.5},
            reference_modality="eit",
        )

        assert shared_clock_shift(session, "ventilator", "default") == 1.0
        assert shared_clock_shift(session, "ventilator", "Monitor") == 12.5
        assert summary["ventilator:Monitor"] == {"start_time_seconds": 12.5}

    def test_a_recording_without_its_own_entry_uses_the_shared_one(self):
        session = self._session()

        session.synchronize_raw_modalities(
            offset_seconds={"ventilator": 3.0}, reference_modality="eit"
        )

        assert shared_clock_shift(session, "ventilator", "Monitor") == 3.0

    def test_the_old_spelling_and_the_names_case_are_kept(self):
        session = self._session()

        session.synchronize_raw_modalities(
            offset_seconds={"vent:Monitor": 2.0}, reference_modality="eit"
        )

        assert session.start_times["ventilator:Monitor"] == 2.0

    def test_a_named_recording_can_be_the_reference(self):
        session = self._session()

        session.synchronize_raw_modalities(
            offset_seconds={"ventilator": 1.0, "ventilator:Monitor": 4.0},
            reference_modality="ventilator:Monitor",
        )

        assert session.start_times["ventilator:Monitor"] == 0.0
        assert session.start_times["ventilator"] == -3.0

    def test_a_name_that_is_not_loaded_is_refused(self):
        session = self._session()

        with pytest.raises(ValueError, match="no ventilator recording named 'pod'"):
            session.synchronize_raw_modalities(offset_seconds={"ventilator:pod": 1.0})

    def test_ventilator_data_inside_the_emg_file_cannot_get_its_own(self):
        session = self._session()
        session.load_ventilator("biopac_export.txt", name="biopac")  # EMG clock

        with pytest.raises(ValueError, match="inside the EMG file"):
            session.synchronize_raw_modalities(
                offset_seconds={"ventilator:biopac": 1.0}
            )
