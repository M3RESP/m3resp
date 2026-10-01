"""Warning when recordings that were never synchronized are compared (#115).

`session.sync_methods` records, per recording, how it was placed on the
shared clock. Steps that compare recordings on different clocks warn
(`UnsynchronizedDataWarning`) when one of them has no record.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pytest

from m3resp.core.events import BreathEvent
from m3resp.core.exceptions import UnsynchronizedDataWarning
from m3resp.core.session import M3Session
from m3resp.workflows.steps.emg.quality_events import evaluate_event_timing
from m3resp.workflows.steps.sync import skip

FS = 10.0


def _payload(n_samples: int = 100) -> dict:
    return {
        "array": np.zeros((3, n_samples)),
        "metadata": {"fs": FS, "labels": ["pressure", "flow", "volume"]},
    }


class _Adapter:
    """Stands in for the EMG adapter: loads fixed payloads and records the
    times `evaluate_event_timing` is given."""

    def load(self, path, **kwargs):
        return _payload()

    def evaluate_event_timing(self, first_event_times, second_event_times):
        self.first, self.second = first_event_times, second_event_times
        delta = second_event_times - first_event_times
        return np.ones(len(delta), dtype=bool), delta


def _session() -> M3Session:
    return M3Session(emg_adapter=_Adapter())


def _breath(modality: str, peak: float) -> BreathEvent:
    return BreathEvent(modality, peak - 0.5, peak + 0.5, peak_time=peak)


def _with_breaths(session: M3Session, *modalities: str) -> M3Session:
    for modality in modalities:
        session.add_events(f"{modality}_breaths", [_breath(modality, 2.0)])
    return session


def _link_warnings(session: M3Session) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session.link_breaths()
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, UnsynchronizedDataWarning)
    ]


class TestLinkBreaths:
    def test_warns_when_nothing_was_synchronized(self):
        session = _with_breaths(_session(), "eit", "emg")

        messages = _link_warnings(session)

        assert len(messages) == 1
        assert (
            "link_breaths compares the EIT recording and the EMG recording"
            in (messages[0])
        )
        assert "they were never synchronized" in messages[0]
        assert "skip_synchronization()" in messages[0]

    def test_no_warning_after_synchronize_raw_modalities(self):
        session = _with_breaths(_session(), "eit", "emg")
        session.synchronize_raw_modalities(offset_seconds={"emg": 0.0})

        assert _link_warnings(session) == []
        assert session.sync_methods == {"eit": "manual", "emg": "manual"}

    def test_no_warning_after_synchronize_multimodal_breaths(self):
        session = _with_breaths(_session(), "eit", "emg")
        session.synchronize_multimodal_breaths(offset_seconds={"emg": 0.0})

        assert _link_warnings(session) == []
        assert session.sync_methods == {"eit": "manual", "emg": "manual"}

    def test_no_warning_after_skip_synchronization(self):
        session = _with_breaths(_session(), "eit", "emg")

        skipped = session.skip_synchronization()

        assert skipped == ["eit", "emg"]
        assert session.sync_methods == {"eit": "none", "emg": "none"}
        assert _link_warnings(session) == []
        assert session.provenance[-2].action == "skip_synchronization"

    def test_no_warning_for_breaths_from_a_single_modality(self):
        session = _with_breaths(_session(), "emg")

        assert _link_warnings(session) == []

    def test_no_warning_when_the_ventilator_came_in_the_emg_file(self):
        # Airway pressure recorded in the EMG file shares the EMG's clock.
        session = _session()
        session.load_emg("biopac.txt")
        session.load_ventilator("biopac.txt")
        _with_breaths(session, "emg", "ventilator")

        assert _link_warnings(session) == []

    def test_names_only_the_recording_that_was_never_synchronized(self):
        session = _with_breaths(_session(), "eit", "emg")
        session.synchronize_raw_modalities(offset_seconds={"emg": 0.0})
        session.load_ventilator("ventilator.txt", source="ventilator")
        _with_breaths(session, "ventilator")

        messages = _link_warnings(session)

        assert len(messages) == 1
        assert "ventilator recording 'default' was never synchronized" in messages[0]


class TestSkipSynchronization:
    def test_already_synchronized_recordings_keep_their_record(self):
        session = _session()
        session.load_emg("emg.txt")
        session.synchronize_raw_modalities(offset_seconds={"emg": 0.0})
        session.load_ventilator("ventilator.txt", source="ventilator")

        skipped = session.skip_synchronization()

        assert skipped == ["ventilator:default"]
        assert session.sync_methods == {"emg": "manual", "ventilator:default": "none"}

    def test_the_workflow_step_calls_the_session_method(self):
        session = _session()
        session.load_emg("emg.txt")

        assert skip(session) == {"skipped_recordings": ["emg"]}
        assert session.sync_methods == {"emg": "none"}


class TestLoadingClearsTheRecord:
    def test_reloading_a_recording_clears_its_start_time_and_record(self):
        # A reloaded file is a new recording, back at its full length: an
        # older start time (e.g. moved by slicing) no longer applies.
        session = _session()
        session.load_emg("emg.txt")
        session.synchronize_raw_modalities(
            offset_seconds={"emg": -2.0}, reference_modality="eit"
        )
        session.slice_emg(1.0)

        session.load_emg("emg.txt")

        assert "emg" not in session.start_times
        assert "emg" not in session.sync_methods
        assert session.start_times["eit"] == 0.0

    def test_reloading_a_standalone_ventilator_clears_only_its_own_record(self):
        session = _session()
        session.load_emg("emg.txt")
        session.load_ventilator("ventilator.txt", source="ventilator")
        session.synchronize_raw_modalities(
            offset_seconds={"ventilator:default": 3.0}, reference_modality="emg"
        )

        session.load_ventilator("ventilator.txt", source="ventilator")

        assert "ventilator:default" not in session.start_times
        assert "ventilator:default" not in session.sync_methods
        assert session.sync_methods["emg"] == "manual"


class TestEvaluateEventTimingUsesTheSharedClock:
    def _run(self, session: M3Session) -> None:
        evaluate_event_timing(
            session,
            np.array([10, 30]),  # EMG breaths at 1 s and 3 s
            {"fs": FS},
            np.array([15, 35]),  # ventilator breaths at 1.5 s and 3.5 s
            {"fs": FS},
        )

    def test_a_standalone_ventilator_is_moved_by_its_start_time(self):
        session = _session()
        session.load_emg("emg.txt")
        session.load_ventilator("ventilator.txt", source="ventilator")
        session.synchronize_raw_modalities(
            offset_seconds={"emg": 0.0, "ventilator": 2.0}, reference_modality="emg"
        )

        self._run(session)

        np.testing.assert_allclose(session.emg_adapter.first, [1.0, 3.0])
        np.testing.assert_allclose(session.emg_adapter.second, [3.5, 5.5])

    def test_ventilator_data_from_the_emg_file_keeps_its_timing(self):
        session = _session()
        session.load_emg("biopac.txt")
        session.load_ventilator("biopac.txt")
        session.synchronize_raw_modalities(
            offset_seconds={"emg": -2.0}, reference_modality="eit"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", UnsynchronizedDataWarning)
            self._run(session)

        delta = session.emg_adapter.second - session.emg_adapter.first
        np.testing.assert_allclose(delta, [0.5, 0.5])

    def test_warns_when_a_standalone_ventilator_was_never_synchronized(self):
        session = _session()
        session.load_emg("emg.txt")
        session.load_ventilator("ventilator.txt", source="ventilator")

        with pytest.warns(UnsynchronizedDataWarning, match="emg.evaluate_event_timing"):
            self._run(session)


def test_the_exported_summary_records_the_synchronization(tmp_path):
    session = _session()
    session.load_emg("emg.txt")
    session.synchronize_raw_modalities(
        offset_seconds={"emg": 1.5}, reference_modality="eit"
    )

    session.export_summary(str(tmp_path))

    with open(os.path.join(tmp_path, "summary.json"), encoding="utf-8") as file:
        summary = json.load(file)
    assert summary["synchronization"] == {
        "start_times": {"eit": 0.0, "emg": 1.5, "ventilator": 0.0},
        "sync_methods": {"emg": "manual"},
    }
