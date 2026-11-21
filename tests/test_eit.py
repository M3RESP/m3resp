from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import pytest
from reference.eitprocessing import EITProcessingAdapter

from m3resp import BreathEvent, M3Session
from m3resp.data import Signal
from m3resp.eit.io import EITFileData
from m3resp.io import load_eit
from m3resp.modalities.eit import EITRecording
from m3resp.modalities.eit import load as load_eit_recording


class FakeCollection(dict):
    def add(self, value: Any, overwrite: bool = False) -> None:
        if not overwrite and value.label in self:
            raise KeyError(value.label)
        self[value.label] = value


class FakeEITData:
    label = "raw"
    time: ClassVar = [0.0, 1.0, 2.0]
    pixel_impedance: ClassVar = [[[1.0]], [[2.0]], [[3.0]]]
    sample_frequency = 1.0

    def get_summed_impedance(self, return_label: str | None = None, **kwargs: Any):
        return FakeContinuousData(return_label or f"summed {self.label}")


class FakeContinuousData:
    def __init__(self, label: str):
        self.label = label
        self.time = [0.0, 1.0, 2.0]
        self.values = [1.0, 2.0, 3.0]


class FakeSequence:
    def __init__(self, continuous_data: FakeCollection | None = None):
        self.eit_data = FakeCollection(raw=FakeEITData())
        self.continuous_data = continuous_data or FakeCollection()
        self.sparse_data = FakeCollection()
        self.interval_data = FakeCollection()


class FakeBreath:
    start_time = 1.0
    middle_time = 1.5
    end_time = 2.0


class FakeIntervals:
    values: ClassVar = [FakeBreath()]


def _file_data(path: Any, vendor: Any = None, **kwargs: Any) -> EITFileData:
    """Three frames of one pixel, as a custom loader gives them."""

    return EITFileData(
        vendor="sentec",
        paths=(Path(str(path)),),
        sample_frequency=1.0,
        pixel_impedance=Signal(
            values=[[[1.0]], [[2.0]], [[3.0]]],
            time=[0.0, 1.0, 2.0],
            sample_frequency=1.0,
            name="raw",
            modality="eit",
        ),
    )


def test_load_eit_sets_preferred_and_legacy_session_slots():
    session = M3Session(eit_loader=_file_data)

    returned = session.load_eit("subject.eit", vendor="sentec")

    assert returned is session.eit
    assert session.eit is session.raw["eit"]
    assert session.eit.vendor == "sentec"
    assert list(session.eit.raw_global_impedance.values) == [1.0, 2.0, 3.0]


def test_global_impedance_reuses_existing_continuous_signal():
    existing = FakeContinuousData("global_impedance_(raw)")
    sequence = FakeSequence(
        continuous_data=FakeCollection({"global_impedance_(raw)": existing})
    )
    adapter = EITProcessingAdapter()

    global_impedance = adapter.get_global_impedance(sequence)

    assert global_impedance is existing


def test_global_impedance_computes_and_stores_when_missing():
    sequence = FakeSequence()
    adapter = EITProcessingAdapter()

    global_impedance = adapter.get_global_impedance(sequence)

    assert global_impedance.label == "global_impedance_(raw)"
    assert sequence.continuous_data["global_impedance_(raw)"] is global_impedance


def test_custom_detector_normalization_still_works():
    adapter = EITProcessingAdapter()

    events = adapter.detect_breaths(
        {"processed": True},
        detector=lambda data: [(1.0, 2.0, 1.5)],
    )

    assert events == [
        BreathEvent(
            modality="eit",
            start_time=1.0,
            end_time=2.0,
            extremum_time=1.5,
            source="eitprocessing",
        )
    ]


def test_default_detection_reads_processed_breath_intervals():
    adapter = EITProcessingAdapter()

    events = adapter.detect_breaths({"breath_intervals": FakeIntervals()})

    assert events[0].modality == "eit"
    assert events[0].source == "eitprocessing.BreathDetection"
    assert events[0].extremum_time == 1.5


def test_top_level_and_modality_load_helpers_return_recordings():
    top_level = load_eit("subject.eit", vendor="sentec", custom_loader=_file_data)
    modality_level = load_eit_recording(
        "subject.eit", vendor="sentec", custom_loader=_file_data
    )

    assert isinstance(top_level, EITRecording)
    assert modality_level.raw_global_impedance.name == "Global impedance (raw)"


def test_eit_real_data_workflow_uses_committed_sample():
    repo_root = Path(__file__).resolve().parents[1]
    pytest.importorskip("eitprocessing")

    eit_path = os.path.join(
        repo_root,
        "tests",
        "data",
        "draeger_synthetic_draeger_20Hz.bin",
    )
    assert os.path.exists(eit_path), f"missing committed EIT fixture: {eit_path}"
    session = M3Session()

    session.load_eit(eit_path, vendor="draeger")

    assert session.eit is not None
    assert len(session.eit.pixel_impedance.values) > 0
    assert len(session.eit.raw_global_impedance.time) > 0
    assert len(session.eit.raw_global_impedance.values) > 0

    processed = session.preprocess_eit()

    assert processed["filtered_global_impedance"] is not None
    assert processed["breath_intervals"] is not None
    assert processed["continuous_tiv"] is not None
    assert processed["eeli"] is not None

    events = session.detect_eit_breaths()

    assert events
    assert all(isinstance(event, BreathEvent) for event in events)
