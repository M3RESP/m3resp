from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from m3resp import BreathEvent, M3Session
from m3resp.adapters import EITProcessingAdapter
from m3resp.io import load_eit
from m3resp.modalities.eit import load as load_eit_recording


class FakeCollection(dict):
    def add(self, value: Any, overwrite: bool = False) -> None:
        if not overwrite and value.label in self:
            raise KeyError(value.label)
        self[value.label] = value


class FakeEITData:
    label = "raw"
    time = [0.0, 1.0, 2.0]
    pixel_impedance = [[[1.0]], [[2.0]], [[3.0]]]
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
    values = [FakeBreath()]


def test_load_eit_sets_preferred_and_legacy_session_slots():
    sequence = FakeSequence()
    session = M3Session(
        eit_adapter=EITProcessingAdapter(loader=lambda *args, **kwargs: sequence)
    )

    returned = session.load_eit("subject.eit", vendor="sentec")

    assert returned is sequence
    assert session.eit is session.raw["eit"]
    assert session.eit.raw is sequence.eit_data["raw"]
    assert (
        session.eit.global_impedance
        is sequence.continuous_data["global_impedance_(raw)"]
    )


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
            peak_time=1.5,
            source="eitprocessing",
        )
    ]


def test_default_detection_reads_processed_breath_intervals():
    adapter = EITProcessingAdapter()

    events = adapter.detect_breaths({"breath_intervals": FakeIntervals()})

    assert events[0].modality == "eit"
    assert events[0].source == "eitprocessing.BreathDetection"
    assert events[0].peak_time == 1.5


def test_top_level_and_modality_load_helpers_return_recordings():
    sequence = FakeSequence()
    adapter = EITProcessingAdapter(loader=lambda *args, **kwargs: sequence)

    top_level = load_eit("subject.eit", vendor="sentec", adapter=adapter)
    modality_level = load_eit_recording("subject.eit", vendor="sentec", adapter=adapter)

    assert top_level.raw is sequence.eit_data["raw"]
    assert modality_level.global_impedance.label == "global_impedance_(raw)"


def test_eit_real_data_pipeline_uses_committed_sample():
    repo_root = Path(__file__).resolve().parents[1]
    sibling_eitprocessing = os.path.join(repo_root.parent, "eitprocessing")
    if (
        os.path.exists(sibling_eitprocessing)
        and str(sibling_eitprocessing) not in sys.path
    ):
        sys.path.insert(0, str(sibling_eitprocessing))

    pytest.importorskip("eitprocessing")

    eit_path = os.path.join(
        repo_root, "data", "source", "draeger_synthetic_draeger_20Hz.bin"
    )
    session = M3Session()

    session.load_eit(eit_path, vendor="draeger")

    assert session.eit is not None
    assert session.eit.raw is not None
    assert len(session.eit.raw) > 0
    assert len(session.eit.global_impedance.time) > 0
    assert len(session.eit.global_impedance.values) > 0

    processed = session.preprocess_eit()

    assert processed["filtered_global_impedance"] is not None
    assert processed["breath_intervals"] is not None
    assert processed["continuous_tiv"] is not None
    assert processed["eeli"] is not None

    events = session.detect_eit_breaths()

    assert events
    assert all(isinstance(event, BreathEvent) for event in events)
