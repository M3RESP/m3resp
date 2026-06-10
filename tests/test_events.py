from __future__ import annotations

from m3resp import (
    BreathEvent,
    Event,
    coerce_breath_event,
    coerce_breath_events,
    coerce_event,
    event_to_dict,
)
from m3resp.export.tables import events_to_rows
from m3resp.synchronization.alignment import (
    align_events_by_modality_offset,
    align_events_manual_offset,
)


class UpstreamBreath:
    modality = "eit"
    start_time = 1.0
    middle_time = 1.5
    end_time = 2.0
    confidence = 0.9
    metadata = {"upstream": True}


def test_event_and_breath_defaults_are_isolated():
    event = Event(name="marker", modality="eit", time=1.0)
    breath = BreathEvent(modality="emg", start_time=0.0, end_time=1.0)

    event.metadata["event"] = True
    breath.metadata["breath"] = True

    assert Event(name="marker", modality="eit", time=1.0).metadata == {}
    assert BreathEvent(modality="emg", start_time=0.0, end_time=1.0).metadata == {}
    assert event_to_dict(event)["metadata"] == {"event": True}
    assert event_to_dict(breath)["metadata"] == {"breath": True}


def test_coerce_event_from_dict():
    event = coerce_event(
        {
            "name": "trigger",
            "modality": "vent",
            "time": 2.5,
            "sample_index": 10,
            "metadata": {"kind": "manual"},
        }
    )

    assert event == Event(
        name="trigger",
        modality="vent",
        time=2.5,
        sample_index=10,
        metadata={"kind": "manual"},
    )


def test_coerce_breath_event_passthrough():
    breath = BreathEvent("emg", 0.0, 1.0, peak_time=0.5)

    assert coerce_breath_event(breath, modality="eit", source="custom") is breath


def test_coerce_breath_event_from_dict():
    breath = coerce_breath_event(
        {
            "start_time": 1,
            "end_time": 2,
            "peak_time": 1.5,
            "confidence": 0.8,
            "metadata": {"peak_index": 42},
        },
        modality="emg",
        source="detector",
    )

    assert breath == BreathEvent(
        modality="emg",
        start_time=1.0,
        end_time=2.0,
        peak_time=1.5,
        source="detector",
        confidence=0.8,
        metadata={"peak_index": 42},
    )


def test_coerce_breath_event_from_tuple():
    breath = coerce_breath_event((1, 2, 1.5), modality="eit", source="detector")

    assert breath == BreathEvent("eit", 1.0, 2.0, peak_time=1.5, source="detector")


def test_coerce_breath_event_from_upstream_object_with_middle_time():
    breath = coerce_breath_event(UpstreamBreath(), source="upstream")

    assert breath == BreathEvent(
        modality="eit",
        start_time=1.0,
        end_time=2.0,
        peak_time=1.5,
        source="upstream",
        confidence=0.9,
        metadata={"upstream": True},
    )


def test_coerce_breath_events_normalizes_iterable():
    events = coerce_breath_events([(0, 1, 0.5), (1, 2, None)], modality="emg")

    assert events == [
        BreathEvent("emg", 0.0, 1.0, peak_time=0.5),
        BreathEvent("emg", 1.0, 2.0),
    ]


def test_mixed_event_rows_and_alignment():
    events = [
        Event(name="trigger", modality="vent", time=1.0),
        BreathEvent("emg", 2.0, 3.0, peak_time=2.5),
    ]

    rows = events_to_rows(events)
    aligned = align_events_manual_offset(events, 0.25)

    assert rows[0]["time"] == 1.0
    assert rows[1]["start_time"] == 2.0
    assert aligned[0].time == 1.25
    assert aligned[1].start_time == 2.25
    assert aligned[1].peak_time == 2.75


def test_manual_offset_preserves_none_peak_time_and_original_events():
    events = [BreathEvent("emg", 2.0, 3.0)]

    aligned = align_events_manual_offset(events, 0.5)

    assert events[0].start_time == 2.0
    assert events[0].peak_time is None
    assert aligned[0].start_time == 2.5
    assert aligned[0].end_time == 3.5
    assert aligned[0].peak_time is None
    assert aligned[0] is not events[0]


def test_alignment_uses_per_modality_offset_map():
    events = [
        BreathEvent("eit", 1.0, 2.0, peak_time=1.5),
        BreathEvent("emg", 1.0, 2.0, peak_time=1.5),
        Event(name="trigger", modality="vent", time=1.0),
    ]

    aligned = align_events_by_modality_offset(
        events,
        {"eit": 0.0, "emg": 0.25, "vent": -0.1},
    )

    assert aligned[0].start_time == 1.0
    assert aligned[1].start_time == 1.25
    assert aligned[1].peak_time == 1.75
    assert aligned[2].time == 0.9
