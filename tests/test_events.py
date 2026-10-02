from __future__ import annotations

import ast
import importlib
import os
from typing import ClassVar

import pytest

import m3resp
import m3resp.data
import m3resp.data.events
from m3resp import (
    BreathEvent,
    Event,
    Interval,
    coerce_breath_event,
    coerce_breath_events,
    coerce_event,
    coerce_interval,
    coerce_intervals,
    event_to_dict,
)
from m3resp.data.events import reuse_matching_breaths
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
    metadata: ClassVar = {"upstream": True}


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


def test_coerce_breath_event_rejects_a_too_short_tuple():
    with pytest.raises(ValueError, match="length-1"):
        coerce_breath_event((1,), modality="eit")


def test_coerce_breath_event_rejects_a_too_long_tuple():
    with pytest.raises(ValueError, match="length-4"):
        coerce_breath_event((1, 2, 3, 4), modality="eit")


def test_coerce_breath_event_rejects_a_string():
    with pytest.raises(TypeError, match="mapping"):
        coerce_breath_event("abc", modality="eit")


def test_coerce_event_rejects_a_too_short_tuple():
    with pytest.raises(ValueError, match="length-2"):
        coerce_event(("trigger", "vent"))


def test_coerce_event_rejects_a_string():
    with pytest.raises(TypeError, match="mapping"):
        coerce_event("trigger")


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


def test_event_types_are_the_same_from_every_import_path():
    assert m3resp.BreathEvent is m3resp.data.BreathEvent
    assert m3resp.BreathEvent is m3resp.data.events.BreathEvent
    assert m3resp.Event is m3resp.data.Event
    assert m3resp.Event is m3resp.data.events.Event


def test_old_core_events_path_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("m3resp.core.events")


def test_data_package_does_not_import_from_core():
    # `m3resp.data` holds the plain data types; `m3resp.core` holds the
    # session, which uses them. The data types must not depend on the session.
    data_folder = os.path.dirname(m3resp.data.__file__)
    offending = []
    for file_name in sorted(os.listdir(data_folder)):
        if not file_name.endswith(".py"):
            continue
        file_path = os.path.join(data_folder, file_name)
        with open(file_path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            offending += [
                f"{file_name}: {name}"
                for name in names
                if name == "m3resp.core" or name.startswith("m3resp.core.")
            ]
    assert offending == []


def test_interval_needs_a_name_and_keeps_its_times():
    interval = Interval("ventilator", 10.0, 12.5, name="occlusion")

    assert interval.duration == 2.5
    assert interval.name == "occlusion"
    with pytest.raises(TypeError, match="name"):
        Interval("ventilator", 10.0, 12.5)


def test_interval_rejects_an_end_before_its_start():
    with pytest.raises(ValueError, match="Interval.end_time"):
        Interval("ventilator", 2.0, 1.0, name="occlusion")


def test_breath_event_is_an_interval_named_breath():
    breath = BreathEvent("eit", 1.0, 2.0, peak_time=1.5)

    assert isinstance(breath, Interval)
    assert breath.name == "breath"
    assert breath.duration == 1.0


def test_breath_event_and_interval_with_the_same_times_are_not_equal():
    breath = BreathEvent("eit", 1.0, 2.0)
    interval = Interval("eit", 1.0, 2.0, name="breath")

    assert breath != interval


def test_settings_after_end_time_must_be_given_by_name():
    with pytest.raises(TypeError):
        BreathEvent("eit", 1.0, 2.0, 1.5)
    with pytest.raises(TypeError):
        Interval("eit", 1.0, 2.0, "noise")


def test_coerce_interval_from_dict_object_and_pair():
    from_dict = coerce_interval(
        {"modality": "emg", "start_time": 1.0, "end_time": 3.0, "name": "noise"}
    )

    class UpstreamInterval:
        start_time = 4.0
        end_time = 5.0

    from_object = coerce_interval(
        UpstreamInterval(), name="noise", modality="eit", source="eitprocessing"
    )
    from_pair = coerce_interval((6.0, 7.0), name="noise", modality="emg")

    assert from_dict == Interval("emg", 1.0, 3.0, name="noise")
    assert from_object == Interval(
        "eit", 4.0, 5.0, name="noise", source="eitprocessing"
    )
    assert from_pair == Interval("emg", 6.0, 7.0, name="noise")


def test_coerce_interval_passes_an_interval_through():
    interval = Interval("emg", 1.0, 2.0, name="noise")

    assert coerce_interval(interval) is interval
    assert coerce_intervals([interval]) == [interval]


def test_coerce_interval_needs_a_name_and_two_times():
    with pytest.raises(ValueError, match="name is required"):
        coerce_interval((1.0, 2.0), modality="emg")
    with pytest.raises(ValueError, match="length-3"):
        coerce_interval((1.0, 2.0, 3.0), name="noise", modality="emg")


def test_alignment_shifts_intervals_like_breaths():
    interval = Interval("emg", 1.0, 2.0, name="noise")

    aligned = align_events_by_modality_offset([interval], {"emg": 0.5})

    assert aligned == [Interval("emg", 1.5, 2.5, name="noise")]
    assert aligned[0].id == interval.id
    assert interval.start_time == 1.0


def test_breath_event_name_is_always_breath():
    with pytest.raises(TypeError, match="name"):
        BreathEvent("eit", 1.0, 2.0, name="occlusion")


def test_coerce_breath_event_refuses_an_interval_that_is_not_a_breath():
    occlusion = Interval("ventilator", 1.0, 3.0, name="occlusion")

    with pytest.raises(ValueError, match="'occlusion' interval as a breath"):
        coerce_breath_event(occlusion)
    with pytest.raises(ValueError, match="'occlusion' interval as a breath"):
        coerce_breath_event(event_to_dict(occlusion))


def test_breath_keeps_its_label_through_a_dictionary():
    breath = BreathEvent("eit", 1.0, 2.0, peak_time=1.5, label="first")

    restored = coerce_breath_event(event_to_dict(breath))

    assert restored == breath
    assert restored.label == "first"
    assert restored.id == breath.id


def test_interval_keeps_its_label_through_a_dictionary():
    noise = Interval("emg", 1.0, 2.0, name="noise", label="burst")

    assert coerce_interval(event_to_dict(noise)) == noise


def test_coerce_interval_keeps_the_turning_point_of_a_breath():
    breath = BreathEvent("eit", 1.0, 2.0, peak_time=1.5, peak_index=75)

    restored = coerce_interval(event_to_dict(breath))

    assert isinstance(restored, BreathEvent)
    assert restored == breath


def test_coerce_interval_keeps_middle_time_from_eitprocessing():
    restored = coerce_interval(UpstreamBreath())

    assert isinstance(restored, BreathEvent)
    assert restored.peak_time == 1.5


def test_coerce_interval_refuses_a_turning_point_it_cannot_keep():
    with pytest.raises(ValueError, match="turning point"):
        coerce_interval(
            {
                "modality": "ventilator",
                "start_time": 1.0,
                "end_time": 2.0,
                "name": "occlusion",
                "peak_time": 1.4,
            }
        )


def test_coerce_needs_both_times_in_a_dictionary():
    with pytest.raises(ValueError, match="no end_time entry"):
        coerce_breath_event({"modality": "eit", "start_time": 1.0})
    with pytest.raises(ValueError, match="no start_time entry"):
        coerce_interval({"end_time": 1.0}, name="noise", modality="emg")


class TestReuseMatchingBreaths:
    """`reuse_matching_breaths` points results at breaths already stored."""

    def test_a_stored_breath_with_the_same_times_is_used(self):
        stored = BreathEvent("eit", 0.0, 1.0, peak_time=0.5)
        found = [
            BreathEvent("eit", 0.0, 1.0, peak_time=0.5),
            BreathEvent("eit", 1.0, 2.0),
        ]

        reused = reuse_matching_breaths(found, [stored])

        assert reused[0] is stored
        assert reused[1] is found[1]

    def test_other_modalities_and_items_that_are_not_breaths_are_ignored(self):
        found = [BreathEvent("eit", 0.0, 1.0)]
        stored = [
            BreathEvent("emg", 0.0, 1.0),
            Event(name="marker", modality="eit", time=0.0),
            {"start_time": 0.0, "end_time": 1.0},
        ]

        assert reuse_matching_breaths(found, stored)[0] is found[0]

    def test_nothing_stored_keeps_the_breaths(self):
        found = [BreathEvent("eit", 0.0, 1.0)]

        assert reuse_matching_breaths(found, None) == found
