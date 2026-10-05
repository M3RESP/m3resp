from __future__ import annotations

import json

import numpy as np
import pytest

from m3resp import (
    BreathEvent,
    Event,
    EventData,
    Interval,
    IntervalData,
    ParameterResult,
)


def _breaths() -> list[BreathEvent]:
    return [
        BreathEvent("eit", 0.0, 3.0, peak_time=1.2),
        BreathEvent("eit", 3.0, 6.5, peak_time=4.4),
    ]


def test_interval_data_keeps_one_value_per_breath():
    data = IntervalData(
        name="tidal_impedance_variation",
        modality="eit",
        intervals=_breaths(),
        values=[1.5, 1.7],
        unit="AU",
    )

    assert len(data) == 2
    assert data.intervals[1].start_time == 3.0
    assert data.values is not None
    assert data.values[1] == 1.7


def test_interval_data_holds_one_pixel_map_per_breath():
    maps = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    maps[0, 0, 0] = np.nan

    data = IntervalData(
        name="pixel_breath_start", modality="eit", intervals=_breaths(), values=maps
    )
    same = IntervalData(
        name="pixel_breath_start",
        modality="eit",
        intervals=_breaths(),
        values=maps.copy(),
    )

    assert data == same
    assert data.to_dict()["values"][1] == maps[1].tolist()


def test_interval_data_needs_one_value_per_interval():
    with pytest.raises(ValueError, match="one value per item"):
        IntervalData(name="x", modality="eit", intervals=_breaths(), values=[1.0])


def test_interval_data_rejects_items_that_are_not_intervals():
    with pytest.raises(TypeError, match="coerce_intervals"):
        IntervalData(name="x", modality="eit", intervals=[(0.0, 1.0)])


def test_interval_data_may_hold_intervals_without_values():
    occlusions = [Interval("ventilator", 10.0, 11.0, name="occlusion")]

    data = IntervalData(name="occlusions", modality="ventilator", intervals=occlusions)

    assert data.values is None
    assert data.to_dict()["values"] is None


def test_interval_data_tidies_unit_and_category():
    data = IntervalData(
        name="x",
        modality="ventilator",
        intervals=[Interval("ventilator", 0.0, 1.0, name="occlusion")],
        values=[2.0],
        unit="AU",
        category="pixel_impedance",
    )

    assert data.unit == "a.u."
    assert data.category == "impedance"


def test_interval_data_to_dict_is_json_ready():
    data = IntervalData(
        name="tidal_impedance_variation",
        modality="eit",
        intervals=_breaths(),
        values=np.array([1.5, 1.7]),
    )

    restored = json.loads(json.dumps(data.to_dict()))

    assert restored["values"] == [1.5, 1.7]
    assert restored["intervals"][0]["peak_time"] == 1.2
    assert restored["intervals"][0]["name"] == "breath"


def test_interval_data_writes_parameter_results_as_dictionaries():
    results = [
        ParameterResult(name="tidal_impedance_variation", value=1.5, modality="eit"),
        ParameterResult(name="tidal_impedance_variation", value=1.7, modality="eit"),
    ]
    data = IntervalData(
        name="tiv_per_breath",
        modality="eit",
        intervals=_breaths(),
        values=results,
    )

    restored = json.loads(json.dumps(data.to_dict()))

    assert [value["value"] for value in restored["values"]] == [1.5, 1.7]
    assert restored["values"][0] == results[0].to_dict()


def test_event_data_keeps_one_value_per_event():
    peaks = [
        Event(name="ecg_peak", modality="emg", time=0.8),
        Event(name="ecg_peak", modality="emg", time=1.6),
    ]

    data = EventData(
        name="ecg_peak_amplitude", modality="emg", events=peaks, values=[3, 4]
    )

    assert len(data) == 2
    assert data.to_dict()["events"][1]["time"] == 1.6
    with pytest.raises(ValueError, match="one value per item"):
        EventData(name="x", modality="emg", events=peaks, values=[1, 2, 3])


def test_event_data_rejects_intervals():
    with pytest.raises(TypeError, match="Event objects"):
        EventData(name="x", modality="eit", events=_breaths())


def test_values_may_not_be_a_dictionary():
    with pytest.raises(TypeError, match="got dict"):
        IntervalData(
            name="x", modality="eit", intervals=_breaths(), values={"a": 1.2, "b": 3.4}
        )


def test_values_may_not_be_a_single_number():
    event = Event(name="ecg_peak", modality="emg", time=0.8)

    with pytest.raises(TypeError, match="single number"):
        EventData(name="x", modality="emg", events=[event], values=np.array(5.0))
    with pytest.raises(TypeError, match="one value per item"):
        EventData(name="x", modality="emg", events=[event], values=5.0)


def test_values_may_come_from_another_device_than_the_breaths():
    ventilator_breaths = [BreathEvent("ventilator", 0.0, 3.0)]

    data = IntervalData(
        name="tidal_impedance_variation",
        modality="eit",
        intervals=ventilator_breaths,
        values=[1.5],
    )

    assert data.modality == "eit"
    assert data.intervals[0].modality == "ventilator"
