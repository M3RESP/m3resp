"""Native EIT slicing, device events and frozen time-boundary results."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from m3resp.data import BreathEvent, Event, EventData, IntervalData, Signal
from m3resp.eit.io import EITFileData, read_eit_file
from m3resp.eit.processing.slicing import slice_file_data
from m3resp.processing.slicing import slice_by_time, time_slice_indices

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "tests", "regression", "golden", "eit")
with open(os.path.join(GOLDEN, "manifest.json"), encoding="utf-8") as handle:
    MANIFEST = json.load(handle)


@pytest.fixture(scope="module", params=sorted(MANIFEST["inputs"]))
def recording(request):
    entry = MANIFEST["inputs"][request.param]
    data = read_eit_file(os.path.join(ROOT, entry["file"]), entry["vendor"])
    return request.param, data


def test_native_time_slices_match_all_golden_windows_and_flags(recording):
    dataset, data = recording
    pixels = data.pixel_impedance
    with np.load(os.path.join(GOLDEN, dataset, "slicing.npz")) as golden:
        for key in golden.files:
            _, window, flags = key.split("/")
            start, end = map(float, window.split("-"))
            first_time = pixels.time[
                np.searchsorted(pixels.time, pixels.time[0] + start)
            ]
            last_time = pixels.time[np.searchsorted(pixels.time, pixels.time[0] + end)]
            selected = slice_by_time(
                pixels,
                start=first_time,
                end=last_time,
                start_inclusive=flags[0] == "1",
                end_inclusive=flags[1] == "1",
                allow_empty=True,
            )
            expected = golden[key]
            indices = [
                int(np.searchsorted(pixels.time, selected.time[0]))
                if len(selected)
                else -1,
                int(np.searchsorted(pixels.time, selected.time[-1]))
                if len(selected)
                else -1,
                len(selected),
            ]
            np.testing.assert_array_equal(indices, expected, err_msg=key)
            if len(selected):
                np.testing.assert_array_equal(
                    selected.values, pixels.values[expected[0] : expected[1] + 1]
                )
            assert not np.shares_memory(selected.time, pixels.time)
            assert not np.shares_memory(selected.values, pixels.values)


@pytest.mark.parametrize("start_flag", [False, True])
@pytest.mark.parametrize("end_flag", [False, True])
def test_between_sample_bounds_match_eitprocessing(start_flag, end_flag):
    pytest.importorskip("eitprocessing")
    from eitprocessing.datahandling.continuousdata import ContinuousData

    time = np.array([10.0, 11.0, 12.0, 14.0, 15.0, 16.0])
    upstream = ContinuousData(
        label="wave",
        name="wave",
        unit="AU",
        category="impedance",
        time=time,
        values=np.arange(6.0),
        sample_frequency=1.0,
    )
    expected = upstream.select_by_time(
        11.3, 14.2, start_inclusive=start_flag, end_inclusive=end_flag
    )
    first, last = time_slice_indices(
        time, start=11.3, end=14.2, start_inclusive=start_flag, end_inclusive=end_flag
    )
    np.testing.assert_array_equal(time[first:last], expected.time)


@pytest.mark.parametrize(
    "time",
    [
        [0.0, 2.0, 1.0],
        np.array([0, 2, 1], dtype=np.uint64),
        [0.0, np.nan],
        [0.0, np.inf],
        [[0.0, 1.0]],
    ],
)
def test_invalid_time_axis_is_refused(time):
    with pytest.raises(ValueError, match="sorted, finite, one-dimensional"):
        time_slice_indices(np.asarray(time))


def _device_data():
    time = np.arange(7.0)
    return EITFileData(
        vendor="timpel",
        path="recording.txt",
        sample_frequency=1.0,
        pixel_impedance=Signal(
            time=time, values=np.ones((7, 2, 2)), source="timpel", sample_frequency=1.0
        ),
        signals={
            "pressure": Signal(
                time=time, values=time, unit="cmH2O", sample_frequency=1.0
            )
        },
        events={
            "marks": EventData(
                name="marks",
                modality="eit",
                events=[
                    Event("mark", "eit", t, label=str(t))
                    for t in [0.0, 1.0, 3.9, 4.0, 6.0]
                ],
                values=[0, 1, 2, 3, 4],
            )
        },
        intervals={
            "breaths": IntervalData(
                name="breaths",
                modality="eit",
                intervals=[
                    BreathEvent("eit", a, b, extremum_time=(a + b) / 2)
                    for a, b in [(0.0, 2.0), (1.0, 4.0), (2.0, 5.0)]
                ],
            )
        },
    )


def test_recording_slice_has_distinct_mark_and_breath_end_rules():
    original = _device_data()

    selected = slice_file_data(original, 1, 5)

    np.testing.assert_array_equal(selected.pixel_impedance.time, [1.0, 2.0, 3.0, 4.0])
    assert [mark.time for mark in selected.events["marks"].events] == [1.0, 3.9]
    assert selected.events["marks"].values == [1, 2]
    assert [mark.label for mark in selected.events["marks"].events] == ["1.0", "3.9"]
    assert (
        selected.events["marks"].events[0].id == original.events["marks"].events[1].id
    )
    breaths = selected.intervals["breaths"].intervals
    assert breaths == [original.intervals["breaths"].intervals[1]]
    assert breaths[0].id == original.intervals["breaths"].intervals[1].id
    assert len(original.pixel_impedance) == 7
    assert selected.signals["pressure"].unit == "cmH2O"


def test_one_frame_keeps_no_marks_and_no_complete_breaths():
    selected = slice_file_data(_device_data(), 1, 2)
    assert selected.events["marks"].events == []
    assert selected.events["marks"].values == []
    assert selected.intervals["breaths"].intervals == []


def test_partial_intervals_are_trimmed_when_enabled():
    original = _device_data()
    original.intervals["breaths"].metadata["default_partial_inclusion"] = True
    selected = slice_file_data(original, 1, 5)
    assert [
        (b.start_time, b.end_time) for b in selected.intervals["breaths"].intervals
    ] == [(1.0, 2.0), (1.0, 4.0), (2.0, 4.0)]


def test_waveforms_with_a_different_sample_count_are_refused():
    original = _device_data()
    original.signals["pressure"] = Signal(time=[0.0], values=[1.0])
    with pytest.raises(ValueError, match="samples; the images"):
        slice_file_data(original, 1, 5)


@pytest.mark.parametrize("start,end", [(0, 1), (20, 201), (100, 400)])
def test_rebuilt_sequence_equals_upstream_index_slice(
    recording, start, end, monkeypatch
):
    pytest.importorskip("eitprocessing")
    from eitprocessing.datahandling.sequence import Sequence

    from m3resp import M3Session
    from m3resp.modalities.eit import keep_frames

    _, data = recording
    session = M3Session()
    session.load_eit(data.path, vendor=data.vendor)
    expected = session.eit.data.select_by_index(start, end)

    def upstream_slicing_was_called(*args, **kwargs):
        pytest.fail("The native recording slice called upstream slicing")

    monkeypatch.setattr(Sequence, "select_by_index", upstream_slicing_was_called)
    keep_frames(session.eit, start, end)

    assert session.eit.data == expected
    assert session.eit.data.name == expected.name
    assert session.eit.data.description == expected.description
    np.testing.assert_array_equal(
        session.eit.pixel_impedance.values, expected.eit_data["raw"].pixel_impedance
    )
    for key, waveform in expected.continuous_data.items():
        rebuilt = session.eit.data.continuous_data[key]
        assert rebuilt.description == waveform.description
        assert rebuilt.locked == waveform.locked


def test_session_device_marks_and_breaths_match_reader_golden(recording):
    pytest.importorskip("eitprocessing")
    from m3resp import M3Session

    dataset, data = recording
    session = M3Session()
    session.load_eit(data.path, vendor=data.vendor)
    with np.load(os.path.join(GOLDEN, dataset, "reader.npz")) as golden:
        for key, marks in session.eit.events.items():
            np.testing.assert_array_equal(
                [e.time for e in marks.events], golden[f"sparse/{key}/time"]
            )
            if marks.values is not None:
                np.testing.assert_array_equal(
                    marks.values, golden[f"sparse/{key}/marker"]
                )
                np.testing.assert_array_equal(
                    [e.label for e in marks.events], golden[f"sparse/{key}/text"]
                )
        for key, data in session.eit.intervals.items():
            breaths = data.intervals
            np.testing.assert_array_equal(
                [(b.start_time, b.end_time) for b in breaths],
                golden[f"interval/{key}/start_end"],
            )
            np.testing.assert_array_equal(
                [(b.start_time, b.extremum_time, b.end_time) for b in breaths],
                golden[f"interval/{key}/breaths"],
            )


def test_workflow_raw_signals_record_the_vendor_source(recording):
    pytest.importorskip("eitprocessing")
    from m3resp import M3Session
    from m3resp.workflows.steps.eit.loading import _loaded_recording_outputs

    _, data = recording
    session = M3Session()
    session.load_eit(data.path, vendor=data.vendor)
    outputs = _loaded_recording_outputs(session)
    assert outputs["raw_eit"].source == data.vendor
    assert outputs["raw_global_impedance"].source == data.vendor
    assert outputs["raw_eit"] is outputs["raw_pixel_impedance_signal"]
    assert outputs["raw_global_impedance"] is outputs["raw_global_impedance_signal"]


def test_sentec_loading_sums_native_pixels(monkeypatch):
    pytest.importorskip("eitprocessing")
    from eitprocessing.datahandling.eitdata import EITData

    from m3resp import M3Session

    entry = MANIFEST["inputs"]["sentec_healthy_volunteer_1b"]

    def upstream_sum_was_called(*args, **kwargs):
        pytest.fail("Loading Sentec called the upstream pixel sum")

    monkeypatch.setattr(EITData, "get_summed_impedance", upstream_sum_was_called)
    session = M3Session()
    session.load_eit(os.path.join(ROOT, entry["file"]), vendor="sentec")
    np.testing.assert_array_equal(
        session.eit.global_impedance.values,
        np.nansum(session.eit.pixel_impedance.values, axis=(1, 2)),
    )


def test_custom_sequence_keeps_interval_values_extra_images_and_waveforms():
    pytest.importorskip("eitprocessing")
    from dataclasses import replace

    from eitprocessing.datahandling.intervaldata import (
        IntervalData as UpstreamIntervals,
    )

    from m3resp.adapters import EITProcessingAdapter
    from m3resp.adapters.eitprocessing_adapter import (
        file_data_to_sequence,
        sequence_to_file_data,
    )

    sequence = file_data_to_sequence(_device_data())
    sequence.eit_data.add(replace(sequence.eit_data["raw"], label="extra"))
    waveform = sequence.continuous_data["pressure"]
    sequence.continuous_data.add(
        replace(waveform, label="extra_wave", parameters={"setting": 1})
    )
    sequence.interval_data.add(
        UpstreamIntervals(
            label="settings",
            name="settings",
            unit="cmH2O",
            category="pressure",
            intervals=[(0.0, 2.0), (2.0, 6.0)],
            values=[5.0, 10.0],
            default_partial_inclusion=True,
        )
    )
    expected = sequence.select_by_index(1, 5)
    native = sequence_to_file_data(sequence)
    np.testing.assert_array_equal(native.intervals["settings"].values, [5.0, 10.0])

    selected = EITProcessingAdapter().slice_sequence(sequence, 1, 5)

    assert selected == expected
    assert selected.continuous_data["extra_wave"].parameters == {}
    np.testing.assert_array_equal(
        selected.interval_data["settings"].values, [5.0, 10.0]
    )


def test_repeated_recording_slices_match_upstream(recording):
    pytest.importorskip("eitprocessing")
    from m3resp import M3Session
    from m3resp.modalities.eit import keep_frames

    _, data = recording
    session = M3Session()
    session.load_eit(data.path, vendor=data.vendor)
    expected = session.eit.data.select_by_index(10, 400).select_by_index(5, 100)
    keep_frames(session.eit, 10, 400)
    keep_frames(session.eit, 5, 100)
    assert session.eit.data == expected
