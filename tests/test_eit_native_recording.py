"""The m3resp fields of a loaded `EITRecording` hold the same data as its Sequence."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from m3resp.adapters import EITProcessingAdapter
from m3resp.adapters.eitprocessing_adapter import (
    file_data_to_sequence,
    sequence_to_file_data,
)
from m3resp.core.session import M3Session
from m3resp.eit.io import EITFileData, join_eit_files, read_eit_file

pytest.importorskip("eitprocessing")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEST_FILES = {
    "draeger": os.path.join(DATA_DIR, "draeger_synthetic_draeger_20Hz.bin"),
    "sentec": os.path.join(DATA_DIR, "sentec_healthy_volunteer_1b.zri"),
    "timpel": os.path.join(DATA_DIR, "timpel_healthy_volunteer_1_60-120s.txt"),
}
# Bytes per frame of the Draeger test file (1200 frames at 20 Hz).
DRAEGER_FRAME_SIZE = 4358


def _write_draeger_frames(target: Path, first: int, stop: int) -> str:
    """Write frames [first, stop) of the Draeger test file to a new file."""

    with open(TEST_FILES["draeger"], "rb") as handle:
        handle.seek(first * DRAEGER_FRAME_SIZE)
        target.write_bytes(handle.read((stop - first) * DRAEGER_FRAME_SIZE))
    return os.fspath(target)


def _assert_native_matches_sequence(native: EITFileData, sequence) -> None:
    """Every m3resp item holds the values of the Sequence item with its key."""

    raw = sequence.eit_data["raw"]
    np.testing.assert_array_equal(native.pixel_impedance.values, raw.pixel_impedance)
    np.testing.assert_array_equal(native.pixel_impedance.time, raw.time)
    assert native.sample_frequency == raw.sample_frequency
    for key, signal in native.signals.items():
        data = sequence.continuous_data[key]
        np.testing.assert_array_equal(signal.values, data.values)
        np.testing.assert_array_equal(signal.time, data.time)
        assert signal.unit == data.unit
    assert list(native.events) == list(sequence.sparse_data)
    for key, marks in native.events.items():
        data = sequence.sparse_data[key]
        np.testing.assert_array_equal([e.time for e in marks.events], data.time)
        if data.values is not None:
            assert list(marks.values or []) == [e.marker for e in data.values]
    assert list(native.intervals) == list(sequence.interval_data)
    for key, breaths in native.intervals.items():
        data = sequence.interval_data[key]
        assert [(b.start_time, b.end_time) for b in breaths.intervals] == [
            (i.start_time, i.end_time) for i in data.intervals
        ]


def _assert_same_file_data(first: EITFileData, second: EITFileData) -> None:
    assert (first.vendor, first.sample_frequency) == (
        second.vendor,
        second.sample_frequency,
    )
    assert first.pixel_impedance == second.pixel_impedance
    assert list(first.signals) == list(second.signals)
    assert first.signals == second.signals
    assert first.events == second.events
    assert first.intervals == second.intervals


class TestLoad:
    @pytest.mark.parametrize("vendor", sorted(TEST_FILES))
    def test_native_fields_are_the_reader_output(self, vendor):
        session = M3Session()

        session.load_eit(TEST_FILES[vendor], vendor=vendor)

        recording = session.eit
        expected = read_eit_file(TEST_FILES[vendor], vendor)
        loaded = EITFileData(
            vendor=vendor,
            path=TEST_FILES[vendor],
            sample_frequency=recording.sample_frequency,
            pixel_impedance=recording.pixel_impedance,
            signals=recording.signals,
            events=recording.events,
            intervals=recording.intervals,
        )
        _assert_same_file_data(loaded, expected)

    @pytest.mark.parametrize("vendor", sorted(TEST_FILES))
    def test_native_fields_hold_the_data_of_the_sequence(self, vendor):
        session = M3Session()
        session.load_eit(TEST_FILES[vendor], vendor=vendor)
        recording = session.eit

        native = EITFileData(
            vendor=vendor,
            path=TEST_FILES[vendor],
            sample_frequency=recording.sample_frequency,
            pixel_impedance=recording.pixel_impedance,
            signals=recording.signals,
            events=recording.events,
            intervals=recording.intervals,
        )
        _assert_native_matches_sequence(native, recording.data)


class TestJoin:
    def test_joined_files_hold_the_data_of_the_joined_sequence(self, tmp_path):
        files = [
            _write_draeger_frames(tmp_path / "part_1.bin", 0, 600),
            _write_draeger_frames(tmp_path / "part_2.bin", 600, 1200),
        ]
        session = M3Session()

        session.load_eit(files, vendor="draeger")

        recording = session.eit
        whole = read_eit_file(TEST_FILES["draeger"], "draeger")
        native = EITFileData(
            vendor="draeger",
            path=files[0],
            sample_frequency=recording.sample_frequency,
            pixel_impedance=recording.pixel_impedance,
            signals=recording.signals,
            events=recording.events,
            intervals=recording.intervals,
        )
        _assert_native_matches_sequence(native, recording.data)
        np.testing.assert_array_equal(
            recording.pixel_impedance.values, whole.pixel_impedance.values
        )
        for key, signal in whole.signals.items():
            np.testing.assert_array_equal(recording.signals[key].values, signal.values)

    def test_a_pause_gives_one_warning(self, tmp_path):
        files = [
            _write_draeger_frames(tmp_path / "early.bin", 0, 400),
            _write_draeger_frames(tmp_path / "late.bin", 800, 1200),
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            M3Session().load_eit(files, vendor="draeger")

        gaps = [w for w in caught if "gap of 20.000 s" in str(w.message)]
        assert len(gaps) == 1

    def test_files_of_different_devices_are_refused(self):
        draeger = read_eit_file(TEST_FILES["draeger"], "draeger")
        sentec = read_eit_file(TEST_FILES["sentec"], "sentec")
        later = EITFileData(
            vendor=sentec.vendor,
            path=sentec.path,
            sample_frequency=sentec.sample_frequency,
            pixel_impedance=sentec.pixel_impedance,
        )
        later.pixel_impedance.time = later.pixel_impedance.time + 1e6

        with pytest.raises(ValueError, match="cannot be joined.*vendor 'draeger'"):
            join_eit_files([draeger, later], ["a.bin", "b.zri"])

    def test_files_out_of_order_are_refused(self, tmp_path):
        files = [
            _write_draeger_frames(tmp_path / "part_1.bin", 0, 600),
            _write_draeger_frames(tmp_path / "part_2.bin", 600, 1200),
        ]
        data = [read_eit_file(path, "draeger") for path in files]

        with pytest.raises(ValueError, match="List the files in recording order"):
            join_eit_files(list(reversed(data)), list(reversed(files)))

    def test_one_file_is_returned_as_it_is(self):
        data = read_eit_file(TEST_FILES["timpel"], "timpel")

        assert join_eit_files([data], [TEST_FILES["timpel"]]) is data

    def test_no_file_is_refused(self):
        with pytest.raises(ValueError, match="at least one EIT file"):
            join_eit_files([], [])


class TestSequenceToFileData:
    @pytest.mark.parametrize("vendor", sorted(TEST_FILES))
    def test_round_trip_gives_the_reader_output(self, vendor):
        read = read_eit_file(TEST_FILES[vendor], vendor)

        back = sequence_to_file_data(file_data_to_sequence(read))

        assert back is not None
        _assert_same_file_data(back, read)

    def test_other_objects_give_none(self):
        assert sequence_to_file_data({"path": "x"}) is None
        assert sequence_to_file_data(object()) is None


class TestCustomLoader:
    def test_loader_may_return_eit_file_data(self):
        read = read_eit_file(TEST_FILES["timpel"], "timpel")
        session = M3Session(
            eit_adapter=EITProcessingAdapter(loader=lambda path, **kwargs: read)
        )

        session.load_eit("anything.txt", loader="custom")

        recording = session.eit
        assert recording.vendor == "timpel"
        assert recording.pixel_impedance is read.pixel_impedance
        _assert_native_matches_sequence(read, recording.data)

    def test_loader_result_without_eit_data_leaves_the_fields_empty(self):
        session = M3Session(
            eit_adapter=EITProcessingAdapter(loader=lambda path, **kwargs: {"p": path})
        )

        session.load_eit("anything.bin", loader="custom")

        recording = session.eit
        assert recording.pixel_impedance is None
        assert recording.signals == recording.events == recording.intervals == {}


def test_cut_recording_keeps_the_native_fields_in_step():
    session = M3Session()
    session.load_eit(TEST_FILES["draeger"], vendor="draeger")
    keys = list(session.eit.signals)

    session.slice_eit(10.0, 20.0)

    recording = session.eit
    assert list(recording.signals) == keys
    np.testing.assert_array_equal(
        recording.pixel_impedance.values, recording.raw.pixel_impedance
    )
    np.testing.assert_array_equal(recording.pixel_impedance.time, recording.raw.time)
    for key, signal in recording.signals.items():
        np.testing.assert_array_equal(
            signal.values, recording.data.continuous_data[key].values
        )
