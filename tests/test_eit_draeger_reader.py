"""`read_draeger` reads a Draeger `.bin` file the same way as eitprocessing 1.8.7."""

from __future__ import annotations

import os
import struct
import warnings

import numpy as np
import pytest

from m3resp.eit.io.draeger import read_draeger

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DRAEGER_FILE = os.path.join(TESTS_DIR, "data", "draeger_synthetic_draeger_20Hz.bin")
GOLDEN_READER = os.path.join(
    TESTS_DIR, "regression", "golden", "eit", "draeger_synthetic_20Hz", "reader.npz"
)

SECONDS_PER_DAY = 24 * 60 * 60
N_MEDIBUS = {"original": 52, "pressure_pod": 58}


def _frame(
    time_s: float,
    *,
    n_medibus: int,
    pixel: float = 1.0,
    min_max_flag: int = 0,
    marker: int = 0,
    text: str = "",
    timing_error: int = 0,
    medibus: float = 2.0,
) -> bytes:
    """Pack one Draeger frame (little endian, no padding)."""

    pixels = np.full(1024, pixel, dtype="<f4")
    pixels[0] = pixel + time_s  # one pixel changes from frame to frame
    medibus_values = np.full(n_medibus, medibus, dtype="<f4")
    return b"".join(
        [
            struct.pack("<d", time_s / SECONDS_PER_DAY),
            struct.pack("<f", 0.0),
            pixels.tobytes(),
            struct.pack("<ii", min_max_flag, marker),
            text.encode("ascii").ljust(30, b" "),
            struct.pack("<i", timing_error),
            medibus_values.tobytes(),
        ]
    )


def _write(path, frames: list[bytes]) -> str:
    path = os.fspath(path)
    with open(path, "wb") as handle:
        handle.write(b"".join(frames))
    return path


def _regular_file(tmp_path, *, n_frames=40, fs=20.0, fmt="original", **kwargs):
    n_medibus = N_MEDIBUS[fmt]
    frames = [
        _frame(1000.0 + index / fs, n_medibus=n_medibus, **kwargs)
        for index in range(n_frames)
    ]
    return _write(tmp_path / f"{fmt}.bin", frames)


def _marked_file(tmp_path) -> str:
    """A file with breath flags, event markers, a NaN Medibus value and a midnight."""

    frames = []
    for index in range(30):
        time_s = SECONDS_PER_DAY - 0.5 + index * 0.05  # crosses midnight
        if time_s >= SECONDS_PER_DAY:
            time_s -= SECONDS_PER_DAY
        marker = 1 if index < 10 else (2 if index < 20 else 3)
        frames.append(
            _frame(
                time_s,
                n_medibus=52,
                min_max_flag=(-1, 0, 1)[index % 3],
                marker=marker,
                text=f"event {marker}",
                medibus=-1e31 if index == 5 else 2.0,
            )
        )
    return _write(tmp_path / "marked.bin", frames)


def _upstream_load(path, **kwargs):
    loading = pytest.importorskip("eitprocessing.datahandling.loading")
    return loading.load_eit_data(path, vendor="draeger", **kwargs)


def _assert_same_as_upstream(ours, sequence) -> None:
    raw = sequence.eit_data["raw"]
    assert ours.sample_frequency == raw.sample_frequency
    np.testing.assert_array_equal(ours.pixel_impedance.time, raw.time)
    np.testing.assert_array_equal(ours.pixel_impedance.values, raw.pixel_impedance)
    assert list(ours.signals) == list(sequence.continuous_data)
    for label, data in sequence.continuous_data.items():
        signal = ours.signals[label]
        assert signal.unit == data.unit
        assert signal.values.dtype == np.asarray(data.values).dtype
        np.testing.assert_array_equal(signal.time, data.time)
        np.testing.assert_array_equal(signal.values, data.values)
    assert list(ours.events) == list(sequence.sparse_data)
    for label, data in sequence.sparse_data.items():
        marks = ours.events[label]
        np.testing.assert_array_equal([e.time for e in marks.events], data.time)
        if data.values is None:
            assert marks.values is None
        else:
            assert list(marks.values) == [e.marker for e in data.values]
            assert [e.label for e in marks.events] == [e.text for e in data.values]
    assert ours.intervals == {}
    assert len(sequence.interval_data) == 0


class TestGoldenFile:
    """The Draeger test file against the saved eitprocessing 1.8.7 results."""

    @pytest.fixture(scope="class")
    def golden(self):
        with np.load(GOLDEN_READER) as stored:
            return dict(stored)

    @pytest.fixture(scope="class")
    def ours(self):
        return read_draeger(DRAEGER_FILE)

    def test_images_time_and_sample_frequency(self, ours, golden):
        assert ours.vendor == str(golden["vendor"])
        assert ours.sample_frequency == float(golden["sample_frequency"])
        np.testing.assert_array_equal(ours.pixel_impedance.time, golden["time"])
        np.testing.assert_array_equal(
            ours.pixel_impedance.values, golden["pixel_impedance"]
        )

    def test_waveforms(self, ours, golden):
        labels = sorted(
            {key.split("/")[1] for key in golden if key.startswith("continuous/")}
        )
        assert sorted(ours.signals) == labels
        for label in labels:
            signal = ours.signals[label]
            assert signal.unit == str(golden[f"continuous/{label}/unit"])
            np.testing.assert_array_equal(
                signal.time, golden[f"continuous/{label}/time"]
            )
            np.testing.assert_array_equal(
                signal.values, golden[f"continuous/{label}/values"]
            )

    def test_breath_flags_and_event_markers(self, ours, golden):
        labels = sorted(
            {key.split("/")[1] for key in golden if key.startswith("sparse/")}
        )
        assert sorted(ours.events) == labels
        for label in labels:
            marks = ours.events[label]
            np.testing.assert_array_equal(
                [e.time for e in marks.events], golden[f"sparse/{label}/time"]
            )
            if f"sparse/{label}/marker" in golden:
                assert list(marks.values) == golden[f"sparse/{label}/marker"].tolist()
                # A NumPy text array drops the "\x00" bytes at the end of each
                # text, so the texts go through the same conversion as in the
                # golden file.
                texts = np.asarray([e.label for e in marks.events])
                assert texts.tolist() == golden[f"sparse/{label}/text"].tolist()

    def test_no_intervals(self, ours, golden):
        assert ours.intervals == {}
        assert not any(key.startswith("interval/") for key in golden)


class TestSameAsEitprocessing:
    """Small written files, read by both readers."""

    @pytest.mark.parametrize("fmt", ["original", "pressure_pod"])
    def test_both_file_formats(self, tmp_path, fmt):
        path = _regular_file(tmp_path, fmt=fmt)

        ours = read_draeger(path)

        _assert_same_as_upstream(ours, _upstream_load(path))
        n_waveforms = 6 if fmt == "original" else 10
        assert len(ours.signals) == n_waveforms + 1  # and the global impedance

    def test_flags_markers_nan_and_midnight(self, tmp_path):
        path = _marked_file(tmp_path)

        ours = read_draeger(path)

        _assert_same_as_upstream(ours, _upstream_load(path))
        assert np.all(np.diff(ours.pixel_impedance.time) > 0)
        assert np.isnan(ours.signals["airway pressure"].values[5])
        assert [e.label for e in ours.events["events_(draeger)"].events] == [
            "event 1",
            "event 2",
            "event 3",
        ]

    @pytest.mark.parametrize(
        ("first_frame", "max_frames"), [(0, 5), (10, None), (11, 9), (25, None)]
    )
    def test_first_frame_and_max_frames(self, tmp_path, first_frame, max_frames):
        path = _marked_file(tmp_path)
        options = {"first_frame": first_frame, "max_frames": max_frames}

        ours = read_draeger(path, **options)

        _assert_same_as_upstream(ours, _upstream_load(path, **options))

    def test_given_sample_frequency(self, tmp_path):
        path = _regular_file(tmp_path)

        with pytest.warns(RuntimeWarning, match="does not match the estimated"):
            ours = read_draeger(path, sample_frequency=25.0)
        with pytest.warns(RuntimeWarning, match="does not match the estimated"):
            sequence = _upstream_load(path, sample_frequency=25.0)

        assert ours.sample_frequency == 25.0
        _assert_same_as_upstream(ours, sequence)


class TestChecks:
    def test_unknown_file_size_raises(self, tmp_path):
        path = _write(tmp_path / "short.bin", [b"\x00" * 100])

        with pytest.raises(OSError, match="does not match the supported"):
            read_draeger(path)

    def test_first_frame_after_the_end_raises(self, tmp_path):
        path = _regular_file(tmp_path, n_frames=4)

        with pytest.raises(ValueError, match="larger than the total number"):
            read_draeger(path, first_frame=5)
        with pytest.raises(ValueError, match="No frames to load"):
            read_draeger(path, first_frame=4)

    def test_too_many_frames_warns(self, tmp_path):
        path = _regular_file(tmp_path, n_frames=4)

        with pytest.warns(RuntimeWarning, match="4 frames will be loaded"):
            ours = read_draeger(path, max_frames=10)

        assert ours.pixel_impedance.n_samples == 4

    def test_timing_error_warns(self, tmp_path):
        path = _regular_file(tmp_path, n_frames=4, timing_error=1)

        with pytest.warns(RuntimeWarning, match="timing error"):
            read_draeger(path)

    def test_time_that_goes_back_raises(self, tmp_path):
        frames = [_frame(t, n_medibus=52) for t in (10.0, 10.05, 10.02)]
        path = _write(tmp_path / "back.bin", frames)

        with pytest.raises(ValueError, match="not strictly monotonically"):
            read_draeger(path)

    def test_one_frame_needs_a_sample_frequency(self, tmp_path):
        path = _regular_file(tmp_path, n_frames=1)

        with pytest.raises(ValueError, match="Could not estimate sample frequency"):
            read_draeger(path)
        with pytest.warns(RuntimeWarning, match="Could not estimate"):
            ours = read_draeger(path, sample_frequency=20)

        assert ours.sample_frequency == 20.0

    def test_sample_frequency_must_be_a_number(self, tmp_path):
        path = _regular_file(tmp_path)

        with pytest.raises(TypeError, match="invalid type"):
            read_draeger(path, sample_frequency="20")  # type: ignore[arg-type]

    def test_close_sample_frequency_gives_no_warning(self, tmp_path):
        path = _regular_file(tmp_path)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ours = read_draeger(path, sample_frequency=20.0)

        assert ours.sample_frequency == 20.0


def test_signals_are_tagged():
    ours = read_draeger(DRAEGER_FILE)

    assert ours.pixel_impedance.channel == "pixel_impedance"
    assert ours.pixel_impedance.modality == "eit"
    assert ours.signals["global_impedance_(raw)"].channel == "global_impedance"
    assert ours.signals["airway pressure"].modality == "ventilator"
    assert {s.source for s in ours.signals.values()} == {"draeger"}
