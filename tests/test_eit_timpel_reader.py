"""`read_timpel` reads a Timpel `.txt` file the same way as eitprocessing 1.8.7."""

from __future__ import annotations

import os

import numpy as np
import pytest

from m3resp.eit.io.timpel import _make_breaths, read_timpel

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TIMPEL_FILE = os.path.join(TESTS_DIR, "data", "timpel_healthy_volunteer_1_60-120s.txt")
GOLDEN_READER = os.path.join(
    TESTS_DIR,
    "regression",
    "golden",
    "eit",
    "timpel_healthy_volunteer_1_60-120s",
    "reader.npz",
)

#: Breath minima and maxima of the written test files, as frame numbers.
MINIMA = (5, 25, 45, 65)
MAXIMA = (2, 15, 35, 55, 60)


def _write_txt(tmp_path, *, n_frames: int = 80, name: str = "test.txt") -> str:
    """Write a small Timpel file with images, waveforms and device marks.

    One pixel is -1000 (no value) in every frame. The minima and maxima
    follow ``MINIMA`` and ``MAXIMA``; every 10th frame has a QRS mark.
    """

    rng = np.random.default_rng(7)
    rows = []
    for index in range(n_frames):
        pixels = rng.normal(100.0, 5.0, 1024).round(4)
        pixels[3] = -1000
        row = [
            *pixels,
            round(5 + np.sin(index / 5), 4),
            round(np.cos(index / 5), 4),
            round(0.4 + 0.1 * np.sin(index / 5), 4),
            1 if index in MINIMA else 0,
            1 if index in MAXIMA else 0,
            1 if index % 10 == 0 else 0,
        ]
        rows.append(",".join(f"{value:g}" for value in row))
    path = os.fspath(tmp_path / name)
    with open(path, "w", encoding="ascii") as handle:
        handle.write("\n".join(rows) + "\n")
    return path


def _upstream():
    return pytest.importorskip("eitprocessing.datahandling.loading.timpel")


def _upstream_load(path, **kwargs):
    loading = pytest.importorskip("eitprocessing.datahandling.loading")
    return loading.load_eit_data(path, vendor="timpel", **kwargs)


def _breath_rows(breaths) -> list[tuple[float, float, float]]:
    return [(b.start_time, b.extremum_time, b.end_time) for b in breaths]


def _upstream_breath_rows(breaths) -> list[tuple[float, float, float]]:
    return [(b.start_time, b.middle_time, b.end_time) for b in breaths]


def _assert_same_as_upstream(ours, sequence) -> None:
    raw = sequence.eit_data["raw"]
    assert ours.sample_frequency == raw.sample_frequency
    np.testing.assert_array_equal(ours.pixel_impedance.time, raw.time)
    np.testing.assert_array_equal(ours.pixel_impedance.values, raw.pixel_impedance)
    assert list(ours.signals) == list(sequence.continuous_data)
    for label, data in sequence.continuous_data.items():
        signal = ours.signals[label]
        assert signal.unit == data.unit
        np.testing.assert_array_equal(signal.time, data.time)
        np.testing.assert_array_equal(signal.values, data.values)
    assert list(ours.events) == list(sequence.sparse_data)
    for label, data in sequence.sparse_data.items():
        times = [e.time for e in ours.events[label].events]
        np.testing.assert_array_equal(times, data.time)
    assert list(ours.intervals) == list(sequence.interval_data)
    for label, data in sequence.interval_data.items():
        intervals = ours.intervals[label].intervals
        assert [(i.start_time, i.end_time) for i in intervals] == [
            (i.start_time, i.end_time) for i in data.intervals
        ]
        assert _breath_rows(intervals) == _upstream_breath_rows(data.values)


class TestGoldenFile:
    """The Timpel test file against the saved eitprocessing 1.8.7 results."""

    @pytest.fixture(scope="class")
    def golden(self):
        with np.load(GOLDEN_READER) as stored:
            return dict(stored)

    @pytest.fixture(scope="class")
    def ours(self):
        return read_timpel(TIMPEL_FILE)

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

    def test_device_marks(self, ours, golden):
        labels = sorted(
            {key.split("/")[1] for key in golden if key.startswith("sparse/")}
        )
        assert sorted(ours.events) == labels
        for label in labels:
            np.testing.assert_array_equal(
                [e.time for e in ours.events[label].events],
                golden[f"sparse/{label}/time"],
            )

    def test_device_breaths(self, ours, golden):
        breaths = ours.intervals["breaths_(timpel)"].intervals

        assert list(ours.intervals) == ["breaths_(timpel)"]
        np.testing.assert_array_equal(
            [(b.start_time, b.end_time) for b in breaths],
            golden["interval/breaths_(timpel)/start_end"],
        )
        np.testing.assert_array_equal(
            _breath_rows(breaths), golden["interval/breaths_(timpel)/breaths"]
        )
        assert len(breaths) == 9


class TestSameAsEitprocessing:
    """Small written files, read by both readers."""

    def test_whole_file(self, tmp_path):
        path = _write_txt(tmp_path)

        ours = read_timpel(path)

        _assert_same_as_upstream(ours, _upstream_load(path))
        assert np.isnan(ours.pixel_impedance.values[:, 0, 3]).all()
        assert len(ours.intervals["breaths_(timpel)"]) == 3

    @pytest.mark.parametrize(
        ("first_frame", "max_frames"), [(0, 30), (10, None), (20, 40), (50, 100)]
    )
    def test_first_frame_and_max_frames(self, tmp_path, first_frame, max_frames):
        path = _write_txt(tmp_path)
        options = {"first_frame": first_frame, "max_frames": max_frames}

        ours = read_timpel(path, **options)

        _assert_same_as_upstream(ours, _upstream_load(path, **options))
        assert ours.pixel_impedance.time[0] == first_frame / 50

    @pytest.mark.parametrize("sample_frequency", [20.0, 0, None])
    def test_sample_frequency(self, tmp_path, sample_frequency):
        path = _write_txt(tmp_path)

        ours = read_timpel(path, sample_frequency=sample_frequency)

        expected = sample_frequency or 50.0
        assert ours.sample_frequency == expected
        _assert_same_as_upstream(
            ours, _upstream_load(path, sample_frequency=sample_frequency)
        )

    def test_too_many_frames_warns(self, tmp_path):
        path = _write_txt(tmp_path)

        with pytest.warns(RuntimeWarning, match="70 frames have been loaded"):
            ours = read_timpel(path, first_frame=10, max_frames=100)

        assert ours.pixel_impedance.n_samples == 70


class TestMakeBreaths:
    """`_make_breaths` gives the breaths of eitprocessing's `_make_breaths`."""

    @pytest.mark.parametrize("seed", range(40))
    def test_random_minima_and_maxima(self, seed):
        upstream = _upstream()
        rng = np.random.default_rng(seed)
        n_frames = 200
        time = np.arange(n_frames) / 50
        gi = rng.normal(0.0, 1.0, n_frames)
        flags = rng.choice(3, size=n_frames, p=[0.9, 0.05, 0.05])
        min_indices = np.nonzero(flags == 1)[0]
        max_indices = np.nonzero(flags == 2)[0]

        ours = _make_breaths(time, min_indices, max_indices, gi)
        try:
            _, theirs = upstream._make_breaths(time, min_indices, max_indices, gi)
        except ValueError:
            # eitprocessing fails when fewer than 2 minima are left; m3resp
            # gives no breaths there (see the Provenance block of timpel.py).
            assert ours == []
        else:
            assert _breath_rows(ours) == _upstream_breath_rows(theirs)

    def test_maximum_between_minima_only_after_cleaning(self):
        upstream = _upstream()
        time = np.arange(10) / 50
        gi = np.array([0, 0, 5, 1, 0, 0, 0, 0, 0, 0], dtype=float)

        ours = _make_breaths(time, np.array([1, 3, 6]), np.array([2, 4]), gi)
        _, theirs = upstream._make_breaths(
            time, np.array([1, 3, 6]), np.array([2, 4]), gi
        )

        assert _breath_rows(ours) == _upstream_breath_rows(theirs)

    def test_fewer_than_two_minima_left_gives_no_breaths(self):
        upstream = _upstream()
        time = np.arange(10) / 50
        gi = np.zeros(10)
        minima, maxima = np.array([2, 5]), np.array([8])

        assert _make_breaths(time, minima, maxima, gi) == []
        with pytest.raises(ValueError, match="not enough values to unpack"):
            upstream._make_breaths(time, minima, maxima, gi)

    def test_too_few_marks_give_no_breaths(self):
        time = np.arange(10) / 50

        assert _make_breaths(time, np.array([2]), np.array([4]), np.zeros(10)) == []
        assert _make_breaths(time, np.array([2, 6]), np.array([]), np.zeros(10)) == []


class TestFewFrames:
    """Bug fix: eitprocessing 1.8.7 raises IndexError with 0 or 1 frames left."""

    def test_no_frame_left_raises_value_error(self, tmp_path):
        path = _write_txt(tmp_path, n_frames=10)

        with pytest.raises(ValueError, match="larger than the total number"):
            read_timpel(path, first_frame=10)
        with pytest.raises(IndexError):
            _upstream_load(path, first_frame=10)

    @pytest.mark.parametrize(("first_frame", "max_frames"), [(9, None), (0, 1), (4, 1)])
    def test_one_frame_is_read(self, tmp_path, first_frame, max_frames):
        path = _write_txt(tmp_path, n_frames=10)
        whole = read_timpel(path)

        ours = read_timpel(path, first_frame=first_frame, max_frames=max_frames)

        assert ours.pixel_impedance.n_samples == 1
        np.testing.assert_array_equal(
            ours.pixel_impedance.values, whole.pixel_impedance.values[[first_frame]]
        )
        assert ours.pixel_impedance.time[0] == whole.pixel_impedance.time[first_frame]
        assert len(ours.intervals["breaths_(timpel)"]) == 0
        with pytest.raises(IndexError):
            _upstream_load(path, first_frame=first_frame, max_frames=max_frames)


class TestChecks:
    def test_wrong_number_of_columns_raises(self, tmp_path):
        path = os.fspath(tmp_path / "narrow.txt")
        with open(path, "w", encoding="ascii") as handle:
            handle.write("1,2,3\n4,5,6\n")

        with pytest.raises(OSError, match="width of 1030 columns"):
            read_timpel(path)

    def test_binary_file_raises(self, tmp_path):
        path = os.fspath(tmp_path / "binary.txt")
        with open(path, "wb") as handle:
            handle.write(b"\xff\xfe\x00\x81" * 100)

        with pytest.raises(OSError, match="could not be read as Timpel data"):
            read_timpel(path)


def test_signals_are_tagged():
    ours = read_timpel(TIMPEL_FILE)

    assert ours.pixel_impedance.channel == "pixel_impedance"
    assert ours.signals["global_impedance_(raw)"].channel == "global_impedance"
    assert ours.signals["global_impedance_(raw)"].category == "impedance"
    assert ours.signals["airway_pressure_(timpel)"].category == "airway_pressure"
    assert ours.signals["flow_(timpel)"].category == "airflow"
    assert ours.signals["volume_(timpel)"].modality == "ventilator"
    assert {s.source for s in ours.signals.values()} == {"timpel"}
