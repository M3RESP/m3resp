"""`read_sentec` reads a Sentec `.zri` file the same way as eitprocessing 1.8.7."""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest

from m3resp.eit.io.sentec import read_sentec

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SENTEC_FILE = os.path.join(TESTS_DIR, "data", "sentec_healthy_volunteer_1b.zri")
GOLDEN_READER = os.path.join(
    TESTS_DIR,
    "regression",
    "golden",
    "eit",
    "sentec_healthy_volunteer_1b",
    "reader.npz",
)

MEASUREMENT, CONFIGURATION = 16, 64
TIMESTAMP, ZERO_REF_IMAGE, SAMPLE_FREQUENCY = 0, 5, 1


def _field(data_id: int, payload: bytes) -> bytes:
    return struct.pack("<BH", data_id, len(payload)) + payload


def _record(domain: int, fields: list[bytes]) -> bytes:
    return struct.pack("<QBB", 0, domain, len(fields)) + b"".join(fields)


def _image_payload(index: int, version: int, width=32, height=32) -> bytes:
    values = np.arange(width * height, dtype="<f4") + index
    quality = b"\x07" if version > 1 else b""
    return quality + struct.pack("<BB", width, height) + values.tobytes()


def _write_zri(
    tmp_path,
    *,
    n_frames: int = 30,
    version: int = 2,
    sample_frequency: float | None = 50.86,
    start_us: int = 1_000_000,
    step_us: int = 19_662,
    name: str = "test.zri",
) -> str:
    """Write a small Sentec file: an optional sample frequency, then frames.

    Each frame record holds a time stamp and an image. An unknown data field
    and an empty data field sit between the frames.
    """

    records = []
    if sample_frequency is not None:
        records.append(
            _record(
                CONFIGURATION,
                [_field(SAMPLE_FREQUENCY, struct.pack("<f", sample_frequency))],
            )
        )
    for index in range(n_frames):
        stamp = (start_us + index * step_us) % (2**32 - 1)
        records.append(
            _record(
                MEASUREMENT,
                [
                    _field(TIMESTAMP, struct.pack("<I", stamp)),
                    _field(9, b"\x01\x02\x03"),
                    _field(ZERO_REF_IMAGE, _image_payload(index, version)),
                    _field(TIMESTAMP + 1, b""),
                ],
            )
        )
    path = os.fspath(tmp_path / name)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<B", version) + b"".join(records))
    return path


def _upstream_load(path, **kwargs):
    loading = pytest.importorskip("eitprocessing.datahandling.loading")
    return loading.load_eit_data(path, vendor="sentec", **kwargs)


def _assert_same_as_upstream(ours, sequence) -> None:
    raw = sequence.eit_data["raw"]
    assert ours.sample_frequency == raw.sample_frequency
    np.testing.assert_array_equal(ours.pixel_impedance.time, raw.time)
    np.testing.assert_array_equal(ours.pixel_impedance.values, raw.pixel_impedance)
    assert ours.signals == {}
    assert len(sequence.continuous_data) == 0
    assert ours.events == {}
    assert len(sequence.sparse_data) == 0
    assert ours.intervals == {}
    assert len(sequence.interval_data) == 0


class TestGoldenFile:
    """The Sentec test file against the saved eitprocessing 1.8.7 results."""

    @pytest.fixture(scope="class")
    def golden(self):
        with np.load(GOLDEN_READER) as stored:
            return dict(stored)

    def test_images_time_and_sample_frequency(self, golden):
        ours = read_sentec(SENTEC_FILE)

        assert ours.vendor == str(golden["vendor"])
        assert ours.sample_frequency == float(golden["sample_frequency"])
        np.testing.assert_array_equal(ours.pixel_impedance.time, golden["time"])
        np.testing.assert_array_equal(
            ours.pixel_impedance.values, golden["pixel_impedance"]
        )

    def test_no_waveforms_or_markers(self, golden):
        ours = read_sentec(SENTEC_FILE)

        assert ours.signals == {} and ours.events == {} and ours.intervals == {}
        assert not any("/" in key for key in golden)


class TestSameAsEitprocessing:
    """Small written files, read by both readers."""

    @pytest.mark.parametrize("version", [1, 2])
    def test_both_file_versions(self, tmp_path, version):
        path = _write_zri(tmp_path, version=version)

        _assert_same_as_upstream(read_sentec(path), _upstream_load(path))

    def test_sign_of_the_image_is_turned_around(self, tmp_path):
        path = _write_zri(tmp_path, n_frames=2)

        ours = read_sentec(path)

        assert ours.pixel_impedance.values[1, 0, 1] == -2.0

    def test_no_sample_frequency_in_the_file(self, tmp_path):
        path = _write_zri(tmp_path, sample_frequency=None)

        ours = read_sentec(path)

        assert ours.sample_frequency == 50.2
        _assert_same_as_upstream(ours, _upstream_load(path))

    def test_time_stamps_that_start_again(self, tmp_path):
        path = _write_zri(tmp_path, start_us=2**32 - 1 - 100_000, step_us=40_000)

        ours = read_sentec(path)

        _assert_same_as_upstream(ours, _upstream_load(path))
        assert np.all(np.diff(ours.pixel_impedance.time) > 0)

    @pytest.mark.parametrize(
        ("first_frame", "max_frames"), [(0, 5), (10, None), (0, 40), (5, 30)]
    )
    def test_first_frame_and_max_frames(self, tmp_path, first_frame, max_frames):
        path = _write_zri(tmp_path)
        options = {"first_frame": first_frame, "max_frames": max_frames}

        ours = read_sentec(path, **options)

        _assert_same_as_upstream(ours, _upstream_load(path, **options))

    def test_given_sample_frequency_differs_from_the_file(self, tmp_path):
        path = _write_zri(tmp_path)

        with pytest.warns(RuntimeWarning, match="differs from value found in file"):
            ours = read_sentec(path, sample_frequency=20.0)
        with pytest.warns(RuntimeWarning, match="differs from value found in file"):
            sequence = _upstream_load(path, sample_frequency=20.0)

        assert ours.sample_frequency == pytest.approx(50.86)
        _assert_same_as_upstream(ours, sequence)


class TestFirstFrameWithMaxFrames:
    """Bug fix: eitprocessing 1.8.7 raises IndexError for these options."""

    @pytest.mark.parametrize(
        ("first_frame", "max_frames"), [(5, 10), (1, 29), (20, 10), (29, 1)]
    )
    def test_gives_the_frames_after_first_frame(
        self, tmp_path, first_frame, max_frames
    ):
        path = _write_zri(tmp_path)
        whole = read_sentec(path)

        ours = read_sentec(path, first_frame=first_frame, max_frames=max_frames)

        frames = slice(first_frame, first_frame + max_frames)
        assert ours.pixel_impedance.n_samples == max_frames
        np.testing.assert_array_equal(
            ours.pixel_impedance.values, whole.pixel_impedance.values[frames]
        )
        np.testing.assert_array_equal(
            ours.pixel_impedance.time, whole.pixel_impedance.time[frames]
        )
        assert ours.sample_frequency == whole.sample_frequency

    def test_same_as_eitprocessing_without_max_frames(self, tmp_path):
        path = _write_zri(tmp_path)

        ours = read_sentec(path, first_frame=5, max_frames=10)
        sequence = _upstream_load(path, first_frame=5)

        raw = sequence.eit_data["raw"]
        np.testing.assert_array_equal(
            ours.pixel_impedance.values, raw.pixel_impedance[:10]
        )
        np.testing.assert_array_equal(ours.pixel_impedance.time, raw.time[:10])

    def test_eitprocessing_raises(self, tmp_path):
        path = _write_zri(tmp_path)

        with pytest.raises(IndexError):
            _upstream_load(path, first_frame=5, max_frames=10)

    def test_too_many_frames_after_first_frame_warns(self, tmp_path):
        path = _write_zri(tmp_path)

        with pytest.warns(RuntimeWarning, match="25 frames were loaded"):
            ours = read_sentec(path, first_frame=5, max_frames=40)

        assert ours.pixel_impedance.n_samples == 25


class TestChecks:
    def test_first_frame_after_the_end_raises(self, tmp_path):
        path = _write_zri(tmp_path, n_frames=4)

        with pytest.raises(ValueError, match="larger than or equal to the number"):
            read_sentec(path, first_frame=4)

    def test_too_many_frames_warns(self, tmp_path):
        path = _write_zri(tmp_path, n_frames=4)

        with pytest.warns(RuntimeWarning, match="4 frames were loaded"):
            ours = read_sentec(path, max_frames=10)

        assert ours.pixel_impedance.n_samples == 4

    def test_wrong_image_size_raises(self, tmp_path):
        payload = b"\x07" + struct.pack("<BB", 32, 31) + bytes(4 * 1024)
        record = _record(MEASUREMENT, [_field(ZERO_REF_IMAGE, payload)])
        path = os.fspath(tmp_path / "bad.zri")
        with open(path, "wb") as handle:
            handle.write(b"\x02" + record)

        with pytest.raises(OSError, match="not equal to the product"):
            read_sentec(path)


def test_images_are_tagged():
    ours = read_sentec(SENTEC_FILE)

    assert ours.pixel_impedance.channel == "pixel_impedance"
    assert ours.pixel_impedance.modality == "eit"
    assert ours.pixel_impedance.source == "sentec"
    assert ours.pixel_impedance.sample_frequency == ours.sample_frequency
