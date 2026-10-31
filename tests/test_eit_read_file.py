"""`read_eit_file` and the `Sequence` built from its result match eitprocessing 1.8.7."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from m3resp.adapters.eitprocessing_adapter import (
    EITProcessingAdapter,
    file_data_to_sequence,
    load_eit_file_as_sequence,
)
from m3resp.eit.io import EITFileData, read_eit_file

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEST_FILES = {
    "draeger": os.path.join(DATA_DIR, "draeger_synthetic_draeger_20Hz.bin"),
    "sentec": os.path.join(DATA_DIR, "sentec_healthy_volunteer_1b.zri"),
    "timpel": os.path.join(DATA_DIR, "timpel_healthy_volunteer_1_60-120s.txt"),
}
OPTIONS: tuple[dict[str, Any], ...] = (
    {},
    # Both together crash in eitprocessing for Sentec (fixed in m3resp,
    # see tests/test_eit_sentec_reader.py), so they are tried one at a time.
    {"first_frame": 100},
    {"max_frames": 500},
    {"sample_frequency": 20.0},
    {"label": "rec", "name": "Recording", "description": "test"},
)


def _upstream_load(path, **kwargs):
    loading = pytest.importorskip("eitprocessing.datahandling.loading")
    return loading.load_eit_data(path, **kwargs)


def _assert_same_sequence(ours, theirs) -> None:
    """Equal by eitprocessing's own `==`, and in the fields it does not compare."""

    assert ours == theirs
    assert ours.description == theirs.description
    if theirs.label.startswith("Sequence_"):
        # No label given: both make one from the id of the object.
        assert ours.label.startswith("Sequence_") and ours.name == ours.label
    else:
        assert (ours.label, ours.name) == (theirs.label, theirs.name)
    for collection in ("eit_data", "continuous_data", "sparse_data", "interval_data"):
        mine, upstream = getattr(ours, collection), getattr(theirs, collection)
        assert list(mine) == list(upstream)
        for key, expected in upstream.items():
            item = mine[key]
            for field in ("label", "name", "description", "path", "nframes"):
                if hasattr(expected, field):
                    assert getattr(item, field) == getattr(expected, field), (
                        key,
                        field,
                    )
            if hasattr(expected, "derived_from"):
                assert len(item.derived_from) == len(expected.derived_from), key
            if hasattr(expected, "locked"):
                assert item.locked == expected.locked, key
            if isinstance(getattr(expected, "values", None), np.ndarray):
                assert item.values.dtype == expected.values.dtype, key


class TestReadEitFile:
    @pytest.mark.parametrize("vendor", sorted(TEST_FILES))
    def test_picks_the_reader_of_the_vendor(self, vendor):
        result = read_eit_file(TEST_FILES[vendor], vendor)

        assert isinstance(result, EITFileData)
        assert result.vendor == vendor

    def test_takes_an_eitprocessing_vendor(self):
        datahandling = pytest.importorskip("eitprocessing.datahandling.eitdata")

        result = read_eit_file(TEST_FILES["draeger"], datahandling.Vendor.DRAEGER)

        assert result.vendor == "draeger"

    @pytest.mark.parametrize("vendor", ["drager", "Draeger", "simulated", "x", None])
    def test_unknown_vendor_raises(self, vendor):
        with pytest.raises(NotImplementedError, match="No loading method for"):
            read_eit_file(TEST_FILES["draeger"], vendor)

    @pytest.mark.parametrize("vendor", ["drager", "Draeger", "x"])
    def test_unknown_vendor_raises_like_eitprocessing(self, vendor):
        with pytest.raises(NotImplementedError, match="No loading method for"):
            _upstream_load(TEST_FILES["draeger"], vendor=vendor)

    def test_simulated_vendor_fails_in_eitprocessing(self):
        # Bug fix: eitprocessing knows the name but has no reader for it.
        with pytest.raises(KeyError):
            _upstream_load(TEST_FILES["draeger"], vendor="simulated")

    @pytest.mark.parametrize(
        ("first_frame", "error"),
        [(1.5, TypeError), ("2", TypeError), (-1, ValueError)],
    )
    def test_wrong_first_frame_raises(self, first_frame, error):
        with pytest.raises(error, match="`first_frame`"):
            read_eit_file(TEST_FILES["draeger"], "draeger", first_frame=first_frame)
        with pytest.raises(error, match="`first_frame`"):
            _upstream_load(
                TEST_FILES["draeger"], vendor="draeger", first_frame=first_frame
            )

    @pytest.mark.parametrize("first_frame", [None, 2.0])
    def test_first_frame_as_none_or_whole_float(self, first_frame):
        result = read_eit_file(
            TEST_FILES["draeger"], "draeger", first_frame=first_frame, max_frames=10
        )

        start = 0 if first_frame is None else 2
        whole = read_eit_file(TEST_FILES["draeger"], "draeger")
        np.testing.assert_array_equal(
            result.pixel_impedance.time, whole.pixel_impedance.time[start : start + 10]
        )

    def test_missing_file_raises(self, tmp_path):
        missing = os.fspath(tmp_path / "missing.bin")

        with pytest.raises(FileNotFoundError):
            read_eit_file(missing, "draeger")
        with pytest.raises(FileNotFoundError):
            _upstream_load(missing, vendor="draeger")


class TestSameSequenceAsEitprocessing:
    @pytest.mark.parametrize("options", OPTIONS, ids=lambda o: ",".join(o) or "none")
    @pytest.mark.parametrize("vendor", sorted(TEST_FILES))
    def test_load_eit_file_as_sequence(self, vendor, options):
        path = TEST_FILES[vendor]

        ours = load_eit_file_as_sequence(path, vendor=vendor, **options)
        theirs = _upstream_load(path, vendor=vendor, **options)

        _assert_same_sequence(ours, theirs)

    def test_file_data_to_sequence_keeps_the_sequence_label(self):
        sequence = file_data_to_sequence(
            read_eit_file(TEST_FILES["timpel"], "timpel"),
            label="a",
            name="b",
            description="c",
        )

        assert (sequence.label, sequence.name, sequence.description) == ("a", "b", "c")

    @pytest.mark.parametrize("vendor", sorted(TEST_FILES))
    def test_adapter_loads_the_same_sequence(self, vendor):
        path = TEST_FILES[vendor]

        ours = EITProcessingAdapter().load_with_eitprocessing(path, vendor=vendor)

        _assert_same_sequence(ours, _upstream_load(path, vendor=vendor))

    def test_unknown_option_raises(self):
        with pytest.raises(TypeError):
            load_eit_file_as_sequence(TEST_FILES["draeger"], "draeger", speed=2)
