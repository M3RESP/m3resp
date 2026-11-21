"""`BinReader` reads numbers and text from a binary EIT file in file order."""

from __future__ import annotations

import io
import struct
from typing import Any

import numpy as np
import pytest

from m3resp.eit.io import BinReader

# Every BinReader method, the struct letter it reads and an example value.
SINGLE_VALUES = [
    ("float32", "f", 1.5),
    ("float64", "d", -2.25e-7),
    ("int32", "i", -123_456),
    ("uint8", "B", 200),
    ("uint16", "H", 60_000),
    ("uint32", "I", 4_000_000_000),
    ("uint64", "Q", 2**63 + 5),
]
ARRAY_VALUES = [
    ("npfloat32", "f", np.float32, [0.5, -1.0, 3.25]),
    ("npfloat64", "d", np.float64, [1e-12, 2.0, -7.5]),
    ("npint32", "i", np.int32, [-1, 0, 2_000_000_000]),
]


def _reader(payload: bytes, endian: Any = None) -> BinReader:
    return BinReader(io.BytesIO(payload), endian=endian)  # type: ignore[arg-type]


@pytest.mark.parametrize("endian", ["little", "big"])
@pytest.mark.parametrize(("method", "letter", "value"), SINGLE_VALUES)
def test_single_value_round_trips(endian, method, letter, value):
    prefix = "<" if endian == "little" else ">"
    reader = _reader(struct.pack(prefix + letter, value), endian=endian)

    assert getattr(reader, method)() == value


@pytest.mark.parametrize("endian", ["little", "big"])
@pytest.mark.parametrize(("method", "letter", "dtype", "values"), ARRAY_VALUES)
def test_array_round_trips(endian, method, letter, dtype, values):
    prefix = "<" if endian == "little" else ">"
    payload = struct.pack(f"{prefix}{len(values)}{letter}", *values)

    result = getattr(_reader(payload, endian=endian), method)(len(values))

    assert result.dtype == dtype
    np.testing.assert_array_equal(result, np.array(values, dtype=dtype))


def test_values_are_read_one_after_another():
    payload = struct.pack("<iBf", 7, 3, 0.25) + b"rest"
    reader = _reader(payload, endian="little")

    assert reader.int32() == 7
    assert reader.uint8() == 3
    assert reader.float32() == 0.25
    assert reader.file_handle.read() == b"rest"


def test_read_list_returns_python_values():
    reader = _reader(struct.pack("<3H", 1, 2, 3), endian="little")

    assert reader.read_list("H", int, 3) == [1, 2, 3]


def test_string_removes_white_space_at_the_end():
    reader = _reader(b"Marker   next", endian="little")

    assert reader.string(length=9) == "Marker"
    assert reader.read_string(4) == "next"


def test_no_byte_order_uses_this_computer():
    payload = struct.pack("=i", -42)

    assert _reader(payload, endian=None).int32() == -42


def test_unknown_byte_order_raises():
    reader = _reader(b"\x00\x00\x00\x00", endian="middle")

    with pytest.raises(ValueError, match="Endian type 'middle' not recognized"):
        reader.int32()


def test_gives_the_same_values_as_eitprocessing():
    upstream = pytest.importorskip("eitprocessing.datahandling.loading.binreader")
    payload = struct.pack("<fdiBHIQ", *(value for *_, value in SINGLE_VALUES))
    payload += struct.pack(
        "<3f3d3i", *(v for *_, values in ARRAY_VALUES for v in values)
    )
    payload += b"text  "

    ours = _reader(payload, endian="little")
    theirs = upstream.BinReader(io.BytesIO(payload), endian="little")
    for method, *_ in SINGLE_VALUES:
        assert getattr(ours, method)() == getattr(theirs, method)()
    for method, _, _, values in ARRAY_VALUES:
        np.testing.assert_array_equal(
            getattr(ours, method)(len(values)), getattr(theirs, method)(len(values))
        )
    assert ours.string(6) == theirs.string(6)
