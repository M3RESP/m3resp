"""Poly5 binary writer for portable synthetic recordings."""

from __future__ import annotations

import struct
from datetime import datetime

import numpy as np

DEFAULT_ONE_DIMENSION = 1
DEFAULT_TWO_DIMENSIONS = 2
DEFAULT_FIRST_AXIS = 0
DEFAULT_SECOND_AXIS = 1
DEFAULT_ROW_VECTOR_SHAPE = (1, -1)
DEFAULT_LITTLE_ENDIAN_FLOAT32 = "<f4"
DEFAULT_POLY5_MAGIC = b"POLY SAMPLE FILEversion 2.03\r\n\x1a"
DEFAULT_POLY5_VERSION = 203
DEFAULT_POLY5_HEADER_FORMAT = "=31sH81phhBHi4xHHHHHHHiHHH64x"
DEFAULT_POLY5_CHANNEL_FORMAT = "=41p4x11pffffH62x"
DEFAULT_POLY5_CHANNEL_BYTES = 136
DEFAULT_POLY5_SIGNAL_BLOCK_HEADER_BYTES = 86
DEFAULT_POLY5_SAMPLES_PER_BLOCK = 256


def _write_poly5_file(
    path: str,
    array: np.ndarray,
    sample_frequency: float,
    labels: list[str],
    units: list[str],
) -> None:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == DEFAULT_ONE_DIMENSION:
        values = values.reshape(DEFAULT_ROW_VECTOR_SHAPE)
    if values.ndim != DEFAULT_TWO_DIMENSIONS:
        raise ValueError("Poly5 export requires a 2D channel-by-sample array")
    if values.shape[DEFAULT_FIRST_AXIS] != len(labels):
        raise ValueError("labels length must match the first data dimension")
    if values.shape[DEFAULT_FIRST_AXIS] != len(units):
        raise ValueError("units length must match the first data dimension")
    if values.shape[DEFAULT_SECOND_AXIS] <= 0:
        raise ValueError("Poly5 export requires at least one sample")

    sample_rate = _coerce_poly5_sample_rate(sample_frequency)
    num_channels = int(values.shape[DEFAULT_FIRST_AXIS])
    num_samples = int(values.shape[DEFAULT_SECOND_AXIS])
    samples_per_block = DEFAULT_POLY5_SAMPLES_PER_BLOCK
    num_data_blocks = int(np.ceil(num_samples / samples_per_block))
    now = datetime.now()

    header = struct.pack(
        DEFAULT_POLY5_HEADER_FORMAT,
        DEFAULT_POLY5_MAGIC,
        DEFAULT_POLY5_VERSION,
        b"Synthetic M3Resp recording",
        sample_rate,
        sample_rate,
        0,
        num_channels * 2,
        num_samples,
        now.year,
        now.month,
        now.day,
        now.weekday(),
        now.hour,
        now.minute,
        now.second,
        num_data_blocks,
        samples_per_block,
        0,
        0,
    )

    with open(path, "wb") as file_obj:
        file_obj.write(header)
        for label, unit in zip(labels, units):
            file_obj.write(_pack_poly5_channel_description(label, unit))
            file_obj.write(bytes(DEFAULT_POLY5_CHANNEL_BYTES))

        for block_index in range(num_data_blocks):
            start = block_index * samples_per_block
            stop = min(start + samples_per_block, num_samples)
            block = values[:, start:stop].T.astype(DEFAULT_LITTLE_ENDIAN_FLOAT32)
            file_obj.write(bytes(DEFAULT_POLY5_SIGNAL_BLOCK_HEADER_BYTES))
            file_obj.write(block.ravel().tobytes())


def _coerce_poly5_sample_rate(sample_frequency: float) -> int:
    sample_rate = int(round(float(sample_frequency)))
    if sample_rate <= 0:
        raise ValueError("Poly5 sample_frequency must be positive")
    if not np.isclose(float(sample_frequency), float(sample_rate)):
        raise ValueError("Poly5 sample_frequency must be an integer number of Hz")
    return sample_rate


def _pack_poly5_channel_description(label: str, unit: str) -> bytes:
    label_bytes = str(label).encode("ascii")
    unit_bytes = str(unit).encode("utf-8")
    if len(label_bytes) > 36:
        raise ValueError("Poly5 channel labels must be at most 36 ASCII bytes")
    if len(unit_bytes) > 10:
        raise ValueError("Poly5 channel units must be at most 10 UTF-8 bytes")
    return struct.pack(
        DEFAULT_POLY5_CHANNEL_FORMAT,
        b"     " + label_bytes,
        unit_bytes,
        0.0,
        0.0,
        1.0,
        0.0,
        0,
    )
