"""Draeger binary frame packing and writing for synthetic EIT data."""

from __future__ import annotations

import struct

import numpy as np

from m3resp.synthetic.config import ExportConfig, SyntheticDraegerData

DEFAULT_ZERO = 0.0
DEFAULT_FLOAT32_DTYPE = np.float32
DEFAULT_DRAEGER_PIXEL_SHAPE = (32, 32)
DEFAULT_DRAEGER_EVENT_TEXT_LENGTH = 30
DEFAULT_LITTLE_ENDIAN_FLOAT32 = "<f4"
DEFAULT_LITTLE_ENDIAN_FLOAT64_PACK = "<d"
DEFAULT_LITTLE_ENDIAN_FLOAT32_PACK = "<f"
DEFAULT_LITTLE_ENDIAN_INT32_PACK = "<i"


def pack_draeger_frame(
    time_seconds: float,
    pixel_frame: np.ndarray,
    min_max_flag: int,
    event_marker: int,
    event_text: str,
    timing_error: int,
    medibus_values: np.ndarray,
    export_config: ExportConfig,
    unused_float32: float = DEFAULT_ZERO,
) -> bytes:
    """Pack one Draeger frame exactly as expected by the Draeger loader."""

    pixel_frame = np.asarray(pixel_frame, dtype=DEFAULT_FLOAT32_DTYPE)
    if pixel_frame.shape != DEFAULT_DRAEGER_PIXEL_SHAPE:
        raise ValueError(f"pixel_frame must have shape {DEFAULT_DRAEGER_PIXEL_SHAPE}")

    time_fraction_of_day = float(time_seconds) / export_config.seconds_per_day
    return b"".join(
        [
            struct.pack(DEFAULT_LITTLE_ENDIAN_FLOAT64_PACK, time_fraction_of_day),
            struct.pack(DEFAULT_LITTLE_ENDIAN_FLOAT32_PACK, float(unused_float32)),
            np.asarray(pixel_frame, dtype=DEFAULT_LITTLE_ENDIAN_FLOAT32)
            .reshape(-1, order="C")
            .tobytes(),
            struct.pack(DEFAULT_LITTLE_ENDIAN_INT32_PACK, int(min_max_flag)),
            struct.pack(DEFAULT_LITTLE_ENDIAN_INT32_PACK, int(event_marker)),
            _encode_event_text(
                event_text,
                length=export_config.draeger_event_text_length,
            ),
            struct.pack(DEFAULT_LITTLE_ENDIAN_INT32_PACK, int(timing_error)),
            np.asarray(medibus_values, dtype=DEFAULT_LITTLE_ENDIAN_FLOAT32)
            .reshape(-1)
            .tobytes(),
        ]
    )


def save_draeger_bin(
    path: str,
    data: SyntheticDraegerData,
    export_config: ExportConfig,
) -> str:
    """Save a complete Draeger-style binary file."""

    n_frames = len(data.time)
    if data.pixel_impedance.shape != (n_frames, *DEFAULT_DRAEGER_PIXEL_SHAPE):
        raise ValueError("pixel_impedance shape mismatch")
    if data.medibus_data.shape[1] != n_frames:
        raise ValueError("medibus_data shape mismatch")

    with open(path, "wb") as file_obj:
        for index in range(n_frames):
            frame_bytes = pack_draeger_frame(
                time_seconds=float(data.time[index]),
                pixel_frame=data.pixel_impedance[index],
                min_max_flag=int(data.min_max_flags[index]),
                event_marker=int(data.event_markers[index]),
                event_text=data.event_texts[index],
                timing_error=int(data.timing_errors[index]),
                medibus_values=data.medibus_data[:, index],
                export_config=export_config,
                unused_float32=float(data.unused_float32[index]),
            )
            if len(frame_bytes) != data.frame_size:
                raise ValueError(
                    f"Packed frame has {len(frame_bytes)} bytes, "
                    f"expected {data.frame_size}"
                )
            file_obj.write(frame_bytes)
    return path


def _encode_event_text(
    text: str,
    length: int = DEFAULT_DRAEGER_EVENT_TEXT_LENGTH,
) -> bytes:
    raw = text.encode("ascii", errors="replace")[:length]
    return raw.ljust(length, b"\x00")
