"""EIT modality containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from m3resp.modalities.time_window import TimeWindow


@dataclass
class EITRecording:
    """Loaded EIT recording with source metadata."""

    data: Any
    path: Path
    vendor: str | None = None
    raw: Any = None
    global_impedance: Any = None


def load(
    path: str | Path,
    vendor: str | None = None,
    *,
    adapter: Any = None,
    **kwargs: Any,
) -> EITRecording:
    """Load an EIT recording through the Stage 1 adapter."""

    from m3resp.adapters.eitprocessing_adapter import EITProcessingAdapter

    eit_adapter = adapter or EITProcessingAdapter()
    sequence = eit_adapter.load(str(path), vendor=vendor, **kwargs)
    raw = None
    global_impedance = None
    if hasattr(sequence, "eit_data"):
        raw = eit_adapter.get_raw_eit(sequence)
    if hasattr(sequence, "continuous_data") and raw is not None:
        global_impedance = eit_adapter.get_global_impedance(sequence)
    return EITRecording(
        data=sequence,
        path=Path(path),
        vendor=vendor,
        raw=raw,
        global_impedance=global_impedance,
    )


def frame_window(
    recording: EITRecording, start_seconds: float, end_seconds: float | None
) -> TimeWindow:
    """The frames of an EIT recording between two times.

    Times are in seconds from the recording's first frame, not the time of
    day stored in the file; ``end_seconds`` None means up to the end. The
    window runs from the first frame at or after the start up to, but not
    including, the first frame at or after the end. Frames are chosen by
    their time stamps, which in Draeger files are not perfectly even. Raises
    ValueError when the window does not lie inside the recording or holds
    no frame.
    """

    time = np.asarray(recording.data.time, dtype=float)
    if time.size < 2:
        raise ValueError("slice_eit needs an EIT recording with at least two frames.")
    frame_interval = float(np.median(np.diff(time)))
    duration = float(time[-1] - time[0]) + frame_interval
    end = duration if end_seconds is None else float(end_seconds)
    start = float(start_seconds)
    if not 0.0 <= start < end <= duration:
        raise ValueError(
            f"slice_eit: need 0 <= start < end <= {duration:.3f} s (the "
            f"recording's length); got start={start_seconds!r}, "
            f"end={end_seconds!r}."
        )

    start_index = int(np.searchsorted(time, time[0] + start, side="left"))
    end_index = (
        time.size
        if end_seconds is None
        else int(np.searchsorted(time, time[0] + end, side="left"))
    )
    if end_index <= start_index:
        raise ValueError(f"slice_eit: no EIT frame lies between {start} s and {end} s.")
    return TimeWindow(
        start_seconds=start,
        end_seconds=end,
        start_index=start_index,
        end_index=end_index,
        n_samples=time.size,
        shift_seconds=float(time[start_index] - time[0]),
    )


def keep_frames(
    recording: EITRecording, start_index: int, end_index: int, *, adapter: Any
) -> None:
    """Keep frames `start_index` up to (not including) `end_index` of an EIT
    recording, in place.

    The cutting is done by `eitprocessing` (`adapter.slice_sequence`, see
    `EITProcessingAdapter`): pixel data, global impedance, the pressure/flow
    channels and the vendor's markers are all cut to the same frames. Frame
    times keep their original values.
    """

    sequence = adapter.slice_sequence(recording.data, start_index, end_index)
    recording.data = sequence
    recording.raw = adapter.get_raw_eit(sequence)
    recording.global_impedance = adapter.get_global_impedance(sequence)
