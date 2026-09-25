"""Cutting one signal to a window of samples or time.

These cut a single signal and return a shorter copy; the signal passed in is
not changed. They work on m3resp's own `Signal`/`TimeSeries`, and on
eitprocessing data (`EITData`, `ContinuousData`, `Sequence`), which cuts
itself.

To cut a whole loaded recording in a session instead, use
`M3Session.slice_emg`, `slice_eit` or `slice_ventilator`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import numpy as np

from m3resp.data.timeseries import TimeSeries


def slice_by_index(data: Any, *, start: int, end: int) -> Any:
    """Keep samples `start` up to (not including) `end` of a signal.

    Works on m3resp's own `Signal`/`TimeSeries` and on eitprocessing data
    (`EITData`, `ContinuousData`, `Sequence`), which cut themselves.
    """

    if isinstance(data, TimeSeries):
        return _slice_time_series(data, slice(start, end))
    return data[slice(start, end)]


def slice_by_time(data: Any, *, start: float, end: float) -> Any:
    """Keep the samples from the first one at or after `start` up to (not
    including) the first one at or after `end`.

    Times are the signal's own time values. For data read from an EIT file
    that is the time of day stored in the file, not seconds from the start.
    """

    if isinstance(data, TimeSeries):
        time = np.asarray(data.time, dtype=float)
        start_index = int(np.searchsorted(time, float(start), side="left"))
        end_index = int(np.searchsorted(time, float(end), side="left"))
        return _slice_time_series(data, slice(start_index, end_index))
    return data.t[slice(start, end)]


def slice_signal_by_mode(
    data: Any,
    *,
    start: float,
    end: float,
    slicing_mode: Literal["index", "time"],
) -> Any:
    """Slice a signal by sample index or time selector."""

    if slicing_mode == "index":
        return slice_by_index(data, start=int(start), end=int(end))
    if slicing_mode == "time":
        return slice_by_time(data, start=float(start), end=float(end))
    raise ValueError("slicing_mode must be 'index' or 'time'.")


def _slice_time_series(data: TimeSeries, kept: slice) -> TimeSeries:
    """A copy of `data` holding only the samples in `kept`, values and time
    axis cut together. Every other field (unit, modality, channel, ...) is
    kept, and the kept sample range is noted in the metadata."""

    start_index, end_index, _ = kept.indices(len(data.time))
    if end_index <= start_index:
        raise ValueError(
            f"Slicing {data.name or type(data).__name__!r} keeps no samples "
            f"(samples {start_index} to {end_index} of {len(data.time)})."
        )
    metadata = dict(data.metadata)
    metadata["slice"] = {"start_index": start_index, "end_index": end_index}
    return dataclasses.replace(
        data,
        values=data.values[start_index:end_index],
        time=data.time[start_index:end_index],
        metadata=metadata,
    )
