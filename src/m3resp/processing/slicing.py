"""Cutting one signal to a window of samples or time.

These cut a single signal and return a shorter copy; the signal passed in is
not changed. They work on m3resp's own `Signal`/`TimeSeries`, and on
eitprocessing data (`EITData`, `ContinuousData`, `Sequence`), which cuts
itself.

To cut a whole loaded recording in a session instead, use
`M3Session.slice_emg`, `slice_eit` or `slice_ventilator`.

Provenance
----------
Portions of this module are derived from eitprocessing.
Source: https://github.com/EIT-ALIVE/eitprocessing
Revision: m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
version 1.8.7. Original: datahandling/mixins/slicing.py::SelectByTime.select_by_time.
Copyright (c) Netherlands eScience Center and Erasmus MC.
License: Apache License, Version 2.0. See top-level NOTICE.md.
Modified for m3resp: return sample bounds separately, validate finite times,
and slice native TimeSeries and Signal objects.
"""

from __future__ import annotations

import bisect
import dataclasses
from typing import Any, Literal

import numpy as np

from m3resp.data.timeseries import TimeSeries


def time_slice_indices(
    time: np.ndarray,
    *,
    start: float | None = None,
    end: float | None = None,
    start_inclusive: bool = False,
    end_inclusive: bool = False,
) -> tuple[int, int]:
    """Return the first sample and the sample after the last one in a window.

    The rules are those of eitprocessing 1.8.7 (``select_by_time``). A
    sample exactly at ``start`` is always kept. When ``start`` falls between
    two samples, ``start_inclusive=True`` keeps the sample before it and
    False starts at the sample after it. ``end_inclusive=True`` also keeps
    the first sample at or after ``end``; False stops before that sample.

    Args:
        time (numpy.ndarray): Sample times in seconds, in increasing order.
        start (float | None): Start of the window, in seconds. None starts
            at the first sample.
        end (float | None): End of the window, in seconds. None ends at the
            last sample.
        start_inclusive (bool): Keep the sample before a ``start`` that
            falls between samples.
        end_inclusive (bool): Keep the first sample at or after ``end``.

    Returns:
        tuple[int, int]: The index of the first kept sample and the index
            after the last kept sample. An empty time axis gives (0, 0).

    Raises:
        ValueError: If ``time`` is not a one-dimensional array of finite
            values in increasing order, or if a bound is not finite.
    """

    time = np.asarray(time)
    if time.ndim != 1 or not np.all(np.isfinite(time)) or np.any(time[1:] < time[:-1]):
        raise ValueError(
            "Slicing by time needs a sorted, finite, one-dimensional time axis."
        )
    if any(value is not None and not np.isfinite(value) for value in (start, end)):
        raise ValueError("Slicing bounds must be finite times in seconds.")
    if not len(time):
        return 0, 0
    if start is None or start < time[0]:
        first = 0
    elif start_inclusive:
        first = bisect.bisect_right(time, start) - 1
    else:
        first = bisect.bisect_left(time, start)
    last = (
        len(time) if end is None else bisect.bisect_left(time, end) + int(end_inclusive)
    )
    return first, last


def slice_by_index(
    data: Any, *, start: int, end: int, allow_empty: bool = False
) -> Any:
    """Keep samples `start` up to (not including) `end` of a signal.

    Works on m3resp's own `Signal`/`TimeSeries` and on eitprocessing data
    (`EITData`, `ContinuousData`, `Sequence`), which cut themselves.
    Native values and times are copied. ``allow_empty=True`` allows an
    empty native result; otherwise an empty selection raises ValueError.
    """

    if isinstance(data, TimeSeries):
        return _slice_time_series(data, slice(start, end), allow_empty=allow_empty)
    return data[slice(start, end)]


def slice_by_time(
    data: Any,
    *,
    start: float | None,
    end: float | None,
    start_inclusive: bool = False,
    end_inclusive: bool = False,
    allow_empty: bool = False,
) -> Any:
    """Keep the samples from the first one at or after `start` up to (not
    including) the first one at or after `end`.

    Times are in seconds on the signal's own clock. The boundary flags
    follow ``time_slice_indices``. ``allow_empty=True`` returns an empty
    native signal when the window keeps no samples.
    """

    if isinstance(data, TimeSeries):
        start_index, end_index = time_slice_indices(
            data.time,
            start=start,
            end=end,
            start_inclusive=start_inclusive,
            end_inclusive=end_inclusive,
        )
        return _slice_time_series(
            data, slice(start_index, end_index), allow_empty=allow_empty
        )
    if start_inclusive or end_inclusive:
        return data.select_by_time(
            start_time=start,
            end_time=end,
            start_inclusive=start_inclusive,
            end_inclusive=end_inclusive,
        )
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


def _slice_time_series(
    data: TimeSeries, kept: slice, *, allow_empty: bool = False
) -> TimeSeries:
    """A copy of `data` holding only the samples in `kept`, values and time
    axis cut together. Every other field (unit, modality, channel, ...) is
    kept, and the kept sample range is noted in the metadata."""

    start_index, end_index, _ = kept.indices(len(data.time))
    if end_index <= start_index and not allow_empty:
        raise ValueError(
            f"Slicing {data.name or type(data).__name__!r} keeps no samples "
            f"(samples {start_index} to {end_index} of {len(data.time)})."
        )
    metadata = dict(data.metadata)
    metadata["slice"] = {"start_index": start_index, "end_index": end_index}
    return dataclasses.replace(
        data,
        values=np.copy(data.values[start_index:end_index]),
        time=np.copy(data.time[start_index:end_index]),
        metadata=metadata,
    )
