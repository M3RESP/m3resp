"""Cut EIT images, waveforms, device marks and device breaths together.

Provenance
----------
Portions of this module are derived from eitprocessing.
Version 1.8.7, revision
6d6da6a825a08201ad01c0a7b68f4de07d4501f3 (m3resp-integration):
``datahandling/sequence.py::Sequence._sliced_copy`` and
``datahandling/intervaldata.py::IntervalData.select_by_time``.
Source: https://github.com/EIT-ALIVE/eitprocessing
Copyright (c) Netherlands eScience Center and Erasmus MC.
License: Apache License, Version 2.0. See top-level NOTICE.md.

Modified for m3resp: use Signal, EventData, IntervalData and EITFileData;
keep their descriptive fields and event identifiers; reject an empty frame
window with a ValueError.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from m3resp.data.event_data import EventData, IntervalData
from m3resp.eit.io.file_data import EITFileData
from m3resp.processing.slicing import slice_by_index, time_slice_indices


def slice_events(data: EventData, start: float, end: float) -> EventData:
    """Keep the device marks from ``start`` up to the mark before ``end``.

    Args:
        data (EventData): Device marks, with times in seconds in
            increasing order.
        start (float): First time to keep, in seconds. A mark at this
            time is kept.
        end (float): End of the window, in seconds. A mark at this time is
            left out.

    Returns:
        EventData: A copy with the kept marks and their values. The
            metadata entry ``"slice"`` gives the index of the first kept
            mark and the index after the last kept mark.

    Raises:
        ValueError: If the mark times are not in increasing order or are
            not finite.
    """

    first, last = time_slice_indices(
        np.array([event.time for event in data.events]), start=start, end=end
    )
    return replace(
        data,
        events=copy.deepcopy(data.events[first:last]),
        values=copy.deepcopy(data.values[first:last])
        if data.values is not None
        else None,
        metadata={**data.metadata, "slice": {"start_index": first, "end_index": last}},
    )


def interval_indices(data: IntervalData, start: float, end: float) -> list[int]:
    """Return the indices of the device intervals inside a time window.

    An interval is kept when its start and its end lie in the window,
    both window ends included. When ``metadata["default_partial_inclusion"]``
    is True, every interval that overlaps the window is kept. A bound of
    exactly 0 s is replaced by the start of the first interval (for
    ``start``) or the end of the last interval (for ``end``), as in
    eitprocessing 1.8.7.

    Args:
        data (IntervalData): Device intervals, for example Timpel breaths.
        start (float): Start of the window, in seconds.
        end (float): End of the window, in seconds.

    Returns:
        list[int]: The indices of the kept intervals, in increasing order.
    """

    start, end = _interval_bounds(data, start, end)
    partial = data.metadata.get("default_partial_inclusion", False)
    if partial:
        return [
            i
            for i, item in enumerate(data.intervals)
            if item.start_time < end and item.end_time > start
        ]
    return [
        i
        for i, item in enumerate(data.intervals)
        if item.start_time >= start and item.end_time <= end
    ]


def _interval_bounds(
    data: IntervalData, start: float, end: float
) -> tuple[float, float]:
    """A zero bound uses the corresponding outer interval bound, as upstream."""

    if not data.intervals:
        return start, end
    return start or data.intervals[0].start_time, end or data.intervals[-1].end_time


def slice_intervals(data: IntervalData, start: float, end: float) -> IntervalData:
    """Keep the device intervals inside a time window.

    The intervals are chosen by ``interval_indices``. A complete breath
    keeps its start, turning point and end. With partial inclusion, an
    interval that overlaps a window end is cut at that end; its turning
    point and its value stay the same.

    Args:
        data (IntervalData): Device intervals, for example Timpel breaths.
        start (float): Start of the window, in seconds.
        end (float): End of the window, in seconds.

    Returns:
        IntervalData: A copy with the kept intervals and their values. Its
            ``metadata["default_partial_inclusion"]`` is False, as in
            eitprocessing 1.8.7.
    """

    indices = interval_indices(data, start, end)
    start, end = _interval_bounds(data, start, end)
    items = [
        replace(
            copy.deepcopy(data.intervals[i]),
            start_time=max(data.intervals[i].start_time, start),
            end_time=min(data.intervals[i].end_time, end),
        )
        for i in indices
    ]
    values = None
    if isinstance(data.values, np.ndarray):
        values = np.copy(data.values[indices])
    elif data.values is not None:
        selected = [copy.deepcopy(data.values[i]) for i in indices]
        values = tuple(selected) if isinstance(data.values, tuple) else selected
    metadata = {**data.metadata, "default_partial_inclusion": False}
    return replace(data, intervals=items, values=values, metadata=metadata)


def slice_file_data(data: EITFileData, start: int, end: int) -> EITFileData:
    """Keep frames ``start`` up to the frame before ``end`` of one recording.

    The waveforms are cut at the same frames as the images. Device marks
    are kept from the time of the first kept frame up to the time of the
    last kept frame; a mark at that last time is left out. A complete
    device breath may end at the last kept frame. For device intervals, a
    bound of exactly 0 s uses the outer interval bound, as in
    eitprocessing 1.8.7. Values and times are copied; times keep their
    original values.

    Args:
        data (EITFileData): The recording: images, waveforms, marks and
            device breaths.
        start (int): Index of the first frame to keep.
        end (int): Index after the last frame to keep.

    Returns:
        EITFileData: The cut recording, with the same vendor, path, frames
            per second and keys.

    Raises:
        ValueError: If no frame is kept, if a waveform has a different
            number of samples from the images, or if the mark times are
            not in increasing order.
    """

    pixels = slice_by_index(data.pixel_impedance, start=start, end=end)
    for key, signal in data.signals.items():
        if len(signal) != len(data.pixel_impedance):
            raise ValueError(
                f"EIT waveform {key!r} has {len(signal)} samples; the images "
                f"have {len(data.pixel_impedance)} frames."
            )
    first, last = float(pixels.time[0]), float(pixels.time[-1])
    return replace(
        data,
        pixel_impedance=pixels,
        signals={
            key: slice_by_index(value, start=start, end=end)
            for key, value in data.signals.items()
        },
        events={
            key: slice_events(value, first, last) for key, value in data.events.items()
        },
        intervals={
            key: slice_intervals(value, first, last)
            for key, value in data.intervals.items()
        },
    )
