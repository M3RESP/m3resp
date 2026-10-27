"""Join the data of EIT files recorded one after the other into one recording.

Two files can be joined when they come from the same kind of device and hold
the same kinds of data: the same vendor, sample frequency and image size, and
the same waveforms, marks and breaths with the same units and categories. The
second file must start after the first one ends. A pause between the files
stays in the time axis and gives a warning.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/datahandling/mixins/equality.py::
                Equivalence.isequivalent;
                eitprocessing/datahandling/eitdata.py::EITData.concatenate;
                eitprocessing/datahandling/continuousdata.py::
                ContinuousData.concatenate;
                eitprocessing/datahandling/sparsedata.py::
                SparseData.concatenate;
                eitprocessing/datahandling/intervaldata.py::
                IntervalData.concatenate
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The checks and the joins work on `EITFileData`: `Signal`s for the
      images and waveforms, `EventData` for the marks and `IntervalData`
      for the breaths.
    - The check compares vendor, sample frequency, image size, the keys of
      each group, and the unit, category and sample frequency of each item.
      A failed check raises ValueError.
    - The order check and the gap warning before the join are m3resp's own.
    - Docstrings and messages are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import dataclasses
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from m3resp.data.event_data import EventData, IntervalData
from m3resp.data.signals import Signal
from m3resp.eit.io.file_data import EITFileData

#: A pause between two files longer than this many frame intervals is
#: reported as a gap.
GAP_TOLERANCE_FRAMES = 0.5


def join_eit_files(files: Sequence[EITFileData], paths: Sequence[str]) -> EITFileData:
    """Join the data of EIT files recorded one after the other, in the given order.

    Args:
        files (Sequence[EITFileData]): The data of each file, in recording
            order.
        paths (Sequence[str]): The file of each item, for messages.

    Returns:
        EITFileData: All frames, waveforms, marks and breaths in one
            recording. ``path`` is the path of the first file.

    Raises:
        ValueError: If no file is given, a file starts before the one before
            it ends, or two files hold different kinds of data.

    Warns:
        UserWarning: If there is a pause between two files.
    """
    if not files:
        raise ValueError("Give at least one EIT file to join.")
    check_file_order(files, paths)
    joined = files[0]
    for after, after_path in zip(files[1:], paths[1:], strict=True):
        problem = _difference(joined, after)
        if problem is not None:
            raise ValueError(
                "These EIT files cannot be joined into one recording: they "
                f"differ in device, image size or sampling rate ({problem}; "
                f"file {after_path!r})."
            )
        joined = _join_two(joined, after)
    return joined


def check_file_order(files: Sequence[EITFileData], paths: Sequence[str]) -> None:
    """Check that each file starts after the one before it ends.

    Args:
        files (Sequence[EITFileData]): The data of each file, in recording
            order.
        paths (Sequence[str]): The file of each item, for messages.

    Raises:
        ValueError: If a file starts at or before the end of the file before
            it (files out of order, or overlapping).

    Warns:
        UserWarning: If the pause between two files is longer than half a
            frame interval.
    """
    for (before, before_path), (after, after_path) in zip(
        zip(files, paths, strict=True),
        zip(files[1:], paths[1:], strict=True),
        strict=False,
    ):
        before_time = np.asarray(before.pixel_impedance.time, dtype=float)
        after_time = np.asarray(after.pixel_impedance.time, dtype=float)
        before_end = float(before_time[-1])
        after_start = float(after_time[0])
        if after_start <= before_end:
            raise ValueError(
                f"EIT file {after_path!r} starts (t = {after_start:.3f} s) "
                f"before {before_path!r} ends (t = {before_end:.3f} s). List "
                "the files in recording order, without overlap."
            )
        frame_interval = (
            float(np.median(np.diff(before_time))) if before_time.size > 1 else 0.0
        )
        gap = after_start - before_end - frame_interval
        if frame_interval > 0 and gap > GAP_TOLERANCE_FRAMES * frame_interval:
            warnings.warn(
                f"There is a gap of {gap:.3f} s between EIT files "
                f"{before_path!r} and {after_path!r}. The time axis keeps the "
                "gap; filters and breath detection run across it.",
                UserWarning,
                stacklevel=3,
            )


def _difference(first: EITFileData, second: EITFileData) -> str | None:
    """Return what keeps two files from being joined, or None."""

    if first.vendor != second.vendor:
        return f"vendor {first.vendor!r} and {second.vendor!r}"
    if first.sample_frequency != second.sample_frequency:
        return (
            f"sample frequency {first.sample_frequency} Hz and "
            f"{second.sample_frequency} Hz"
        )
    first_shape = first.pixel_impedance.values.shape[1:]
    second_shape = second.pixel_impedance.values.shape[1:]
    if first_shape != second_shape:
        return f"image size {first_shape} and {second_shape}"
    for group in ("signals", "events", "intervals"):
        first_items = getattr(first, group)
        second_items = getattr(second, group)
        if set(first_items) != set(second_items):
            return f"{group} {sorted(first_items)} and {sorted(second_items)}"
        for key, item in first_items.items():
            other = second_items[key]
            for field in ("unit", "category", "sample_frequency"):
                if getattr(item, field, None) != getattr(other, field, None):
                    return (
                        f"{field} of {key!r}: {getattr(item, field, None)!r} and "
                        f"{getattr(other, field, None)!r}"
                    )
    return None


def _join_two(first: EITFileData, second: EITFileData) -> EITFileData:
    """Join two files that passed the checks."""

    return dataclasses.replace(
        first,
        pixel_impedance=_join_signals(first.pixel_impedance, second.pixel_impedance),
        signals={
            key: _join_signals(signal, second.signals[key])
            for key, signal in first.signals.items()
        },
        events={
            key: _join_events(marks, second.events[key])
            for key, marks in first.events.items()
        },
        intervals={
            key: _join_intervals(breaths, second.intervals[key])
            for key, breaths in first.intervals.items()
        },
    )


def _join_signals(first: Signal, second: Signal) -> Signal:
    """Values and time of two signals one after the other; other fields from the first."""

    return dataclasses.replace(
        first,
        values=np.concatenate((first.values, second.values), axis=0),
        time=np.concatenate((first.time, second.time)),
        metadata=dict(first.metadata),
    )


def _join_events(first: EventData, second: EventData) -> EventData:
    """The marks of two files one after the other.

    Raises:
        ValueError: If the second file has a mark at or before the last mark
            of the first.
    """

    if not len(first):
        return copy.deepcopy(second)
    if not len(second):
        return copy.deepcopy(first)
    if second.events[0].time <= first.events[-1].time:
        raise ValueError(
            f"The marks {second.name!r} of the second file start before those "
            "of the first file end."
        )
    return dataclasses.replace(
        first,
        events=[*first.events, *second.events],
        values=_join_values(first.values, second.values),
        metadata=dict(first.metadata),
    )


def _join_intervals(first: IntervalData, second: IntervalData) -> IntervalData:
    """The intervals of two files one after the other.

    Raises:
        ValueError: If the first interval of the second file starts before
            the last interval of the first file ends.
    """

    if not len(first):
        return copy.deepcopy(second)
    if not len(second):
        return copy.deepcopy(first)
    if second.intervals[0].start_time < first.intervals[-1].end_time:
        raise ValueError(
            f"The intervals {second.name!r} of the second file start before "
            "those of the first file end."
        )
    return dataclasses.replace(
        first,
        intervals=[*first.intervals, *second.intervals],
        values=_join_values(first.values, second.values),
        metadata=dict(first.metadata),
    )


def _join_values(first: Any, second: Any) -> Any:
    """Join two value lists or arrays; None stays None."""

    if first is None:
        return None
    if isinstance(first, np.ndarray):
        return np.concatenate((first, second))
    return list(first) + list(second)
