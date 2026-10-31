"""Build the eitprocessing objects for a recording cut with m3resp.

Provenance
----------
Portions of this module are derived from eitprocessing.
Source: https://github.com/EIT-ALIVE/eitprocessing
Revision: m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
version 1.8.7. Originals: datahandling/sequence.py::Sequence._sliced_copy;
datahandling/eitdata.py::EITData._sliced_copy;
datahandling/continuousdata.py::ContinuousData._sliced_copy;
datahandling/sparsedata.py::SparseData._sliced_copy;
datahandling/intervaldata.py::IntervalData.select_by_time.
Copyright (c) Netherlands eScience Center and Erasmus MC.
License: Apache License, Version 2.0. See top-level NOTICE.md.
Modified for m3resp: take values and times from native sliced data and
rebuild the upstream objects at the adapter boundary.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from m3resp.eit.io.file_data import EITFileData
from m3resp.eit.processing.slicing import interval_indices
from m3resp.processing.slicing import slice_by_index, time_slice_indices

from ._file_data import sequence_to_file_data
from ._signals import continuous_data_to_signal, eit_data_to_signal


def sequence_from_sliced_data(
    sequence: Any, original: EITFileData, sliced: EITFileData, start: int, end: int
) -> Any:
    """Build the eitprocessing ``Sequence`` of a recording cut with m3resp.

    The images, waveforms, marks and breaths come from ``sliced``. Images
    and waveforms that exist only in ``sequence`` (for example filtered
    images) are cut at the same frames. File paths and keys stay the same.
    Labels, names and descriptions are those eitprocessing 1.8.7 gives a
    cut ``Sequence``.

    Args:
        sequence (Any): The eitprocessing ``Sequence`` before the cut.
        original (EITFileData): The m3resp data before the cut.
        sliced (EITFileData): The m3resp data after the cut, from
            ``slice_file_data``.
        start (int): Index of the first kept frame.
        end (int): Index after the last kept frame.

    Returns:
        Any: A new eitprocessing ``Sequence`` with the kept frames.
    """

    images = type(sequence.eit_data)(sequence.eit_data.data_type)
    for key, value in sequence.eit_data.items():
        signal = (
            sliced.pixel_impedance
            if key == "raw"
            else slice_by_index(eit_data_to_signal(value), start=start, end=end)
        )
        images.add(
            type(value)(
                path=value.path,
                nframes=len(signal),
                vendor=value.vendor,
                time=np.copy(signal.time),
                sample_frequency=value.sample_frequency,
                label=value.label,
                pixel_impedance=signal.values,
            )
        )

    continuous = type(sequence.continuous_data)(sequence.continuous_data.data_type)
    for key, value in sequence.continuous_data.items():
        signal = sliced.signals.get(key)
        if signal is None:
            signal = slice_by_index(
                continuous_data_to_signal(
                    value,
                    modality="eit",
                    channel=None,
                    processing_state="raw",
                    category=value.category,
                ),
                start=start,
                end=end,
            )
        continuous.add(
            type(value)(
                label=value.label,
                name=value.name,
                unit=value.unit,
                category=value.category,
                description=f"Slice ({start}-{end}) of <{value.description}>",
                derived_from=[*value.derived_from, value],
                time=np.copy(signal.time),
                values=np.copy(signal.values),
                sample_frequency=signal.sample_frequency,
            )
        )

    first, last = (
        float(sliced.pixel_impedance.time[0]),
        float(sliced.pixel_impedance.time[-1]),
    )
    sparse = type(sequence.sparse_data)(sequence.sparse_data.data_type)
    for key, value in sequence.sparse_data.items():
        if not len(value.time):
            sparse.add(copy.deepcopy(value))
            continue
        marks = sliced.events[key]
        i, j = time_slice_indices(value.time, start=first, end=last)
        sparse.add(
            type(value)(
                label=value.label,
                name=value.name,
                unit=value.unit,
                category=value.category,
                description=f"Slice ({i}-{j}) of <{value.description}>",
                derived_from=[*value.derived_from, value],
                time=np.array([event.time for event in marks.events]),
                values=copy.deepcopy(value.values[i:j])
                if value.values is not None
                else None,
            )
        )

    intervals = type(sequence.interval_data)(sequence.interval_data.data_type)
    for key, value in sequence.interval_data.items():
        indices = interval_indices(original.intervals[key], first, last)
        intervals.add(
            type(value)(
                label=value.label,
                name=value.name,
                unit=value.unit,
                category=value.category,
                derived_from=[*value.derived_from, value],
                intervals=[
                    (item.start_time, item.end_time)
                    for item in sliced.intervals[key].intervals
                ],
                values=[copy.deepcopy(value.values[i]) for i in indices]
                if value.values is not None
                else None,
            )
        )

    return type(sequence)(
        label=sequence.label,
        name=f"Sliced copy of <{sequence.name}>",
        description=f"Sliced copy of <{sequence.description}>",
        eit_data=images,
        continuous_data=continuous,
        sparse_data=sparse,
        interval_data=intervals,
    )


def slice_sequence(sequence: Any, start: int, end: int) -> Any:
    """Keep frames ``start`` up to the frame before ``end`` of a ``Sequence``.

    The ``Sequence`` is turned into m3resp data, cut with
    ``slice_file_data`` and built again with ``sequence_from_sliced_data``.

    Args:
        sequence (Any): An eitprocessing ``Sequence`` with raw images.
        start (int): Index of the first frame to keep.
        end (int): Index after the last frame to keep.

    Returns:
        Any: A new eitprocessing ``Sequence`` with the kept frames.

    Raises:
        TypeError: If ``sequence`` has no raw images with a time axis.
        ValueError: If no frame is kept.
    """

    from m3resp.eit.processing.slicing import slice_file_data

    native = sequence_to_file_data(sequence)
    if native is None:
        raise TypeError("Slicing EIT data needs raw images and their time axis.")
    sliced = slice_file_data(native, start, end)
    return sequence_from_sliced_data(sequence, native, sliced, start, end)
