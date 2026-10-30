"""Read a Timpel ``.txt`` EIT file.

A Timpel file is text, with one frame per line and 1,030 numbers per line,
split by commas:

- columns 0 to 1023: the 32 by 32 image; -1000 means "no value" (NaN);
- column 1024: airway pressure (cmH2O);
- column 1025: flow (L/s);
- column 1026: volume (L);
- column 1027: 1 marks a breath minimum found by the device;
- column 1028: 1 marks a breath maximum found by the device;
- column 1029: 1 marks a QRS complex (a heart beat) found by the device.

The file holds no time. The time is the frame number divided by the sample
frequency (50 Hz). The reader also makes breaths from the minima and
maxima of the device.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/datahandling/loading/timpel.py::
                load_from_single_path, _make_breaths, TIMPEL_SAMPLE_FREQUENCY,
                _COLUMN_WIDTH, _NAN_VALUE
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - `load_from_single_path` is `read_timpel`. It returns an
      `EITFileData` of m3resp types: the images and the waveforms are
      `Signal`s, the minima, maxima and QRS complexes are `EventData`, and
      the breaths are `BreathEvent`s in an `IntervalData`. The dictionary
      keys are the eitprocessing labels.
    - Bug fix: the text is read with `numpy.loadtxt(..., ndmin=2)`, and the
      "no frames" check comes before the column check. With 0 or 1 frames
      left after `first_frame`, `numpy.loadtxt` gives an array with fewer
      than 2 dimensions, and eitprocessing 1.8.7 stops with an IndexError.
      With the fix, 0 frames give the ValueError the original meant to give,
      and 1 frame is read.
    - Bug fix: when the cleaning of the minima leaves fewer than 2 minima,
      `_make_breaths` returns no breaths. eitprocessing 1.8.7 stops there
      with a ValueError (an empty list is unpacked); it returns no breaths
      when the file has fewer than 2 minima from the start.
    - The results for all other inputs are unchanged by the two fixes.
    - Docstrings and comments are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from numpy.typing import NDArray

from m3resp.data.event_data import EventData, IntervalData
from m3resp.data.events import BreathEvent, Event
from m3resp.data.signals import Signal
from m3resp.eit.io.file_data import EITFileData

VENDOR = "timpel"

#: The number of values on each line.
_COLUMN_WIDTH = 1030

#: An image value that means "no value".
_NAN_VALUE = -1000

#: The sample frequency in Hz when none is given.
TIMPEL_SAMPLE_FREQUENCY = 50


def read_timpel(
    path: str | os.PathLike[str],
    sample_frequency: float | None = TIMPEL_SAMPLE_FREQUENCY,
    first_frame: int = 0,
    max_frames: int | None = None,
) -> EITFileData:
    """Read one Timpel ``.txt`` file.

    Args:
        path (str | os.PathLike[str]): The ``.txt`` file.
        sample_frequency (float | None): The sample frequency in Hz. ``None``
            or 0 uses 50 Hz.
        first_frame (int): The index of the first frame (line) to read,
            counted from 0. The time of that frame is
            ``first_frame / sample_frequency``.
        max_frames (int | None): The largest number of frames to read.
            ``None`` reads up to the end of the file.

    Returns:
        EITFileData: The images, the global impedance, the airway pressure,
            flow and volume, the minima, maxima and QRS complexes found by
            the device, and the breaths made from the minima and maxima.
            Time is in seconds from the first line of the file.

    Raises:
        OSError: If the file is not text, or a line does not hold 1,030
            values.
        ValueError: If no frame is left after ``first_frame``, or the
            cleaned minima and maxima do not give one maximum per breath
            (see ``_make_breaths``).

    Warns:
        RuntimeWarning: If fewer than ``max_frames`` frames are left after
            ``first_frame``.
    """
    path = os.fspath(path)
    if not sample_frequency:
        sample_frequency = TIMPEL_SAMPLE_FREQUENCY

    try:
        # Bug fix (see Provenance): ndmin=2 keeps 0 or 1 lines as a table.
        data: NDArray = np.loadtxt(
            path,
            dtype=float,
            delimiter=",",
            skiprows=first_frame,
            max_rows=max_frames,
            ndmin=2,
        )
    except UnicodeDecodeError as e:
        msg = (
            f"File {path} could not be read as Timpel data.\n"
            "Make sure this is a valid and uncorrupted Timpel data file.\n"
            f"Original error message: {e}"
        )
        raise OSError(msg) from e

    if data.shape[0] == 0:
        msg = (
            f"Invalid input: `first_frame` {first_frame} is larger than the "
            "total number of frames in the file."
        )
        raise ValueError(msg)
    if data.shape[1] != _COLUMN_WIDTH:
        msg = (
            f"Input does not have a width of {_COLUMN_WIDTH} columns.\n"
            "Make sure this is a valid and uncorrupted Timpel data file."
        )
        raise OSError(msg)

    if max_frames and data.shape[0] == max_frames:
        nframes = max_frames
    else:
        if max_frames:
            warnings.warn(
                f"The number of frames requested ({max_frames}) is larger "
                f"than the available number ({data.shape[0]}) of frames after "
                f"the first frame selected ({first_frame}).\n"
                f"{data.shape[0]} frames have been loaded.",
                RuntimeWarning,
                stacklevel=2,
            )
        nframes = data.shape[0]

    # The time is made for all frames from the first line on, then cut.
    # Adding first_frame / sample_frequency to a shorter range gives other
    # rounding errors in the last digits.
    time = np.arange(nframes + first_frame) / sample_frequency
    time = time[first_frame:]

    pixel_impedance = data[:, :1024]
    pixel_impedance = np.reshape(pixel_impedance, (-1, 32, 32), order="C")

    pixel_impedance = np.where(pixel_impedance == _NAN_VALUE, np.nan, pixel_impedance)

    # The waveforms keep the value as given: the default is the whole number
    # 50, as in eitprocessing. The images hold it as a float, as eitprocessing
    # EITData does.
    frequency = sample_frequency
    pixel_signal = Signal(
        values=pixel_impedance,
        time=time,
        sample_frequency=float(frequency),
        name="raw",
        modality="eit",
        category="impedance",
        channel="pixel_impedance",
        source=VENDOR,
    )

    global_impedance = np.nansum(pixel_impedance, axis=(1, 2))
    signals = {
        "global_impedance_(raw)": Signal(
            values=global_impedance,
            time=time,
            sample_frequency=frequency,
            unit="a.u.",
            name="Global impedance",
            metadata={"description": "Global impedance calculated from raw EIT data"},
            modality="eit",
            category="global_impedance",
            channel="global_impedance",
            source=VENDOR,
        ),
        "airway_pressure_(timpel)": _waveform(
            data[:, 1024],
            time,
            frequency,
            unit="cmH2O",
            name="Airway pressure",
            category="pressure",
            description="Airway pressure measured by Timpel device",
        ),
        "flow_(timpel)": _waveform(
            data[:, 1025],
            time,
            frequency,
            unit="L/s",
            name="Flow",
            category="flow",
            description="Flow measures by Timpel device",
        ),
        "volume_(timpel)": _waveform(
            data[:, 1026],
            time,
            frequency,
            unit="L",
            name="Volume",
            category="volume",
            description="Volume measured by Timpel device",
        ),
    }

    min_indices = np.nonzero(data[:, 1027] == 1)[0]
    max_indices = np.nonzero(data[:, 1028] == 1)[0]
    qrs_indices = np.nonzero(data[:, 1029] == 1)[0]
    events = {
        "minvalues_(timpel)": _device_marks(
            time[min_indices],
            name="minvalues_(timpel)",
            category="minvalue",
            description="Minimum values detected by Timpel device.",
        ),
        "maxvalues_(timpel)": _device_marks(
            time[max_indices],
            name="maxvalues_(timpel)",
            category="maxvalue",
            description="Maximum values detected by Timpel device.",
        ),
        "qrscomplexes_(timpel)": _device_marks(
            time[qrs_indices],
            name="qrscomplexes_(timpel)",
            category="qrs_complex",
            description="QRS complexes detected by Timpel device",
        ),
    }

    breaths = _make_breaths(time, min_indices, max_indices, global_impedance)
    intervals = {
        "breaths_(timpel)": IntervalData(
            name="breaths_(timpel)",
            modality="eit",
            intervals=list(breaths),
            category="breaths",
            metadata={
                "description": "Breaths (Timpel)",
                "default_partial_inclusion": False,
            },
        )
    }

    return EITFileData(
        vendor=VENDOR,
        path=path,
        sample_frequency=float(frequency),
        pixel_impedance=pixel_signal,
        signals=signals,
        events=events,
        intervals=intervals,
    )


def _waveform(
    values: NDArray,
    time: NDArray,
    sample_frequency: float,
    *,
    unit: str,
    name: str,
    category: str,
    description: str,
) -> Signal:
    """Return one ventilator waveform of the file as a ``Signal``."""

    return Signal(
        values=values,
        time=time,
        sample_frequency=sample_frequency,
        unit=unit,
        name=name,
        metadata={"description": description},
        modality="ventilator",
        category=category,
        source=VENDOR,
    )


def _device_marks(
    times: NDArray, *, name: str, category: str, description: str
) -> EventData:
    """Return the times of one kind of device mark as ``EventData``."""

    return EventData(
        name=name,
        modality="eit",
        events=[
            Event(name=category, modality="eit", time=float(mark_time))
            for mark_time in times
        ],
        category=category,
        metadata={"description": description},
    )


def _make_breaths(
    time: np.ndarray,
    min_indices: np.ndarray,
    max_indices: np.ndarray,
    gi: np.ndarray,
) -> list[BreathEvent]:
    """Make breaths from the minima and maxima found by the device.

    A breath runs from one minimum to the next, with a maximum in between.
    Maxima before the first or after the last minimum are left out. When two
    minima have no maximum between them, the higher minimum is removed. When
    two minima have more than one maximum between them, the lower of the
    first two maxima is removed.

    Known issue, copied from eitprocessing 1.8.7: ``peak_values`` comes
    from all maxima, but ``peak_indices`` was first cut at the edges. When
    maxima are cut at the start, the two lists are out of step, so the
    wrong one of two close maxima can be removed.

    Args:
        time (np.ndarray): The frame times in seconds.
        min_indices (np.ndarray): Frame indices of the minima.
        max_indices (np.ndarray): Frame indices of the maxima.
        gi (np.ndarray): The global impedance, used to compare minima and
            maxima.

    Returns:
        list[BreathEvent]: One breath per pair of minima that are next to
            each other, with the maximum as ``extremum_time``. Empty when
            fewer than 2 minima are left or there is no maximum.

    Raises:
        ValueError: If the cleaned minima and maxima do not give one maximum
            per breath, for example when a frame is both a minimum and a
            maximum.
    """
    if len(min_indices) < 2 or len(max_indices) < 1:
        return []

    valley_indices = min_indices.copy()
    peak_indices = max_indices.copy()

    keep_peaks = peak_indices > valley_indices[0]
    peak_indices = peak_indices[keep_peaks]

    keep_peaks = peak_indices < valley_indices[-1]
    peak_indices = peak_indices[keep_peaks]

    valley_values = gi[min_indices]
    peak_values = gi[max_indices]

    current_valley_index = 0
    while current_valley_index < len(valley_indices) - 1:
        start_index = valley_indices[current_valley_index]
        end_index = valley_indices[current_valley_index + 1]
        peaks_between_valleys = np.argwhere(
            (peak_indices > start_index) & (peak_indices < end_index),
        )
        if not len(peaks_between_valleys):
            # No maximum between two minima: remove the higher minimum.
            delete_valley_index = (
                current_valley_index
                if valley_values[current_valley_index]
                > valley_values[current_valley_index + 1]
                else current_valley_index + 1
            )
            valley_indices = np.delete(valley_indices, delete_valley_index)
            valley_values = np.delete(valley_values, delete_valley_index)
            continue

        if len(peaks_between_valleys) > 1:
            # More than one maximum between two minima: remove the lower one.
            delete_peak_index = (
                peaks_between_valleys[0]
                if peak_values[peaks_between_valleys[0]]
                < peak_values[peaks_between_valleys[1]]
                else peaks_between_valleys[1]
            )
            peak_indices = np.delete(peak_indices, delete_peak_index)
            peak_values = np.delete(peak_values, delete_peak_index)
            continue

        current_valley_index += 1

    # Bug fix (see Provenance): fewer than 2 minima left give no breaths.
    return [
        BreathEvent(
            "eit",
            float(time[start]),
            float(time[end]),
            extremum_time=float(time[middle]),
        )
        for start, end, middle in zip(
            valley_indices[:-1], valley_indices[1:], peak_indices, strict=True
        )
    ]
