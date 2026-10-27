"""Read a Draeger ``.bin`` EIT file.

A Draeger file is a row of frames of the same size. Each frame holds:

- the time, as a fraction of a day;
- one 32 by 32 image of the impedance;
- a flag that marks a breath minimum (-1) or maximum (1) found by the device;
- an event marker number and a text of 30 characters, set by the user;
- a timing error flag;
- the Medibus values, which the ventilator sends to the EIT device.

The frame size tells the file format. An "original" frame (4,358 bytes)
holds 52 Medibus values. A "pressure pod" frame (4,382 bytes) holds 58: the
pod adds the airway, esophageal, transpulmonary and gastric pressure. The
reader keeps the Medibus values that change in each frame (pressure, flow,
volume and CO2). It reads the other Medibus values (settings such as PEEP)
and does not keep them.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/datahandling/loading/draeger.py::
                load_from_single_path, _estimate_sample_frequency,
                _convert_medibus_data, _read_frame, _MedibusField,
                _bin_file_formats
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - `load_from_single_path` is `read_draeger`. It returns an
      `EITFileData` of m3resp types: the images and the waveforms are
      `Signal`s, the breath minima and maxima and the event markers are
      `EventData`. The dictionary keys are the eitprocessing labels.
    - An event marker is an `Event` with the text as `label` and the number
      in `EventData.values`.
    - The path is read with `os.path`.
    - Docstrings and comments are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import mmap
import os
import sys
import warnings
from typing import Any, NamedTuple
from warnings import catch_warnings

import numpy as np
import scipy as sp
from numpy.typing import NDArray

from m3resp.data.event_data import EventData
from m3resp.data.events import Event
from m3resp.data.signals import Signal
from m3resp.eit.io.binreader import BinReader
from m3resp.eit.io.file_data import EITFileData

VENDOR = "draeger"

#: Medibus values below this number mean "not measured" and become NaN.
NAN_VALUE_INDICATOR = -1e30

#: The estimated sample frequency is rounded to this number of digits.
SAMPLE_FREQUENCY_ESTIMATION_PRECISION = 4


def read_draeger(
    path: str | os.PathLike[str],
    sample_frequency: float | None = None,
    first_frame: int = 0,
    max_frames: int | None = None,
) -> EITFileData:
    """Read one Draeger ``.bin`` file.

    The time in the file is the time of day. It starts again at 0 after
    midnight, so the reader adds a day at each such jump. The sample
    frequency comes from a straight line fit of time against frame number,
    rounded to 4 digits.

    Args:
        path (str | os.PathLike[str]): The ``.bin`` file.
        sample_frequency (float | None): The sample frequency in Hz. If it
            differs from the estimate by more than 1 in 10,000, a warning is
            given, and this value is used. ``None`` uses the estimate.
        first_frame (int): The index of the first frame to read, counted
            from 0.
        max_frames (int | None): The largest number of frames to read.
            ``None`` reads up to the end of the file.

    Returns:
        EITFileData: The images, the global impedance, the Medibus
            waveforms, the breath minima and maxima found by the device and
            the event markers. Time is in seconds of the day.

    Raises:
        OSError: If the file size is not a whole number of frames of a known
            format.
        ValueError: If ``first_frame`` is after the last frame, no frame is
            left to read, the time does not increase from frame to frame, or
            the sample frequency cannot be estimated and none is given.
        TypeError: If ``sample_frequency`` is not a number.

    Warns:
        RuntimeWarning: If ``max_frames`` is more than the frames left after
            ``first_frame``, a frame has a timing error, or the given sample
            frequency differs from the estimate.
    """
    path = os.fspath(path)
    file_size = os.path.getsize(path)

    frame_size: int
    medibus_fields: list[_MedibusField]

    # The file format is the one whose frame size divides the file size.
    for _file_format_data in _bin_file_formats.values():
        frame_size = _file_format_data["frame_size"]
        if file_size % frame_size == 0:
            medibus_fields = _file_format_data["medibus_fields"]
            break
    else:
        msg = (
            f"File size {file_size} of file {path!s} does not match the "
            "supported *.bin file formats.\n"
            "Currently this package does not support loading files containing "
            "esophageal pressure or other non-standard data. "
            "Make sure this is a valid and uncorrupted Dräger data file."
        )
        raise OSError(msg)
    total_frames = file_size // frame_size

    if (f0 := first_frame) > (fn := total_frames):
        msg = (
            f"Invalid input: `first_frame` ({f0}) is larger than the total "
            f"number of frames in the file ({fn})."
        )
        raise ValueError(msg)

    n_frames = min(
        total_frames - first_frame,
        max_frames if max_frames is not None else sys.maxsize,
    )
    if n_frames < 1:
        msg = f"No frames to load with `{first_frame=}` and `{max_frames=}`."
        raise ValueError(msg)

    if max_frames and max_frames != n_frames:
        msg = (
            f"The number of frames requested ({max_frames}) is larger "
            f"than the available number ({n_frames}) of frames after "
            f"the first frame selected ({first_frame}, total frames: "
            f"{total_frames}).\n {n_frames} frames will be loaded."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # The frame before the first frame is read too, only to see whether the
    # first frame starts a new event marker. Its data is not kept.
    load_dummy_frame = first_frame > 0
    first_frame_to_load = first_frame - 1 if load_dummy_frame else 0

    pixel_impedance = np.zeros((n_frames, 32, 32))
    time = np.zeros((n_frames,))
    events: list[tuple[float, int, str]] = []
    phases: list[tuple[float, int]] = []
    medibus_data = np.zeros((len(medibus_fields), n_frames), dtype=np.float32)

    with (
        open(path, "rb") as fo,
        mmap.mmap(fo.fileno(), length=0, access=mmap.ACCESS_READ) as fh,
    ):
        fh.seek(first_frame_to_load * frame_size)
        reader = BinReader(fh)
        previous_marker = None

        first_index = -1 if load_dummy_frame else 0
        for index in range(first_index, n_frames):
            previous_marker = _read_frame(
                reader,
                index,
                time,
                pixel_impedance,
                medibus_data,
                len(medibus_fields),
                events,
                phases,
                previous_marker,
            )

    # The time of day starts again after 24 hours.
    time = np.unwrap(time, period=24 * 60 * 60)

    if not np.all(np.diff(time) > 0):
        msg = "The time axis is not strictly monotonically increasing."
        raise ValueError(msg)

    sample_frequency = _estimate_sample_frequency(time, sample_frequency)

    pixel_signal = Signal(
        values=pixel_impedance,
        time=time,
        sample_frequency=sample_frequency,
        name="raw",
        modality="eit",
        category="impedance",
        channel="pixel_impedance",
        source=VENDOR,
    )
    signals = _convert_medibus_data(
        medibus_data, medibus_fields, time, sample_frequency
    )
    signals["global_impedance_(raw)"] = Signal(
        values=np.nansum(pixel_impedance, axis=(1, 2)),
        time=time,
        sample_frequency=sample_frequency,
        unit="a.u.",
        name="Global impedance (raw)",
        modality="eit",
        category="impedance",
        channel="global_impedance",
        source=VENDOR,
    )

    marks = {
        "minvalues_(draeger)": _phase_events(phases, -1, "minvalue"),
        "maxvalues_(draeger)": _phase_events(phases, 1, "maxvalue"),
        "events_(draeger)": EventData(
            name="events_(draeger)",
            modality="eit",
            events=[
                Event(name="event", modality="eit", time=event_time, label=text)
                for event_time, _, text in events
            ],
            values=[marker for _, marker, _ in events],
            category="event",
            metadata={"description": "Events loaded from Draeger data"},
        ),
    }

    return EITFileData(
        vendor=VENDOR,
        path=path,
        sample_frequency=sample_frequency,
        pixel_impedance=pixel_signal,
        signals=signals,
        events=marks,
    )


def _phase_events(
    phases: list[tuple[float, int]], flag: int, category: str
) -> EventData:
    """Return the times of one min/max flag value as ``EventData``."""

    kind = "Minimum" if flag == -1 else "Maximum"
    label = f"{category}s_(draeger)"
    return EventData(
        name=label,
        modality="eit",
        events=[
            Event(name=category, modality="eit", time=float(phase_time))
            for phase_time, phase_flag in phases
            if phase_flag == flag
        ],
        category=category,
        metadata={"description": f"{kind} values detected by Draeger device."},
    )


def _estimate_sample_frequency(
    time: np.ndarray, sample_frequency: float | None
) -> float:
    """Estimate the sample frequency from the time, and compare it with the given one.

    Args:
        time (np.ndarray): The frame times in seconds.
        sample_frequency (float | None): The sample frequency the user gave,
            in Hz, or ``None``.

    Returns:
        float: The given sample frequency, or else the estimate rounded to
            4 digits.

    Raises:
        ValueError: If the estimate fails (too few frames) and no sample
            frequency is given.
        TypeError: If ``sample_frequency`` is not a number.

    Warns:
        RuntimeWarning: If the estimate fails and a sample frequency is
            given, or the given value differs from the estimate by more
            than 1 in 10,000.
    """
    with catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        unrounded_estimated_sample_frequency = (
            1 / sp.stats.linregress(np.arange(len(time)), time).slope
        )

    if np.isnan(unrounded_estimated_sample_frequency):
        msg = (
            "Could not estimate sample frequency from time axis, "
            f"which could be due to too few data points ({len(time)})."
        )
        if sample_frequency is not None:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            return float(sample_frequency)

        raise ValueError(msg)

    # Round to a number of digits, counted from the first digit that is not 0.
    estimated_sample_frequency = round(
        unrounded_estimated_sample_frequency,
        -math.ceil(np.log10(abs(unrounded_estimated_sample_frequency)))
        + SAMPLE_FREQUENCY_ESTIMATION_PRECISION,
    )

    if sample_frequency is None:
        return float(estimated_sample_frequency)
    if not isinstance(sample_frequency, (int, float)):
        msg = (
            f"Provided sample frequency has invalid type {type(sample_frequency)}; "
            "should be int or float."
        )
        raise TypeError(msg)

    if not np.isclose(
        sample_frequency,
        unrounded_estimated_sample_frequency,
        rtol=10**-SAMPLE_FREQUENCY_ESTIMATION_PRECISION,
        atol=0,
    ):
        digits = SAMPLE_FREQUENCY_ESTIMATION_PRECISION + 2
        msg = (
            "Provided sample frequency "
            f"({sample_frequency:.{digits}f} Hz) "
            "does not match the estimated sample frequency "
            f"({unrounded_estimated_sample_frequency:.{digits}f} Hz) "
            f"within {SAMPLE_FREQUENCY_ESTIMATION_PRECISION} digits. "
            "Note that the estimate might not be as accurate for very short signals."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    return float(sample_frequency)


def _convert_medibus_data(
    medibus_data: NDArray,
    medibus_fields: list[_MedibusField],
    time: NDArray,
    sample_frequency: float,
) -> dict[str, Signal]:
    """Turn the Medibus waveforms into ``Signal``s.

    Values below ``NAN_VALUE_INDICATOR`` become NaN, in ``medibus_data``
    itself. Only the fields marked ``continuous`` are kept.

    Args:
        medibus_data (NDArray): One row of float32 values per Medibus field.
        medibus_fields (list[_MedibusField]): Name, unit and kind of each row.
        time (NDArray): The frame times in seconds.
        sample_frequency (float): The sample frequency in Hz.

    Returns:
        dict[str, Signal]: One ``Signal`` per waveform, by field name.
    """
    signals: dict[str, Signal] = {}

    for field_info, data in zip(medibus_fields, medibus_data, strict=True):
        data[data < NAN_VALUE_INDICATOR] = np.nan
        if field_info.continuous:
            signals[field_info.signal_name] = Signal(
                values=data,
                time=time,
                sample_frequency=sample_frequency,
                unit=field_info.unit,
                name=field_info.signal_name,
                metadata={
                    "description": (
                        f"Continuous {field_info.signal_name} data loaded from file"
                    )
                },
                modality="ventilator",
                category=field_info.signal_name,
                source=VENDOR,
            )

    return signals


def _read_frame(
    reader: BinReader,
    index: int,
    time: NDArray,
    pixel_impedance: NDArray,
    medibus_data: NDArray,
    n_medibus_fields: int,
    events: list[tuple[float, int, str]],
    phases: list[tuple[float, int]],
    previous_marker: int | None,
) -> int:
    """Read one frame and store its data at ``index``.

    With a negative ``index`` nothing is stored. In both cases the event
    marker of the frame is returned.

    Args:
        reader (BinReader): The reader, at the start of the frame.
        index (int): Where to store the frame, or -1 to store nothing.
        time (NDArray): Frame times in seconds; ``time[index]`` is set.
        pixel_impedance (NDArray): Images; ``pixel_impedance[index]`` is set.
        medibus_data (NDArray): Medibus values; ``medibus_data[:, index]`` is
            set.
        n_medibus_fields (int): The number of Medibus values in a frame.
        events (list[tuple[float, int, str]]): Time, marker number and text
            of each new event marker; a new one is added here.
        phases (list[tuple[float, int]]): Time and flag of each breath
            minimum (-1) or maximum (1); a new one is added here.
        previous_marker (int | None): The event marker of the frame before.

    Returns:
        int: The event marker of this frame.

    Warns:
        RuntimeWarning: If the frame has a timing error.
    """
    frame_time = reader.float64() * 24 * 60 * 60
    _ = reader.float32()
    frame_pixel_impedance = reader.npfloat32(length=1024)
    frame_pixel_impedance = np.reshape(frame_pixel_impedance, (32, 32), "C")
    min_max_flag = reader.int32()
    event_marker = reader.int32()
    event_text = reader.string(length=30)
    timing_error = reader.int32()

    frame_medibus_data = reader.npfloat32(length=n_medibus_fields)

    if index < 0:
        return event_marker

    time[index] = frame_time
    pixel_impedance[index, :, :] = frame_pixel_impedance
    medibus_data[:, index] = frame_medibus_data

    # The event marker number stays the same until the next event. A higher
    # number than in the frame before starts a new event.
    if ((previous_marker is not None) and (event_marker > previous_marker)) or (
        index == 0 and event_text
    ):
        events.append((frame_time, event_marker, event_text))
    if timing_error:
        warnings.warn(
            "A timing error was encountered during loading.",
            RuntimeWarning,
            stacklevel=2,
        )
    if min_max_flag in (1, -1):
        phases.append((frame_time, min_max_flag))

    return event_marker


class _MedibusField(NamedTuple):
    """Name, unit and kind of one Medibus value.

    ``continuous`` is True for a waveform (a new value in each frame) and
    False for a setting or a value per breath.
    """

    signal_name: str
    unit: str
    continuous: bool


_bin_file_formats: dict[str, dict[str, Any]] = {
    "original": {
        "frame_size": 4358,
        "medibus_fields": [
            _MedibusField("airway pressure", "mbar", True),
            _MedibusField("flow", "L/min", True),
            _MedibusField("volume", "mL", True),
            _MedibusField("CO2 (%)", "%", True),
            _MedibusField("CO2 (kPa)", "kPa", True),
            _MedibusField("CO2 (mmHg)", "mmHg", True),
            _MedibusField("dynamic compliance", "mL/mbar", False),
            _MedibusField("resistance", "mbar/L/s", False),
            _MedibusField("r^2", "", False),
            _MedibusField("spontaneous inspiratory time", "s", False),
            _MedibusField("minimal pressure", "mbar", False),
            _MedibusField("P0.1", "mbar", False),
            _MedibusField("mean pressure", "mbar", False),
            _MedibusField("plateau pressure", "mbar", False),
            _MedibusField("PEEP", "mbar", False),
            _MedibusField("intrinsic PEEP", "mbar", False),
            _MedibusField("mandatory respiratory rate", "/min", False),
            _MedibusField("mandatory minute volume", "L/min", False),
            _MedibusField("peak inspiratory pressure", "mbar", False),
            _MedibusField("mandatory tidal volume", "L", False),
            _MedibusField("spontaneous tidal volume", "L", False),
            _MedibusField("trapped volume", "mL", False),
            _MedibusField("mandatory expiratory tidal volume", "mL", False),
            _MedibusField("spontaneous expiratory tidal volume", "mL", False),
            _MedibusField("mandatory inspiratory tidal volume", "mL", False),
            _MedibusField("tidal volume", "mL", False),
            _MedibusField("spontaneous inspiratory tidal volume", "mL", False),
            _MedibusField("negative inspiratory force", "mbar", False),
            _MedibusField("leak minute volume", "L/min", False),
            _MedibusField("leak percentage", "%", False),
            _MedibusField("spontaneous respiratory rate", "/min", False),
            _MedibusField("percentage of spontaneous minute volume", "%", False),
            _MedibusField("spontaneous minute volume", "L/min", False),
            _MedibusField("minute volume", "L/min", False),
            _MedibusField("airway temperature", "degrees C", False),
            _MedibusField("rapid shallow breating index", "1/min/L", False),
            _MedibusField("respiratory rate", "/min", False),
            _MedibusField("inspiratory:expiratory ratio", "", False),
            _MedibusField("CO2 flow", "mL/min", False),
            _MedibusField("dead space volume", "mL", False),
            _MedibusField(
                "percentage dead space of expiratory tidal volume", "%", False
            ),
            _MedibusField("end-tidal CO2", "%", False),
            _MedibusField("end-tidal CO2", "kPa", False),
            _MedibusField("end-tidal CO2", "mmHg", False),
            _MedibusField("fraction inspired O2", "%", False),
            _MedibusField("spontaneous inspiratory:expiratory ratio", "", False),
            _MedibusField("elastance", "mbar/L", False),
            _MedibusField("time constant", "s", False),
            _MedibusField(
                "ratio between upper 20% pressure range and total dynamic compliance",
                "",
                False,
            ),
            _MedibusField("end-inspiratory pressure", "mbar", False),
            _MedibusField("expiratory tidal volume", "mL", False),
            _MedibusField("time at low pressure", "s", False),
        ],
    },
    "pressure_pod": {
        "frame_size": 4382,
        "medibus_fields": [
            _MedibusField("airway pressure", "mbar", True),
            _MedibusField("flow", "L/min", True),
            _MedibusField("volume", "mL", True),
            _MedibusField("CO2 (%)", "%", True),
            _MedibusField("CO2 (kPa)", "kPa", True),
            _MedibusField("CO2 (mmHg)", "mmHg", True),
            _MedibusField("dynamic compliance", "mL/mbar", False),
            _MedibusField("resistance", "mbar/L/s", False),
            _MedibusField("r^2", "", False),
            _MedibusField("spontaneous inspiratory time", "s", False),
            _MedibusField("minimal pressure", "mbar", False),
            _MedibusField("P0.1", "mbar", False),
            _MedibusField("mean pressure", "mbar", False),
            _MedibusField("plateau pressure", "mbar", False),
            _MedibusField("PEEP", "mbar", False),
            _MedibusField("intrinsic PEEP", "mbar", False),
            _MedibusField("mandatory respiratory rate", "/min", False),
            _MedibusField("mandatory minute volume", "L/min", False),
            _MedibusField("peak inspiratory pressure", "mbar", False),
            _MedibusField("mandatory tidal volume", "L", False),
            _MedibusField("spontaneous tidal volume", "L", False),
            _MedibusField("trapped volume", "mL", False),
            _MedibusField("mandatory expiratory tidal volume", "mL", False),
            _MedibusField("spontaneous expiratory tidal volume", "mL", False),
            _MedibusField("mandatory inspiratory tidal volume", "mL", False),
            _MedibusField("tidal volume", "mL", False),
            _MedibusField("spontaneous inspiratory tidal volume", "mL", False),
            _MedibusField("negative inspiratory force", "mbar", False),
            _MedibusField("leak minute volume", "L/min", False),
            _MedibusField("leak percentage", "%", False),
            _MedibusField("spontaneous respiratory rate", "/min", False),
            _MedibusField("percentage of spontaneous minute volume", "%", False),
            _MedibusField("spontaneous minute volume", "L/min", False),
            _MedibusField("minute volume", "L/min", False),
            _MedibusField("airway temperature", "degrees C", False),
            _MedibusField("rapid shallow breating index", "1/min/L", False),
            _MedibusField("respiratory rate", "/min", False),
            _MedibusField("inspiratory:expiratory ratio", "", False),
            _MedibusField("CO2 flow", "mL/min", False),
            _MedibusField("dead space volume", "mL", False),
            _MedibusField(
                "percentage dead space of expiratory tidal volume", "%", False
            ),
            _MedibusField("end-tidal CO2", "%", False),
            _MedibusField("end-tidal CO2", "kPa", False),
            _MedibusField("end-tidal CO2", "mmHg", False),
            _MedibusField("fraction inspired O2", "%", False),
            _MedibusField("spontaneous inspiratory:expiratory ratio", "", False),
            _MedibusField("elastance", "mbar/L", False),
            _MedibusField("time constant", "s", False),
            _MedibusField(
                "ratio between upper 20% pressure range and total dynamic compliance",
                "",
                False,
            ),
            _MedibusField("end-inspiratory pressure", "mbar", False),
            _MedibusField("expiratory tidal volume", "mL", False),
            _MedibusField("high pressure", "mbar", False),
            _MedibusField("low pressure", "mbar", False),
            _MedibusField("time at low pressure", "s", False),
            _MedibusField("airway pressure (pod)", "mbar", True),
            _MedibusField("esophageal pressure (pod)", "mbar", True),
            _MedibusField("transpulmonary pressure (pod)", "mbar", True),
            _MedibusField("gastric pressure/auxiliary pressure (pod)", "mbar", True),
        ],
    },
}
