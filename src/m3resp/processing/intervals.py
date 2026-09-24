"""Shared interval and onset/offset primitives.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from ReSurfEMG.

    Source:     https://github.com/resurfemg-org/ReSurfEMG
    Revision:   m3resp-integration (c63668689030e4581d5f985e7d09d3a8c01e7a77)
    Original:   resurfemg/postprocessing/event_detection.py::
                onoffpeak_baseline_crossing, onoffpeak_slope_extrapolation
    Copyright:  Copyright (c) 2022 Netherlands eScience Center and
                University of Twente
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - `onoffpeak_baseline_crossing` renamed to
      `onoff_from_baseline_crossings`, `onoffpeak_slope_extrapolation` renamed
      to `onoff_from_slope`.
    - Parameters renamed and reorganized as keyword-only arguments.
    - `baseline_crossings` extracted as a named function from the crossing
      calculation both upstream functions perform inline.
    - `sample_intervals_to_breath_events` below is independent M3RESP code
      (conversion to `BreathEvent`), not derived from ReSurfEMG.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from m3resp.core.events import BreathEvent
from m3resp.emg.processing.shared import _compute_derivative

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType


def baseline_crossings(signal: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Detect baseline crossings.

    Return indices where a signal crosses its baseline.

    Args:
        signal (numpy.ndarray): the signal array.
        baseline (numpy.ndarray): the baseline array.

    Returns:
        numpy.ndarray: Indices where the signal crosses its baseline.
    """
    return np.nonzero(np.diff(np.sign(np.asarray(signal) - np.asarray(baseline))) != 0)[
        0
    ]


def _validate_peak_indices(peak_indexes: np.ndarray, **kwargs) -> tuple[bool, bool]:
    # check the validity of the peaks
    peak_number = kwargs.get("peak_number", 0)
    start_index = kwargs.get("start_index", 0)
    valid_starts = kwargs.get("valid_starts", np.array([True]))
    valid_ends = kwargs.get("valid_ends", np.array([True]))
    peak_ends = kwargs.get("peak_ends", np.array([0]))
    # upslope and downslope are only used for slope-based peak detection,
    # so we check if they are present in kwargs
    upslope = "previous_downslope" in kwargs and "new_upslope" in kwargs
    previous_downslope = kwargs.get("previous_downslope", 0)
    new_upslope = kwargs.get("new_upslope", 0)

    end_index = peak_ends[peak_number]
    peak_index = peak_indexes[peak_number]
    # If the start index is after the peak index, mark the start as invalid
    if peak_number > 0 and start_index > peak_index:
        valid_starts[peak_number] = False
    # If the end index is before the peak index, mark the end as invalid
    if end_index < peak_index:
        valid_ends[peak_number] = False
    # If the current peak's end index is after the next peak's start index,
    # mark the current peak's end as invalid
    if (
        peak_number < (len(peak_indexes) - 1)
        and end_index > peak_indexes[peak_number + 1]
    ):
        valid_ends[peak_number] = False

    # check if the start of the peak is before the end of the previous peak
    if (
        peak_number > 0
        and start_index < peak_ends[peak_number - 1]
        and valid_ends[peak_number - 1]
        and valid_starts[peak_number]
    ):
        # case: upslope-based on- and offset detection
        if upslope:
            # if the new upslope is less than the previous downslope, the previous
            # peak's end is valid and the current peak's start is invalid
            previous_end_validity = new_upslope <= -previous_downslope
            current_start_validity = not previous_end_validity
        # case: baseline-crossing-based on- and offset detection
        else:
            # if the distance from the peak index to the start index is less than or
            # equal to the distance from the previous peak's end index to the previous
            # peak's index,the current peak's start is valid and the previous peak's
            # end is invalid
            current_start_validity = (
                peak_index - start_index
                <= peak_ends[peak_number - 1] - peak_indexes[peak_number - 1]
            )
            previous_end_validity = not current_start_validity
        valid_ends[peak_number - 1] = previous_end_validity
        valid_starts[peak_number] = current_start_validity

    return valid_starts[peak_number], valid_ends[peak_number]


def onoff_from_baseline_crossings(
    signal: np.ndarray,
    baseline: np.ndarray,
    peak_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[bool]]:
    """Find peak starts/ends by nearest baseline crossings around peaks.

    The start of each peak is the last crossing of the baseline before it,
    and the end is the first crossing after it. Starts and ends that fall on
    the wrong side of a peak, or overlap a neighbouring peak's window, are
    marked invalid. Any baseline can be used, e.g. the moving or slope-sum
    baseline of an EMG envelope, or the moving baseline of airway pressure.

    Args:
        signal (numpy.ndarray): Envelope signal.
        baseline (numpy.ndarray): Baseline signal of EMG data for baseline detection.
        peak_indices (numpy.ndarray): List of peak indices for which to find on-
            and offset.

    Returns:
        tuple:
            - numpy.ndarray: List of start indices of the peaks.
            - numpy.ndarray: List of end indices of the peaks.
            - numpy.ndarray: List of boolean values for valid starts.
            - numpy.ndarray: List of boolean values for valid ends.
            - numpy.ndarray: List of boolean values for valid peaks.
    """
    crossings = baseline_crossings(signal, baseline)
    _peak_indexes = np.asarray(peak_indices, dtype=int)
    peak_starts = np.zeros((len(_peak_indexes),), dtype=int)
    peak_ends = np.zeros((len(_peak_indexes),), dtype=int)
    valid_starts = np.array([True for _ in range(len(_peak_indexes))])
    valid_ends = np.array([True for _ in range(len(_peak_indexes))])

    for peak_number, peak_index in enumerate(_peak_indexes):
        delta_samples = peak_index - crossings[crossings < peak_index]
        if len(delta_samples) < 1:
            peak_starts[peak_number] = 0
            crossings_after = crossings[crossings > peak_index]
            peak_ends[peak_number] = int(crossings_after[0])
        else:
            crossing_index = np.argmin(delta_samples)
            peak_starts[peak_number] = int(crossings[crossing_index])
            if crossing_index < len(crossings) - 1:
                peak_ends[peak_number] = int(crossings[crossing_index + 1])
            else:
                peak_ends[peak_number] = len(signal) - 1

        valid_starts[peak_number], valid_ends[peak_number] = _validate_peak_indices(
            _peak_indexes,
            peak_number=peak_number,
            start_index=peak_starts[peak_number],
            valid_starts=valid_starts,
            valid_ends=valid_ends,
        )

    valid_peaks = [
        valids[0] and valids[1]
        for valids in zip(valid_starts, valid_ends, strict=False)
    ]

    return peak_starts, peak_ends, valid_starts, valid_ends, valid_peaks


def onoff_from_slope(
    signal: np.ndarray,
    *,
    sample_frequency: float,
    peak_indices: np.ndarray,
    slope_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[bool]]:
    """Calculate the on- and offsets of peaks using slope extrapolation.

    This function calculates the peak on- and offsets of a signal by extra-
    polating the maximum slopes in the `slope_window_s` to the zero crossings.
    The validity arrays provide feedback on the validity of the detected on-
    and offsets, aiming to prevent onsets after peak indices, offsets before
    peak indices, and overlapping peaks.

    Args:
        signal (numpy.ndarray): Signal to identify on- and offsets in.
        sample_frequency (int): Sampling rate of the signal.
        peak_indices (numpy.ndarray): List of peak indices for which to find on- and
            offset.
        slope_window (int): How many samples on each side to use for detecting the
            local maximum slope.

    Returns:
        tuple:
            - numpy.ndarray: List of start indices of the peaks.
            - numpy.ndarray: List of end indices of the peaks.
            - numpy.ndarray: List of boolean values for valid starts.
            - numpy.ndarray: List of boolean values for valid ends.
            - numpy.ndarray: List of boolean values for valid peaks.
    """
    scipy_signal = _scipy_signal()
    _signal = np.asarray(signal)
    _peak_indexes = np.asarray(peak_indices, dtype=int)
    derivative = _compute_derivative(_signal, sample_frequency)

    # get the local minima and maxima in the derivative
    max_upslope_indices = scipy_signal.argrelextrema(
        derivative, np.greater, order=slope_window
    )[0]
    max_downslope_indices = scipy_signal.argrelextrema(
        derivative, np.less, order=slope_window
    )[0]

    peak_starts = np.zeros((len(_peak_indexes),), dtype=int)
    peak_ends = np.zeros((len(_peak_indexes),), dtype=int)
    valid_starts = np.array([True for _ in range(len(_peak_indexes))])
    valid_ends = np.array([True for _ in range(len(_peak_indexes))])
    previous_downslope = 0
    new_upslope = 0
    max_downslope_index = 0
    for peak_number, peak_index in enumerate(_peak_indexes):
        # if there are no local derivative maxima before the current peak,
        # set the start index to 0
        if len(max_upslope_indices[max_upslope_indices < peak_index]) < 1:
            start_index = 0
        else:
            # get the closest local derivative maximum before the current peak
            max_upslope_index = int(
                max_upslope_indices[max_upslope_indices < peak_index][-1]
            )
            # store the amplitude of the derivative there
            new_upslope = derivative[max_upslope_index]
            # and the corresponding amplitude of the signal
            y_value = _signal[max_upslope_index]
            dy_dt_value = derivative[max_upslope_index]
            # calculate y/dy (newton's method)
            upslope_index_delta = int(
                np.array(y_value * sample_frequency // dy_dt_value, dtype=int).astype(
                    np.int64
                )
            )
            # find the root: x
            start_index = max(0, max_upslope_index - upslope_index_delta)
        peak_starts[peak_number] = start_index
        # do the same for the downslope after the peak
        if len(max_downslope_indices[max_downslope_indices > peak_index]) < 1:
            end_index = len(_signal) - 1
        else:
            max_downslope_index = int(
                max_downslope_indices[max_downslope_indices > peak_index][0]
            )
            if peak_number > 0:
                previous_downslope = derivative[max_downslope_index]
            y_value = _signal[max_downslope_index]
            dy_dt_value = derivative[max_downslope_index]
            downslope_index_delta = int(
                np.array(y_value * sample_frequency // dy_dt_value, dtype=int).astype(
                    np.int64
                )
            )
            end_index = min(
                len(_signal) - 1, max_downslope_index - downslope_index_delta
            )
        peak_ends[peak_number] = end_index

        # check the validity of the peaks
        valid_starts[peak_number], valid_ends[peak_number] = _validate_peak_indices(
            _peak_indexes,
            peak_number=peak_number,
            start_index=peak_starts[peak_number],
            valid_starts=valid_starts,
            valid_ends=valid_ends,
            previous_downslope=previous_downslope,
            new_upslope=new_upslope,
        )
    # valid peaks have valid starts and ends
    valid_peaks = [
        detection[0] and detection[1]
        for detection in zip(valid_starts, valid_ends, strict=True)
    ]
    return peak_starts, peak_ends, valid_starts, valid_ends, valid_peaks


def sample_intervals_to_breath_events(
    *,
    start_indices: Sequence[int],
    end_indices: Sequence[int],
    peak_indices: Sequence[int] | None = None,
    sample_frequency: float | None = None,
    time: Sequence[float] | None = None,
    modality: str,
    source: str | None = None,
) -> list[BreathEvent]:
    """Convert sample-index breath intervals into common `BreathEvent` objects."""
    starts = np.asarray(start_indices, dtype=int)
    ends = np.asarray(end_indices, dtype=int)
    peaks = None if peak_indices is None else np.asarray(peak_indices, dtype=int)
    return [
        BreathEvent(
            modality=modality,
            start_time=_sample_to_time(
                start, sample_frequency=sample_frequency, time=time
            ),
            end_time=_sample_to_time(end, sample_frequency=sample_frequency, time=time),
            peak_time=(
                None
                if peaks is None
                else _sample_to_time(
                    peaks[index], sample_frequency=sample_frequency, time=time
                )
            ),
            start_index=int(start),
            peak_index=None if peaks is None else int(peaks[index]),
            end_index=int(end),
            source=source,
        )
        for index, (start, end) in enumerate(zip(starts, ends, strict=True))
    ]


def _sample_to_time(
    sample_index: int,
    *,
    sample_frequency: float | None,
    time: Sequence[float] | None,
) -> float:
    if time is not None:
        return float(time[sample_index])
    if sample_frequency is None:
        msg = "sample_frequency or time is required"
        raise ValueError(msg)
    return float(sample_index) / float(sample_frequency)


def _scipy_signal() -> ModuleType:
    try:
        from scipy import signal  # noqa: PLC0415
    except ImportError as exc:
        from m3resp.core.exceptions import OptionalDependencyError  # noqa: PLC0415

        msg = "Slope-based interval detection requires SciPy. Install `scipy` to "
        "use `m3resp.processing.intervals`."
        raise OptionalDependencyError(msg) from exc
    return signal
