"""Shared interval and onset/offset primitives."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from m3resp.core.events import BreathEvent


def baseline_crossings(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Return indices where a signal crosses its baseline."""

    return np.nonzero(np.diff(np.sign(np.asarray(values) - np.asarray(baseline))) != 0)[
        0
    ]


def onoff_from_baseline_crossings(
    values: np.ndarray,
    baseline: np.ndarray,
    peak_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[bool]]:
    """Find peak starts/ends by nearest baseline crossings around peaks.


    This function calculates the peaks of each breath using the
    slopesum baseline of envelope data.

    Args:
        values (numpy.ndarray): Envelope signal.
        baseline (numpy.ndarray): Baseline signal of EMG data for baseline detection.
        peak_indices (numpy.ndarray): List of peak indices for which to find on- and offset.

    Returns:
        tuple:
            - numpy.ndarray: List of start indices of the peaks.
            - numpy.ndarray: List of end indices of the peaks.
            - numpy.ndarray: List of boolean values for valid starts.
            - numpy.ndarray: List of boolean values for valid ends.
            - numpy.ndarray: List of boolean values for valid peaks.
    """

    crossings = baseline_crossings(values, baseline)
    peaks = np.asarray(peak_indices, dtype=int)
    starts = np.zeros((len(peaks),), dtype=int)
    ends = np.zeros((len(peaks),), dtype=int)
    valid_starts = np.array([True for _ in range(len(peaks))])
    valid_ends = np.array([True for _ in range(len(peaks))])

    for peak_number, peak_index in enumerate(peaks):
        delta_samples = peak_index - crossings[crossings < peak_index]
        if len(delta_samples) < 1:
            starts[peak_number] = 0
            crossings_after = crossings[crossings > peak_index]
            ends[peak_number] = (
                int(crossings_after[0])
                if len(crossings_after) >= 1
                else len(values) - 1
            )
        else:
            crossing_index = np.argmin(delta_samples)
            starts[peak_number] = int(crossings[crossing_index])
            if crossing_index < len(crossings) - 1:
                ends[peak_number] = int(crossings[crossing_index + 1])
            else:
                ends[peak_number] = len(values) - 1

        if peak_number > 0 and starts[peak_number] > peak_index:
            valid_starts[peak_number] = False

        if (
            peak_number < (len(peaks) - 1)
            and ends[peak_number] > peaks[peak_number + 1]
        ):
            valid_ends[peak_number] = False

        if peak_number > 0 and starts[peak_number] <= ends[peak_number - 1]:
            if not valid_starts[peak_number] or not valid_ends[peak_number - 1]:
                pass
            elif (
                peak_index - starts[peak_number]
                > ends[peak_number - 1] - peaks[peak_number - 1]
            ):
                valid_starts[peak_number] = False
            else:
                valid_ends[peak_number - 1] = False

    valid_peaks = [
        detection[0] and detection[1]
        for detection in zip(valid_starts, valid_ends, strict=True)
    ]
    return starts, ends, valid_starts, valid_ends, valid_peaks


def onoff_from_slope(
    values: np.ndarray,
    *,
    sample_frequency: float,
    peak_indices: np.ndarray,
    slope_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[bool]]:
    """Find peak starts/ends by extrapolating local maximum slopes."""

    signal = _scipy_signal()
    data = np.asarray(values)
    peaks = np.asarray(peak_indices, dtype=int)
    derivative = (data[1:] - data[:-1]) * sample_frequency

    max_upslope_indices = signal.argrelextrema(
        derivative, np.greater, order=slope_window
    )[0]
    max_downslope_indices = signal.argrelextrema(
        derivative, np.less, order=slope_window
    )[0]

    starts = np.zeros((len(peaks),), dtype=int)
    ends = np.zeros((len(peaks),), dtype=int)
    valid_starts = np.array([True for _ in range(len(peaks))])
    valid_ends = np.array([True for _ in range(len(peaks))])
    previous_downslope = 0

    for peak_number, peak_index in enumerate(peaks):
        if len(max_upslope_indices[max_upslope_indices < peak_index]) < 1:
            start_index = 0
            new_upslope = 0
        else:
            max_upslope_index = int(
                max_upslope_indices[max_upslope_indices < peak_index][-1]
            )
            new_upslope = derivative[max_upslope_index]
            y_value = data[max_upslope_index]
            dy_dt_value = derivative[max_upslope_index]
            upslope_index_delta = int(
                np.array(y_value * sample_frequency // dy_dt_value, dtype=int).astype(
                    np.int64
                )
            )
            start_index = max(0, max_upslope_index - upslope_index_delta)
        starts[peak_number] = start_index

        if len(max_downslope_indices[max_downslope_indices > peak_index]) < 1:
            end_index = len(data) - 1
        else:
            max_downslope_index = int(
                max_downslope_indices[max_downslope_indices > peak_index][0]
            )
            y_value = data[max_downslope_index]
            dy_dt_value = derivative[max_downslope_index]
            downslope_index_delta = int(
                np.array(y_value * sample_frequency // dy_dt_value, dtype=int).astype(
                    np.int64
                )
            )
            end_index = min(len(data) - 1, max_downslope_index - downslope_index_delta)
        ends[peak_number] = end_index

        if start_index > peak_index:
            valid_starts[peak_number] = False
        if end_index < peak_index:
            valid_ends[peak_number] = False
        if peak_number < (len(peaks) - 1) and end_index > peaks[peak_number + 1]:
            valid_ends[peak_number] = False

        if peak_number > 0 and start_index < ends[peak_number - 1]:
            if not valid_ends[peak_number - 1]:
                pass
            elif new_upslope > -previous_downslope:
                valid_ends[peak_number - 1] = False
            else:
                valid_starts[peak_number] = False

        if len(max_downslope_indices[max_downslope_indices > peak_index]) >= 1:
            previous_downslope = derivative[max_downslope_index]

    valid_peaks = [
        detection[0] and detection[1]
        for detection in zip(valid_starts, valid_ends, strict=True)
    ]
    return starts, ends, valid_starts, valid_ends, valid_peaks


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
        raise ValueError("sample_frequency or time is required")
    return float(sample_index) / float(sample_frequency)


def _scipy_signal():
    try:
        from scipy import signal
    except ImportError as exc:
        from m3resp.core.exceptions import OptionalDependencyError

        raise OptionalDependencyError(
            "Slope-based interval detection requires SciPy. Install `scipy` to "
            "use `m3resp.processing.intervals`."
        ) from exc
    return signal
