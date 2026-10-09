"""Shared filtering primitives for EIT, EMG, and ventilator signals.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   1.8.7
    Original:   eitprocessing/filters/butterworth_filters.py::
                ButterworthFilter.apply_filter
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - Extracted from the `ButterworthFilter` class into the free function
      `butterworth_filter`, taking the filter type and cutoff directly.
    - Optional `captures` mapping records the filter parameters for provenance.

Portions of this module are derived from ReSurfEMG.

    Source:     https://github.com/resurfemg-org/ReSurfEMG
    Revision:   m3resp-integration (c63668689030e4581d5f985e7d09d3a8c01e7a77)
    Original:   resurfemg/preprocessing/filtering.py::emg_bandpass_butter,
                emg_lowpass_butter, emg_highpass_butter, notch_filter,
                compute_power_loss
    Copyright:  Copyright (c) 2022 Netherlands eScience Center and
                University of Twente
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - `emg_bandpass_butter`/`emg_lowpass_butter`/`emg_highpass_butter` renamed
      to `bandpass_filter`/`lowpass_filter`/`highpass_filter` and reduced to
      thin wrappers over `butterworth_filter`, so EIT, EMG and ventilator
      signals share one filter implementation.
    - Parameters renamed and reorganized as keyword-only arguments.
    - `compute_power_loss` differs from upstream. The
      upstream version sums the whole `(frequencies, density)` pair returned
      by `scipy.signal.welch` instead of the density alone, and inverts the
      power ratio. This version unpacks the pair and uses
      `100 * (1 - processed / original)`. Upstream has been notified; until
      that is resolved these two functions give different results.
    - `harmonic_notch_filter`, `bandstop_filter` and the validation helpers
      below are independent M3RESP code.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any, Literal, TypeGuard, get_args

import numpy as np

from m3resp.core.exceptions import OptionalDependencyError

ButterworthFilterType = Literal["lowpass", "highpass", "bandpass", "bandstop"]


def butterworth_filter(
    values: np.ndarray,
    *,
    filter_type: ButterworthFilterType,
    cutoff_frequency: float | Sequence[float],
    sample_frequency: float,
    order: int,
    axis: int = 0,
    captures: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply a zero-phase Butterworth filter along one axis.

    Applies the filter forwards and backwards using SciPy second-order
    sections. Values retain their physical units.

    Args:
        values (numpy.ndarray): Finite input samples, with time along `axis`.
        filter_type (ButterworthFilterType): "lowpass", "highpass", "bandpass",
            or "bandstop".
        cutoff_frequency (float | Sequence[float]): Cutoff in Hz. Lowpass and
            highpass require one number; bandpass and bandstop require two
            numbers (low, high), with low < high. Every cutoff must be positive
            and below half the sampling rate.
        sample_frequency (float): Positive sampling rate in Hz.
        order (int): Positive integer filter order for each pass.
        axis (int): Time axis to filter. Defaults to 0.
        captures (dict[str, Any] | None): Optional dictionary updated with
            unfiltered_data, filtered_data, sample_frequency, and the relevant
            low_pass_frequency, high_pass_frequency or frequency_bands.
            Bandstop cutoffs are appended to frequency_bands.

    Returns:
        numpy.ndarray: Filtered values with the input shape and units.

    Raises:
        TypeError: If a cutoff or sampling rate has an invalid type, or order
            is a non-integer. Boolean values are rejected for these arguments.
        ValueError: If the filter type, order, sampling rate, or cutoffs are
            invalid, samples contain NaN or infinity, or the time axis has too
            few samples for SciPy's edge padding.
        OptionalDependencyError: If SciPy is unavailable.
    """

    scipy_signal = _scipy_signal()
    allowed_types = get_args(ButterworthFilterType)
    if filter_type not in allowed_types:
        raise ValueError(
            f"filter_type must be one of {allowed_types}, got {filter_type!r}"
        )
    cutoff = _normalize_cutoff_frequency(filter_type, cutoff_frequency)
    _validate_common_filter_arguments(
        sample_frequency=sample_frequency,
        order=order,
    )

    data = np.asarray(values)
    if not np.all(np.isfinite(data)):
        raise ValueError(
            "Input data contains NaN or infinite values. Fill gaps before "
            "applying a Butterworth filter."
        )

    capture_value(captures, "unfiltered_data", data)
    capture_value(captures, "sample_frequency", sample_frequency)
    _capture_butterworth_parameters(captures, filter_type, cutoff)

    sos = scipy_signal.butter(
        N=order,
        Wn=cutoff,
        btype=filter_type,
        fs=sample_frequency,
        analog=False,
        output="sos",
    )
    filtered = scipy_signal.sosfiltfilt(sos, data, axis=axis)
    capture_value(captures, "filtered_data", filtered)
    return filtered


def lowpass_filter(
    values: np.ndarray,
    *,
    cutoff_frequency: float,
    sample_frequency: float,
    order: int,
    axis: int = 0,
    captures: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply a low-pass Butterworth filter.

    Args:
        values (numpy.ndarray): Input data to filter.
        cutoff_frequency (float): Lowpass cutoff frequency of the filter.
        sample_frequency (float): Sampling rate of the input data.
        order (int): Order of the filter.
        axis (int): Axis along which to apply the filter.
        captures (dict, optional): Dictionary to store captured values.
            If None (default), no values will be captured.

    Returns:
        numpy.ndarray: Filtered data.
    """

    return butterworth_filter(
        values,
        filter_type="lowpass",
        cutoff_frequency=cutoff_frequency,
        sample_frequency=sample_frequency,
        order=order,
        axis=axis,
        captures=captures,
    )


def highpass_filter(
    values: np.ndarray,
    *,
    cutoff_frequency: float,
    sample_frequency: float,
    order: int,
    axis: int = 0,
    captures: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply a high-pass Butterworth filter.

    Args:
        values (numpy.ndarray): Input data to filter.
        cutoff_frequency (float): Highpass cutoff frequency of the filter.
        sample_frequency (float): Sampling rate of the input data.
        order (int): Order of the filter.
        axis (int): Axis along which to apply the filter.
        captures (dict, optional): Dictionary to store captured values.
            If None (default), no values will be captured.

    Returns:
        numpy.ndarray: Filtered data.
    """

    return butterworth_filter(
        values,
        filter_type="highpass",
        cutoff_frequency=cutoff_frequency,
        sample_frequency=sample_frequency,
        order=order,
        axis=axis,
        captures=captures,
    )


def bandpass_filter(
    values: np.ndarray,
    *,
    cutoff_frequency: Sequence[float],
    sample_frequency: float,
    order: int,
    axis: int = 0,
    captures: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply a band-pass Butterworth filter.

    Args:
        values (numpy.ndarray): Input data to filter.
        cutoff_frequency (tuple): A tuple of two floats representing the lower and upper
            cutoff frequencies.
        sample_frequency (float): Sampling rate of the input data.
        order (int): Order of the filter.
        axis (int): Axis along which to apply the filter.
        captures (dict, optional): Dictionary to store captured values.
            If None (default), no values will be captured.

    Returns:
        numpy.ndarray: Filtered data.
    """

    return butterworth_filter(
        values,
        filter_type="bandpass",
        cutoff_frequency=cutoff_frequency,
        sample_frequency=sample_frequency,
        order=order,
        axis=axis,
        captures=captures,
    )


def bandstop_filter(
    values: np.ndarray,
    *,
    cutoff_frequency: Sequence[float],
    sample_frequency: float,
    order: int,
    axis: int = 0,
    captures: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply a band-stop Butterworth filter.

    Args:
        values (numpy.ndarray): Input data to filter.
        cutoff_frequency (tuple): A tuple of two floats representing the lower and upper
            cutoff frequencies.
        sample_frequency (float): Sampling rate of the input data.
        order (int): Order of the filter.
        axis (int): Axis along which to apply the filter.
        captures (dict, optional): Dictionary to store captured values.
            If None (default), no values will be captured.

    Returns:
        numpy.ndarray: Filtered data.
    """

    return butterworth_filter(
        values,
        filter_type="bandstop",
        cutoff_frequency=cutoff_frequency,
        sample_frequency=sample_frequency,
        order=order,
        axis=axis,
        captures=captures,
    )


def notch_filter(
    values: np.ndarray,
    *,
    frequency: float,
    sample_frequency: float,
    quality_factor: float,
    axis: int = 0,
) -> np.ndarray:
    """Apply an IIR notch filter at one frequency.

    Args:
        values (numpy.ndarray): Input data to filter.
        frequency (float): Frequency to remove.
        sample_frequency (float): Sampling rate of the input data.
        quality_factor (float): Quality factor of the filter: the notch
            frequency divided by the width of the removed band.
        axis (int): Axis along which to apply the filter.

    Returns:
        numpy.ndarray: Filtered data.
    """

    scipy_signal = _scipy_signal()
    b_notch, a_notch = scipy_signal.iirnotch(
        frequency,
        quality_factor,
        sample_frequency,
    )
    return scipy_signal.filtfilt(b_notch, a_notch, values, axis=axis)


def harmonic_notch_filter(
    values: np.ndarray,
    *,
    base_frequency: float,
    sample_frequency: float,
    max_frequency: float | None = None,
    distance: float | None = None,
    order: int = 4,
    quality_factor: float = 30.0,
    axis: int = 0,
    captures: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply notch or band-stop filtering at harmonics of a base frequency.

    Args:
        values (numpy.ndarray): Input data to filter.
        base_frequency (float): Base frequency of the harmonics.
        sample_frequency (float): Sampling rate of the input data.
        max_frequency (float, optional): Maximum frequency to filter.
            If None (default), the Nyquist frequency is used.
        distance (float, optional): Distance from the harmonic to the cutoff frequency.
            If None (default), a single notch is applied at each harmonic;
            otherwise a band-stop filter from harmonic - distance to
            harmonic + distance.
        order (int): Order of the band-stop filter. Used only when `distance`
            is set.
        quality_factor (float): Quality factor of the notch filter. Used only
            when `distance` is None.
        axis (int): Axis along which to apply the filter.
        captures (dict, optional): Dictionary to store captured values.
            If None (default), no values will be captured.

    Returns:
        numpy.ndarray: Filtered data.
    """

    if base_frequency <= 0:
        raise ValueError("base_frequency must be positive")

    nyquist = sample_frequency / 2
    stop_frequency = min(max_frequency or nyquist, nyquist)
    filtered = np.asarray(values)
    harmonic_i = 1
    harmonic = harmonic_i * base_frequency
    while harmonic < stop_frequency:
        if distance is None:
            filtered = notch_filter(
                filtered,
                frequency=harmonic,
                sample_frequency=sample_frequency,
                quality_factor=quality_factor,
                axis=axis,
            )
        else:
            low = max(harmonic - distance, np.finfo(float).eps)
            high = min(harmonic + distance, nyquist - np.finfo(float).eps)
            if low < high:
                filtered = bandstop_filter(
                    filtered,
                    cutoff_frequency=(low, high),
                    sample_frequency=sample_frequency,
                    order=order,
                    axis=axis,
                    captures=captures,
                )
        harmonic_i += 1
        harmonic = harmonic_i * base_frequency
    return filtered


def compute_power_loss(
    original: np.ndarray,
    processed: np.ndarray,
    *,
    original_frequency: float,
    processed_frequency: float,
    n_segment: int | None = None,
    percent_overlap: float = 25,
) -> float:
    """Compute percentage power loss after processing.

    Args:
        original (numpy.ndarray): Original data.
        processed (numpy.ndarray): Processed data.
        original_frequency (float): Sampling frequency of the original data.
        processed_frequency (float): Sampling frequency of the processed data.
        n_segment (int, optional): Width of the window for Welch's method in samples.
            If None, it is set to half the sampling frequency.
        percent_overlap (float): Overlap between segments, as a percentage of
            the original sampling frequency (not of `n_segment`), matching
            ReSurfEMG. With the default `n_segment` of fs/2, the default of 25
            gives segments that overlap by half. Defaults to 25.

    Returns:
        float: Percentage power loss after processing.
    """

    scipy_signal = _scipy_signal()
    if n_segment is None:
        n_segment = int(original_frequency) // 2
    noverlap = int(percent_overlap / 100 * original_frequency)

    _, original_density = scipy_signal.welch(
        original,
        original_frequency,
        nperseg=n_segment,
        noverlap=noverlap,
    )
    _, processed_density = scipy_signal.welch(
        processed,
        processed_frequency,
        nperseg=n_segment,
        noverlap=noverlap,
    )
    return float(100 * (1 - (np.sum(processed_density) / np.sum(original_density))))


def _normalize_cutoff_frequency(
    filter_type: ButterworthFilterType,
    cutoff_frequency: float | Sequence[float],
) -> float | tuple[float, float]:
    """Convert numeric cutoffs to one float or a two-float tuple.

    Raises TypeError for invalid numeric types and ValueError for a sequence
    whose length differs from two. Frequency limits are checked by SciPy.
    """

    if filter_type in {"lowpass", "highpass"}:
        if not _is_number(cutoff_frequency):
            raise TypeError("cutoff_frequency must be numeric for low/high pass")
        return float(cutoff_frequency)

    if isinstance(cutoff_frequency, np.ndarray):
        cutoff_frequency = cutoff_frequency.tolist()
    if not isinstance(cutoff_frequency, Sequence) or isinstance(
        cutoff_frequency,
        str | bytes,
    ):
        raise TypeError("cutoff_frequency must be a two-value sequence")
    if len(cutoff_frequency) != 2:
        raise ValueError("cutoff_frequency must contain two values")
    low, high = cutoff_frequency
    if not _is_number(low) or not _is_number(high):
        raise TypeError("cutoff_frequency values must be numeric")
    return (float(low), float(high))


def _validate_common_filter_arguments(
    *,
    sample_frequency: float,
    order: int,
) -> None:
    """Require a positive integer order and a positive numeric sampling rate.

    Raises TypeError for invalid types, including boolean values, and
    ValueError for zero or negative values.
    """

    if not _is_whole_number(order):
        raise TypeError("order must be a whole number")
    if order < 1:
        raise ValueError("order must be positive")
    if not _is_number(sample_frequency):
        raise TypeError("sample_frequency must be numeric")
    if sample_frequency <= 0:
        raise ValueError("sample_frequency must be positive")


def _is_number(value: Any) -> TypeGuard[float]:
    """Return True for Python and numpy numbers, but not for True or False."""
    return isinstance(value, Real) and not isinstance(value, bool | np.bool_)


def _is_whole_number(value: Any) -> TypeGuard[int]:
    """Return True for Python and numpy whole numbers, but not for True or False."""
    return isinstance(value, Integral) and not isinstance(value, bool | np.bool_)


def capture_value(
    captures: dict[str, Any] | None,
    key: str,
    value: Any,
    *,
    append_to_list: bool = False,
) -> None:
    """Capture a value in a dictionary.

    Args:
        captures (dict or None): Dictionary to capture values in.
            If None, no values will be captured.
        key (str): Key to use for capturing the value.
        value (Any): Value to capture.
        append_to_list (bool): If True, the value will be appended to a list
            under the given key.
    """
    if captures is None:
        return
    if append_to_list:
        captures.setdefault(key, []).append(value)
    else:
        captures[key] = value


def _capture_butterworth_parameters(
    captures: dict[str, Any] | None,
    filter_type: ButterworthFilterType,
    cutoff: float | tuple[float, float],
) -> None:
    """Record the filter's cutoff frequencies in Hz when captures is supplied."""

    match filter_type:
        case "lowpass":
            capture_value(captures, "low_pass_frequency", cutoff)
        case "highpass":
            capture_value(captures, "high_pass_frequency", cutoff)
        case "bandpass":
            if not isinstance(cutoff, tuple):
                raise TypeError("bandpass cutoff must be (low, high)")
            capture_value(captures, "low_pass_frequency", cutoff[1])
            capture_value(captures, "high_pass_frequency", cutoff[0])
        case "bandstop":
            capture_value(captures, "frequency_bands", cutoff, append_to_list=True)


def _scipy_signal():
    try:
        from scipy import signal
    except ImportError as exc:
        raise OptionalDependencyError(
            "Filtering requires SciPy. Install `scipy` to use "
            "`m3resp.processing.filters`."
        ) from exc
    return signal
