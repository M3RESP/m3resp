"""Shared filtering primitives for EIT, EMG, and ventilator signals."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from m3resp.core.exceptions import OptionalDependencyError

FilterType = Literal["lowpass", "highpass", "bandpass", "bandstop"]


def butterworth_filter(
    values: np.ndarray,
    *,
    filter_type: FilterType,
    cutoff_frequency: float | Sequence[float],
    sample_frequency: float,
    order: int,
    axis: int = 0,
    captures: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply a zero-phase Butterworth filter using second-order sections."""

    scipy_signal = _scipy_signal()
    cutoff = _normalize_cutoff_frequency(filter_type, cutoff_frequency)
    _validate_common_filter_arguments(
        sample_frequency=sample_frequency,
        order=order,
    )

    data = np.asarray(values)
    if np.any(np.isnan(data)):
        raise ValueError(
            "Input data contains NaN-values. Fill gaps before applying a "
            "Butterworth filter."
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
    """Apply a low-pass Butterworth filter."""

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
    """Apply a high-pass Butterworth filter."""

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
    """Apply a band-pass Butterworth filter."""

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
    """Apply a band-stop Butterworth filter."""

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
    """Apply an IIR notch filter at one frequency."""

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
    """Apply notch or band-stop filtering at harmonics of a base frequency."""

    if base_frequency <= 0:
        raise ValueError("base_frequency must be positive")

    nyquist = sample_frequency / 2
    stop_frequency = min(max_frequency or nyquist, nyquist)
    filtered = np.asarray(values)
    harmonic = base_frequency
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
        harmonic += base_frequency
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
    """Compute percentage power loss after processing."""

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
    filter_type: FilterType,
    cutoff_frequency: float | Sequence[float],
) -> float | tuple[float, float]:
    if filter_type in {"lowpass", "highpass"}:
        if not isinstance(cutoff_frequency, int | float):
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
    if not isinstance(low, int | float) or not isinstance(high, int | float):
        raise TypeError("cutoff_frequency values must be numeric")
    return (float(low), float(high))


def _validate_common_filter_arguments(
    *,
    sample_frequency: float,
    order: int,
) -> None:
    if not isinstance(order, int):
        raise TypeError("order must be an int")
    if order < 1:
        raise ValueError("order must be positive")
    if not isinstance(sample_frequency, int | float):
        raise TypeError("sample_frequency must be numeric")
    if sample_frequency <= 0:
        raise ValueError("sample_frequency must be positive")


def capture_value(
    captures: dict[str, Any] | None,
    key: str,
    value: Any,
    *,
    append_to_list: bool = False,
) -> None:
    if captures is None:
        return
    if append_to_list:
        captures.setdefault(key, []).append(value)
    else:
        captures[key] = value


def _capture_butterworth_parameters(
    captures: dict[str, Any] | None,
    filter_type: FilterType,
    cutoff: float | tuple[float, float],
) -> None:
    match filter_type:
        case "lowpass":
            capture_value(captures, "low_pass_frequency", cutoff)
        case "highpass":
            capture_value(captures, "high_pass_frequency", cutoff)
        case "bandpass":
            assert isinstance(cutoff, tuple), "bandpass cutoff must be (low, high)"
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
