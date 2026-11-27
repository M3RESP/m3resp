"""Butterworth filter of the EIT pixel impedance, one pixel at a time along time.

``butterworth_filter_pixels`` filters each pixel of the pixel impedance along time with
``m3resp.processing.filters.butterworth_filter``, forwards and backwards.
A pixel without any value stays without value. The order must be 1 or more.
An order above ``MAX_EIT_BUTTERWORTH_ORDER`` (10) gives a warning; eitprocessing
1.8.7 refuses such an order unless ``ignore_max_order=True``.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from m3resp.data.signals import Signal
from m3resp.processing.captures import capture_value
from m3resp.processing.filters import ButterworthFilterType, butterworth_filter

#: Public name of the function, stored as the method of each result.
METHOD = "m3resp.eit.processing.butterworth_filter_pixels"

#: Highest Butterworth order without a warning, as in eitprocessing.
MAX_EIT_BUTTERWORTH_ORDER = 10


def check_eit_butterworth_order(order: int) -> None:
    """Check the Butterworth order of a filter of the EIT pixel impedance.

    The filter runs forwards and backwards, so the effective order is twice
    the given order. An order above 10 can give unstable or wrong results.

    Args:
        order (int): The order of the filter.

    Raises:
        ValueError: If ``order`` is below 1.

    Warns:
        UserWarning: If ``order`` is above 10.
    """

    if order < 1:
        raise ValueError(
            "The order of a Butterworth filter of EIT pixels must be 1 or more, "
            f"got {order}."
        )
    if order > MAX_EIT_BUTTERWORTH_ORDER:
        warnings.warn(
            f"A Butterworth order of {order} is above {MAX_EIT_BUTTERWORTH_ORDER}. "
            f"The filter runs forwards and backwards, so the effective order is "
            f"{2 * order}, which can give unstable results. Check the filtered "
            "data.",
            UserWarning,
            stacklevel=3,
        )


def filter_pixels_preserving_gaps(
    pixel_impedance: Any,
    *,
    operation: str,
    apply: Callable[[np.ndarray], Any],
    captures: dict[str, Any] | None = None,
) -> np.ndarray:
    """Filter the measured pixels with ``apply``; unmeasured pixels stay missing.

    A pixel that was never measured (outside the electrode plane, or
    switched off) is NaN for the whole recording, and it is NaN in the
    result too. A Butterworth filter cannot run on NaN, so these pixels are
    set to 0 during the call and back to NaN in the result. ``apply``
    filters each pixel along time (axis 0) on its own, so the 0 values
    cannot reach a measured pixel.

    A pixel with NaN for part of the recording only cannot be filtered
    without new values in the gap, so it gives a ValueError.

    Args:
        pixel_impedance (Any): Pixel impedance, time along the first axis.
        operation (str): Name of the calling step or function, for messages.
        apply (Callable[[numpy.ndarray], Any]): The filter, applied to the
            pixels along time.
        captures (dict[str, Any] | None): The captures that ``apply``
            writes. Arrays in it with the shape of ``pixel_impedance`` get NaN at the
            unmeasured pixels too.

    Returns:
        numpy.ndarray: The filtered pixel impedance, NaN at the unmeasured pixels.

    Raises:
        ValueError: If a pixel has NaN for part of the recording only.
    """

    values = np.asarray(pixel_impedance, dtype=float)
    missing = np.isnan(values)
    if not missing.any():
        return np.asarray(apply(values))

    # Axis 0 is time; the other axes are the pixel grid.
    n_samples = values.shape[0]
    missing_per_pixel = missing.sum(axis=0)
    partly_missing = (missing_per_pixel > 0) & (missing_per_pixel < n_samples)
    if partly_missing.any():
        examples = ", ".join(
            "(" + ", ".join(str(int(axis)) for axis in position) + ")"
            for position in np.argwhere(partly_missing)[:5]
        )
        raise ValueError(
            f"{operation}: {int(partly_missing.sum())} pixel(s) are missing "
            f"samples for part of the recording only (e.g. at {examples}). "
            "A Butterworth filter cannot span a gap, so these cannot be "
            "filtered without inventing the missing samples. Repair or drop "
            "those pixels first. Pixels missing for the whole recording need "
            "no action - they stay missing through the filter."
        )

    unmeasured = missing_per_pixel == n_samples
    placeholder = values.copy()
    placeholder[:, unmeasured] = 0.0
    filtered = np.array(apply(placeholder), dtype=float, copy=True)
    filtered[:, unmeasured] = np.nan

    # The 0 values are also in the captured pixel impedance before and after the
    # filter; set the unmeasured pixels to NaN there too.
    for key, captured in list((captures or {}).items()):
        recorded = np.asarray(captured)
        if recorded.shape == values.shape:
            blanked = recorded.astype(float, copy=True)
            blanked[:, unmeasured] = np.nan
            captures[key] = blanked  # type: ignore[index]

    return filtered


def butterworth_filter_pixels(
    pixel_impedance: Signal,
    *,
    mode: ButterworthFilterType,
    cutoff_frequency: float | Sequence[float],
    order: int,
    operation: str,
    captures: dict[str, Any] | None = None,
) -> Signal:
    """Filter each pixel of the pixel impedance along time with a Butterworth filter.

    The filter is ``m3resp.processing.filters.butterworth_filter``, run
    forwards and backwards. A pixel without any value stays without value.

    Args:
        pixel_impedance (Signal): Pixel impedance (frames, rows, columns),
            time along the first axis, frames per second in ``sample_frequency``.
        mode (ButterworthFilterType): ``"lowpass"``, ``"highpass"``,
            ``"bandpass"`` or ``"bandstop"``.
        cutoff_frequency (float | Sequence[float]): One cutoff in Hz for
            ``"lowpass"`` and ``"highpass"``; (low, high) in Hz for the two
            band modes.
        order (int): Butterworth order, 1 or more. Above 10 gives a warning.
        operation (str): Name of the calling step or function, for messages.
        captures (dict[str, Any] | None): When a dictionary, it receives
            ``sample_frequency``, the cutoffs in Hz, and the pixel impedance
            before and after the filter as ``unfiltered_data`` and
            ``filtered_data``.

    Returns:
        Signal: The filtered pixel impedance, named
            ``"<Mode>-filtered EIT data"``, with processing state
            ``"intermediate"`` and method
            ``"m3resp.eit.processing.butterworth_filter_pixels"``. The
            metadata of ``pixel_impedance`` is copied.

    Raises:
        ValueError: If the order is below 1, if a pixel has gaps among
            otherwise present samples, or if the cutoffs are invalid.

    Warns:
        UserWarning: If the order is above 10.
    """

    check_eit_butterworth_order(order)
    filtered_pixels = filter_pixels_preserving_gaps(
        pixel_impedance.values,
        operation=operation,
        apply=lambda pixels: butterworth_filter(
            pixels,
            filter_type=mode,
            cutoff_frequency=cutoff_frequency,
            sample_frequency=float(pixel_impedance.sample_frequency or 0.0),
            order=order,
            axis=0,
            captures=captures,
        ),
        captures=captures,
    )
    name = f"{mode.title()}-filtered EIT data"
    filtered = dataclasses.replace(
        pixel_impedance,
        values=filtered_pixels,
        time=np.copy(pixel_impedance.time),
        name=name,
        metadata=dict(pixel_impedance.metadata),
        processing_state="intermediate",
        method=METHOD,
    )
    capture_value(captures, "unfiltered_data", pixel_impedance)
    capture_value(captures, "filtered_data", filtered)
    return filtered
