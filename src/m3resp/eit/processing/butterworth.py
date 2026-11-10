"""Butterworth filter of EIT images, one pixel at a time along time.

``butterworth_filter_images`` filters each pixel of EIT images with
``m3resp.processing.filters.butterworth_filter``, forwards and backwards.
A pixel without any value stays without value. The order must be from 1 to
``MAX_EIT_BUTTERWORTH_ORDER`` (10), as in eitprocessing 1.8.7.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from m3resp.data.signals import Signal
from m3resp.processing.filters import ButterworthFilterType, butterworth_filter

#: Highest order of a Butterworth filter of EIT images, as in eitprocessing.
MAX_EIT_BUTTERWORTH_ORDER = 10


def check_eit_butterworth_order(order: int) -> None:
    """Refuse a Butterworth order outside 1 to 10 for EIT images.

    The filter runs forwards and backwards, so the effective order is twice
    the given order. Higher orders can give unstable or wrong results.

    Args:
        order (int): The order of the filter.

    Raises:
        ValueError: If ``order`` is below 1 or above 10.
    """

    if not 1 <= order <= MAX_EIT_BUTTERWORTH_ORDER:
        raise ValueError(
            "The order of a Butterworth filter of EIT images must be from 1 to "
            f"{MAX_EIT_BUTTERWORTH_ORDER}, got {order}. Higher orders can give "
            "unstable results."
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
        pixel_impedance (Any): EIT images, time along the first axis.
        operation (str): Name of the calling step or function, for messages.
        apply (Callable[[numpy.ndarray], Any]): The filter, applied to the
            images along time.
        captures (dict[str, Any] | None): The captures that ``apply``
            writes. Arrays in it with the shape of the images get NaN at the
            unmeasured pixels too.

    Returns:
        numpy.ndarray: The filtered images, NaN at the unmeasured pixels.

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

    # The 0 values are also in the captured images before and after the
    # filter; set the unmeasured pixels to NaN there too.
    for key, captured in list((captures or {}).items()):
        recorded = np.asarray(captured)
        if recorded.shape == values.shape:
            blanked = recorded.astype(float, copy=True)
            blanked[:, unmeasured] = np.nan
            captures[key] = blanked  # type: ignore[index]

    return filtered


def butterworth_filter_images(
    images: Signal,
    *,
    mode: ButterworthFilterType,
    cutoff_frequency: float | Sequence[float],
    order: int,
    operation: str,
) -> tuple[Signal, dict[str, Any]]:
    """Filter each pixel of EIT images along time with a Butterworth filter.

    The filter is ``m3resp.processing.filters.butterworth_filter``, run
    forwards and backwards. A pixel without any value stays without value.

    Args:
        images (Signal): EIT images (frames, rows, columns), time along the
            first axis, frames per second in ``sample_frequency``.
        mode (ButterworthFilterType): ``"lowpass"``, ``"highpass"``,
            ``"bandpass"`` or ``"bandstop"``.
        cutoff_frequency (float | Sequence[float]): One cutoff in Hz for
            ``"lowpass"`` and ``"highpass"``; (low, high) in Hz for the two
            band modes.
        order (int): Butterworth order, 1 to 10.
        operation (str): Name of the calling step or function, for messages.

    Returns:
        tuple[Signal, dict[str, Any]]: The filtered images and the captures.
            The images are named ``"<Mode>-filtered EIT data"``, with
            processing state ``"intermediate"`` and method
            ``"m3resp.processing.filters.butterworth_filter"``; the metadata
            of ``images`` is copied. The captures hold ``sample_frequency``,
            the cutoffs in Hz, and the images before and after the filter as
            ``unfiltered_data`` and ``filtered_data``.

    Raises:
        ValueError: If the order is outside 1 to 10, if a pixel has gaps
            among otherwise present samples, or if the cutoffs are invalid.
    """

    check_eit_butterworth_order(order)
    captures: dict[str, Any] = {}
    filtered_pixels = filter_pixels_preserving_gaps(
        images.values,
        operation=operation,
        apply=lambda pixels: butterworth_filter(
            pixels,
            filter_type=mode,
            cutoff_frequency=cutoff_frequency,
            sample_frequency=float(images.sample_frequency or 0.0),
            order=order,
            axis=0,
            captures=captures,
        ),
        captures=captures,
    )
    name = f"{mode.title()}-filtered EIT data"
    filtered = dataclasses.replace(
        images,
        values=filtered_pixels,
        time=np.copy(images.time),
        name=name,
        metadata=dict(images.metadata),
        processing_state="intermediate",
        method="m3resp.processing.filters.butterworth_filter",
    )
    captures["unfiltered_data"] = images
    captures["filtered_data"] = filtered
    return filtered, captures
