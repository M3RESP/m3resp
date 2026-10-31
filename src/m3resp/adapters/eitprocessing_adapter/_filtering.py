"""Butterworth filter of EIT images, as the EIT step and ``preprocess`` use it."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from m3resp.data.signals import Signal
from m3resp.processing.filters import ButterworthFilterType, butterworth_filter

from ._shared import check_eit_butterworth_order, filter_pixels_preserving_gaps
from ._signals import EITPROCESSING_FIELDS, eitprocessing_fields


def butterworth_filter_images(
    images: Signal,
    *,
    mode: ButterworthFilterType,
    cutoff_frequency: float | Sequence[float],
    order: int,
    label: str,
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
        label (str): Label of the filtered images in an eitprocessing
            ``Sequence``.
        operation (str): Name of the calling step or function, for messages.

    Returns:
        tuple[Signal, dict[str, Any]]: The filtered images and the captures.
            The images are named ``"<Mode>-filtered EIT data"``, with
            processing state ``"intermediate"`` and method
            ``"m3resp.processing.filters.butterworth_filter"``; their
            ``metadata["eitprocessing_fields"]`` holds the new label, name and
            description. The captures hold ``sample_frequency``, the cutoffs
            in Hz, and the images before and after the filter as
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
    fields = {
        **eitprocessing_fields(images),
        "type": "EITData",
        "label": label,
        "name": name,
        "description": f"EIT data filtered with a {mode} Butterworth filter.",
    }
    filtered = dataclasses.replace(
        images,
        values=filtered_pixels,
        time=np.copy(images.time),
        name=name,
        metadata={**images.metadata, EITPROCESSING_FIELDS: fields},
        processing_state="intermediate",
        method="m3resp.processing.filters.butterworth_filter",
    )
    captures["unfiltered_data"] = images
    captures["filtered_data"] = filtered
    return filtered, captures
