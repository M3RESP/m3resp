"""Butterworth filter of EIT images, with the eitprocessing fields of the result."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from m3resp.data.signals import Signal
from m3resp.eit.processing import butterworth
from m3resp.processing.filters import ButterworthFilterType

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
    """Filter EIT images with ``m3resp.eit.processing.butterworth``.

    The result is that of
    ``m3resp.eit.processing.butterworth.butterworth_filter_images``, with
    the label, name and description for an eitprocessing ``Sequence`` added
    to ``metadata["eitprocessing_fields"]`` of the filtered images.

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
        tuple[Signal, dict[str, Any]]: The filtered images and the captures;
            ``captures["filtered_data"]`` holds the same filtered images.

    Raises:
        ValueError: If the order is outside 1 to 10, if a pixel has gaps
            among otherwise present samples, or if the cutoffs are invalid.
    """

    filtered, captures = butterworth.butterworth_filter_images(
        images,
        mode=mode,
        cutoff_frequency=cutoff_frequency,
        order=order,
        operation=operation,
    )
    fields = {
        **eitprocessing_fields(images),
        "type": "EITData",
        "label": label,
        "name": filtered.name,
        "description": f"EIT data filtered with a {mode} Butterworth filter.",
    }
    filtered = dataclasses.replace(
        filtered, metadata={**filtered.metadata, EITPROCESSING_FIELDS: fields}
    )
    captures["filtered_data"] = filtered
    return filtered, captures
