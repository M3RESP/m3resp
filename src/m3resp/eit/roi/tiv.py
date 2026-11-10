"""Functional lung space from the mean tidal impedance variation (TIV).

``tiv_lungspace`` computes the TIV of each pixel in each breath, with the
breath times of the global breaths for every pixel. The mean over the breaths
gives one TIV per pixel. A pixel is in the lung space when its mean TIV is at
least ``threshold`` times the largest mean TIV (15 % by default).

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/roi/tiv.py::TIVLungspace
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The class `TIVLungspace` and its method `apply` are one function,
      `tiv_lungspace`, with the EIT images as a `Signal`. The global breaths
      are an input (eitprocessing runs `BreathDetection` with the default
      settings on the timing data).
    - The TIV per breath comes from `pixel_tidal_impedance_variation`
      (eitprocessing: `TIV.compute_pixel_parameter` with
      `tiv_timing="continuous"`); the numbers are the same.
    - The capture "TIV per breath" is an `IntervalData` with one `PixelMap`
      per breath; "mean TIV" is a `PixelMap` of kind "tiv".
    - The mask has a `name`, and `metadata` with the settings and the count
      and fraction of included pixels.
    - Error messages are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from m3resp.data import BreathEvent, IntervalData, PixelMap, PixelMask, Signal
from m3resp.eit.processing import pixel_tidal_impedance_variation
from m3resp.eit.roi._shared import (
    check_per_breath,
    check_threshold,
    finish_mask,
    pixel_maps_of,
)

NAME = "tiv_lungspace_mask"
METHOD = "m3resp.eit.roi.tiv_lungspace"


def tiv_lungspace(
    pixels: Signal,
    breaths: Sequence[BreathEvent],
    *,
    threshold: float = 0.15,
    name: str = NAME,
    captures: dict[str, Any] | None = None,
) -> PixelMask:
    """Return the pixels whose mean TIV is at least a fraction of the largest.

    Args:
        pixels (Signal): EIT images, values with shape (time, row, column),
            with time in seconds.
        breaths (Sequence[BreathEvent]): The global breaths, with times on
            the time axis of ``pixels``, for example the breaths of the
            global impedance.
        threshold (float): Fraction of the largest mean TIV that a pixel
            must reach, between 0 and 1.
        name (str): Name of the mask.
        captures (dict | None): When a dictionary, it receives
            ``"TIV per breath"`` (``IntervalData`` with one ``PixelMap`` per
            breath) and ``"mean TIV"`` (``PixelMap``).

    Returns:
        PixelMask: 1 for the pixels of the lung space, NaN elsewhere.
            ``metadata`` holds ``threshold``, ``breath_count``,
            ``included_pixel_count`` and ``included_pixel_fraction``.

    Raises:
        TypeError: If ``threshold`` is not a float.
        ValueError: If ``threshold`` is not between 0 and 1, there are no
            breaths, or every TIV value is NaN; also the errors of
            :func:`~m3resp.eit.processing.pixel_tidal_impedance_variation`.
    """

    check_threshold(threshold, name="threshold", function="tiv_lungspace")
    tiv_per_breath = pixel_tidal_impedance_variation(pixels, breaths)
    _, mask = mean_map_and_mask(
        tiv_per_breath,
        quantity="TIV",
        threshold=threshold,
        captures=captures,
    )
    return finish_mask(
        mask,
        name=name,
        method=METHOD,
        settings={
            "threshold": threshold,
            "breath_count": len(tiv_per_breath.intervals),
        },
    )


def mean_map_and_mask(
    per_breath: IntervalData,
    *,
    quantity: str,
    threshold: float,
    captures: dict[str, Any] | None,
) -> tuple[PixelMap, PixelMask]:
    """Return the mean map over the breaths and its threshold mask.

    Args:
        per_breath (IntervalData): One ``PixelMap`` of TIV or amplitude
            values per breath.
        quantity (str): ``"TIV"`` (map of kind ``"tiv"``, no warning for
            negative values) or ``"amplitude"`` (kind ``"amplitude"``, with
            a warning for negative values).
        threshold (float): Fraction of the largest mean value.
        captures (dict | None): Receives ``"<quantity> per breath"`` and
            ``"mean <quantity>"``.

    Returns:
        tuple[PixelMap, PixelMask]: The mean map and the mask of the pixels
            whose mean is at least ``threshold`` times the largest mean.

    Raises:
        ValueError: If there are no breaths, or every value is NaN.
    """

    grids = check_per_breath(per_breath, quantity=quantity)
    if captures is not None:
        captures[f"{quantity} per breath"] = per_breath
    first = pixel_maps_of(per_breath)[0]
    mean = PixelMap.from_aggregate(
        grids,
        np.nanmean,
        name=f"mean {quantity}",
        modality=first.modality,
        category=first.category,
        unit=first.unit,
        kind="tiv" if quantity == "TIV" else "amplitude",
        suppress_negative_warning=quantity == "TIV",
    )
    if captures is not None:
        captures[f"mean {quantity}"] = mean
    return mean, mean.create_mask_from_threshold(threshold, fraction_of_max=True)
