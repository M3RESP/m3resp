"""Functional lung space from the mean amplitude of each pixel.

``amplitude_lungspace`` finds the breaths of each pixel with the "phase
shift" mode of :func:`~m3resp.eit.processing.find_pixel_breaths`, so each
pixel has its own breath times and a positive amplitude. The amplitude of a
pixel in a breath is its TIV between its own breath start and end of
inspiration. The first and the last global breath have no pixel breaths, so
their values are NaN. The mean over the breaths gives one amplitude per
pixel. A pixel is in the lung space when its mean amplitude is at least
``threshold`` times the largest mean amplitude (15 % by default).

Warning: this lung space can include reconstruction artefacts, so it is not
advised on its own. :func:`~m3resp.eit.roi.watershed_lungspace` uses the
amplitude together with the TIV.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/roi/amplitude.py::AmplitudeLungspace
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The class `AmplitudeLungspace` and its method `apply` are one function,
      `amplitude_lungspace`, with the EIT images and the waveform as
      `Signal`s. The global breaths are an input (eitprocessing runs
      `BreathDetection` with the default settings on the timing data).
    - The amplitude per breath comes from `find_pixel_breaths` with
      "phase shift" and `pixel_tidal_impedance_variation` (eitprocessing:
      `TIV.compute_pixel_parameter` with `tiv_timing="pixel"` and
      `PixelBreath(phase_correction_mode="phase shift")`); the numbers are
      the same.
    - The capture "amplitude per breath" is an `IntervalData` with one
      `PixelMap` per breath; "mean amplitude" is a `PixelMap` of kind
      "amplitude".
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

from m3resp.data import BreathEvent, IntervalData, PixelMask, Signal
from m3resp.eit.processing import (
    find_pixel_breaths,
    pixel_tidal_impedance_variation,
)
from m3resp.eit.roi._shared import check_threshold, finish_mask
from m3resp.eit.roi.tiv import mean_map_and_mask

NAME = "amplitude_lungspace_mask"
METHOD = "m3resp.eit.roi.amplitude_lungspace"


def amplitude_lungspace(
    pixels: Signal,
    signal: Signal,
    breaths: Sequence[BreathEvent],
    *,
    threshold: float = 0.15,
    name: str = NAME,
    captures: dict[str, Any] | None = None,
) -> PixelMask:
    """Return the pixels whose mean amplitude is at least a fraction of the largest.

    Args:
        pixels (Signal): EIT images, values with shape (time, row, column),
            with time in seconds.
        signal (Signal): The waveform of the breaths, for example the global
            impedance; the pixel breath search uses it for the phase lag of
            each pixel.
        breaths (Sequence[BreathEvent]): The global breaths of ``signal``.
        threshold (float): Fraction of the largest mean amplitude that a
            pixel must reach, between 0 and 1.
        name (str): Name of the mask.
        captures (dict | None): When a dictionary, it receives
            ``"amplitude per breath"`` (``IntervalData`` with one
            ``PixelMap`` per global breath) and ``"mean amplitude"``
            (``PixelMap``).

    Returns:
        PixelMask: 1 for the pixels of the lung space, NaN elsewhere.
            ``metadata`` holds ``threshold``, ``breath_count``,
            ``included_pixel_count`` and ``included_pixel_fraction``.

    Raises:
        TypeError: If ``threshold`` is not a float.
        ValueError: If ``threshold`` is not between 0 and 1, there are no
            breaths, or every amplitude value is NaN; also the errors of
            :func:`~m3resp.eit.processing.find_pixel_breaths`.

    Warns:
        UserWarning: If a mean amplitude is negative.
        RuntimeWarning: The warnings of the pixel breath search.
    """

    check_threshold(threshold, name="threshold", function="amplitude_lungspace")
    amplitude_per_breath = pixel_amplitude_per_breath(pixels, signal, breaths)
    _, mask = mean_map_and_mask(
        amplitude_per_breath,
        quantity="amplitude",
        threshold=threshold,
        captures=captures,
    )
    return finish_mask(
        mask,
        name=name,
        method=METHOD,
        settings={
            "threshold": threshold,
            "breath_count": len(amplitude_per_breath.intervals),
        },
    )


def pixel_amplitude_per_breath(
    pixels: Signal, signal: Signal, breaths: Sequence[BreathEvent]
) -> IntervalData:
    """Return the amplitude of each pixel in each global breath.

    Args:
        pixels (Signal): EIT images (time, row, column).
        signal (Signal): The waveform of the breaths.
        breaths (Sequence[BreathEvent]): The global breaths of ``signal``.

    Returns:
        IntervalData: One ``PixelMap`` per global breath: the TIV of each
            pixel between its own breath start and end of inspiration, found
            with the "phase shift" mode. NaN where a pixel has no breath.
    """

    pixel_breaths = find_pixel_breaths(
        pixels, signal, breaths, phase_correction_mode="phase shift"
    )
    return pixel_tidal_impedance_variation(
        pixels, pixel_breaths, name="pixel_amplitudes"
    )
