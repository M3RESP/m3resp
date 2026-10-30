"""Functional lung space with the watershed method.

A lung space from the TIV alone can miss pixels whose TIV is low because of
pendelluft (air that moves between lung regions during the breath). A lung
space from the amplitude alone can include too many pixels.
``watershed_lungspace`` uses both:

1. It computes the TIV of each pixel per global breath (as
   :func:`~m3resp.eit.roi.tiv_lungspace`) and the amplitude of each pixel per
   global breath (as :func:`~m3resp.eit.roi.amplitude_lungspace`). It keeps
   the breaths that have values in both; in practice, the first and the last
   breath drop out.
2. Over these breaths it computes the mean TIV and the mean amplitude of each
   pixel. The TIV mask holds the pixels with a mean TIV of at least
   ``threshold_fraction`` times the largest mean TIV. The amplitude mask holds
   the pixels with a mean amplitude of at least that same TIV value.
3. The local maxima of the mean amplitude are the centres of the watershed
   regions. Each pixel belongs to the region of the maximum it "flows up"
   to. Neighbouring maxima get one number. Pixels with a NaN mean amplitude
   belong to no region (number 0) and can not be a maximum.
4. The lung space is the union of the regions whose maximum lies inside the
   TIV mask, multiplied by the amplitude mask.

The TIV mask of step 2 uses the breaths of step 1 only, so it can differ from
the mask of :func:`~m3resp.eit.roi.tiv_lungspace`, which uses all breaths.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/roi/watershed.py::WatershedLungspace
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The class `WatershedLungspace` and its method `apply` are one function,
      `watershed_lungspace`, with the EIT images and the waveform as
      `Signal`s. The global breaths are an input (eitprocessing runs
      `BreathDetection` with the default settings on the timing data).
    - The pixel TIV and the pixel amplitude per breath are computed once
      (eitprocessing runs `TIVLungspace.apply` and
      `AmplitudeLungspace.apply` in full, and then uses only their
      captures). The masks and the captures are the same.
    - The "no breaths" ValueError comes only from breaths without TIV or
      amplitude values; other errors of the pixel TIV and the pixel
      amplitude reach the user unchanged.
    - The warning for a negative mean amplitude checks the mean over the
      shared breaths (eitprocessing checks the mean over all breaths).
    - The captures "mean tiv" and "mean amplitude" are named
      "mean TIV of shared breaths" and "mean amplitude of shared breaths".
    - The captures have m3resp types: `PixelMap` for the mean maps,
      `PixelMask` for the masks and "included region", a `PixelMap` of kind
      "integer" for "watershed regions", a plain `PixelMap` for
      "included watershed regions".
    - "included watershed regions" holds the included regions only, with
      NaN elsewhere (eitprocessing gives the `PixelMask` object to
      `np.where`, which reads it as one true value, so it keeps every
      region).
    - Pixels with a NaN mean amplitude are left out of the search for local
      maxima and of the watershed (the `labels` and `mask` arguments of
      scikit-image); they get region number 0. eitprocessing searches the
      maxima with NaN read as 0 and gives the NaN pixels to the watershed,
      so the region borders near those pixels depend on how scikit-image
      treats NaN.
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
from scipy import ndimage
from skimage import feature, segmentation

from m3resp.data import BreathEvent, PixelMap, PixelMask, Signal
from m3resp.eit.processing import pixel_tidal_impedance_variation
from m3resp.eit.roi._shared import check_per_breath, check_threshold, finish_mask
from m3resp.eit.roi.amplitude import pixel_amplitude_per_breath

NAME = "watershed_lungspace_mask"
METHOD = "m3resp.eit.roi.watershed_lungspace"


def watershed_lungspace(
    pixels: Signal,
    signal: Signal,
    breaths: Sequence[BreathEvent],
    *,
    threshold_fraction: float = 0.15,
    name: str = NAME,
    captures: dict[str, Any] | None = None,
) -> PixelMask:
    """Return the functional lung space found with the watershed method.

    Args:
        pixels (Signal): EIT images, values with shape (time, row, column),
            with time in seconds.
        signal (Signal): The waveform of the breaths, for example the global
            impedance; the pixel breath search uses it for the phase lag of
            each pixel.
        breaths (Sequence[BreathEvent]): The global breaths of ``signal``.
        threshold_fraction (float): Fraction of the largest mean TIV for the
            TIV mask, between 0 and 1. The same TIV value is the threshold of
            the amplitude mask.
        name (str): Name of the mask.
        captures (dict | None): When a dictionary, it receives
            ``"mean TIV of shared breaths"`` and
            ``"mean amplitude of shared breaths"`` (``PixelMap``),
            ``"functional tiv mask"``, ``"functional amplitude mask"`` and
            ``"included region"`` (``PixelMask``), ``"local peaks"`` (row and
            column of each local maximum), ``"included marker indices"``
            (numbers of the maxima inside the TIV mask),
            ``"watershed regions"`` (``PixelMap`` of region numbers, 0 for
            pixels with a NaN mean amplitude),
            ``"included peaks"`` and ``"excluded peaks"`` (row and column of
            the maxima inside and outside the TIV mask) and
            ``"included watershed regions"`` (the region numbers of the
            included regions, NaN elsewhere).

    Returns:
        PixelMask: 1 for the pixels of the lung space, NaN elsewhere.
            ``metadata`` holds ``threshold_fraction``, ``breath_count`` (the
            breaths with TIV and amplitude), ``included_pixel_count`` and
            ``included_pixel_fraction``.

    Raises:
        TypeError: If ``threshold_fraction`` is not a float.
        ValueError: If ``threshold_fraction`` is not between 0 and 1, no
            breath has TIV or amplitude values, no breath has both, or no
            local maximum lies inside the TIV mask. The errors of the pixel
            TIV and the pixel amplitude (for example images that are not
            (time, row, column)) are raised as they are.

    Warns:
        UserWarning: If the mean amplitude of a pixel over the shared
            breaths is negative.
        RuntimeWarning: The warnings of the pixel breath search.
    """

    check_threshold(
        threshold_fraction, name="threshold_fraction", function="watershed_lungspace"
    )
    tiv = pixel_tidal_impedance_variation(pixels, breaths)
    amplitude = pixel_amplitude_per_breath(pixels, signal, breaths)
    try:
        tiv_grids = check_per_breath(tiv, quantity="TIV")
        amplitude_grids = check_per_breath(amplitude, quantity="amplitude")
    except ValueError as error:
        raise ValueError(
            "No breaths with TIV or amplitude values were found, so no "
            "functional lung space can be found."
        ) from error

    shared = ~np.all(np.isnan(tiv_grids), axis=(1, 2)) & ~np.all(
        np.isnan(amplitude_grids), axis=(1, 2)
    )
    if not np.any(shared):
        raise ValueError(
            "No breath has both TIV and amplitude values, so no functional "
            "lung space can be found."
        )

    fields = {"modality": pixels.modality, "category": "impedance", "unit": pixels.unit}
    mean_tiv = PixelMap.from_aggregate(
        tiv_grids[shared],
        np.nanmean,
        name="mean TIV of shared breaths",
        kind="tiv",
        suppress_negative_warning=True,
        **fields,
    )
    tiv_threshold: dict[str, Any] = {}
    tiv_mask = mean_tiv.create_mask_from_threshold(
        threshold_fraction,
        fraction_of_max=True,
        name="functional tiv mask",
        captures=tiv_threshold,
    )
    _capture(captures, "mean TIV of shared breaths", mean_tiv)
    _capture(captures, "functional tiv mask", tiv_mask)

    mean_amplitude = PixelMap.from_aggregate(
        amplitude_grids[shared],
        np.nanmean,
        name="mean amplitude of shared breaths",
        kind="amplitude",
        **fields,
    )
    amplitude_mask = mean_amplitude.create_mask_from_threshold(
        tiv_threshold["actual threshold"],
        fraction_of_max=False,
        name="functional amplitude mask",
    )
    _capture(captures, "mean amplitude of shared breaths", mean_amplitude)
    _capture(captures, "functional amplitude mask", amplitude_mask)

    # Local maxima of the mean amplitude, border included, among the pixels
    # with a mean amplitude. Neighbouring maxima get one number; these
    # numbers mark the regions.
    has_amplitude = ~np.isnan(mean_amplitude.values)
    image = mean_amplitude.to_non_nan_array()
    local_peaks = feature.peak_local_max(
        image, exclude_border=False, labels=has_amplitude.astype(int)
    )
    is_peak = np.zeros(mean_amplitude.shape, dtype=bool)
    is_peak[tuple(local_peaks.T)] = True
    peak_numbers, _ = ndimage.label(is_peak)
    markers = PixelMap(
        name="markers", values=np.where(peak_numbers == 0, np.nan, peak_numbers)
    )
    _capture(captures, "local peaks", local_peaks)

    markers_in_tiv_mask = tiv_mask.apply(markers).values
    included_markers = markers_in_tiv_mask[~np.isnan(markers_in_tiv_mask)]
    if not len(included_markers):
        raise ValueError(
            "No local maximum of the mean amplitude lies inside the TIV mask, "
            "so no functional lung space can be found."
        )
    _capture(captures, "included marker indices", included_markers)

    regions = segmentation.watershed(
        image=-image, markers=markers.to_integer_array(), mask=has_amplitude
    )
    is_included = np.isin(regions, included_markers)
    included_region = PixelMask(name="included region", values=is_included)
    _capture(captures, "included region", included_region)

    if captures is not None:
        included_peak_numbers = np.isin(peak_numbers, included_markers)
        _capture(
            captures,
            "watershed regions",
            PixelMap(name="watershed regions", values=regions, kind="integer"),
        )
        _capture(captures, "included peaks", np.argwhere(included_peak_numbers))
        _capture(
            captures,
            "excluded peaks",
            np.argwhere(is_peak & ~included_peak_numbers),
        )
        _capture(
            captures,
            "included watershed regions",
            PixelMap(
                name="included watershed regions",
                values=np.where(is_included, regions, np.nan),
            ),
        )

    return finish_mask(
        included_region * amplitude_mask,
        name=name,
        method=METHOD,
        settings={
            "threshold_fraction": threshold_fraction,
            "breath_count": int(shared.sum()),
        },
    )


def _capture(captures: dict[str, Any] | None, key: str, value: Any) -> None:
    if captures is not None:
        captures[key] = value
