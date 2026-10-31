"""Tidal impedance variation (TIV): the change in impedance during one breath.

``tidal_impedance_variation`` gives one TIV value per breath of an impedance
waveform, for example the global impedance. The breaths are given as
``BreathEvent``s, so they can come from ``find_breaths``, from the EIT
device, or from another modality. ``pixel_tidal_impedance_variation``
gives one map of TIV values per breath, one value for each pixel of EIT
images. Each pixel uses the times of the global breaths, or the times of its
own breaths (``PixelBreaths``). Three methods are available:

- ``"inspiratory"`` (default): impedance at the end of inspiration
  (``extremum_time``) minus impedance at the start of the breath.
- ``"expiratory"``: impedance at the end of inspiration minus impedance at
  the end of the breath.
- ``"mean"``: impedance at the end of inspiration minus the mean of the
  impedance at the start and at the end of the breath.

The value is a TIV only when the breath times mark the start of inspiration,
the end of inspiration and the end of expiration in the impedance. This is
true for EIT breaths. Breaths of another modality can mark other moments.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/parameters/tidal_impedance_variation.py::
                TIV.compute_continuous_parameter;
                TIV.compute_pixel_parameter; TIV._calculate_tiv_values
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The class `TIV` and its method `compute_continuous_parameter` are one
      function, `tidal_impedance_variation`, with the waveform as a `Signal`
      and the breaths as a list of `BreathEvent`s. The function finds no
      breaths itself (eitprocessing runs its own `BreathDetection`).
    - `compute_pixel_parameter` is the function
      `pixel_tidal_impedance_variation`, with the EIT images as a `Signal`.
      The breaths are a list of `BreathEvent`s (eitprocessing
      `tiv_timing="continuous"`) or a `PixelBreaths` (eitprocessing
      `tiv_timing="pixel"`, which runs its own `PixelBreath`). The function
      finds no breaths itself.
    - The result is an m3resp `IntervalData` with the given breaths and the
      unit of the signal (eitprocessing: a `SparseData` with no unit). For
      pixels, the result has one `PixelMap` per breath and the global
      breaths; the times of each pixel stay in the `PixelBreaths`
      (eitprocessing: a list of middle times per pixel, packed to the front
      when a pixel has no breath, so out of step with the values).
    - The TIV of all pixels is computed at once on arrays (eitprocessing: a
      loop over the pixels).
    - Each breath time is read at the nearest sample (eitprocessing: the
      first sample at or after the time). Both give the same sample for
      breaths found on the same waveform.
    - New: a breath partly outside the data or in a gap of the time axis,
      or with no extremum time, gets NaN, with a warning.
    - New: the data category must be an impedance category.
    - Docstrings and error texts are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np

from m3resp.data.event_data import IntervalData
from m3resp.data.events import BreathEvent, Interval
from m3resp.data.pixel_breaths import PixelBreaths
from m3resp.data.signals import Signal
from m3resp.eit.processing._breath_samples import (
    breath_samples,
    breath_source_metadata,
    check_breaths,
    check_impedance_waveform,
    check_pixel_breaths,
    check_pixel_impedance,
    pixel_maps_per_breath,
    values_at,
)

#: The TIV methods.
TIV_METHODS: Final = ("inspiratory", "expiratory", "mean")
#: Name of the result, as in eitprocessing.
NAME: Final = "continuous_tivs"
METHOD: Final = "m3resp.eit.processing.tidal_impedance_variation"
#: Name of the pixel result, as in eitprocessing.
PIXEL_NAME: Final = "pixel_tivs"
PIXEL_METHOD: Final = "m3resp.eit.processing.pixel_tidal_impedance_variation"


def tidal_impedance_variation(
    signal: Signal,
    breaths: Sequence[BreathEvent],
    *,
    method: str = "inspiratory",
    name: str = NAME,
) -> IntervalData:
    """Return the tidal impedance variation (TIV) of each breath.

    Args:
        signal (Signal): Impedance waveform, one value per sample, with time
            in seconds. Its ``category`` must be None or an impedance
            category.
        breaths (Sequence[BreathEvent]): The breaths, with times on the time
            axis of ``signal``. Each time is read at the nearest sample.
        method (str): ``"inspiratory"``, ``"expiratory"`` or ``"mean"`` (see
            the module text).
        name (str): Name of the result.

    Returns:
        IntervalData: The given breaths, in the same order, with one TIV
            value each (floats), in the unit of ``signal``. The value is NaN
            for a breath partly outside the data or in a gap of the time
            axis, for a breath with no ``extremum_time``, and where the
            waveform is NaN. ``metadata`` holds ``tiv_method``,
            ``signal_name``, ``breath_modality`` and ``breath_source``.

    Raises:
        TypeError: If ``signal`` is not a ``Signal`` or a breath is not a
            ``BreathEvent``.
        ValueError: If ``method`` is not one of the three methods, the
            category is not an impedance category, or the values do not have
            one axis.

    Warns:
        UserWarning: When breaths lie partly outside the data or in a gap
            of the time axis, or have no ``extremum_time``.
    """

    function = "tidal_impedance_variation"
    _check_method(method)
    data = check_impedance_waveform(signal, function=function)
    checked = check_breaths(breaths, function=function)
    samples = breath_samples(
        signal,
        checked,
        points=("start_time", "extremum_time", "end_time"),
        function=function,
    )

    return IntervalData(
        name=name,
        modality=signal.modality,
        intervals=list[Interval](checked),
        values=_tiv_values(data, samples, method),
        category="impedance",
        unit=signal.unit,
        method=METHOD,
        metadata={
            "tiv_method": method,
            "signal_name": signal.name,
            **breath_source_metadata(checked),
        },
    )


def pixel_tidal_impedance_variation(
    pixels: Signal,
    breaths: Sequence[BreathEvent] | PixelBreaths,
    *,
    method: str = "inspiratory",
    name: str = PIXEL_NAME,
) -> IntervalData:
    """Return the TIV of each pixel, as one map per breath.

    Args:
        pixels (Signal): EIT images, values with shape (time, row, column),
            with time in seconds. Its ``category`` must be None or an
            impedance category.
        breaths (Sequence[BreathEvent] | PixelBreaths): The breaths, with
            times on the time axis of ``pixels``. A list of ``BreathEvent``
            objects gives the same times to every pixel (for example the
            breaths of the global impedance). A ``PixelBreaths`` gives each
            pixel the times of its own breaths. Each time is read at the
            nearest sample.
        method (str): ``"inspiratory"``, ``"expiratory"`` or ``"mean"`` (see
            the module text).
        name (str): Name of the result and of each ``PixelMap``.

    Returns:
        IntervalData: The global breaths (the given ``BreathEvent``
            objects, or ``PixelBreaths.breaths``), in the same order, with
            one ``PixelMap`` of TIV values each, in the unit of ``pixels``. A
            pixel is NaN where its breath is partly outside the data or in a
            gap of the time axis, where its image value is NaN, and, for
            ``PixelBreaths``, where the pixel has no breath (always the
            first and the last breath). ``metadata`` holds ``tiv_method``,
            ``tiv_timing`` (``"continuous"`` for a list of breaths,
            ``"pixel"`` for ``PixelBreaths``), ``signal_name``,
            ``breath_modality`` and ``breath_source``.

    Raises:
        TypeError: If ``pixels`` is not a ``Signal`` or a breath is not a
            ``BreathEvent``.
        ValueError: If ``method`` is not one of the three methods, the
            category is not an impedance category, the values do not have
            the shape (time, row, column), or the grid of ``PixelBreaths``
            differs from the grid of ``pixels``.

    Warns:
        UserWarning: When breaths lie partly outside the data or in a gap
            of the time axis, or have no ``extremum_time``.
    """

    function = "pixel_tidal_impedance_variation"
    _check_method(method)
    data = check_pixel_impedance(pixels, function=function)
    checked, global_breaths, timing = check_pixel_breaths(
        breaths, data, function=function
    )
    samples = breath_samples(
        pixels,
        checked,
        points=("start_time", "extremum_time", "end_time"),
        function=function,
    )

    return pixel_maps_per_breath(
        _tiv_values(data, samples, method),
        pixels,
        global_breaths,
        name=name,
        method=PIXEL_METHOD,
        metadata={
            "tiv_method": method,
            "tiv_timing": timing,
            "signal_name": pixels.name,
            **breath_source_metadata(checked),
        },
    )


def _check_method(method: str) -> None:
    """Raise ``ValueError`` if ``method`` is not one of ``TIV_METHODS``.

    Args:
        method (str): The TIV method to check.

    Raises:
        ValueError: If ``method`` is not one of ``TIV_METHODS``.
    """

    if method not in TIV_METHODS:
        raise ValueError(
            f"Unknown TIV method {method!r}; use one of {', '.join(TIV_METHODS)}."
        )


def _tiv_values(
    data: np.ndarray, samples: dict[str, np.ndarray], method: str
) -> np.ndarray:
    """Return the TIV at the samples of each breath, NaN where there is no sample.

    Args:
        data (np.ndarray): The impedance, shape (time,) for a waveform or
            (time, row, column) for EIT images.
        samples (dict[str, np.ndarray]): The sample indices of
            ``"start_time"``, ``"extremum_time"`` and ``"end_time"``, from
            ``breath_samples``.
        method (str): ``"inspiratory"``, ``"expiratory"`` or ``"mean"``.

    Returns:
        np.ndarray: One TIV per breath, shape (breath,) for a waveform or
            (breath, row, column) for EIT images, in the unit of ``data``.
    """

    start = values_at(data, samples["start_time"])
    end_of_inspiration = values_at(data, samples["extremum_time"])
    end = values_at(data, samples["end_time"])
    if method == "inspiratory":
        return end_of_inspiration - start
    if method == "expiratory":
        return end_of_inspiration - end
    return end_of_inspiration - (start + end) / 2
