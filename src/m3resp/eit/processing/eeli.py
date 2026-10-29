"""End-expiratory lung impedance (EELI): the impedance at the end of each breath.

``end_expiratory_lung_impedance`` gives one EELI value per breath of an
impedance waveform, for example the global impedance: the impedance at the
end of the breath (``end_time``). The breaths are given as ``BreathEvent``s,
so they can come from ``find_breaths``, from the EIT device, or from another
modality.

``pixel_end_expiratory_lung_impedance`` gives one map of EELI values per
breath, one value for each pixel of EIT images: the impedance of the pixel at
the end of the global breath, or at the end of its own breath
(``PixelBreaths``). It is m3resp code, with no eitprocessing version.

The value is an EELI only when the breath end marks the end of expiration in
the impedance. This is true for EIT breaths. Breaths of another modality can
mark other moments.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/parameters/eeli.py::EELI.compute_parameter
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The class `EELI` and its method `compute_parameter` are one function,
      `end_expiratory_lung_impedance`, with the waveform as a `Signal` and
      the breaths as a list of `BreathEvent`s. The function finds no breaths
      itself (eitprocessing runs its own `BreathDetection`).
    - The result is an m3resp `IntervalData` with the given breaths and the
      unit of the signal (eitprocessing: a `SparseData` with no unit and
      empty settings).
    - Each breath end is read at the nearest sample. eitprocessing reads
      only a sample at exactly the breath end time, and gives fewer values
      than breaths, with no message, when a breath end lies between two
      samples. Both give the same values for breaths found on the same
      waveform.
    - New: a breath partly outside the data or in a gap of the time axis
      gets NaN, with a warning.
    - The category check reads the m3resp `Signal.category` (eitprocessing:
      its category tree). The impedance categories of eitprocessing pass.
    - Docstrings and error texts are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

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

#: Name of the result, as in eitprocessing.
NAME: Final = "continuous_eelis"
METHOD: Final = "m3resp.eit.processing.end_expiratory_lung_impedance"
#: Name of the pixel result.
PIXEL_NAME: Final = "pixel_eelis"
PIXEL_METHOD: Final = "m3resp.eit.processing.pixel_end_expiratory_lung_impedance"


def end_expiratory_lung_impedance(
    signal: Signal,
    breaths: Sequence[BreathEvent],
    *,
    name: str = NAME,
) -> IntervalData:
    """Return the end-expiratory lung impedance (EELI) of each breath.

    Args:
        signal (Signal): Impedance waveform, one value per sample, with time
            in seconds. Its ``category`` must be None or an impedance
            category.
        breaths (Sequence[BreathEvent]): The breaths, with times on the time
            axis of ``signal``. The breath end is read at the nearest sample.
        name (str): Name of the result.

    Returns:
        IntervalData: The given breaths, in the same order, with one EELI
            value each (floats), in the unit of ``signal``. The value is NaN
            for a breath partly outside the data or in a gap of the time
            axis, and where the waveform is NaN. ``metadata`` holds
            ``signal_name``, ``breath_modality`` and ``breath_source``.

    Raises:
        TypeError: If ``signal`` is not a ``Signal`` or a breath is not a
            ``BreathEvent``.
        ValueError: If the category is not an impedance category, or the
            values do not have one axis.

    Warns:
        UserWarning: When breaths lie partly outside the data or in a gap
            of the time axis.
    """

    function = "end_expiratory_lung_impedance"
    data = check_impedance_waveform(signal, function=function)
    checked = check_breaths(breaths, function=function)
    end = breath_samples(
        signal, checked, points=("start_time", "end_time"), function=function
    )["end_time"]

    return IntervalData(
        name=name,
        modality=signal.modality,
        intervals=list[Interval](checked),
        values=values_at(data, end),
        category="impedance",
        unit=signal.unit,
        method=METHOD,
        metadata={"signal_name": signal.name, **breath_source_metadata(checked)},
    )


def pixel_end_expiratory_lung_impedance(
    pixels: Signal,
    breaths: Sequence[BreathEvent] | PixelBreaths,
    *,
    name: str = PIXEL_NAME,
) -> IntervalData:
    """Return the EELI of each pixel, as one map per breath.

    Args:
        pixels (Signal): EIT images, values with shape (time, row, column),
            with time in seconds. Its ``category`` must be None or an
            impedance category.
        breaths (Sequence[BreathEvent] | PixelBreaths): The breaths, with
            times on the time axis of ``pixels``. A list of ``BreathEvent``
            objects gives the same breath end to every pixel (for example
            the breaths of the global impedance). A ``PixelBreaths`` gives
            each pixel the end of its own breath. The breath end is read at
            the nearest sample.
        name (str): Name of the result and of each ``PixelMap``.

    Returns:
        IntervalData: The global breaths (the given ``BreathEvent``
            objects, or ``PixelBreaths.breaths``), in the same order, with
            one ``PixelMap`` of EELI values each, in the unit of ``pixels``. A
            pixel is NaN where its breath is partly outside the data or in a
            gap of the time axis, where its image value is NaN, and, for
            ``PixelBreaths``, where the pixel has no breath (always the
            first and the last breath). ``metadata`` holds ``eeli_timing``
            (``"continuous"`` for a list of breaths, ``"pixel"`` for
            ``PixelBreaths``), ``signal_name``, ``breath_modality`` and
            ``breath_source``.

    Raises:
        TypeError: If ``pixels`` is not a ``Signal`` or a breath is not a
            ``BreathEvent``.
        ValueError: If the category is not an impedance category, the
            values do not have the shape (time, row, column), or the grid of
            ``PixelBreaths`` differs from the grid of ``pixels``.

    Warns:
        UserWarning: When breaths lie partly outside the data or in a gap
            of the time axis.
    """

    function = "pixel_end_expiratory_lung_impedance"
    data = check_pixel_impedance(pixels, function=function)
    checked, global_breaths, timing = check_pixel_breaths(
        breaths, data, function=function
    )
    end = breath_samples(
        pixels, checked, points=("start_time", "end_time"), function=function
    )["end_time"]

    return pixel_maps_per_breath(
        values_at(data, end),
        pixels,
        global_breaths,
        name=name,
        method=PIXEL_METHOD,
        metadata={
            "eeli_timing": timing,
            "signal_name": pixels.name,
            **breath_source_metadata(checked),
        },
    )
