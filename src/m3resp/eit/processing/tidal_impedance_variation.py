"""Tidal impedance variation (TIV): the change in impedance during one breath.

``tidal_impedance_variation`` gives one TIV value per breath of an impedance
waveform, for example the global impedance. The breaths are given as
``BreathEvent``s, so they can come from ``find_breaths``, from the EIT
device, or from another modality. Three methods are available:

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
                TIV.compute_continuous_parameter
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The class `TIV` and its method `compute_continuous_parameter` are one
      function, `tidal_impedance_variation`, with the waveform as a `Signal`
      and the breaths as a list of `BreathEvent`s. The function finds no
      breaths itself (eitprocessing runs its own `BreathDetection`).
    - The result is an m3resp `IntervalData` with the given breaths and the
      unit of the signal (eitprocessing: a `SparseData` with no unit).
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
from m3resp.data.signals import Signal
from m3resp.eit.processing._breath_samples import (
    NO_SAMPLE,
    breath_samples,
    breath_source_metadata,
    check_breaths,
    check_impedance_waveform,
)

#: The TIV methods.
TIV_METHODS: Final = ("inspiratory", "expiratory", "mean")
#: Name of the result, as in eitprocessing.
NAME: Final = "continuous_tivs"
METHOD: Final = "m3resp.eit.processing.tidal_impedance_variation"


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
    if method not in TIV_METHODS:
        raise ValueError(
            f"Unknown TIV method {method!r}; use one of {', '.join(TIV_METHODS)}."
        )
    data = check_impedance_waveform(signal, function=function)
    checked = check_breaths(breaths, function=function)
    samples = breath_samples(
        signal,
        checked,
        points=("start_time", "extremum_time", "end_time"),
        function=function,
    )

    values = np.full(len(checked), np.nan)
    found = samples["start_time"] != NO_SAMPLE
    start = data[samples["start_time"][found]]
    end_of_inspiration = data[samples["extremum_time"][found]]
    end = data[samples["end_time"][found]]
    if method == "inspiratory":
        values[found] = end_of_inspiration - start
    elif method == "expiratory":
        values[found] = end_of_inspiration - end
    else:
        values[found] = end_of_inspiration - (start + end) / 2

    return IntervalData(
        name=name,
        modality=signal.modality,
        intervals=list[Interval](checked),
        values=values,
        category="impedance",
        unit=signal.unit,
        method=METHOD,
        metadata={
            "tiv_method": method,
            "signal_name": signal.name,
            **breath_source_metadata(checked),
        },
    )
