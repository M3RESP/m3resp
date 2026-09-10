"""Shared ventilator-waveform primitives.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from ReSurfEMG.

    Source:     https://github.com/resurfemg-org/ReSurfEMG
    Revision:   m3resp-integration (c63668689030e4581d5f985e7d09d3a8c01e7a77)
    Original:   resurfemg/data_connector/data_classes.py::
                VentilatorDataGroup.find_peep
    Copyright:  Copyright (c) 2022 Netherlands eScience Center and
                University of Twente
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - Extracted from the `VentilatorDataGroup` class into a free function
      taking the pressure and volume arrays directly.
    - Renamed to `estimate_peep` to fit M3RESP naming conventions.
    - Missing or unusable end-expiratory samples raise
      `MissingModalityDataError` instead of producing a silent `nan`.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from m3resp.core.exceptions import MissingModalityDataError
from m3resp.processing.peaks import detect_peaks


def estimate_peep(
    pressure: np.ndarray,
    volume: np.ndarray,
    *,
    round_to_integer: bool = True,
) -> float:
    """Estimate PEEP as the end-expiratory airway pressure (Warnaar et al. 2024).

    End-expiration is located as the local minima of the ventilator volume
    signal; PEEP is the median airway pressure at those samples, rounded to
    the nearest integer because ventilators are set in whole cmH2O.

    This is deliberately *not* the median of the whole pressure trace: that
    median includes inspiration and therefore sits above PEEP, which biases
    every threshold derived from it (see `detect_occluded_breath_peaks`).

    Matches ReSurfEMG's `VentilatorDataGroup.find_peep`. Note that Warnaar et
    al. describe the rounding as downward while the reference implementation
    rounds to nearest; rounding down can place PEEP below the lowest pressure
    in the record when the measured end-expiratory pressure sits just under
    the set value, which collapses the occlusion-detection thresholds. Nearest
    is used here, matching the implementation.

    Raises `MissingModalityDataError` when the volume signal holds no usable
    end-expiratory minima, rather than silently falling back to a whole-trace
    statistic. Pass an explicit PEEP in that case.
    """

    pressure = np.asarray(pressure, dtype=float)
    volume = np.asarray(volume, dtype=float)
    if pressure.shape != volume.shape:
        raise ValueError(
            "PEEP estimation needs airway pressure and volume sampled on the "
            f"same time base; got {pressure.shape} and {volume.shape}."
        )

    end_expiration = detect_peaks(volume, invert=True)
    if len(end_expiration) == 0:
        raise MissingModalityDataError(
            "Cannot estimate PEEP: no end-expiratory minima found in the "
            "ventilator volume signal. Pass an explicit `peep` value."
        )

    end_expiratory_pressure = pressure[end_expiration]
    if np.all(np.isnan(end_expiratory_pressure)):
        raise MissingModalityDataError(
            "Cannot estimate PEEP: airway pressure is missing at every "
            "end-expiratory sample. Pass an explicit `peep` value."
        )

    peep = float(np.nanmedian(end_expiratory_pressure))
    return float(np.round(peep)) if round_to_integer else peep
