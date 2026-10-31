"""MDN filter: remove the heart signal from EIT data.

MDN stands for "multiple digital notch". The heart beat adds a small signal
at the heart rate and at its multiples (harmonics: 2, 3, ... times the heart
rate) to the EIT data. The filter removes it in two parts:

1. For each harmonic up to the noise frequency limit, a band-stop filter
   removes the band from (harmonic − notch distance) to (harmonic + notch
   distance). For the first harmonic, the lower edge of the band is at least
   halfway between the breathing rate and the heart rate, so the breathing
   signal stays.
2. A low-pass filter removes everything above the noise frequency limit.

Each filter is a Butterworth filter of the given order, applied forwards and
backwards (``scipy.signal.sosfiltfilt``), so the data keeps its timing.
The default settings are a noise frequency limit of 220 per minute
(3.67 Hz), a notch distance of 10 per minute (0.167 Hz) and order 10.

The filter was made for EIT data and tested in a limited number of cases. It
can work less well for other data, other patient groups or other forms of
ventilation.

All rates and frequencies are in Hz. Divide a rate per minute by 60 to get
Hz.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/filters/mdn.py::MDNFilter
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The class `MDNFilter` and its method `apply` are one function,
      `mdn_filter`, with the settings as keyword arguments and the data as a
      `Signal` (time along the first axis).
    - The captures `unfiltered_data` and `filtered_data` are the input and
      the output `Signal`.
    - New: a warning when a pixel has no value (NaN) at some samples. The
      filter gives NaN at every sample of such a pixel, as upstream.
    - New: `mdn_frequency_bands` gives the bands without filtering.
    - Docstrings and warning texts are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Any, Final

import numpy as np
from scipy import signal as scipy_signal

from m3resp.data.signals import Signal
from m3resp.processing.captures import capture_value

MINUTE: Final = 60

#: Default noise frequency limit in Hz (220 per minute).
NOISE_FREQUENCY_LIMIT: Final = 220 / MINUTE
#: Default notch distance in Hz (10 per minute).
NOTCH_DISTANCE: Final = 10 / MINUTE
#: Default Butterworth order of each filter.
ORDER: Final = 10

#: A breathing rate above this value (85 per minute) gives a warning.
UPPER_RESPIRATORY_RATE_LIMIT: Final = 85 / MINUTE
#: A heart rate above this value (210 per minute) gives a warning.
UPPER_HEART_RATE_LIMIT: Final = 210 / MINUTE


def mdn_filter(
    signal: Signal,
    *,
    respiratory_rate: float,
    heart_rate: float,
    noise_frequency_limit: float = NOISE_FREQUENCY_LIMIT,
    notch_distance: float = NOTCH_DISTANCE,
    order: int = ORDER,
    name: str | None = None,
    captures: dict[str, Any] | None = None,
) -> Signal:
    """Remove the heart signal from EIT images or an impedance waveform.

    Args:
        signal (Signal): EIT images (frames, rows, columns) or one waveform,
            with time along the first axis and the samples per second in
            ``sample_frequency``.
        respiratory_rate (float): Breathing rate in Hz, above 0 and below
            ``heart_rate``.
        heart_rate (float): Heart rate in Hz, above 0.
        noise_frequency_limit (float): Low-pass edge in Hz. The filter
            removes the harmonics of the heart rate up to this frequency, and
            everything above it.
        notch_distance (float): Half the width of each removed band, in Hz.
        order (int): Butterworth order of each band-stop and low-pass filter.
        name (str | None): The name of the result. None keeps the name of
            ``signal``.
        captures (dict | None): A dictionary that receives the in-between
            results: ``"low_pass_frequency"`` (Hz), ``"unfiltered_data"``
            (``signal``), ``"n_harmonics"`` (number of removed harmonics),
            ``"frequency_bands"`` (one (low, high) pair in Hz per harmonic;
            missing when no harmonic is removed) and ``"filtered_data"`` (the
            result). None stores nothing.

    Returns:
        Signal: The filtered data, with the shape, unit, frames per second
            and device maker (``source``) of ``signal`` and a copy of its
            time. Processing state ``"intermediate"``, method
            ``"m3resp.eit.processing.mdn_filter"``. A pixel with NaN at one or
            more samples is NaN at every sample.

    Raises:
        ValueError: If a rate is 0 or below, if the breathing rate is equal
            to or higher than the heart rate, if ``sample_frequency`` is
            missing or 0, or if a band edge is at or above half the sample
            frequency (from SciPy).

    Warns:
        UserWarning: When the breathing rate is above 85 per minute or the
            heart rate is above 210 per minute (a sign that the rate is per
            minute, while the filter needs Hz), and when a pixel has NaN at
            some samples only.
    """

    _check_rates(respiratory_rate, heart_rate)
    capture_value(captures, "low_pass_frequency", noise_frequency_limit)
    capture_value(captures, "unfiltered_data", signal)

    sample_frequency = signal.sample_frequency
    if not sample_frequency:
        raise ValueError(
            f"mdn_filter needs the samples per second of {signal.name!r} "
            "(sample_frequency)."
        )
    values = np.asarray(signal.values)
    _warn_partly_missing(values, signal.name)

    bands = _frequency_bands(
        respiratory_rate, heart_rate, noise_frequency_limit, notch_distance
    )
    capture_value(captures, "n_harmonics", len(bands))

    filtered = values
    for lower_limit, upper_limit in bands:
        sos = scipy_signal.butter(
            N=order,
            Wn=[lower_limit, upper_limit],
            fs=sample_frequency,
            btype="bandstop",
            output="sos",
        )
        capture_value(
            captures, "frequency_bands", (lower_limit, upper_limit), append_to_list=True
        )
        filtered = scipy_signal.sosfiltfilt(sos, filtered, axis=0)

    sos = scipy_signal.butter(
        N=order,
        Wn=noise_frequency_limit,
        fs=sample_frequency,
        btype="low",
        output="sos",
    )
    filtered = scipy_signal.sosfiltfilt(sos, filtered, axis=0)

    result = dataclasses.replace(
        signal,
        values=filtered,
        time=np.copy(signal.time),
        name=signal.name if name is None else name,
        metadata=dict(signal.metadata),
        processing_state="intermediate",
        method="m3resp.eit.processing.mdn_filter",
    )
    capture_value(captures, "filtered_data", result)
    return result


def mdn_frequency_bands(
    *,
    respiratory_rate: float,
    heart_rate: float,
    noise_frequency_limit: float = NOISE_FREQUENCY_LIMIT,
    notch_distance: float = NOTCH_DISTANCE,
) -> list[tuple[float, float]]:
    """Return the bands that ``mdn_filter`` removes, without filtering.

    The number of bands is the number of removed harmonics of the heart rate.

    Args:
        respiratory_rate (float): Breathing rate in Hz, above 0 and below
            ``heart_rate``.
        heart_rate (float): Heart rate in Hz, above 0.
        noise_frequency_limit (float): Low-pass edge of the filter in Hz.
        notch_distance (float): Half the width of each band, in Hz.

    Returns:
        list[tuple[float, float]]: One (low, high) pair in Hz per harmonic,
            from the first harmonic up. Empty when the heart rate is above the
            noise frequency limit plus the notch distance.

    Raises:
        ValueError: If a rate is 0 or below, or if the breathing rate is
            equal to or higher than the heart rate.

    Warns:
        UserWarning: When the breathing rate is above 85 per minute or the
            heart rate is above 210 per minute.
    """

    _check_rates(respiratory_rate, heart_rate)
    return _frequency_bands(
        respiratory_rate, heart_rate, noise_frequency_limit, notch_distance
    )


def _frequency_bands(
    respiratory_rate: float,
    heart_rate: float,
    noise_frequency_limit: float,
    notch_distance: float,
) -> list[tuple[float, float]]:
    """Return one (low, high) band in Hz per removed harmonic of the heart rate."""

    # The last harmonic is the last one whose band starts below the limit.
    n_harmonics = math.floor((noise_frequency_limit + notch_distance) / heart_rate)
    bands = []
    for harmonic in range(1, n_harmonics + 1):
        lower_limit = heart_rate * harmonic - notch_distance
        upper_limit = heart_rate * harmonic + notch_distance
        if harmonic == 1:
            # Keep the breathing signal: start at least halfway between the
            # breathing rate and the heart rate.
            lower_limit = max(lower_limit, (heart_rate + respiratory_rate) / 2)
        bands.append((lower_limit, upper_limit))
    return bands


def _check_rates(respiratory_rate: float, heart_rate: float) -> None:
    """Check the two rates, in the order and with the messages of eitprocessing.

    Raises:
        ValueError: If a rate is 0 or below, or if the breathing rate is
            equal to or higher than the heart rate.

    Warns:
        UserWarning: When a rate is above its upper limit.
    """

    if respiratory_rate > UPPER_RESPIRATORY_RATE_LIMIT:
        warnings.warn(
            f"The breathing rate ({respiratory_rate:.2f} Hz) is higher than "
            f"{UPPER_RESPIRATORY_RATE_LIMIT * MINUTE:g} per minute "
            f"({UPPER_RESPIRATORY_RATE_LIMIT:.2f} Hz). Make sure the rate is "
            "in Hz, not per minute.",
            UserWarning,
            stacklevel=3,
        )
    if respiratory_rate <= 0:
        raise ValueError(
            f"The breathing rate ({respiratory_rate:.2f} Hz) must be above 0."
        )
    if heart_rate > UPPER_HEART_RATE_LIMIT:
        warnings.warn(
            f"The heart rate ({heart_rate:.2f} Hz) is higher than "
            f"{UPPER_HEART_RATE_LIMIT * MINUTE:g} per minute "
            f"({UPPER_HEART_RATE_LIMIT:.2f} Hz). Make sure this is correct "
            "and that the rate is in Hz.",
            UserWarning,
            stacklevel=3,
        )
    if heart_rate <= 0:
        raise ValueError(f"The heart rate ({heart_rate:.2f} Hz) must be above 0.")
    if respiratory_rate >= heart_rate:
        raise ValueError(
            f"The breathing rate ({respiratory_rate:.2f} Hz) must be below the "
            f"heart rate ({heart_rate:.2f} Hz)."
        )


def _warn_partly_missing(values: np.ndarray, name: str | None) -> None:
    """Warn when a pixel has NaN at some samples, but not at all samples.

    The filter spreads one NaN sample over the whole pixel. A pixel that is
    NaN at every sample (outside the measured area) gives no warning.
    """

    if values.size == 0 or not np.issubdtype(values.dtype, np.floating):
        return
    n_samples = values.shape[0]
    missing_per_pixel = np.atleast_1d(np.isnan(values).sum(axis=0))
    partly_missing = (missing_per_pixel > 0) & (missing_per_pixel < n_samples)
    if not partly_missing.any():
        return
    if values.ndim == 1:
        where = f"the waveform has NaN at {int(missing_per_pixel[0])} of {n_samples} samples"
    else:
        examples = ", ".join(
            f"({', '.join(str(int(axis)) for axis in position)}): "
            f"{int(missing_per_pixel[tuple(position)])} samples"
            for position in np.argwhere(partly_missing)[:5]
        )
        where = (
            f"{int(partly_missing.sum())} pixel(s) have NaN at some of the "
            f"{n_samples} samples, for example {examples}"
        )
    warnings.warn(
        f"In {name!r}, {where}. The MDN filter gives NaN at every sample of "
        "these pixels.",
        UserWarning,
        stacklevel=3,
    )
