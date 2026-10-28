"""Breathing rate and heart rate from EIT images.

Both rates come from frequency spectra of the EIT images (Welch's method with
a Hann window):

- The breathing rate is the frequency with the most power in the spectrum of
  the summed pixels, inside the breathing-rate range.
- For the heart rate, the spectrum of each pixel is divided by its total
  power, and these spectra are averaged over the pixels. The spectrum of the
  summed pixels, also divided by its total power, is subtracted from this
  average. The heart rate is the frequency with the most power in this
  difference, inside the heart-rate range. In many pixels with a small
  breathing signal the heart signal is relatively strong, so the heart rate
  stands out in the difference.

By default each rate is refined with a parabola through the highest point of
the spectrum and its two neighbours. This gives a rate between two frequency
steps of the spectrum.

The pixels are summed in single precision (float32), as in eitprocessing.
This sum can differ a little from the global impedance of
``m3resp.eit.processing.global_impedance``, which is summed in double
precision.

Use at least 30 s of data for reliable rates. The rates are most reliable
when the end-expiratory impedance is stable, the images are not filtered and
both rates are steady. When a rate changes during the data, the result is
usually an average rate.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/features/rate_detection.py::RateDetection,
                DEFAULT_RATE_LIMITS
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The settings are the record `RateDetectionSettings`; the calculation is
      the function `detect_rates`, with the EIT images as a `Signal`.
    - A rate limit of 0 Hz is kept as given; only a missing limit (None) is
      replaced by the default limit of the subject type.
    - A ValueError when the images have no frames per second, when they do
      not have three axes, or when no frequency of the spectrum lies inside
      a rate range.
    - The rates are returned as Python floats.
    - Docstrings and warning texts are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
from dataclasses import KW_ONLY, dataclass
from typing import Any, Final, Literal, cast

import numpy as np
from scipy import signal

from m3resp.data.signals import Signal
from m3resp.processing.captures import capture_value

MINUTE: Final = 60
MIN_WELCH_WINDOW_LENGTH: Final = 10  # seconds

#: Default rate limits in Hz for each subject type.
DEFAULT_RATE_LIMITS: Final[dict[str, dict[str, float]]] = {
    "min_heart_rate": {"adult": 40 / MINUTE, "neonate": 90 / MINUTE},
    "max_heart_rate": {"adult": 200 / MINUTE, "neonate": 210 / MINUTE},
    "min_respiratory_rate": {"adult": 6 / MINUTE, "neonate": 15 / MINUTE},
    "max_respiratory_rate": {"adult": 60 / MINUTE, "neonate": 85 / MINUTE},
}

SubjectType = Literal["adult", "neonate"]


@dataclass(frozen=True)
class RateDetectionSettings:
    """Settings of the breathing-rate and heart-rate detection.

    Attributes:
        subject_type (str): ``"adult"`` or ``"neonate"``. It sets the default
            rate limits (``DEFAULT_RATE_LIMITS``): adult breathing 6 to 60
            and heart 40 to 200 per minute, neonate breathing 15 to 85 and
            heart 90 to 210 per minute.
        welch_window (float): Length of one Welch window in seconds, at least
            10 s. When the data is shorter, the window is the whole data.
        welch_overlap (float): Part of a Welch window that overlaps the next
            window, from 0 up to but not including 1. 0.5 is half a window.
        min_heart_rate (float | None): Lowest heart rate in Hz. None gives
            the default of the subject type.
        max_heart_rate (float | None): Highest heart rate in Hz. None gives
            the default of the subject type.
        min_respiratory_rate (float | None): Lowest breathing rate in Hz.
            None gives the default of the subject type.
        max_respiratory_rate (float | None): Highest breathing rate in Hz.
            None gives the default of the subject type.
        refine_estimated_frequency (bool): If True, refine each rate with a
            parabola through the highest point of the spectrum and its two
            neighbours. If False, each rate is the frequency of the highest
            point.

    Raises:
        ValueError: If ``welch_overlap`` is below 0 or 1 or more, if
            ``welch_window`` is shorter than 10 s, or if ``subject_type`` is
            not ``"adult"`` or ``"neonate"``.
    """

    subject_type: SubjectType
    _: KW_ONLY
    welch_window: float = 30.0
    welch_overlap: float = 0.5
    min_heart_rate: float | None = None
    max_heart_rate: float | None = None
    min_respiratory_rate: float | None = None
    max_respiratory_rate: float | None = None
    refine_estimated_frequency: bool = True

    def __post_init__(self) -> None:
        """Check the settings and fill in the default rate limits."""

        if self.welch_overlap >= 1:
            raise ValueError(
                f"welch_overlap must be less than 1 (100%), got {self.welch_overlap}."
            )
        if self.welch_overlap < 0:
            raise ValueError(
                f"welch_overlap must be at least 0 (0%), got {self.welch_overlap}."
            )
        if self.welch_window < MIN_WELCH_WINDOW_LENGTH:
            raise ValueError(
                f"welch_window must be at least {MIN_WELCH_WINDOW_LENGTH} s, "
                f"got {self.welch_window} s."
            )
        if self.subject_type not in ("adult", "neonate"):
            raise ValueError(
                f"Invalid subject type: {self.subject_type!r}. Must be 'adult' "
                "or 'neonate'."
            )
        for name, limits in DEFAULT_RATE_LIMITS.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, limits[self.subject_type])


def detect_rates(
    pixel_impedance: Signal,
    settings: RateDetectionSettings,
    *,
    captures: dict[str, Any] | None = None,
    suppress_length_warnings: bool = False,
    suppress_edge_case_warning: bool = False,
) -> tuple[float, float]:
    """Return the breathing rate and the heart rate of EIT images, in Hz.

    Multiply by 60 to get breaths or beats per minute.

    Args:
        pixel_impedance (Signal): EIT images, values with shape (frames,
            rows, columns), with the number of frames per second in
            ``sample_frequency``.
        settings (RateDetectionSettings): The settings.
        captures (dict | None): A dictionary that receives the in-between
            results for a figure: ``"frequencies"`` (Hz),
            ``"normalized_total_power"``, ``"average_normalized_pixel_power"``,
            ``"diff_total_averaged_power"``, ``"estimated_respiratory_rate"``
            and ``"estimated_heart_rate"`` (Hz). None stores nothing.
        suppress_length_warnings (bool): If True, give no warning when the
            data is shorter than the Welch window.
        suppress_edge_case_warning (bool): If True, give no warning when the
            highest point of a spectrum is at the edge of the rate range.

    Returns:
        tuple[float, float]: The breathing rate and the heart rate in Hz.

    Raises:
        ValueError: If the values do not have three axes, if
            ``sample_frequency`` is missing, or if no frequency of the
            spectrum lies inside a rate range.

    Warns:
        UserWarning: When the data is shorter than the Welch window. The
            window is then the whole data.
        RuntimeWarning: When the highest point of a spectrum is at the edge
            of the rate range. The rate is then not refined.
    """

    sample_frequency = _frames_per_second(pixel_impedance)
    pixels = np.asarray(pixel_impedance.values).astype(np.float32)
    summed_impedance = np.nansum(pixels, axis=(1, 2))

    segment_length = int(settings.welch_window * sample_frequency)
    if len(summed_impedance) < segment_length:
        if not suppress_length_warnings:
            warnings.warn(
                "The Welch window is longer than the data. The window is "
                "shortened to the length of the data.",
                UserWarning,
                stacklevel=2,
            )
        segment_length = len(summed_impedance)
    overlap_length = int(segment_length * settings.welch_overlap)
    hann_window: Any = signal.windows.hann(segment_length, sym=False)

    frequencies, total_power = signal.welch(
        summed_impedance,
        sample_frequency,
        nperseg=segment_length,
        noverlap=overlap_length,
        detrend="constant",
        window=hann_window,
    )
    normalized_total_power = total_power / np.sum(total_power)

    # A pixel that is 0 in every frame is outside the measured area.
    pixels[:, np.all(pixels == 0, axis=0)] = np.nan
    _, pixel_power_spectra = signal.welch(
        pixels,
        sample_frequency,
        nperseg=segment_length,
        noverlap=overlap_length,
        detrend="constant",
        axis=0,
        window=hann_window,
    )
    normalized_power_spectra = np.divide(
        pixel_power_spectra, np.nansum(pixel_power_spectra, axis=0, keepdims=True)
    )
    average_normalized_pixel_power = np.nanmean(normalized_power_spectra, axis=(1, 2))
    diff_total_averaged_power = average_normalized_pixel_power - normalized_total_power

    respiratory_rate = _peak_frequency(
        frequencies,
        total_power,
        cast(float, settings.min_respiratory_rate),
        cast(float, settings.max_respiratory_rate),
        refine=settings.refine_estimated_frequency,
        suppress_edge_case_warning=suppress_edge_case_warning,
        what="breathing",
    )
    heart_rate = _peak_frequency(
        frequencies,
        diff_total_averaged_power,
        cast(float, settings.min_heart_rate),
        cast(float, settings.max_heart_rate),
        refine=settings.refine_estimated_frequency,
        suppress_edge_case_warning=suppress_edge_case_warning,
        what="heart",
    )

    capture_value(captures, "frequencies", frequencies)
    capture_value(captures, "normalized_total_power", normalized_total_power)
    capture_value(
        captures, "average_normalized_pixel_power", average_normalized_pixel_power
    )
    capture_value(captures, "diff_total_averaged_power", diff_total_averaged_power)
    capture_value(captures, "estimated_respiratory_rate", respiratory_rate)
    capture_value(captures, "estimated_heart_rate", heart_rate)
    return respiratory_rate, heart_rate


def _frames_per_second(pixel_impedance: Signal) -> float:
    """Return the frames per second of EIT images after checking their shape."""

    values = np.asarray(pixel_impedance.values)
    if values.ndim != 3:
        raise ValueError(
            "detect_rates needs EIT images with shape (frames, rows, columns); "
            f"{pixel_impedance.name!r} has shape {values.shape}."
        )
    if pixel_impedance.sample_frequency is None:
        raise ValueError(
            f"detect_rates needs the frames per second of {pixel_impedance.name!r} "
            "(sample_frequency)."
        )
    return float(pixel_impedance.sample_frequency)


def _peak_frequency(
    frequencies: np.ndarray,
    power: np.ndarray,
    min_rate: float,
    max_rate: float,
    *,
    refine: bool,
    suppress_edge_case_warning: bool,
    what: str,
) -> float:
    """Return the frequency with the most power inside a rate range.

    Args:
        frequencies (np.ndarray): Frequencies of the spectrum in Hz.
        power (np.ndarray): Power at each frequency.
        min_rate (float): Lowest rate in Hz.
        max_rate (float): Highest rate in Hz.
        refine (bool): If True, move the result to the top of a parabola
            through the highest point and its two neighbours.
        suppress_edge_case_warning (bool): If True, give no warning when the
            highest point is at the edge of the range.
        what (str): ``"breathing"`` or ``"heart"``, for the messages.

    Returns:
        float: The rate in Hz.

    Raises:
        ValueError: If no frequency lies inside the range.
    """

    in_range = (frequencies >= min_rate) & (frequencies <= max_rate)
    if not np.any(in_range):
        raise ValueError(
            f"No frequency of the spectrum lies inside the {what}-rate range "
            f"{min_rate * MINUTE:g} to {max_rate * MINUTE:g} per minute. Use a "
            "longer Welch window or a wider range."
        )
    frequency_range = frequencies[in_range]
    power_range = power[in_range]
    index_max_power = int(np.argmax(power_range))

    if not refine:
        return float(frequency_range[index_max_power])

    if index_max_power in (0, len(power_range) - 1):
        if not suppress_edge_case_warning:
            warnings.warn(
                f"The highest power of the {what}-rate spectrum is at the edge "
                "of the range, so the rate is not refined.",
                RuntimeWarning,
                stacklevel=3,
            )
        return float(frequency_range[index_max_power])

    y0, y1, y2 = power_range[index_max_power - 1 : index_max_power + 2]
    x0, x1, x2 = frequency_range[index_max_power - 1 : index_max_power + 2]
    denominator = y0 - 2 * y1 + y2
    if denominator == 0:
        return float(x1)
    delta = 0.5 * (y0 - y2) / denominator
    return float(x1 + delta * (x2 - x0) / 2)
