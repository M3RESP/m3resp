"""Estimate breathing rate and heart rate from EIT data.

Both rates come from frequency spectra of the EIT data, made with Welch's
method and a Hann window:

- The breathing rate is the frequency with the most power in the spectrum of
  the summed pixels, inside the breathing-rate range.
- For the heart rate, the power spectrum of each pixel is normalized by
  dividing it by its total power. The power spectrum of the sum of all pixels
  is also normalized, and it is subtracted from the average normalized pixel
  spectrum. The heart rate is the frequency with the most power in this
  difference, inside the heart-rate range.

The heart rate stands out in this difference because the heart signal is
relatively strong in many pixels with a small breathing signal.

By default, each rate is refined with a parabola through the highest point of
the spectrum and its two neighbours. This gives a rate between two frequency
steps of the spectrum.

The pixels are converted to single precision (float32) before the spectra
are made, as in eitprocessing, so both packages give the same rates. On the
three test files, double precision changes the rates by less than 1e-6 of
their value. The sum of the pixels can differ a little from the global
impedance of ``m3resp.eit.processing.global_impedance``, which is summed in
double precision.

Use at least 30 s of data for reliable rates. The rates are most reliable
when the end-expiratory impedance is stable, the data is not filtered and
both rates are steady. When a rate changes during the data, the result is
usually near the average rate.

Reference: J. J. Wisse, P. Somhorst, J. Behr, A. R. van Nieuw Amerongen,
D. Gommers and A. H. Jonkman, "Improved filtering methods to suppress
cardiovascular contamination in electrical impedance tomography recordings",
Physiol. Meas. 45, 055010 (2024), https://doi.org/10.1088/1361-6579/ad46e3.

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
      the function `detect_rates`, with the pixel impedance as a `Signal`.
    - A rate limit of 0 Hz is kept as given; only a missing limit (None) is
      replaced by the default limit of the subject type.
    - A ValueError when the pixel impedance has no frames per second, when it does
      not have three axes, or when no frequency of the spectrum lies inside
      a rate range.
    - The subject type can be None; then all four rate limits must be given.
    - A rate at the edge of its range gives a warning also when the rate is
      not refined.
    - The rates are returned as Python floats.
    - Docstrings and warning texts are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
from dataclasses import KW_ONLY, dataclass
from typing import Any, Literal, cast, get_args

import numpy as np
from scipy import signal

from m3resp.data.signals import Signal
from m3resp.processing.captures import capture_value

#: Public name of the function, stored as the method of each result.
METHOD = "m3resp.eit.processing.detect_rates"

MINUTE = 60
MIN_WELCH_WINDOW_LENGTH = 10  # seconds

#: Default rate limits in Hz for each subject type.
DEFAULT_RATE_LIMITS: dict[str, dict[str, float]] = {
    "min_heart_rate": {"adult": 40 / MINUTE, "neonate": 90 / MINUTE},
    "max_heart_rate": {"adult": 200 / MINUTE, "neonate": 210 / MINUTE},
    "min_respiratory_rate": {"adult": 6 / MINUTE, "neonate": 15 / MINUTE},
    "max_respiratory_rate": {"adult": 60 / MINUTE, "neonate": 85 / MINUTE},
}

SubjectType = Literal["adult", "neonate"]
#: The subject types with default rate limits.
SUBJECT_TYPES: tuple[str, ...] = get_args(SubjectType)


@dataclass(frozen=True)
class RateDetectionSettings:
    """Settings of the breathing-rate and heart-rate detection.

    Attributes:
        subject_type (str | None): ``"adult"``, ``"neonate"`` or None. It
            sets the default rate limits (``DEFAULT_RATE_LIMITS``): adult
            breathing 6 to 60 and heart 40 to 200 per minute, neonate
            breathing 15 to 85 and heart 90 to 210 per minute. With None, for
            example for children or animals, give all four rate limits.
        welch_window (float): Length of one Welch window in seconds, at least
            10 s. When the data is shorter, the window is the whole data.
        welch_overlap (float): Part of a Welch window that overlaps the next
            window, from 0 up to but not including 1. 0.5 is half a window.
        min_heart_rate (float | None): Lowest heart rate in Hz. None gives
            the default of the subject type; required when ``subject_type``
            is None.
        max_heart_rate (float | None): Highest heart rate in Hz. None gives
            the default of the subject type; required when ``subject_type``
            is None.
        min_respiratory_rate (float | None): Lowest breathing rate in Hz.
            None gives the default of the subject type; required when
            ``subject_type`` is None.
        max_respiratory_rate (float | None): Highest breathing rate in Hz.
            None gives the default of the subject type; required when
            ``subject_type`` is None.
        refine_estimated_frequency (bool): If True, refine each rate with a
            parabola through the highest point of the spectrum and its two
            neighbours. If False, each rate is the frequency of the highest
            point.

    Raises:
        ValueError: If ``welch_overlap`` is below 0 or 1 or more, if
            ``welch_window`` is shorter than 10 s, if ``subject_type`` is
            not ``"adult"``, ``"neonate"`` or None, or if ``subject_type`` is
            None and a rate limit is missing.
    """

    subject_type: SubjectType | None
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
        if self.subject_type is None:
            missing = [
                name for name in DEFAULT_RATE_LIMITS if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    "Without a subject type, give all four rate limits in Hz. "
                    f"Missing: {', '.join(missing)}."
                )
            return
        if self.subject_type not in SUBJECT_TYPES:
            raise ValueError(
                f"Invalid subject type: {self.subject_type!r}. Must be 'adult', "
                "'neonate' or None."
            )
        for name, limits in DEFAULT_RATE_LIMITS.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, limits[self.subject_type])


def detect_rates(
    pixel_impedance: Signal,
    *,
    settings: RateDetectionSettings,
    captures: dict[str, Any] | None = None,
    suppress_length_warnings: bool = False,
    suppress_edge_case_warning: bool = False,
) -> tuple[float, float]:
    """Return the breathing rate and the heart rate of the pixel impedance, in Hz.

    Multiply by 60 to get breaths or beats per minute. The method is from
    Wisse et al., Physiol. Meas. 45, 055010 (2024),
    https://doi.org/10.1088/1361-6579/ad46e3. See the module docstring.

    Args:
        pixel_impedance (Signal): Pixel impedance, values with shape (frames,
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

    _check_eit_pixels(pixel_impedance)
    sample_frequency = float(cast(float, pixel_impedance.sample_frequency))
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


def _check_eit_pixels(pixel_impedance: Signal) -> None:
    """Raise ``ValueError`` if the pixel impedance has the wrong shape or no frame rate.

    Args:
        pixel_impedance (Signal): The pixel impedance.

    Raises:
        ValueError: If the values do not have the shape (frames, rows,
            columns), or if ``sample_frequency`` is missing.
    """

    values = np.asarray(pixel_impedance.values)
    if values.ndim != 3:
        raise ValueError(
            "detect_rates needs pixel impedance with shape (frames, rows, columns); "
            f"{pixel_impedance.name!r} has shape {values.shape}."
        )
    if pixel_impedance.sample_frequency is None:
        raise ValueError(
            f"detect_rates needs the frames per second of {pixel_impedance.name!r} "
            "(sample_frequency)."
        )


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

    at_edge = index_max_power in (0, len(power_range) - 1)
    if at_edge and not suppress_edge_case_warning:
        warnings.warn(
            f"The highest power of the {what}-rate spectrum is at the edge of "
            f"the range ({min_rate * MINUTE:g} to {max_rate * MINUTE:g} per "
            "minute), so the rate is probably wrong. Check the data or the "
            "range. A rate at the edge is not refined.",
            RuntimeWarning,
            stacklevel=3,
        )
    if not refine or at_edge:
        return float(frequency_range[index_max_power])

    y0, y1, y2 = power_range[index_max_power - 1 : index_max_power + 2]
    x0, x1, x2 = frequency_range[index_max_power - 1 : index_max_power + 2]
    denominator = y0 - 2 * y1 + y2
    if denominator == 0:
        return float(x1)
    delta = 0.5 * (y0 - y2) / denominator
    return float(x1 + delta * (x2 - x0) / 2)
