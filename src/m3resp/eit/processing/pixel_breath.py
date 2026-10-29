"""Pixel breaths: the start, end of inspiration and end of each breath in each pixel.

Each pixel of an EIT image has its own impedance waveform. Its breaths are
close in time to the breaths of a global (or regional) impedance waveform,
but can start and end a little earlier or later. ``find_pixel_breaths``
finds the breaths of each pixel from the global breaths:

1. The mean TIV of each pixel over the global breaths gives the sign of the
   pixel. A pixel with a negative mean TIV has an impedance that goes down
   during inspiration.
2. The search windows lie between the ends of inspiration
   (``extremum_time``) of two neighbouring global breaths. The start of a
   pixel breath is the lowest impedance in a window, the end is the start
   of the next pixel breath, and the end of inspiration is the highest
   impedance between start and end.
3. The phase correction mode sets what happens to pixels that are out of
   phase with the global waveform:

   - ``"negative amplitude"`` (default): a pixel with a negative mean TIV
     is turned upside down. Its start is the highest impedance in a window,
     and its end of inspiration the lowest.
   - ``"phase shift"``: every pixel keeps a positive TIV. The lag of the
     peak of the cross-correlation between the pixel and the global
     waveform that is nearest to zero, within one mean breath duration,
     moves the search windows.
   - ``"none"`` or None: no correction. This can give very short or very
     long pixel breaths.

The first and the last global breath never have pixel breaths, because a
pixel breath needs the global breath before it and after it. A pixel breath
is left out when its start is not before its end of inspiration, its end of
inspiration not before its end, or one of its two search windows falls
partly outside the data. When more than 25 % of the breaths of a pixel are
left out, the pixel has no breaths at all.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/features/pixel_breath.py::PixelBreath
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The class `PixelBreath` and its method `find_pixel_breaths` are one
      function, `find_pixel_breaths`, with the EIT images and the waveform
      as `Signal`s. The global breaths are an input (eitprocessing runs its
      own `BreathDetection`). The result is a `PixelBreaths`, with NaN times
      for a pixel with no breath (eitprocessing: an `IntervalData` of
      `Breath` objects and None).
    - The sign of each pixel comes from its inspiratory TIV, as in
      `pixel_tidal_impedance_variation`, on `sign_breaths`, or on `breaths`
      when `sign_breaths` is None.
      eitprocessing computes it with `TIV()` on the breaths of a second
      `BreathDetection` with the default settings.
    - The warnings for left-out pixels come once per kind (no
      cross-correlation peak; too many breaths left out) with the number
      of pixels and their positions (eitprocessing: one warning per pixel).
      Every left-out pixel and its reason is in the result `metadata`. The
      message for too many breaths gives the real limit, 25 %
      (eitprocessing: "more than half").
    - The waveform is detrended once, and only in the "phase shift" mode
      (eitprocessing: once per pixel, in every mode; the detrended data is
      used only for the cross-correlation).
    - Bug fix: with fewer than two global breaths, the result has no pixel
      breaths (eitprocessing raises a ValueError).
    - Bug fix: a pixel breath with a search window partly outside the data
      is left out. eitprocessing moves the end of inspiration of such a
      global breath to the first or the last sample, and keeps the pixel
      breaths of the moved window; two moved breaths give an empty window
      and a ValueError. A global breath whose end of inspiration lies
      outside the time axis of the data counts as outside, also before the
      first sample.
    - Bug fix: the 25 % rule counts each left-out pixel breath once.
      eitprocessing counts a breath twice when it is left out for two
      reasons, and counts global breaths that are no pixel breaths. A pixel
      with every breath left out is listed under
      ``"too_many_breaths_skipped"`` (eitprocessing: no warning).
    - New: the waveform must be on the time axis of the images.
    - New: a warning when no pixel has a breath.
    - Docstrings and error texts are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import numpy as np
from scipy import signal as scipy_signal

from m3resp.data.events import BreathEvent
from m3resp.data.pixel_breaths import PixelBreaths
from m3resp.data.signals import Signal
from m3resp.eit.processing._breath_samples import (
    breath_samples,
    breath_source_metadata,
    check_breaths,
    check_impedance_waveform,
    check_pixel_impedance,
)
from m3resp.eit.processing.tidal_impedance_variation import _tiv_values

#: The phase correction modes. None is the same as ``"none"``.
PhaseCorrectionMode = Literal["negative amplitude", "phase shift", "none", None]  # noqa: PYI061
PHASE_CORRECTION_MODES: Final = ("negative amplitude", "phase shift", "none", None)
#: Largest lag of the cross-correlation, in mean breath durations.
MAX_XCORR_LAG: Final = 1
#: Largest part of the breaths of a pixel that can be left out.
ALLOW_FRACTION_BREATHS_SKIPPED: Final = 0.25
#: Name of the result, as in eitprocessing.
NAME: Final = "pixel_breaths"
METHOD: Final = "m3resp.eit.processing.find_pixel_breaths"

#: The reasons a pixel has no breaths, as keys of ``metadata["skipped_pixels"]``.
NO_TIV: Final = "no_tiv"
FLAT: Final = "flat"
NO_CORRELATION_PEAK: Final = "no_correlation_peak"
TOO_MANY_BREATHS_SKIPPED: Final = "too_many_breaths_skipped"


def find_pixel_breaths(
    pixels: Signal,
    signal: Signal,
    breaths: Sequence[BreathEvent],
    *,
    phase_correction_mode: PhaseCorrectionMode = "negative amplitude",
    name: str = NAME,
    sign_breaths: Sequence[BreathEvent] | None = None,
) -> PixelBreaths:
    """Return the breaths of each pixel of EIT images.

    Args:
        pixels (Signal): EIT images, values with shape (time, row, column),
            with time in seconds. Its ``category`` must be None or an
            impedance category.
        signal (Signal): The impedance waveform the global breaths were
            found in: the global impedance, or a regional impedance. One
            value per sample, on the time axis of ``pixels`` (the same
            number of samples, each sample time less than half a sample
            from the time of ``pixels``). The "phase shift" mode
            cross-correlates it with each pixel.
        breaths (Sequence[BreathEvent]): The global breaths, in time order.
            Their ``extremum_time`` (end of inspiration) sets the search
            windows; each is read at the first sample at or after it. An
            ``extremum_time`` outside the time axis of ``pixels`` gives no
            search window.
        phase_correction_mode (PhaseCorrectionMode): ``"negative
            amplitude"``, ``"phase shift"``, ``"none"`` or None (see the
            module text).
        name (str): Name of the result.
        sign_breaths (Sequence[BreathEvent] | None): The breaths for the
            mean TIV that gives the sign of each pixel. None uses
            ``breaths``. eitprocessing uses the breaths of the global
            impedance found with the default settings of the breath
            detection.

    Returns:
        PixelBreaths: The given global breaths, and the start, end of
            inspiration and end time of the breath of each pixel, in
            seconds (sample times of ``pixels``). NaN where a pixel has no
            breath: always the first and the last global breath, a
            left-out breath, and every breath of a left-out pixel.
            ``metadata`` holds ``phase_correction_mode``, ``signal_name``,
            ``breath_modality``, ``breath_source`` and ``skipped_pixels``:
            for each reason a list of ``[row, column]``. The reasons are
            ``"no_tiv"`` (the TIV is NaN in every breath), ``"flat"`` (the
            impedance does not change), ``"no_correlation_peak"`` and
            ``"too_many_breaths_skipped"``.

    Raises:
        TypeError: If ``pixels`` or ``signal`` is not a ``Signal`` or a
            breath is not a ``BreathEvent``.
        ValueError: If the phase correction mode is unknown, a category is
            not an impedance category, the values do not have the expected
            shapes, ``signal`` is not on the time axis of ``pixels``, or a
            global breath has no ``extremum_time``.

    Warns:
        RuntimeWarning: Once when pixels have no cross-correlation peak, and
            once when more than 25 % of the breaths of pixels are left out,
            with the number of pixels and their positions. Once when no
            pixel has a breath, with the number of pixels for each reason.
        UserWarning: When breaths of ``sign_breaths`` (or ``breaths``) lie
            partly outside the data.
    """

    function = "find_pixel_breaths"
    if phase_correction_mode not in PHASE_CORRECTION_MODES:
        named = ", ".join(repr(mode) for mode in PHASE_CORRECTION_MODES)
        raise ValueError(
            f"Unknown phase correction mode {phase_correction_mode!r}; use one "
            f"of {named}."
        )
    pixel_impedance = check_pixel_impedance(pixels, function=function)
    waveform = check_impedance_waveform(signal, function=function)
    time = np.asarray(pixels.time)
    _check_same_time_axis(time, np.asarray(signal.time), signal.name, function=function)
    checked = check_breaths(breaths, function=function)
    middle_times = [breath.extremum_time for breath in checked]
    if any(middle_time is None for middle_time in middle_times):
        raise ValueError(
            f"{function} needs the extremum_time (end of inspiration) of every "
            "global breath."
        )
    checked_sign = (
        checked
        if sign_breaths is None
        else check_breaths(sign_breaths, function=function)
    )

    n_breaths = len(checked)
    _, n_rows, n_columns = pixel_impedance.shape
    start = np.full((n_breaths, n_rows, n_columns), np.nan)
    middle = np.full_like(start, np.nan)
    end = np.full_like(start, np.nan)
    skipped: dict[str, list[list[int]]] = {
        NO_TIV: [],
        FLAT: [],
        NO_CORRELATION_PEAK: [],
        TOO_MANY_BREATHS_SKIPPED: [],
    }

    if n_breaths >= 2 and len(time):
        middle_array = np.asarray(middle_times, dtype=float)
        indices_breath_middles = np.searchsorted(time, middle_array)
        middle_in_data = (middle_array >= time[0]) & (middle_array <= time[-1])
        sign_samples = breath_samples(
            pixels,
            checked_sign,
            points=("start_time", "extremum_time", "end_time"),
            function=function,
        )
        mean_tiv_pixel, all_nan_mask = _mean_tiv(
            _tiv_values(pixel_impedance, sign_samples, "inspiratory")
        )
        allow_negative_amplitude = phase_correction_mode == "negative amplitude"
        correct_for_phase_shift = phase_correction_mode == "phase shift"
        if correct_for_phase_shift:
            lags = scipy_signal.correlation_lags(
                len(waveform), len(waveform), mode="same"
            )
            detrended = scipy_signal.detrend(np.copy(waveform), type="linear")
            max_lag = MAX_XCORR_LAG * np.mean(np.diff(indices_breath_middles))
            lag_range = (lags >= -max_lag) & (lags <= max_lag)

        for row, column in itertools.product(range(n_rows), range(n_columns)):
            if all_nan_mask[row, column]:
                skipped[NO_TIV].append([row, column])
                continue
            pixel = pixel_impedance[:, row, column]
            if np.std(pixel) == 0:
                skipped[FLAT].append([row, column])
                continue

            lagged_indices_breath_middles = indices_breath_middles
            if allow_negative_amplitude and mean_tiv_pixel[row, column] < 0:
                start_function, middle_function = np.argmax, np.argmin
            else:
                start_function, middle_function = np.argmin, np.argmax
                if correct_for_phase_shift:
                    lag = _correlation_lag(detrended, pixel, lags, lag_range)
                    if lag is None:
                        skipped[NO_CORRELATION_PEAK].append([row, column])
                        continue
                    lagged_indices_breath_middles = indices_breath_middles - lag

            found, starts, middles, ends = _pixel_breath_indices(
                pixel,
                lagged_indices_breath_middles,
                middle_in_data,
                start_function,
                middle_function,
            )
            n_left_out = int((~found).sum())
            if n_left_out > len(found) * ALLOW_FRACTION_BREATHS_SKIPPED:
                skipped[TOO_MANY_BREATHS_SKIPPED].append([row, column])
                continue

            # Pixel breath i belongs to global breath i + 1: the first and
            # the last global breath have no pixel breath.
            keep = np.flatnonzero(found)
            start[keep + 1, row, column] = time[starts[keep]]
            middle[keep + 1, row, column] = time[middles[keep]]
            end[keep + 1, row, column] = time[ends[keep]]

    _warn_for_skipped_pixels(skipped, function=function)
    if n_breaths > 2 and len(time) and np.isnan(start).all():
        _warn_for_no_pixel_breaths(skipped, function=function)
    return PixelBreaths(
        name=name,
        breaths=checked,
        start_time=start,
        extremum_time=middle,
        end_time=end,
        modality=pixels.modality,
        method=METHOD,
        metadata={
            "phase_correction_mode": phase_correction_mode,
            "signal_name": signal.name,
            **breath_source_metadata(checked),
            "skipped_pixels": skipped,
        },
    )


def _mean_tiv(tiv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean TIV of each pixel and the pixels whose TIV is always NaN.

    Args:
        tiv (np.ndarray): The TIV of each pixel in each breath, shape
            (breath, row, column).

    Returns:
        tuple[np.ndarray, np.ndarray]: The mean TIV of each pixel, without
            the NaN values (NaN for a pixel with only NaN values), and True
            for the pixels with only NaN values. Both of shape (row, column).
    """

    all_nan_mask = np.isnan(tiv).all(axis=0)
    mean_tiv_pixel = np.full(tiv.shape[1:], np.nan)
    if not all_nan_mask.all():
        mean_tiv_pixel[~all_nan_mask] = np.nanmean(tiv[:, ~all_nan_mask], axis=0)
    return mean_tiv_pixel, all_nan_mask


def _check_same_time_axis(
    time: np.ndarray, signal_time: np.ndarray, signal_name: str | None, *, function: str
) -> None:
    """Raise ``ValueError`` if the waveform is not on the time axis of the images.

    The two time axes must have the same number of samples, and each pair of
    sample times must differ by less than half a sample.

    Args:
        time (np.ndarray): Sample times of the EIT images, in seconds.
        signal_time (np.ndarray): Sample times of the waveform, in seconds.
        signal_name (str | None): Name of the waveform, for the message.
        function (str): Name of the calling function, for the message.

    Raises:
        ValueError: If the number of samples differs, or a pair of sample
            times differs by half a sample or more.
    """

    same = len(signal_time) == len(time)
    if same and len(time) > 1:
        half_sample = 0.5 * float(np.median(np.diff(time)))
        same = bool(np.all(np.abs(signal_time - time) < half_sample))
    if not same:
        raise ValueError(
            f"{function} needs the waveform {signal_name!r} on the time axis of "
            f"the EIT images. The waveform has {len(signal_time)} samples "
            f"({_time_text(signal_time)}), the images {len(time)} samples "
            f"({_time_text(time)})."
        )


def _time_text(time: np.ndarray) -> str:
    """Return the first and the last sample time, for a message.

    Args:
        time (np.ndarray): Sample times in seconds.

    Returns:
        str: For example ``"0 s to 59.95 s"``, or ``"none"`` with no samples.
    """

    if not len(time):
        return "none"
    return f"{time[0]:g} s to {time[-1]:g} s"


def _correlation_lag(
    detrended: np.ndarray,
    pixel: np.ndarray,
    lags: np.ndarray,
    lag_range: np.ndarray,
) -> Any:
    """Return the lag of the cross-correlation peak nearest to 0, or None.

    Args:
        detrended (np.ndarray): The detrended global waveform.
        pixel (np.ndarray): The impedance of one pixel.
        lags (np.ndarray): The lag of each value of the cross-correlation,
            in samples.
        lag_range (np.ndarray): True for the lags that can be used.

    Returns:
        Any: The lag in samples (an integer). Of two peaks at the same
            distance from 0, the one with the higher cross-correlation. None
            when there is no peak.
    """

    if not np.all(np.isnan(pixel)):
        pixel = scipy_signal.detrend(pixel, type="linear")
    xcorr = scipy_signal.correlate(detrended, pixel, mode="same")
    peaks, _ = scipy_signal.find_peaks(
        xcorr[lag_range], height=0, prominence=xcorr.std()
    )
    peak_lags = lags[lag_range][peaks]
    peak_distances = np.abs(peak_lags)
    if len(peak_distances) == 0:
        return None
    candidates = peak_lags[peak_distances == np.min(peak_distances)]
    if len(candidates) == 1:
        return candidates[0]
    # Two peaks at the same distance from 0, one on each side.
    return candidates[np.argmax(xcorr[np.searchsorted(lags, candidates)])]


def _pixel_breath_indices(
    pixel: np.ndarray,
    indices_breath_middles: np.ndarray,
    middle_in_data: np.ndarray,
    start_function: Callable[..., Any],
    middle_function: Callable[..., Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the sample indices of the breaths of one pixel.

    Search window ``j`` runs from the end of inspiration of global breath
    ``j`` up to (and without) the end of inspiration of global breath
    ``j + 1``. Pixel breath ``k`` starts in window ``k`` and ends in window
    ``k + 1``, so it belongs to global breath ``k + 1``.

    Args:
        pixel (np.ndarray): The impedance of one pixel.
        indices_breath_middles (np.ndarray): The sample index of the end of
            inspiration of each global breath, moved by the lag of the pixel
            in the "phase shift" mode.
        middle_in_data (np.ndarray): True for the global breaths whose end
            of inspiration lies inside the time axis of the data.
        start_function (Callable[..., Any]): ``np.argmin`` or ``np.argmax``,
            for the start and the end of each pixel breath.
        middle_function (Callable[..., Any]): The other one, for the end of
            inspiration.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: For each
            pixel breath (two fewer than the global breaths): True when the
            pixel breath is found, and the sample index of its start, end of
            inspiration and end (integers). A pixel breath is not found when
            one of its two windows lies partly outside the data or is empty,
            or its three times are not in the order start, end of
            inspiration, end. Its indices are then 0.
    """

    n_samples = len(pixel)
    inside = (
        middle_in_data
        & (indices_breath_middles >= 0)
        & (indices_breath_middles < n_samples)
    )
    window_found = (
        inside[:-1]
        & inside[1:]
        & (indices_breath_middles[1:] > indices_breath_middles[:-1])
    )
    outsides = np.zeros(len(window_found), dtype=int)
    for window in np.flatnonzero(window_found):
        first = indices_breath_middles[window]
        last = indices_breath_middles[window + 1]
        outsides[window] = start_function(pixel[first:last]) + first

    found = window_found[:-1] & window_found[1:]
    starts = outsides[:-1]
    ends = outsides[1:]
    middles = np.zeros(len(found), dtype=int)
    for index in np.flatnonzero(found):
        first, last = starts[index], ends[index]
        middles[index] = middle_function(pixel[first:last]) + first
    found &= (starts < middles) & (middles < ends)
    return found, starts, middles, ends


def _warn_for_skipped_pixels(
    skipped: dict[str, list[list[int]]], *, function: str
) -> None:
    """Warn once per kind of left-out pixel, with the count and positions.

    Args:
        skipped (dict[str, list[list[int]]]): For each reason, the
            ``[row, column]`` of the left-out pixels. Only
            ``"no_correlation_peak"`` and ``"too_many_breaths_skipped"`` give
            a warning.
        function (str): Name of the calling function, for the message.

    Warns:
        RuntimeWarning: Once for each of these two reasons that has pixels.
    """

    messages = {
        NO_CORRELATION_PEAK: (
            "have no cross-correlation peak with the waveform, so they have no "
            "breaths. This can come from unstable pixel impedance"
        ),
        TOO_MANY_BREATHS_SKIPPED: (
            f"have more than {ALLOW_FRACTION_BREATHS_SKIPPED:.0%} of their "
            "breaths left out, so they have no breaths"
        ),
    }
    for reason, message in messages.items():
        positions = skipped[reason]
        if positions:
            listed = ", ".join(f"({row}, {column})" for row, column in positions)
            warnings.warn(
                f"{function}: {len(positions)} pixel(s) {message}: {listed}.",
                RuntimeWarning,
                stacklevel=3,
            )


def _warn_for_no_pixel_breaths(
    skipped: dict[str, list[list[int]]], *, function: str
) -> None:
    """Warn that no pixel has a breath, with the number of pixels per reason.

    Args:
        skipped (dict[str, list[list[int]]]): For each reason, the
            ``[row, column]`` of the left-out pixels.
        function (str): Name of the calling function, for the message.

    Warns:
        RuntimeWarning: Always.
    """

    reasons = {
        NO_TIV: "the TIV is NaN in every breath for the sign of the pixel",
        FLAT: "the impedance does not change",
        NO_CORRELATION_PEAK: "no cross-correlation peak",
        TOO_MANY_BREATHS_SKIPPED: (
            f"more than {ALLOW_FRACTION_BREATHS_SKIPPED:.0%} of the breaths left out"
        ),
    }
    counts = "; ".join(
        f"{len(skipped[reason])} pixel(s): {text}"
        for reason, text in reasons.items()
        if skipped[reason]
    )
    warnings.warn(
        f"{function}: no pixel has a breath, so every time of the result is "
        f"NaN. {counts}.",
        RuntimeWarning,
        stacklevel=3,
    )
