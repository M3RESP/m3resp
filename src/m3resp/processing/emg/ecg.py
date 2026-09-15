"""ECG-artifact-handling methods of `ReSurfEMGAdapter`.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from ReSurfEMG.

    Source:     https://github.com/resurfemg-org/ReSurfEMG
    Revision:   m3resp-integration (c63668689030e4581d5f985e7d09d3a8c01e7a77)
    Original:   resurfemg/preprocessing/ecg_removal::detect_ecg_peaks, gating,
                wavelet_denoising
    Copyright:  Copyright (c) 2022 Netherlands eScience Center and
                University of Twente
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - ``detect_ecg_peaks`: compacted percentiles calculation into a single line.
    - `gating`: renamed to `gate_ecg` and refactored to use a `_GateContext` dataclass
        and related helpers.
    - `wavelet_denoising`: renamed to `wavelet_denoise_ecg`
    - Parameters renamed and reorganized as keyword-only arguments.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks

from m3resp.processing.filters import bandpass_filter
from m3resp.processing.windows import rolling_rms  # TODO: check correctness

from .shared import (
    _require_1d_array,
    _require_index_array,
    _require_integer_valued_sample_frequency,
)

_GATE_FILL_ZEROS = 0
_GATE_FILL_INTERP = 1
_GATE_FILL_PRIOR_MEAN = 2
_GATE_FILL_RMS = 3


@dataclass
class _GateContext:
    array_original: np.ndarray
    array_gated: np.ndarray
    peaks: np.ndarray
    max_samples: int
    gate_width_samples: int
    half_gate_width: float
    gate_start_indexes: np.ndarray
    gate_end_indexes: np.ndarray
    gate_mask: np.ndarray
    gated_indexes: np.ndarray


def _find_corresponding_peaks(gating_context: _GateContext) -> np.ndarray:
    return np.clip(
        a=np.searchsorted(
            a=gating_context.gate_start_indexes,
            v=gating_context.gated_indexes,
            side="right",
        )
        - 1,
        a_min=0,
        a_max=len(gating_context.peaks) - 1,
    )


def _interpolate_missing_samples(
    gating_context: _GateContext, mask: np.ndarray
) -> None:
    missing = gating_context.gated_indexes[mask]
    if missing.size == 0:
        return
    # first, handle the boundary conditions
    if missing[0] == 0:
        gating_context.array_gated[0] = 0.0
        missing = missing[1:]
    if missing.size and missing[-1] == gating_context.max_samples - 1:
        gating_context.array_gated[-1] = 0.0
        missing = missing[:-1]
    # if no remaining samples need interpolation, return
    if missing.size == 0:
        return
    known = np.ones(shape=gating_context.max_samples, dtype=bool)
    known[missing] = False
    gating_context.array_gated[missing] = np.interp(
        missing, np.flatnonzero(known), gating_context.array_gated[known]
    )


def _gate_fill_zeros(gating_context: _GateContext) -> np.ndarray:
    gating_context.array_gated[np.array(gating_context.gate_mask, dtype=np.bool_)] = 0.0
    return gating_context.array_gated


def _gate_fill_interp(gating_context: _GateContext) -> np.ndarray:
    if gating_context.gated_indexes.size == 0:
        return gating_context.array_gated
    # get the indices of the samples right before and after each gate
    pre = np.clip(
        gating_context.gate_start_indexes - 1,
        0,
        gating_context.max_samples,
        dtype=np.int64,
    )
    post = np.clip(
        gating_context.gate_end_indexes + 1,
        0,
        gating_context.max_samples,
        dtype=np.int64,
    )
    # pre_values and post_values are the corresponding values of the array for each
    # pre and post gate index, and 0 elsewhere
    pre_values = np.where(pre, gating_context.array_original[pre], 0.0)
    post_values = np.where(post, gating_context.array_original[post], 0.0)
    # find the peak each gate is associated with
    parent_peak = _find_corresponding_peaks(gating_context)
    # compute the interpolation fraction for each element within the gate.
    # formula: (distance of the sample from the peak + half gate width) / gate width
    frac = (
        gating_context.gated_indexes
        - gating_context.peaks[parent_peak]
        + gating_context.half_gate_width
    ) / float(gating_context.gate_width_samples)
    # finally, interpolate
    gating_context.array_gated[gating_context.gated_indexes] = (
        1.0 - frac
    ) * pre_values[parent_peak] + frac * post_values[parent_peak]
    return gating_context.array_gated


def _windowed_mean(gating_context: _GateContext, starts: np.ndarray) -> np.ndarray:
    """Compute the window mean.

    Compute the mean of the original signal over a window of width
    `gate_width_samples` starting at each index in
    `mean_window_start_indexes`.
    """
    idx = starts[:, None] + np.arange(gating_context.gate_width_samples)
    oob = (idx < 0) | (idx >= gating_context.max_samples)  # was `or` — a bug
    values = gating_context.array_original[
        np.clip(idx, 0, gating_context.max_samples - 1)
    ]
    masked = np.ma.array(values, mask=oob | np.isnan(values))
    return masked.mean(axis=1).filled(np.nan)


def _gate_fill_prior_mean(gating_context: _GateContext) -> np.ndarray:
    # get the starting index of the window over which to compute
    # the mean for each gate
    prior_start = gating_context.peaks - gating_context.half_gate_width * 3
    prior_means = _windowed_mean(gating_context, prior_start)
    post_means = _windowed_mean(
        gating_context, gating_context.peaks + gating_context.half_gate_width
    )
    #  use the prior means where available, otherwise use the post means
    fill = np.where(prior_start < 0, post_means, prior_means)
    parent_peak = _find_corresponding_peaks(gating_context)
    gating_context.array_gated[gating_context.gated_indexes] = fill[parent_peak]
    return gating_context.array_gated


def _gate_fill_rms(gating_context: _GateContext) -> np.ndarray:

    rms = np.copy(gating_context.array_gated)
    rms[np.array(gating_context.gate_mask, dtype=np.bool_)] = np.nan
    rms = rolling_rms(values=rms, window_length=gating_context.gate_width_samples)
    window = max(1, 2 * int(1.5 * gating_context.gate_width_samples) + 1)
    closed = "neither" if gating_context.gate_width_samples % 2 == 0 else "left"
    local_mean = (
        pd.Series(rms)
        .rolling(window=window, center=True, min_periods=1, closed=closed)
        .mean()
        .to_numpy()
    )
    fill_values = local_mean[gating_context.gated_indexes]
    # identify missing samples
    missing = np.isnan(fill_values)
    gating_context.array_gated[gating_context.gated_indexes[~missing]] = fill_values[
        ~missing
    ]

    # the missing samples need to be filled
    # by interpolating the values from the surrounding samples
    _interpolate_missing_samples(gating_context, missing)
    return gating_context.array_gated


_GATE_FILLERS = {
    _GATE_FILL_ZEROS: _gate_fill_zeros,
    _GATE_FILL_INTERP: _gate_fill_interp,
    _GATE_FILL_PRIOR_MEAN: _gate_fill_prior_mean,
    _GATE_FILL_RMS: _gate_fill_rms,
}


class _EcgMixin:
    def detect_ecg_peaks(
        self,
        signal: np.ndarray,
        *,
        sample_frequency: float,
        peak_fraction: float = 0.4,
        peak_width_samples: int | None = None,
        peak_distance_samples: int | None = None,
        apply_bandpass_filter: bool = True,
    ) -> np.ndarray:
        """Detect ECG peak sample indices in `signal`.

        Args:
            signal: ECG signals to detect the ECG peaks in (ECG or an
        ECG-contaminated EMG channel).
            sample_frequency (float): Sampling rate of the EMG signals.
            peak_fraction (float): ECG peaks amplitude threshold relative to the
                specified fraction of the min-max values in the ECG signal.
            peak_width_samples (int, optional): ECG peaks width threshold in samples.
            peak_distance_samples (int, optional): Minimum time between ECG peaks,
                in samples.
            apply_bandpass_filter (bool): Bandpass filter the ecg_raw between 1-500 Hz
                before peak detection.

        Returns:
            numpy.ndarray: ECG peak indices.
        """
        _signal = _require_1d_array("signal", signal)
        fs = _require_integer_valued_sample_frequency(sample_frequency)

        if peak_width_samples is None:
            peak_width_samples = fs // 1000

        if peak_distance_samples is None:
            peak_distance_samples = fs // 3

        if apply_bandpass_filter:
            lp_cutoff = min([500, 0.95 * fs / 2])
            cutoff_frequencies = (1, lp_cutoff)
            _signal = bandpass_filter(
                _signal, cutoff_frequency=cutoff_frequencies, sample_frequency=fs
            )

        _signal_rms = rolling_rms(_signal, window_length=fs // 200)
        [min_rms, max_rms] = np.percentile(_signal_rms, [1, 99])

        peak_height = peak_fraction * (max_rms - min_rms)
        return find_peaks(
            _signal_rms,
            height=peak_height,
            width=peak_width_samples,
            distance=peak_distance_samples,
        )[0]

    def _build_gate_context(
        self,
        signal: np.ndarray,
        peak_indices: np.ndarray,
        gate_width_samples: int,
        fill_method: int,
    ) -> _GateContext:
        array_original = _require_1d_array("signal", signal)
        peaks = np.sort(_require_index_array("peak_indices", peak_indices))
        max_samples = len(array_original)
        # all methods require the gate mask, so we can compute it here
        # to avoid rewriting the same logic in each method.
        # the half_gate_width is computed via integer division for methods 0, 1, and 2,
        # but via float division for methods 3 and 4.
        # The rest of the mask computation is identical for all methods.
        half_gate_width = (
            gate_width_samples / 2
            if fill_method == _GATE_FILL_RMS  # was `is` — use `==`
            else gate_width_samples // 2
        )
        starts = np.clip(peaks - half_gate_width, 0, max_samples, dtype=np.int64)
        ends = np.clip(peaks + half_gate_width, 0, max_samples, dtype=np.int64)
        delta = np.zeros(max_samples + 1, dtype=np.int64)
        np.add.at(delta, starts, 1)  # +1 entering each gate
        np.add.at(delta, ends, -1)  # -1 exiting each gate
        # now delta is >0 when a gate starts, and < 0 when a gate ends.
        # Compute the boolean mask of the gates.
        # The cumulative sum of delta is >0 inside a gate, and 0 outside.
        gate_mask = np.cumsum(delta)[:max_samples] > 0
        # get the indexes of the samples that are inside the gates
        gated_indexes = np.nonzero(np.array(gate_mask, dtype=np.bool_))[0]

        return _GateContext(
            array_original,
            np.copy(array_original),
            peaks,
            max_samples,
            gate_width_samples,
            half_gate_width,
            starts,
            ends,
            gate_mask,
            gated_indexes,
        )

    def gate_ecg(
        self,
        signal: np.ndarray,
        peak_indices: np.ndarray,
        *,
        gate_width_samples: int = 205,
        fill_method: int = 1,
    ) -> np.ndarray:
        """Gating removal of QRS complexes.

        Eliminate peaks (e.g. QRS) from emg_raw using gates
        of width gate_width. The gate either filled by zeros or interpolation.
        The filling method for the gate is encoded as follows:
        - 0: Filled with zeros
        - 1: Interpolation of the samples before and after the gate
        - 2: Fill with average of prior segment if exists,
            otherwise fill with the average of post segment
        - 3: Fill with running average of RMS (default)

        Args:
            signal (numpy.ndarray): Signal to process.
            peak_indices (list or numpy.ndarray): List of individual peak index places
                to be gated.
            gate_width_samples (int): Width of the gate.
            fill_method (int): Filling method of gate.

        Returns:
            numpy.ndarray: The gated result.
        """
        if fill_method not in _GATE_FILLERS:
            msg = (
                f"fill_method must be one of {sorted(_GATE_FILLERS)};"
                f" got {fill_method!r}."
            )
            raise ValueError(msg)
        if gate_width_samples <= 0:
            msg_0 = f"gate_width_samples must be positive; got {gate_width_samples!r}."
            raise ValueError(msg_0)
        # build the gating context: depending on the selected fill method, construct the
        #  necessary gating mask and other relevant data structures
        gating_context = self._build_gate_context(
            signal, peak_indices, gate_width_samples, fill_method
        )
        # now, fill the gates according to the selected method
        return _GATE_FILLERS[fill_method](gating_context)

    # TODO
    def wavelet_denoise_ecg(
        self,
        signal: np.ndarray,
        peak_indices: np.ndarray,
        *,
        sample_frequency: float,
        hard_thresholding: bool = True,
        levels: int = 4,
        wavelet_type: str = "db2",
        fixed_threshold: float = 4.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Wavelet denoising of ECG artifacts.

        Shrinkage Denoising using a-trous wavelet decomposition (SWT). NB: This
        function assumes that the emg_raw has already been preprocessed for
        removal of baseline, powerline, and aliasing. N.B. This is a Python
        implementation of the SWT, as previously implemented in MATLAB by Jan
        Graßhoff. See Copyright notice below.
        --------------------------------------------------------------------------
        Copyright 2019 Institute for Electrical Engineering in Medicine,
        University of Luebeck
        Jan Graßhoff

        Permission is hereby granted, free of charge, to any person obtaining a
        copy of this software and associated documentation files (the "Software"),
        to deal in the Software without restriction, including without limitation
        the rights to use, copy, modify, merge, publish, distribute, sublicense,
        and/or sell copies of the Software, and to permit persons to whom the
        Software is furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included
        in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
        OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
        FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
        DEALINGS IN THE SOFTWARE.

        Args:
            signal: 1D raw EMG data.
            peak_indices: List of R-peak indices.
            sample_frequency (int): Sampling rate of emg_raw.
            hard_thresholding (bool): True for hard thresholding (default),
                False for soft.
            levels (int): Decomposition level (default: 4).
            wavelet_type (str): Wavelet type (default: "db2", see pywt.swt help).
            fixed_threshold (float): Fixed threshold multiplier for wavelet
                coefficients.

        Returns:
            tuple:
                - cleaned_signal (numpy.ndarray): Cleaned EMG signal.
                - decomposition (numpy.ndarray): Wavelet decomposition.
                - thresholds (numpy.ndarray): Threshold values.
                - gate_mask (numpy.ndarray): Gated signal based on R-peaks,
                    where gate == 1.
        """

        def estimate_noise(signal: np.ndarray, window_length: int) -> np.ndarray:
            """Estimate noise level.

            Args:
                    signal (numpy.ndarray): Wavelet-decomposed signal.
                    window_length (int): Window length for noise estimation.

            Returns:
                    numpy.ndarray: Estimated noise level.
            """
            nb_levels = signal.shape[0]
            std_estimate = np.zeros_like(signal)

            for nb_level in range(nb_levels):
                # estimate std from MAD: std ~ MAD/0.6745
                std_estimate[nb_level, :] = (
                    pd.Series(np.abs(signal[nb_level, :]))
                    .rolling(window=window_length, min_periods=1, center=True)
                    .median()
                    .to_numpy(dtype=float)
                    / 0.6745
                )

                # current on- and offset effects
                std_estimate[nb_level, : window_length // 2] = std_estimate[
                    nb_level, window_length // 2
                ]
                std_estimate[nb_level, -window_length // 2] = std_estimate[
                    nb_level, -window_length // 2
                ]
            return std_estimate

        def get_gate_windows(
            rpeak_bool_vec: np.ndarray, window_length: int
        ) -> np.ndarray:
            """Generate gate windows for the peaks.

            Args:
                    rpeak_bool_vec (numpy.ndarray): 1D array, where R-peak location == 1
                    window_length (int): Number of samples to gate around peaks.

            Returns:
                    numpy.ndarray: Gated signal based on R-peaks, where gate == 1.
            """
            window_length = int(np.floor(window_length / 2) * 2)
            rpeak_indexes = np.where(rpeak_bool_vec == 1)[0]
            gate_windows = np.zeros_like(rpeak_bool_vec)
            for rpeak_index in rpeak_indexes:
                gate_windows[
                    max(rpeak_index - window_length // 2, 0) : min(
                        rpeak_index + window_length // 2, len(rpeak_bool_vec)
                    )
                ] = 1

            return gate_windows

        def threshold_wavelets(
            data: np.ndarray, hard_thresholding: bool, threshold: float | np.ndarray
        ) -> np.ndarray:
            """Threshold wavelet coefficients.

            Apply thresholding to data based on 'soft' or 'hard' option.

            Args:
                    data (numpy.ndarray): Input data.
                    hard_thresholding (bool): True for hard thresholding, False for soft
                    threshold (float): Threshold value.

            Returns:
                    numpy.ndarray: Thresholded data.
            """
            if hard_thresholding is True:
                # Hard thresohlding
                data[np.abs(data) < threshold] = 0
            elif hard_thresholding is False:
                # Soft thresholding
                data = np.sign(data) * np.maximum(np.abs(data) - threshold, 0)
            return data

        _signal = _require_1d_array("signal", signal)
        _peak_indices = _require_index_array("peak_indices", peak_indices)
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        if levels <= 0:
            msg = f"levels must be positive; got {levels!r}."
            raise ValueError(msg)

        # calculate the gate windows
        rpeak_bool_vec = np.zeros(_signal.shape, dtype=bool)
        rpeak_bool_vec[_peak_indices] = 1
        gate_bool_array = get_gate_windows(rpeak_bool_vec, window_length=fs // 10)

        # Signal extension by zero padding
        pow_2_levels = 2**levels
        n_samples = len(_signal)
        n_samples_extended = int(np.ceil(n_samples / pow_2_levels) * pow_2_levels)
        zero_padding = np.zeros(n_samples_extended - n_samples)
        padded_signal = np.concatenate((_signal, zero_padding))
        gate_bool_array = np.concatenate((gate_bool_array, zero_padding))

        # Wavelet decomposition of emg_raw using Stationary Wavelet Transform (SWT)
        coeffs = pywt.swt(data=padded_signal, wavelet=wavelet_type, level=levels)
        coeffs_unpacked = np.array([[sub_band[0], sub_band[1]] for sub_band in coeffs])
        swc = np.vstack(
            (
                coeffs_unpacked[:, 1, :],
                coeffs_unpacked[levels - 1, 0, :],
            )
        )

        # Gate out R-peaks in wavelet subbands
        coeffs_gated = np.copy(swc)
        coeffs_gated[:, gate_bool_array == 1] = np.nan

        # Custom threshold coefficients
        window_length = 15 * fs
        std_estimate = estimate_noise(coeffs_gated[:-1], window_length=window_length)

        thresholds = np.zeros_like(swc)
        wxd = np.copy(coeffs_unpacked)

        for level in range(levels):
            threshold = fixed_threshold & std_estimate[level, :]
            wxd[level, 1, :] = threshold_wavelets(
                coeffs_unpacked[level, 1, :], hard_thresholding, threshold
            )

        # Wavelet reconstruction
        reconstructed_signal = pywt.iswt(
            coeffs=[tuple(sub_band) for sub_band in wxd], wavelet=wavelet_type
        )

        # Return results
        return (
            _signal - reconstructed_signal[:n_samples],
            swc,
            thresholds[:, :n_samples],
            gate_bool_array[:n_samples],
        )
