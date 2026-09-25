"""Quality-assessment methods of `ReSurfEMGAdapter`.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from ReSurfEMG.

    Source:     https://github.com/resurfemg-org/ReSurfEMG
    Revision:   m3resp-integration (c63668689030e4581d5f985e7d09d3a8c01e7a77)
    Original:   resurfemg/portprocessing/quality_assessment.py::snr_pseudo, pocc_quality
                interpeak_dist, percentage_under_baseline, detect_local_high_aub,
                detect_extreme_time_products, detect_non_consecutive_manoeuvres,
                evaluate_bell_curve_error, evaluate_event_timing,
                evaluate_respiratory_rates;
    Copyright:  Copyright (c) 2022 Netherlands eScience Center and
                University of Twente
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - `evaluate_bell_curve_error`: changed the formulation of t_delta computation
        the original np.median(peak_array[1:] - peak_array[:-1]) to
        np.median(np.diff(peak_array)). Simplified the nan/inf checks.
    - `interpeak_dist`: renamed to `interpeak_distance`.
    - `detect_extreme_time_products`: simplified the return statement to use logical
        operators instead of np.all and np.array.
    - Parameters renamed and reorganized as keyword-only arguments.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

from m3resp.processing.metrics import area_under_baseline, window_integral

from .shared import (
    _require_1d_array,
    _require_equal_length,
    _require_finite_positive,
    _require_index_array,
    _require_integer_valued_sample_frequency,
    _require_percentile,
)


def _bell_curve(
    x_values: np.ndarray, amplitudes: float, time_shift: float, steepness_factor: float
) -> np.ndarray:
    """Calculate a shifted, smoothed and amplified bell curve.

    This function calculates a bell curve on the samples of x, shifted by
    b, amplified by a for a standard amplitude of 1.

    Args:
        x_values (numpy.ndarray): X values to calculate the bell_curve for.
        amplitudes (float): Amplitude of the bell-curve.
        time_shift (float): Time shift of the bell-curve along the x-axis.
        steepness_factor (float): Steepness factor of bell-curve.

    Returns:
        numpy.ndarray: Bell curve values.
    """
    return amplitudes * np.exp(-((x_values - time_shift) ** 2) / steepness_factor**2)


class _QualityMixin:
    def snr_pseudo(
        self,
        envelope: np.ndarray,
        peak_indices: np.ndarray,
        baseline: np.ndarray,
        *,
        sample_frequency: float,
    ) -> np.ndarray:
        # DONE
        """Compute the pseudo peak-by-peak signal-to-noise ratio.

        Pseudo signal-to-noise ratio per peak: peak height relative to a
        local baseline window.

        Approximate the signal-to-noise ratio (SNR) of the signal based
        on the peak height relative to the baseline.

        Args:
            envelope: Envelope of the signal to evaluate.
            peak_indices: List of individual peak indices.
            baseline: Baseline signal to evaluate SNR to.
            sample_frequency: Sampling rate.

        Returns:
            numpy.ndarray: The SNR per peak.
        """
        array = _require_1d_array("envelope", envelope)
        peaks = _require_index_array("peak_indices", peak_indices)
        baseline_array = _require_1d_array("baseline", baseline)
        fs = _require_integer_valued_sample_frequency(sample_frequency)

        peak_heights = np.zeros(shape=(len(peaks),))
        noise_heights = np.zeros(shape=(len(peaks),))

        for peak_number, peak_index in enumerate(peaks):
            peak_heights[peak_number] = array[peak_index]
            peak_start_index = max([0, peak_index - fs])
            peak_end_index = min([len(array), peak_index + fs])
            noise_heights[peak_number] = np.median(
                baseline_array[peak_start_index:peak_end_index]
            )

        return np.divide(peak_heights, noise_heights)

    def pocc_quality(
        self,
        pressure_signal: np.ndarray,
        pocc_peak_indices: np.ndarray,
        pocc_end_indices: np.ndarray,
        pocc_time_products: np.ndarray,
        *,
        dp_up_10_threshold: float = 0.0,
        dp_up_90_threshold: float = 2.0,
        dp_up_90_norm_threshold: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray]:
        # DONE
        """Evaluate the quality of occlusion pressures (Pocc).

        Evaluation of occlusion pressure (Pocc) quality, in accordance with Warnaar
        et al. (2024). Poccs are labelled invalid if too many negative deflections
        happen in the upslope (first decile < 0), or if the upslope is to steep
        (high absolute or relative 9th decile), indicating occlusion release before
        the patient's inspiriratory effort has ended.

        Args:
            pressure_signal: Airway pressure signal.
            pocc_peak_indices: List of individual peak indices.
            pocc_end_indices: List of individual peak end indices.
            pocc_time_products: List of pressure-time products for each occlusion.
            dp_up_10_threshold (float): Minimum first decile of upslope after the
                (negative) occlusion pressure peak.
            dp_up_90_threshold (float): Maximum 9th decile of upslope after the
                (negative) occlusion pressure peak.
            dp_up_90_norm_threshold (float): Maximum normalised 9th decile of upslope
                after the (negative) occlusion pressure peak.

        Returns:
            tuple:
                - numpy.ndarray: `valid_poccs`, Boolean list of valid Pocc peaks.
                - numpy.ndarray: `criteria_matrix`, Matrix of the calculated criteria.
                    rows are `dp_up_10`, `dp_up_90`, `dp_up_90_norm`, in that order.
        """
        _pressure_signal = _require_1d_array("pressure_signal", pressure_signal)
        _pocc_peak_indices = _require_index_array(
            "pocc_peak_indices", pocc_peak_indices
        )
        _pocc_end_indices = _require_index_array("pocc_end_indices", pocc_end_indices)
        _pocc_time_products = _require_1d_array(
            "pocc_time_products", pocc_time_products
        )
        _require_equal_length(
            ("pocc_peak_indices", _pocc_peak_indices),
            ("pocc_end_indices", _pocc_end_indices),
            ("pocc_time_products", _pocc_time_products),
        )

        dp_up_10 = np.zeros(shape=(len(_pocc_peak_indices),))
        dp_up_90 = np.zeros(shape=(len(_pocc_peak_indices),))
        dp_up_90_norm = np.zeros(shape=(len(_pocc_peak_indices),))

        for peak_index, pocc_peak in enumerate(_pocc_peak_indices):
            pocc_end_index = _pocc_end_indices[peak_index]
            dp = (
                _pressure_signal[pocc_peak + 1 : pocc_end_index]
                - _pressure_signal[pocc_peak : pocc_end_index - 1]
            )
            dp_up_10[peak_index] = np.percentile(dp, 10)
            dp_up_90[peak_index] = np.percentile(dp, 90)
            dp_up_90_norm[peak_index] = dp_up_90[peak_index] / np.sqrt(
                _pocc_time_products[peak_index]
            )

        criteria_matrix = np.array([dp_up_10, dp_up_90, dp_up_90_norm])
        criteria_bool_matrix = np.array(
            [
                dp_up_10 <= dp_up_10_threshold,
                dp_up_90 > dp_up_90_threshold,
                dp_up_90_norm > dp_up_90_norm_threshold,
            ]
        )

        valid_poccs = ~np.any(criteria_bool_matrix, axis=0)
        return np.asarray(valid_poccs), np.asarray(criteria_matrix)

    def interpeak_distance(
        self,
        ecg_peak_indices: np.ndarray,
        emg_peak_indices: np.ndarray,
        *,
        threshold: float = 1.1,
    ) -> bool:
        """Check the interpeak distance ratio of ECG and EMG peaks.

        Calculate the median interpeak distances for ECG and EMG and check if their
        ratio is above the given threshold, i.e. if cardiac frequency is higher
        than respiratory frequency (True) (Warnaar et al. 2024)

        Args:
            ecg_peak_indices: Indices of ECG peaks.
            emg_peak_indices: Indices of EMG peaks.
            threshold (float): The threshold value to compare against. Default is 1.1.

        Returns:
            bool: Boolean value indicating if the interpeak distance is valid.
        """
        _ecg_peak_indices = _require_index_array("ecg_peak_indices", ecg_peak_indices)
        _emg_peak_indices = _require_index_array("emg_peak_indices", emg_peak_indices)
        if len(_ecg_peak_indices) < 2 or len(_emg_peak_indices) < 2:
            msg = (
                "interpeak_distance needs at least two peaks in each of "
                "ecg_peak_indices and emg_peak_indices."
            )
            raise ValueError(msg)

        t_delta_ecg_med = np.median(np.diff(_ecg_peak_indices))
        t_delta_emg_med = np.median(np.diff(_emg_peak_indices))
        t_delta_relative = t_delta_emg_med / t_delta_ecg_med

        return bool(t_delta_relative >= threshold)

    def percentage_under_baseline(
        self,
        signal: np.ndarray,
        peak_indices: np.ndarray,
        start_indices: np.ndarray,
        end_indices: np.ndarray,
        baseline: np.ndarray,
        *,
        sample_frequency: float,
        aub_window_samples: int | None = None,
        reference_signal: np.ndarray | None = None,
        aub_threshold: float = 40.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the percentage of the area under the baseline for each peak.

        Calculate the percentage area under the baseline, in accordance with
        Warnaar et al. (2024).

        Args:
            signal: Signal in which the peaks are detected.
            peak_indices: List of individual peak indices.
            start_indices: List of individual peak start indices.
            end_indices: List of individual peak end indices.
            baseline: Running baseline of the signal.
            sample_frequency (float): Sampling frequency of the signal.
            aub_window_samples (int, optional): Number of samples before and after
                peak_indices to look for the nadir.
            reference_signal (numpy.ndarray, optional): Signal in which the nadir
                is searched.
            aub_threshold (float): Maximum AUB error percentage for a peak.

        Returns:
            tuple:
                - numpy.ndarray: `valid_time_products`: Boolean list of valid
                    time products.
                - numpy.ndarray: `percentages_aub`: List of calculated AUB percentages.
                - numpy.ndarray: `reference_values`: Reference signal nadir
                    values per breath.
        """
        _signal = _require_1d_array("signal", signal)
        _peak_indices = _require_index_array("peak_indices", peak_indices)
        _start_indices = _require_index_array("start_indices", start_indices)
        _end_indices = _require_index_array("end_indices", end_indices)
        _baseline = _require_1d_array("baseline", baseline)
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        _require_equal_length(
            ("peak_indices", _peak_indices),
            ("start_indices", _start_indices),
            ("end_indices", _end_indices),
        )
        _reference_signal = (
            _signal
            if reference_signal is None
            else _require_1d_array("reference_signal", reference_signal)
        )

        if aub_window_samples is None:
            aub_window_samples = 5 * fs

        time_products = window_integral(
            values=_signal,
            sample_frequency=fs,
            start_indices=_start_indices,
            end_indices=_end_indices,
            baseline=_baseline,
        )

        areas_under_baseline, references = area_under_baseline(
            values=_signal,
            sample_frequency=fs,
            peak_indices=_peak_indices,
            start_indices=_start_indices,
            end_indices=_end_indices,
            window=aub_window_samples,
            baseline=_baseline,
            reference_values=_reference_signal,
        )

        percentages = (
            areas_under_baseline / (time_products + areas_under_baseline) * 100
        )
        valid = percentages <= aub_threshold

        return np.asarray(valid), np.asarray(percentages), np.asarray(references)

    def detect_local_high_aub(
        self,
        aub_values: np.ndarray,
        *,
        threshold_percentile: float = 75.0,
        threshold_factor: float = 4.0,
    ) -> np.ndarray:
        """Flag area-under-baseline values that are locally high outliers.

        Detect local upward deflections in the area under the baseline.

        Args:
            aub_values (numpy.ndarray): List of area under the baseline values.
            threshold_percentile (float): Percentile for detecting high baseline.
            threshold_factor (float): Multiplication factor for threshold_percentile.

        Returns:
            numpy.ndarray: Boolean list of AUB values under threshold.
        """
        _aub_values = _require_1d_array("aub_values", aub_values)
        _require_percentile("threshold_percentile", threshold_percentile)

        threshold = threshold_factor * np.percentile(_aub_values, threshold_percentile)
        return np.asarray(_aub_values < threshold)

    def detect_extreme_time_products(
        self,
        time_products: np.ndarray,
        *,
        upper_percentile: float = 95.0,
        upper_factor: float = 10.0,
        lower_percentile: float = 5.0,
        lower_factor: float = 0.1,
    ) -> np.ndarray:
        """Detect extreme time products.

        Flag time-product (area above baseline) values outside the
        expected percentile-based bounds.

        Args:
            time_products (numpy.ndarray): List of time product values.
            upper_percentile (float): Percentile for detecting high time products.
            upper_factor (float): Multiplication factor for upper_percentile.
            lower_percentile (float): Percentile for detecting low time products.
            lower_factor (float): Multiplication factor for lower_percentile.
        """
        _time_products = _require_1d_array("time_products", time_products)
        _require_percentile("upper_percentile", upper_percentile)
        _require_percentile("lower_percentile", lower_percentile)

        upper_threshold = upper_factor * np.percentile(_time_products, upper_percentile)
        lower_threshold = lower_factor * np.percentile(_time_products, lower_percentile)

        return (_time_products < upper_threshold) & (_time_products > lower_threshold)

    def detect_non_consecutive_manoeuvres(
        self,
        ventilator_breath_indices: np.ndarray,
        manoeuvre_indices: np.ndarray,
    ) -> np.ndarray:
        """Detect non-consecutive manoeuvres.

        Flag manoeuvres (e.g. Pocc) with no supported ventilator breathS
        between them and the next manoeuvre.

        Input are the ventilator breaths, to be detected with the
        function post_processing.event_detecton.detect_supported_breaths
        If no supported breaths are detected in between two manoeuvres,
        valid_manoeuvres is "True".
        Note: fs of both signals should be equal.

        Args:
            ventilator_breath_indices (numpy.ndarray): List of supported breath indices.
            manoeuvre_indices (numpy.ndarray): List of manoeuvres indices.

        Returns:
            numpy.ndarray: Boolean list of valid manoeuvres.
        """
        _ventilator_breath_indices = _require_index_array(
            "ventilator_breath_indices", ventilator_breath_indices
        )
        _manoeuvre_indices = _require_index_array(
            "manoeuvre_indices", manoeuvre_indices
        )

        consecutive_manoeuvres = np.zeros(len(_manoeuvre_indices), dtype=bool)

        for index, _ in enumerate(_manoeuvre_indices):
            if index > 0:
                # check for supported breaths in between two Poccs
                intermediate_breaths = np.equal(
                    (_manoeuvre_indices[index - 1] < _ventilator_breath_indices),
                    (_ventilator_breath_indices < _manoeuvre_indices[index]),
                )

                # If no supported breaths are detected in between, detect a "double dip"
                intermediate_breath_count = np.sum(intermediate_breaths)
                if intermediate_breath_count > 0:
                    consecutive_manoeuvres[index] = False
                else:
                    consecutive_manoeuvres[index] = True
            else:
                consecutive_manoeuvres[index] = False

        return np.logical_not(consecutive_manoeuvres)

    def evaluate_bell_curve_error(
        self,
        peak_indices: np.ndarray,
        start_indices: np.ndarray,
        end_indices: np.ndarray,
        signal: np.ndarray,
        time_products: np.ndarray,
        *,
        sample_frequency: float,
        bell_window_samples: int | None = None,
        bell_threshold: float = 40.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the bell-curve error of signal peaks.

        Calculate the bell-curve error of signal peaks, in accordance with Warnaar
        et al. (2024).

        Args:
            peak_indices: List of peak indices.
            start_indices: List of onset indices.
            end_indices: List of offset indices.
            signal: Filtered signal.
            time_products: List of area under the curves per peak.
            sample_frequency (float): Sampling frequency of the signal.
            bell_window_samples (int, optional): Number of samples before and
                after peak_indices to look for the nadir.
            bell_threshold (float): Maximum bell error percentage for a valid peak.

        Returns:
            tuple:
                - numpy.ndarray: `valid_peak`: Boolean list of valid peaks.
                - numpy.ndarray: `percentage_bell_error`: Calculated bell errors
                    in percentage.
                - numpy.ndarray: `bell_error`: Calculated bell errors.
                - numpy.ndarray: `y_nadir`: Minimum value of the baseline.
                - numpy.ndarray: `fitted_parameters`: Fitted bell curve parameters.
        """
        _peak_indices = _require_index_array("peak_indices", peak_indices)
        _start_indices = _require_index_array("start_indices", start_indices)
        _end_indices = _require_index_array("end_indices", end_indices)
        _signal = _require_1d_array("signal", signal)
        _time_products = _require_1d_array("time_products", time_products)
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        _require_equal_length(
            ("peak_indices", _peak_indices),
            ("start_indices", _start_indices),
            ("end_indices", _end_indices),
            ("time_products", _time_products),
        )

        if bell_window_samples is None:
            bell_window_samples = 5 * fs

        time = np.array([i / fs for i in range(len(_signal))])

        bell_error = np.zeros((len(_peak_indices),))
        percentage_bell_error = np.zeros((len(_peak_indices),))
        fitted_parameters = np.zeros((len(_peak_indices), 3))
        y_nadir = np.zeros((len(_peak_indices),))

        for peak_number, (
            peak_index,
            start_index,
            end_index,
            time_product,
        ) in enumerate(
            zip(
                _peak_indices, _start_indices, _end_indices, _time_products, strict=True
            )
        ):
            baseline_start_index = max(0, peak_index - bell_window_samples)
            baseline_end_index = min(len(_signal) - 1, peak_index + bell_window_samples)
            y_nadir[peak_number] = np.min(
                _signal[baseline_start_index:baseline_end_index]
            )

            plus_index = (
                3 - (end_index - start_index) if end_index - start_index < 3 else 0
            )

            x_data = time[start_index : end_index + 1 + plus_index]
            y_data = (
                _signal[start_index : end_index + 1 + plus_index] - y_nadir[peak_number]
            )

            if not (np.isfinite(x_data).all() and np.isfinite(y_data).all()):
                msg = (
                    "NaNs or Infs detected in x_data or y_data for peak "
                    f"{peak_number}. Skipping this peak."
                )
                warnings.warn(msg)
                bell_error[peak_number] = np.nan
                continue

            # TODO port math operations module!
            try:
                popt, *_ = curve_fit(
                    _bell_curve,
                    x_data,
                    y_data,
                    bounds=(
                        [0.0, time[peak_index] - 0.5, 0.0],
                        [np.inf, time[peak_index] + 0.5, np.inf],
                    ),
                )
            except RuntimeError as e:
                msg = (
                    f"Curve fitting failed for peak {peak_number} with error: {e}."
                    "Skipping this peak."
                )
                warnings.warn(msg)
                bell_error[peak_number] = np.nan
                popt = np.array([np.nan, np.nan, np.nan])
                continue

            bell_error[peak_number] = trapezoid(
                np.abs(
                    _signal[start_index : end_index + 1]
                    - (
                        _bell_curve(time[start_index : end_index + 1], *popt)
                        + y_nadir[peak_number]
                    )
                ),
                dx=1 / fs,
            )

            percentage_bell_error[peak_number] = (
                bell_error[peak_number] / time_product * 100
            )
            fitted_parameters[peak_number, :] = popt

        valid_peak = percentage_bell_error <= bell_threshold

        return (
            np.asarray(valid_peak),
            np.asarray(bell_error),
            np.asarray(percentage_bell_error),
            np.asarray(y_nadir),
            np.asarray(fitted_parameters),
        )

    def evaluate_event_timing(
        self,
        first_event_times: np.ndarray,
        second_event_times: np.ndarray,
        *,
        min_delta: float = 0.0,
        max_delta: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the timing of two sets of events.

        Evaluate whether the timing of the events in `first_event_times` preceeds the
        events in `second_event_times` minimally by `min_delta` and maximally by
        `max_delta`. `first_event_times` and `second_event_times`
        should be the same length.

        Args:
            first_event_times: Timing of the events that should happen first.
            second_event_times: Timing of the events that should happen second.
            min_delta (float): The minimum time event 1 should precede event 2.
            max_delta (float, optional): The maximum time by which event 1
                should precede event 2.

        Returns:
            tuple:
                - correct_timing (numpy.ndarray): Boolean list of correct timing.
                - delta_time (numpy.ndarray): List of delta times between the events.
        """
        _first_event_times = _require_1d_array("first_event_times", first_event_times)
        _second_event_times = _require_1d_array(
            "second_event_times", second_event_times
        )
        _require_equal_length(
            ("first_event_times", _first_event_times),
            ("second_event_times", _second_event_times),
        )

        delta_time = _second_event_times - _first_event_times
        min_crit = delta_time >= min_delta
        if max_delta is not None:
            max_crit = delta_time <= max_delta
            correct_timing = np.all(np.array([min_crit, max_crit]), axis=0)
        else:
            correct_timing = min_crit

        return np.asarray(correct_timing), np.asarray(delta_time)

    def evaluate_respiratory_rates(
        self,
        emg_breath_indices: np.ndarray,
        recording_duration_seconds: float,
        ventilator_respiratory_rate: float,
        *,
        minimum_fraction: float = 0.1,
    ) -> tuple[float, bool]:
        """Evaluate the EMG respiratory rate against the ventilatory one.

        This function evaluates fraction of detected EMG breaths relative to the
        ventilatory respiratory rate.

        Args:
            emg_breath_indices: EMG breath indices.
            recording_duration_seconds (float): Recording time in seconds.
            ventilator_respiratory_rate (float): Ventilatory respiratory rate
                (breath/min).
            minimum_fraction (float): Required minimum detected fraction of EMG breaths.

        Returns:
            tuple:
                - detected_fraction (float): Fraction of detected EMG breaths.
                - criterion_met (bool): Boolean indicating if the fraction is above
                    the minimum.
        """
        _emg_breath_indices = _require_index_array(
            "emg_breath_indices", emg_breath_indices
        )
        _require_finite_positive(
            "recording_duration_seconds", recording_duration_seconds
        )
        _require_finite_positive(
            "ventilator_respiratory_rate", ventilator_respiratory_rate
        )

        detected_fraction = float(
            len(_emg_breath_indices)
            / (ventilator_respiratory_rate * recording_duration_seconds / 60)
        )
        criterion_met = detected_fraction >= minimum_fraction
        return float(detected_fraction), bool(criterion_met)
