"""Quality-assessment methods of `ReSurfEMGAdapter`."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._shared import (
    _emg_optional_dependency_error,
    _require_1d_array,
    _require_equal_length,
    _require_finite_positive,
    _require_index_array,
    _require_integer_valued_sample_frequency,
    _require_percentile,
)


class _QualityMixin:
    def snr_pseudo(
        self,
        envelope: Any,
        peak_indices: Any,
        baseline: Any,
        *,
        sample_frequency: float,
    ) -> np.ndarray:
        """Pseudo signal-to-noise ratio per peak: peak height relative to a
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

        try:
            from resurfemg.postprocessing.quality_assessment import snr_pseudo
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("envelope", envelope)
        peaks = _require_index_array("peak_indices", peak_indices)
        baseline_array = _require_1d_array("baseline", baseline)
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        return np.asarray(snr_pseudo(array, peaks, baseline_array, fs))

    def pocc_quality(
        self,
        pressure_signal: Any,
        pocc_peak_indices: Any,
        pocc_end_indices: Any,
        pocc_time_products: Any,
        *,
        dp_up_10_threshold: float = 0.0,
        dp_up_90_threshold: float = 2.0,
        dp_up_90_norm_threshold: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluation of occlusion pressure (Pocc) quality, in accordance with Warnaar
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
                - valid_poccs (numpy.ndarray): Boolean list of valid Pocc peaks.
                - criteria_matrix (numpy.ndarray): Matrix of the calculated criteria.
                    rows are `dp_up_10`, `dp_up_90`, `dp_up_90_norm`, in that order.
        """

        try:
            from resurfemg.postprocessing.quality_assessment import pocc_quality
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        pressure = _require_1d_array("pressure_signal", pressure_signal)
        peaks = _require_index_array("pocc_peak_indices", pocc_peak_indices)
        ends = _require_index_array("pocc_end_indices", pocc_end_indices)
        time_products = _require_1d_array("pocc_time_products", pocc_time_products)
        _require_equal_length(
            ("pocc_peak_indices", peaks),
            ("pocc_end_indices", ends),
            ("pocc_time_products", time_products),
        )
        valid_poccs, criteria_matrix = pocc_quality(
            pressure,
            peaks,
            ends,
            time_products,
            dp_up_10_threshold=dp_up_10_threshold,
            dp_up_90_threshold=dp_up_90_threshold,
            dp_up_90_norm_threshold=dp_up_90_norm_threshold,
        )
        return np.asarray(valid_poccs), np.asarray(criteria_matrix)

    def interpeak_distance(
        self,
        ecg_peak_indices: Any,
        emg_peak_indices: Any,
        *,
        threshold: float = 1.1,
    ) -> bool:
        """Whether the median ECG-to-median-EMG interpeak distance ratio is
        within `threshold` (Warnaar et al. 2024).

        Calculate the median interpeak distances for ECG and EMG and check if their
        ratio is above the given threshold, i.e. if cardiac frequency is higher
        than respiratory frequency (True)

        Args:
            ecg_peak_indices: Indices of ECG peaks.
            emg_peak_indices: Indices of EMG peaks.
            threshold (float): The threshold value to compare against. Default is 1.1.

        Returns:
            bool: Boolean value indicating if the interpeak distance is valid.
        """

        try:
            from resurfemg.postprocessing.quality_assessment import interpeak_dist
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        ecg_peaks = _require_index_array("ecg_peak_indices", ecg_peak_indices)
        emg_peaks = _require_index_array("emg_peak_indices", emg_peak_indices)
        if len(ecg_peaks) < 2 or len(emg_peaks) < 2:
            raise ValueError(
                "interpeak_distance needs at least two peaks in each of "
                "ecg_peak_indices and emg_peak_indices."
            )
        return bool(interpeak_dist(ecg_peaks, emg_peaks, threshold=threshold))

    def percentage_under_baseline(
        self,
        signal: Any,
        peak_indices: Any,
        start_indices: Any,
        end_indices: Any,
        baseline: Any,
        *,
        sample_frequency: float,
        aub_window_samples: int | None = None,
        reference_signal: Any | None = None,
        aub_threshold: float = 40.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the percentage area under the baseline, in accordance with
        Warnaar et al. (2024).

        Args:
            signal: Signal in which the peaks are detected.
            sample_frequency: Sampling frequency.
            peak_indices: List of individual peak indices.
            start_indices: List of individual peak start indices.
            end_indices: List of individual peak end indices.
            baseline: Running baseline of the signal.
            sample_frequency (float): Sampling frequency of the signal.
            aub_window_samples (int, optional): Number of samples before and after peak_indices
                to look for the nadir.
            reference_signal (numpy.ndarray, optional): Signal in which the nadir is searched.
            aub_threshold (float): Maximum AUB error percentage for a peak.

        Returns:
            tuple:
                - valid_time_products (numpy.ndarray): Boolean list of valid time products.
                - percentages_aub (numpy.ndarray): List of calculated AUB percentages.
                - reference_values (numpy.ndarray): Reference signal nadir values per breath.
        """

        try:
            from resurfemg.postprocessing.quality_assessment import (
                percentage_under_baseline,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("signal", signal)
        peaks = _require_index_array("peak_indices", peak_indices)
        starts = _require_index_array("start_indices", start_indices)
        ends = _require_index_array("end_indices", end_indices)
        baseline_array = _require_1d_array("baseline", baseline)
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        _require_equal_length(
            ("peak_indices", peaks),
            ("start_indices", starts),
            ("end_indices", ends),
        )
        reference = (
            None
            if reference_signal is None
            else _require_1d_array("reference_signal", reference_signal)
        )
        valid, percentages, reference_values = percentage_under_baseline(
            array,
            fs,
            peaks,
            starts,
            ends,
            baseline_array,
            aub_window_s=aub_window_samples,
            ref_signal=reference,
            aub_threshold=aub_threshold,
        )
        return np.asarray(valid), np.asarray(percentages), np.asarray(reference_values)

    def detect_local_high_aub(
        self,
        aub_values: Any,
        *,
        threshold_percentile: float = 75.0,
        threshold_factor: float = 4.0,
    ) -> np.ndarray:
        """Flag area-under-baseline values that are locally high outliers.

        Detect local upward deflections in the area under the baseline.

        Args:
            aub_values: List of area under the baseline values.
            threshold_percentile (float): Percentile for detecting high baseline.
            threshold_factor (float): Multiplication factor for threshold_percentile.

        Returns:
            numpy.ndarray: Boolean list of AUB values under threshold.
        """

        try:
            from resurfemg.postprocessing.quality_assessment import (
                detect_local_high_aub,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        values = _require_1d_array("aub_values", aub_values)
        _require_percentile("threshold_percentile", threshold_percentile)
        return np.asarray(
            detect_local_high_aub(
                values,
                threshold_percentile=threshold_percentile,
                threshold_factor=threshold_factor,
            )
        )

    def detect_extreme_time_products(
        self,
        time_products: Any,
        *,
        upper_percentile: float = 95.0,
        upper_factor: float = 10.0,
        lower_percentile: float = 5.0,
        lower_factor: float = 0.1,
    ) -> np.ndarray:
        """Flag time-product (area above baseline) values outside the
        expected percentile-based bounds.

        Args:
            time_products (numpy.ndarray): List of time product values.
            upper_percentile (float): Percentile for detecting high time products.
            upper_factor (float): Multiplication factor for upper_percentile.
            lower_percentile (float): Percentile for detecting low time products.
            lower_factor (float): Multiplication factor for lower_percentile.
        """

        try:
            from resurfemg.postprocessing.quality_assessment import (
                detect_extreme_time_products,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        values = _require_1d_array("time_products", time_products)
        _require_percentile("upper_percentile", upper_percentile)
        _require_percentile("lower_percentile", lower_percentile)
        return np.asarray(
            detect_extreme_time_products(
                values,
                upper_percentile=upper_percentile,
                upper_factor=upper_factor,
                lower_percentile=lower_percentile,
                lower_factor=lower_factor,
            )
        )

    def detect_non_consecutive_manoeuvres(
        self,
        ventilator_breath_indices: Any,
        manoeuvre_indices: Any,
    ) -> np.ndarray:
        """Detect non-consecutive manoeuvres.

        Flag manoeuvres (e.g. Pocc) with no supported ventilator breath
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

        try:
            from resurfemg.postprocessing.quality_assessment import (
                detect_non_consecutive_manoeuvres,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        breaths = _require_index_array(
            "ventilator_breath_indices", ventilator_breath_indices
        )
        manoeuvres = _require_index_array("manoeuvre_indices", manoeuvre_indices)
        return np.asarray(detect_non_consecutive_manoeuvres(breaths, manoeuvres))

    def evaluate_bell_curve_error(
        self,
        peak_indices: Any,
        start_indices: Any,
        end_indices: Any,
        signal: Any,
        time_products: Any,
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
            bell_window_samples (int, optional): Number of samples before and after peak_indices
                to look for the nadir.
            bell_threshold (float): Maximum bell error percentage for a valid peak.

        Returns:
            tuple:
                - valid_peak (numpy.ndarray): Boolean list of valid peaks.
                - percentage_bell_error (numpy.ndarray): Calculated bell errors in percentage.
                - bell_error (numpy.ndarray): Calculated bell errors.
                - y_min (numpy.ndarray): Minimum value of the baseline.
                - fitted_parameters (numpy.ndarray): Fitted bell curve parameters.
        """

        try:
            from resurfemg.postprocessing.quality_assessment import (
                evaluate_bell_curve_error,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        peaks = _require_index_array("peak_indices", peak_indices)
        starts = _require_index_array("start_indices", start_indices)
        ends = _require_index_array("end_indices", end_indices)
        array = _require_1d_array("signal", signal)
        products = _require_1d_array("time_products", time_products)
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        _require_equal_length(
            ("peak_indices", peaks),
            ("start_indices", starts),
            ("end_indices", ends),
            ("time_products", products),
        )
        (
            valid_peak,
            percentage_bell_error,
            bell_error,
            y_min,
            fitted_parameters,
        ) = evaluate_bell_curve_error(
            peaks,
            starts,
            ends,
            array,
            fs,
            products,
            bell_window_s=bell_window_samples,
            bell_threshold=bell_threshold,
        )
        return (
            np.asarray(valid_peak),
            np.asarray(percentage_bell_error),
            np.asarray(bell_error),
            np.asarray(y_min),
            np.asarray(fitted_parameters),
        )

    def evaluate_event_timing(
        self,
        first_event_times: Any,
        second_event_times: Any,
        *,
        min_delta: float = 0.0,
        max_delta: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the timing of two sets of events.

        Evaluate whether the timing of the events in `first_event_times` preceeds the
        events in `second_event_times` minimally by `min_delta` and maximally by
        `max_delta`. `first_event_times` and `second_event_times` should be the same length.

        Args:
            first_event_times: Timing of the events that should happen first.
            second_event_times: Timing of the events that should happen second.
            min_delta (float): The minimum time event 1 should precede event 2.
            max_delta (float, optional): The maximum time event 1 should precede event 2.

        Returns:
            tuple:
                - correct_timing (numpy.ndarray): Boolean list of correct timing.
                - delta_time (numpy.ndarray): List of delta times between the events.
        """

        try:
            from resurfemg.postprocessing.quality_assessment import (
                evaluate_event_timing,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        first = _require_1d_array("first_event_times", first_event_times)
        second = _require_1d_array("second_event_times", second_event_times)
        _require_equal_length(
            ("first_event_times", first), ("second_event_times", second)
        )
        correct_timing, delta_time = evaluate_event_timing(
            first,
            second,
            delta_min=min_delta,
            delta_max=max_delta,
        )
        return np.asarray(correct_timing), np.asarray(delta_time)

    def evaluate_respiratory_rates(
        self,
        emg_breath_indices: Any,
        recording_duration_seconds: float,
        ventilator_respiratory_rate: float,
        *,
        minimum_fraction: float = 0.1,
    ) -> tuple[float, bool]:
        """Evaluate the respiratory rate of detected EMG breaths relative to the ventilatory respiratory rate.

        This function evaluates fraction of detected EMG breaths relative to the
        ventilatory respiratory rate.

        Args:
            emg_breath_indices: EMG breath indices.
            recording_duration_seconds (float): Recording time in seconds.
            ventilator_respiratory_rate (float): Ventilatory respiratory rate (breath/min).
            minimum_fraction (float): Required minimum detected fraction of EMG breaths.

        Returns:
            tuple:
                - detected_fraction (float): Fraction of detected EMG breaths.
                - criterion_met (bool): Boolean indicating if the fraction is above the minimum.
        """

        try:
            from resurfemg.postprocessing.quality_assessment import (
                evaluate_respiratory_rates,
            )
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        breaths = _require_index_array("emg_breath_indices", emg_breath_indices)
        _require_finite_positive(
            "recording_duration_seconds", recording_duration_seconds
        )
        _require_finite_positive(
            "ventilator_respiratory_rate", ventilator_respiratory_rate
        )
        detected_fraction, criterion_met = evaluate_respiratory_rates(
            breaths,
            recording_duration_seconds,
            ventilator_respiratory_rate,
            min_fraction=minimum_fraction,
        )
        return float(detected_fraction), bool(criterion_met)
