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
        local baseline window."""

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
        """Evaluate occlusion-manoeuvre (Pocc) quality (Warnaar et al. 2024).

        Returns `(valid_poccs, criteria_matrix)`; `criteria_matrix` rows are
        `dp_up_10`, `dp_up_90`, `dp_up_90_norm`, in that order.
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
        within `threshold` (Warnaar et al. 2024)."""

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
        """Percentage area-under-baseline per breath (Warnaar et al. 2024).

        Returns `(valid_time_products, percentages_aub, reference_values)` -
        `reference_values` (upstream's undocumented third return value) are
        the nadir reference levels used per breath.
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
        """Flag area-under-baseline values that are locally high outliers."""

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
        expected percentile-based bounds."""

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
        """Flag manoeuvres (e.g. Pocc) with no supported ventilator breath
        between them and the next manoeuvre."""

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
        """How well each breath's envelope fits a bell curve
        (Warnaar et al. 2024).

        Returns `(valid_peak, percentage_bell_error, bell_error, y_min,
        fitted_parameters)`.
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
        """Whether `first_event_times` precede the paired `second_event_times`
        by between `min_delta` and `max_delta` seconds.

        Returns `(correct_timing, delta_time)`.
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
        """Fraction of expected EMG breaths actually detected, relative to
        the ventilator's respiratory rate.

        Returns `(detected_fraction, criterion_met)`.
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
