"""Baseline-estimation methods of `ReSurfEMGAdapter`."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._shared import (
    _emg_optional_dependency_error,
    _require_1d_array,
    _require_integer_valued_sample_frequency,
    _require_percentile,
    _require_positive_int,
)


class _BaselineMixin:
    def moving_baseline(
        self,
        envelope: Any,
        *,
        window_samples: int,
        step_samples: int,
        percentile: float = 33.0,
    ) -> np.ndarray:
        """Compute a moving baseline over `envelope` (Grasshoff et al. 2021).

        Args:
            envelope: Envelope of the EMG signal.
            window_samples (int): Number of samples in the moving window.
            step_samples (int): Number of consecutive samples with the same baseline value.
            percentile (float): Percentile to use for baseline estimation.

        Returns:
            numpy.ndarray: Moving baseline of the envelope.
        """

        try:
            from resurfemg.postprocessing.baseline import moving_baseline
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("envelope", envelope)
        _require_positive_int("window_samples", window_samples)
        _require_positive_int("step_samples", step_samples)
        _require_percentile("percentile", percentile)
        return np.asarray(
            moving_baseline(
                array,
                window_samples,
                step_samples,
                set_percentile=percentile,
            )
        )

    def slopesum_baseline(
        self,
        envelope: Any,
        *,
        window_samples: int,
        step_samples: int,
        sample_frequency: float,
        percentile: float = 33.0,
        augmented_percentile: float = 25.0,
        moving_average_samples: int | None = None,
        percentile_window_samples: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """Compute the slope-sum (augmented) baseline over `envelope`.

        Args:
            envelope: Envelope of the EMG signal.
            window_samples (int): Number of samples in the moving window.
            step_samples (int): Number of consecutive samples with the same baseline value.
            sample_frequency (float): Sampling frequency of the signal.
            percentile (float): Percentile to use for baseline estimation.
            augmented_percentile (float): Percentile to use for augmented baseline estimation.
            moving_average_samples (int, optional): Number of samples for the moving average.
            percentile_window_samples (int, optional): Number of samples for the percentile window.

        Returns:
            tuple:
                - baseline (numpy.ndarray): Slope-sum baseline of the envelope.
                - running_mean (numpy.ndarray): Running mean baseline.
                - running_std (numpy.ndarray): Running standard deviation of the baseline.
                - running_series (pandas.Series): Running series of the baseline. running_series
                    is upstream's `pandas.Series`, kept for the existing compatibility output;
                    use the other three arrays for native/export use.
        """

        try:
            from resurfemg.postprocessing.baseline import slopesum_baseline
        except ImportError as exc:
            raise _emg_optional_dependency_error() from exc

        array = _require_1d_array("envelope", envelope)
        _require_positive_int("window_samples", window_samples)
        _require_positive_int("step_samples", step_samples)
        _require_percentile("percentile", percentile)
        _require_percentile("augmented_percentile", augmented_percentile)
        # `slopesum_baseline` derives `ma_window = fs // 2` internally and
        # feeds it straight into a pandas rolling window when
        # `moving_average_samples` is omitted - same int-only constraint as
        # `_require_integer_valued_sample_frequency`'s docstring describes.
        fs = _require_integer_valued_sample_frequency(sample_frequency)
        baseline, running_mean, running_std, running_series = slopesum_baseline(
            array,
            window_samples,
            step_samples,
            fs,
            set_percentile=percentile,
            augm_percentile=augmented_percentile,
            ma_window=moving_average_samples,
            perc_window=percentile_window_samples,
        )
        return (
            np.asarray(baseline),
            np.asarray(running_mean),
            np.asarray(running_std),
            running_series,
        )
