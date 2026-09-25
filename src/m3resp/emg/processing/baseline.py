"""Baseline-estimation  methods of `M3Resp.processing.emg`.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from ReSurfEMG.

    Source:     https://github.com/resurfemg-org/ReSurfEMG
    Revision:   m3resp-integration (c63668689030e4581d5f985e7d09d3a8c01e7a77)
    Original:   resurfemg/portprocessing/baseline.py::moving_baseline, slopesum_baseline
    Copyright:  Copyright (c) 2022 Netherlands eScience Center and
                University of Twente
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - Parameters renamed and reorganized as keyword-only arguments.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from m3resp.emg.processing.shared import (
    _require_1d_array,
    _require_integer_valued_sample_frequency,
    _require_percentile,
    _require_positive_int,
)
from m3resp.processing.windows import compute_derivative

_MIN_SAMPLES_FOR_LAST_COPY = 2


class _BaselineMixin:
    def moving_baseline(
        self,
        envelope: np.ndarray,
        *,
        window_samples: int,
        step_samples: int,
        percentile: float = 33.0,
    ) -> np.ndarray:
        """Compute a moving baseline over `envelope` (Grasshoff et al. 2021).

        Args:
            envelope: Envelope of the EMG signal.
            window_samples (int): Number of samples in the moving window.
            step_samples (int): Number of consecutive samples with the same
                baseline value.
            percentile (float): Percentile to use for baseline estimation.

        Returns:
            numpy.ndarray: Moving baseline of the envelope.
        """
        array = _require_1d_array("envelope", envelope)
        _require_positive_int("window_samples", window_samples)
        _require_positive_int("step_samples", step_samples)
        _require_percentile("percentile", percentile)

        n_samples = len(array)
        moving_baseline = np.zeros(array.shape, dtype=np.float64)

        for index in range(0, n_samples, step_samples):
            start_index = max([0, index - window_samples // 2])
            end_index = min([n_samples, index + window_samples // 2])
            moving_baseline[index : min([index + step_samples, n_samples])] = (
                np.percentile(array[start_index:end_index], percentile)
            )

        return np.asarray(moving_baseline)

    def slopesum_baseline(
        self,
        envelope: np.ndarray,
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
            step_samples (int): Number of consecutive samples with the same
                baseline value.
            sample_frequency (float): Sampling frequency of the signal.
            percentile (float): Percentile to use for baseline estimation.
            augmented_percentile (float): Percentile to use for augmented
                baseline estimation.
            moving_average_samples (int, optional): Number of samples for
                the moving average.
            percentile_window_samples (int, optional): Number of samples
                for the percentile window.

        Returns:
            tuple:
                - baseline (numpy.ndarray): Slope-sum baseline of the envelope.
                - running_mean (numpy.ndarray): Running mean baseline.
                - running_std (numpy.ndarray): Running standard deviation of the
                    baseline.
                - running_series (pandas.Series): Running series of the baseline.
                    The `running_series` is upstream's `pandas.Series`, kept for the
                    existing compatibility output; use the other three arrays for
                    native/export use.
        """
        array = _require_1d_array("envelope", envelope)
        _require_positive_int("window_samples", window_samples)
        _require_positive_int("step_samples", step_samples)
        _require_percentile("percentile", percentile)
        _require_percentile("augmented_percentile", augmented_percentile)
        # TODO check the comments here
        # `slopesum_baseline` derives `ma_window = fs // 2` internally and
        # feeds it straight into a pandas rolling window when
        # `moving_average_samples` is omitted - same int-only constraint as
        # `_require_integer_valued_sample_frequency`'s docstring describes.
        fs = _require_integer_valued_sample_frequency(sample_frequency)

        n_samples = len(array)

        if moving_average_samples is None:
            moving_average_samples = fs // 2
        if percentile_window_samples is None:
            percentile_window_samples = fs

        moving_baseline = self.moving_baseline(
            envelope=array,
            window_samples=window_samples,
            step_samples=step_samples,
            percentile=percentile,
        )

        running_series = pd.Series(moving_baseline)

        running_baseline = running_series.rolling(
            window=window_samples,
            min_periods=1,
            center=True,
        )

        running_std = np.asarray(running_baseline.std(), dtype=np.float64)
        running_mean = np.asarray(running_baseline.mean(), dtype=np.float64)

        array_derivative = compute_derivative(
            array - moving_baseline, fs, moving_average_samples
        )

        augmented_array = array[:-1] + np.abs(array_derivative)

        baseline = np.zeros(array.shape, dtype=np.float64)

        for index in range(0, n_samples, percentile_window_samples):
            start_index = max([0, index - window_samples])
            end_index = min([n_samples - 1, index + window_samples])
            baseline[
                index : min([index + percentile_window_samples, n_samples - 1])
            ] = 1.2 * np.nanpercentile(
                augmented_array[start_index:end_index], augmented_percentile
            )

        if len(baseline) >= _MIN_SAMPLES_FOR_LAST_COPY:
            baseline[-1] = baseline[-2]

        return (
            np.asarray(baseline),
            np.asarray(running_mean),
            np.asarray(running_std),
            running_series,
        )
