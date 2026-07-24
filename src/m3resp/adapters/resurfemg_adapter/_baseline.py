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
        """Compute a moving baseline over `envelope` (Grasshoff et al. 2021)."""

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

        Returns `(baseline, running_mean, running_std, running_series)` -
        `running_series` is upstream's `pandas.Series`, kept for the
        existing compatibility output; use the other three arrays for
        native/export use.
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
