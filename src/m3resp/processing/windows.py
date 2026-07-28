"""Shared rolling-window and envelope primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

from m3resp.core.exceptions import OptionalDependencyError

PaddingType = Literal[
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "symmetric",
    "wrap",
    "empty",
]


def moving_average(
    values: np.ndarray,
    *,
    window_size: int,
    window_function: Callable[[int], np.ndarray] | None = None,
    padding_type: PaddingType = "edge",
) -> np.ndarray:
    """Apply an EIT-style centered moving average with padded boundaries."""

    data = np.asarray(values)
    window_size = _normalize_odd_window_size(window_size)
    if window_size > len(data):
        window_size = int((len(data) - 1) / 2) * 2 + 1

    if window_function is not None:
        weights = np.asarray(window_function(window_size))
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(window_size) / window_size

    padding_length = (window_size - 1) // 2
    padded_data = np.pad(data, padding_length, mode=padding_type)
    return np.convolve(padded_data, weights, mode="valid")


def running_smoother(values: np.ndarray) -> np.ndarray:
    """Smooth values with ReSurfEMG's running smoother."""

    data = np.asarray(values)
    n_samples = len(data) // 10
    new_values = np.convolve(abs(data), np.ones(n_samples), "valid") / n_samples
    zeros = np.zeros(n_samples - 1)
    return np.hstack((new_values, zeros))


def rolling_rms(
    values: np.ndarray,
    *,
    window_length: int,
    center: bool = True,
    min_periods: int = 1,
) -> np.ndarray:
    """Compute a rolling root-mean-square envelope."""

    squared = pd.Series(np.power(values, 2))
    return np.asarray(
        np.sqrt(
            squared.rolling(
                window=window_length,
                min_periods=min_periods,
                center=center,
            ).mean()
        )
    )


def naive_rolling_rms(values: np.ndarray, *, window_length: int) -> np.ndarray:
    """Compute a cumulative-sum RMS envelope without edge padding."""

    cumulative = np.cumsum(np.abs(values) ** 2)
    return np.sqrt(
        (cumulative[window_length:] - cumulative[:-window_length]) / window_length
    )


def rolling_arv(
    values: np.ndarray,
    *,
    window_length: int,
    center: bool = True,
    min_periods: int = 1,
) -> np.ndarray:
    """Compute a rolling average rectified value envelope."""

    absolute = pd.Series(np.abs(values))
    return np.asarray(
        absolute.rolling(
            window=window_length,
            min_periods=min_periods,
            center=center,
        ).mean()
    )


def rolling_rms_ci(
    values: np.ndarray,
    *,
    window_length: int,
    alpha: float = 0.05,
    center: bool = True,
    min_periods: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate rolling confidence intervals for RMS values."""

    stats = _scipy_stats()
    squared = pd.Series(np.power(values, 2))
    mean_square = (
        squared.rolling(
            window=window_length,
            min_periods=min_periods,
            center=center,
        )
        .mean()
        .values
    )
    sem = (
        squared.rolling(
            window=window_length,
            min_periods=min_periods,
            center=center,
        )
        .sem()
        .values
    )
    ci = stats.t.interval(1 - alpha, window_length - 1, mean_square, sem)
    return np.sqrt(ci[0]), np.sqrt(ci[1])


def rolling_arv_ci(
    values: np.ndarray,
    *,
    window_length: int,
    alpha: float = 0.05,
    center: bool = True,
    min_periods: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate rolling confidence intervals for ARV values."""

    stats = _scipy_stats()
    absolute = pd.Series(np.abs(values))
    arv = (
        absolute.rolling(
            window=window_length,
            min_periods=min_periods,
            center=center,
        )
        .mean()
        .values
    )
    sem = (
        pd.Series(values)
        .rolling(
            window=window_length,
            min_periods=min_periods,
            center=center,
        )
        .sem()
        .values
    )
    return stats.t.interval(1 - alpha, window_length - 1, arv, sem)


def _normalize_odd_window_size(window_size: int) -> int:
    if not isinstance(window_size, int):
        raise TypeError("window_size must be an int")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    return window_size + 1 if window_size % 2 == 0 else window_size


def _scipy_stats():
    try:
        from scipy import stats
    except ImportError as exc:
        raise OptionalDependencyError(
            "Rolling confidence intervals require SciPy. Install `scipy` to "
            "use `m3resp.processing.windows` confidence interval helpers."
        ) from exc
    return stats
