"""Drift generation and timing-drift shift helpers for synthetic signals."""

from __future__ import annotations

from typing import cast

import numpy as np

from m3resp.synthetic.config import (
    DriftConfig,
    SyntheticGeneratorConfig,
    TimingDriftConfig,
)

DEFAULT_ZERO = 0.0
DEFAULT_ONE = 1.0
DEFAULT_TWO = 2.0
DEFAULT_FLOAT32_DTYPE = np.float32
DEFAULT_FIRST_AXIS = 0
DEFAULT_SECOND_AXIS = 1


def generate_drift(time_seconds: np.ndarray, config: DriftConfig) -> np.ndarray:
    """Generate a named drift component for any time series."""

    if not config.enabled:
        return np.zeros_like(time_seconds, dtype=float)
    if config.kind != "linear" and config.amplitude == DEFAULT_ZERO:
        return np.zeros_like(time_seconds, dtype=float)

    if config.kind == "sinusoidal":
        return config.amplitude * config.primary_weight * np.sin(
            2 * np.pi * config.frequency_hz * time_seconds
        ) + config.amplitude * config.secondary_weight * np.sin(
            DEFAULT_TWO * np.pi * config.secondary_frequency_hz * time_seconds
            + config.phase_radians
        )
    if config.kind == "linear":
        centered_time = time_seconds - float(time_seconds[0])
        if config.slope_per_second is not None:
            slope = config.slope_per_second
        else:
            slope = config.amplitude / max(
                float(time_seconds[-1] - time_seconds[0]),
                DEFAULT_ONE,
            )
        return slope * centered_time
    if config.kind == "constant":
        return np.full_like(time_seconds, config.amplitude, dtype=float)

    raise ValueError("drift.kind must be one of: sinusoidal, linear, constant")


def is_timing_drift_enabled(config: TimingDriftConfig | None) -> bool:
    """Return whether a modality-specific timing drift should be applied."""

    return (
        config is not None
        and config.enabled
        and float(config.time_shift_seconds) != DEFAULT_ZERO
    )


def shift_array_in_time(
    array: np.ndarray,
    time_seconds: np.ndarray,
    config: TimingDriftConfig | None,
    *,
    sample_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift a sampled array in time and crop along the requested sample axis."""

    if not is_timing_drift_enabled(config):
        return np.asarray(time_seconds, dtype=float).copy(), np.asarray(array).copy()
    active_config = cast(TimingDriftConfig, config)

    values = np.asarray(array)
    moved = np.moveaxis(values, sample_axis, DEFAULT_FIRST_AXIS)
    flat = moved.reshape((moved.shape[DEFAULT_FIRST_AXIS], -1))
    shifted_columns = []
    shifted_time: np.ndarray | None = None
    for index in range(flat.shape[DEFAULT_SECOND_AXIS]):
        column_time, shifted_column = shift_signal_in_time_cropped(
            flat[:, index],
            time_seconds,
            active_config,
        )
        if shifted_time is None:
            shifted_time = column_time
        shifted_columns.append(shifted_column)

    if shifted_time is None:
        shifted_time = np.asarray(time_seconds[:0], dtype=float)
    shifted = np.stack(shifted_columns, axis=DEFAULT_SECOND_AXIS).astype(
        DEFAULT_FLOAT32_DTYPE
    )
    shifted = shifted.reshape((len(shifted_time), *moved.shape[1:]))
    return shifted_time, np.moveaxis(
        shifted,
        DEFAULT_FIRST_AXIS,
        sample_axis,
    ).astype(values.dtype)


def shift_signal_in_time_cropped(
    signal: np.ndarray,
    time_seconds: np.ndarray,
    config: TimingDriftConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop samples exposed by a temporal shift without inserting fill values."""

    if signal.shape != time_seconds.shape:
        raise ValueError("signal and time_seconds must have the same shape")
    if len(time_seconds) == int(DEFAULT_ZERO):
        return np.asarray(time_seconds, dtype=float), np.asarray(signal, dtype=float)

    shift_samples = _time_shift_sample_count(time_seconds, config)
    if shift_samples == int(DEFAULT_ZERO):
        return np.asarray(time_seconds, dtype=float).copy(), np.asarray(
            signal,
            dtype=float,
        ).copy()

    if shift_samples > int(DEFAULT_ZERO):
        sample_slice = slice(shift_samples, None)
    else:
        sample_slice = slice(None, shift_samples)
    return np.asarray(time_seconds[sample_slice], dtype=float), np.asarray(
        signal[sample_slice],
        dtype=float,
    )


def _time_shift_sample_count(
    time_seconds: np.ndarray,
    config: TimingDriftConfig,
) -> int:
    if len(time_seconds) < int(DEFAULT_TWO):
        return int(DEFAULT_ZERO)
    sample_period = float(np.median(np.diff(time_seconds)))
    if sample_period <= DEFAULT_ZERO:
        raise ValueError("time_seconds must be strictly increasing")
    return round(float(config.time_shift_seconds) / sample_period)


def _shift_eit_signal_components(
    time_seconds: np.ndarray,
    components: dict[str, np.ndarray],
    config: TimingDriftConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    shifted_components: dict[str, np.ndarray] = {}
    shifted_time: np.ndarray | None = None
    for label, values in components.items():
        component_time, shifted_values = shift_signal_in_time_cropped(
            values,
            time_seconds,
            config,
        )
        if shifted_time is None:
            shifted_time = component_time
        shifted_components[label] = shifted_values.astype(values.dtype)

    if shifted_time is None:
        shifted_time = np.asarray(time_seconds[:0], dtype=float)
    shifted_signal = np.zeros_like(shifted_time, dtype=float)
    for values in shifted_components.values():
        shifted_signal = shifted_signal + values
    return shifted_time, shifted_signal, shifted_components


def _eit_timing_drift_config(
    config: SyntheticGeneratorConfig,
) -> TimingDriftConfig | None:
    if config.eit.timing_drift is not None:
        return (
            config.eit.timing_drift
            if is_timing_drift_enabled(config.eit.timing_drift)
            else None
        )
    return None
