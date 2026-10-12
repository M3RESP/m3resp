"""Put a signal onto another time axis by linear interpolation.

Use this to compare two recordings sample by sample, for example an EMG
envelope on the time points of an EIT signal. The new values are read off
the straight line between the two nearest original samples. No anti-aliasing
filter is applied: when the new time axis has a lower sampling rate than the
signal, low-pass filter the signal first, below half the new rate. Otherwise
fast changes in the signal show up as false slow changes in the result.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from numpy.typing import ArrayLike

from m3resp.data.timeseries import TimeSeries


def interpolate_to_time[SignalT: TimeSeries](
    signal: SignalT, time: ArrayLike
) -> SignalT:
    """Return a copy of `signal` with its values at the time points `time`.

    Each new value is linearly interpolated between the two original samples
    around it. Values with more than one number per time point (channels,
    EIT pixel frames) are interpolated per number along the time axis.

    Time points before the signal's first sample or after its last sample
    get NaN, with a warning that says how many there are.

    Args:
        signal: `Signal` or `TimeSeries` with at least two samples and a
            strictly increasing time axis in seconds.
        time: New time points in seconds, on the same clock as
            `signal.time`, strictly increasing. Pass another signal's
            `time` to put both on the same time points.

    Returns:
        A copy of `signal` with the new `time`, the interpolated `values`
        (as floats), `sample_frequency` set to 1 / the median time step of
        `time` (in Hz; None for a single time point) and
        ``metadata["interpolation"] = "linear"``. All other fields are copied.

    Raises:
        ValueError: If `time` is not one-dimensional, is empty, holds NaN or
            infinite values, or is not strictly increasing; if `signal` has
            fewer than two samples; or if `signal.time` holds NaN or
            infinite values or is not strictly increasing.
    """

    new_time = np.asarray(time, dtype=float)
    _check_time_axis(new_time, "time")
    if signal.n_samples < 2:
        raise ValueError(
            f"interpolate_to_time needs a signal with at least 2 samples, got "
            f"{signal.n_samples}."
        )
    old_time = np.asarray(signal.time, dtype=float)
    _check_time_axis(old_time, "signal.time")

    old_values = np.asarray(signal.values, dtype=float)
    flat_values = old_values.reshape(old_values.shape[0], -1)
    new_flat = np.empty((new_time.size, flat_values.shape[1]), dtype=float)
    for column in range(flat_values.shape[1]):
        new_flat[:, column] = np.interp(new_time, old_time, flat_values[:, column])

    outside = (new_time < old_time[0]) | (new_time > old_time[-1])
    outside_count = int(np.count_nonzero(outside))
    if outside_count:
        new_flat[outside, :] = np.nan
        warnings.warn(
            f"{outside_count} of {new_time.size} time points lie outside the "
            f"signal's recording ({old_time[0]} s to {old_time[-1]} s); their "
            "values are NaN.",
            stacklevel=2,
        )

    new_values = new_flat.reshape((new_time.size, *old_values.shape[1:]))
    sample_frequency = (
        1.0 / float(np.median(np.diff(new_time))) if new_time.size > 1 else None
    )
    return dataclasses.replace(
        signal,
        values=new_values,
        time=new_time,
        sample_frequency=sample_frequency,
        metadata={**signal.metadata, "interpolation": "linear"},
    )


def _check_time_axis(time: np.ndarray, label: str) -> None:
    """Raise ValueError unless `time` is a 1D, finite, strictly increasing axis."""

    if time.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got shape {time.shape}.")
    if time.size == 0:
        raise ValueError(f"{label} has no time points.")
    if not np.all(np.isfinite(time)):
        raise ValueError(f"{label} holds NaN or infinite values.")
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"{label} must be strictly increasing.")
