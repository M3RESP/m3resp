"""Resampling to a common time base (plan_stage2.md Sec 20, Milestone 2.5)."""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np

from m3resp.data.signals import Signal


def resample_signal(signal: Signal, target_frequency_hz: float) -> Signal:
    """Return a copy of ``signal`` resampled to ``target_frequency_hz``.

    Uses linear interpolation (`numpy.interp`) rather than a
    filtering-based resampler: this is a synchronization convenience for
    putting two modalities on a common time base, not a signal-processing
    step, so it deliberately does not apply anti-aliasing.
    """

    if target_frequency_hz <= 0:
        raise ValueError("target_frequency_hz must be positive")

    if signal.n_samples == 0:
        return replace(signal, sample_frequency=target_frequency_hz)

    start_time = float(signal.time[0])
    end_time = float(signal.time[-1])
    if end_time <= start_time:
        # Nothing can be interpolated across a span of zero or negative
        # length, so the samples come back untouched. Say so, because the
        # returned signal still carries the requested rate in its metadata
        # and would otherwise claim a rate its samples do not have.
        warnings.warn(
            "Signal spans no time (first timestamp "
            f"{start_time} s, last {end_time} s), so its samples were left "
            f"unchanged; only the recorded rate was set to "
            f"{target_frequency_hz} Hz.",
            stacklevel=2,
        )
        return replace(signal, sample_frequency=target_frequency_hz)

    sample_count = round((end_time - start_time) * target_frequency_hz) + 1
    new_time = start_time + np.arange(sample_count) / target_frequency_hz
    new_values = np.interp(new_time, signal.time, signal.values)

    return replace(
        signal,
        values=new_values,
        time=new_time,
        sample_frequency=target_frequency_hz,
    )
