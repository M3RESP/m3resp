"""Automatic time-offset estimation between simultaneously recorded modalities.

When two devices record the same subject on independent clocks -- for example a
Draeger EIT device (~50 Hz frame rate) and a Biopac amplifier (2 kHz) carrying
airway pressure (Paw) plus diaphragm sEMG -- the recordings share no common
timestamp and must be aligned before analysis. :meth:`M3Session.
synchronize_raw_modalities` can *apply* a known offset (it crops the modalities
onto a common window), but it does not *find* it. This module supplies the
missing estimation stage.

Two complementary estimators are provided, mirroring the interactive marimo
viewer in ``tools/visualization_tool/multimodal_vis.py``:

* :func:`estimate_offset_from_interference` -- a
  **recording-protocol-specific absolute anchor**. In the Annemijn recording, the
  way the EIT and Biopac acquisition were run left a visible high-frequency
  artifact in the raw sEMG while EIT was active; that artifact drops when EIT
  stops. This is not a general EIT-device feature and should only be used when a
  clear on-to-off artifact is present. Detecting that edge pins the EIT
  recording's end onto the reference (Biopac) clock, which -- knowing the EIT
  recording duration -- yields the offset directly.
* :func:`refine_offset_by_crosscorrelation` -- a **relative refinement**. EIT
  global impedance and airway pressure both track the breath cycle. Cross-
  correlating them over an overlap window snaps the alignment to within a
  breath. Because breathing is quasi-periodic (~3-5 s), this only disambiguates
  reliably once a coarse anchor has placed you within one breath.

:func:`estimate_sync_offset` orchestrates the two according to a ``method``
string. Everything here is numpy-only and operates on plain arrays; the
``*_from_signal`` wrappers accept :class:`~m3resp.data.timeseries.TimeSeries`.

Offset convention
-----------------
Throughout this module an *offset* is the timestamp, **on the reference clock**,
at which the target recording's ``t = 0`` occurs. Equivalently, a sample at
target-relative time ``t`` maps to reference time ``offset + t``. For the EIT +
Biopac case the reference clock is the Biopac timeline and the target is the EIT
recording, so ``offset`` is "Biopac seconds at EIT t=0" -- the same quantity the
marimo viewer exposes as *Effective offset*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from m3resp.data.timeseries import TimeSeries

__all__ = [
    "CrossCorrelationOffsetResult",
    "InterferenceOffsetResult",
    "SyncOffsetResult",
    "estimate_offset_from_interference",
    "estimate_offset_from_interference_signal",
    "estimate_sync_offset",
    "interference_power",
    "refine_offset_by_crosscorrelation",
    "refine_offset_by_crosscorrelation_signals",
]

#: Methods accepted by :func:`estimate_sync_offset`.
_METHODS = frozenset(
    {"manual", "interference", "crosscorrelation", "interference+crosscorrelation"}
)


@dataclass(frozen=True)
class InterferenceOffsetResult:
    """Outcome of :func:`estimate_offset_from_interference`.

    ``offset_seconds`` and ``edge_time_seconds`` are ``None`` when no clear
    interference edge was found (``detected is False``); callers should then fall
    back to a manual offset. ``power_time``/``power_values`` are the decimated
    interference-power trace, kept for plotting/verification.
    """

    detected: bool
    offset_seconds: float | None
    edge_time_seconds: float | None
    reference_duration_seconds: float
    threshold: float | None
    high_power: float
    low_power: float
    drop: float
    power_time: np.ndarray = field(repr=False)
    power_values: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class CrossCorrelationOffsetResult:
    """Outcome of :func:`refine_offset_by_crosscorrelation`.

    ``lag_seconds`` is the correction added to the base offset; a positive value
    shifts the target later on the reference clock. ``refined_offset_seconds`` is
    ``base_offset_seconds + lag_seconds``. ``lags_seconds``/``correlation`` are
    the searched cross-correlation curve, kept for plotting/verification.
    """

    lag_seconds: float
    refined_offset_seconds: float
    peak_correlation: float
    lags_seconds: np.ndarray = field(repr=False)
    correlation: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class SyncOffsetResult:
    """Combined result returned by :func:`estimate_sync_offset`."""

    offset_seconds: float
    method: str
    source: str
    interference: InterferenceOffsetResult | None = None
    crosscorrelation: CrossCorrelationOffsetResult | None = None


def _sliding_mean(values: np.ndarray, window: float) -> np.ndarray:
    """Centre-agnostic sliding mean of ``values`` over ``window`` samples.

    Uses a cumulative-sum trick so cost is O(n) regardless of window length
    (important: the sEMG trace can be millions of samples). The result is padded
    at the front to match the input length.
    """

    window = max(1, int(window))
    if values.size == 0:
        return values.astype(float, copy=True)
    cumsum = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    out = (cumsum[window:] - cumsum[:-window]) / window
    if out.size == 0:
        return np.full(values.size, float(np.mean(values)))
    pad = values.size - out.size
    if pad > 0:
        out = np.concatenate([np.full(pad, out[0]), out])
    return out


def interference_power(
    emg_values: np.ndarray,
    sample_frequency: float,
    *,
    fast_window_seconds: float = 0.5,
    smooth_window_seconds: float = 2.0,
) -> np.ndarray:
    """Return a high-frequency power envelope of an sEMG trace.

    The first difference emphasises high-frequency content (the EIT interference
    band sits well above the respiratory sEMG envelope); its magnitude is then
    smoothed twice -- a short window to build a power measure and a longer window
    to suppress burst-to-burst variation so the EIT on/off transition dominates.
    """

    values = np.asarray(emg_values, dtype=float)
    if values.size == 0:
        raise ValueError("emg_values must be non-empty")
    if sample_frequency <= 0:
        raise ValueError("sample_frequency must be positive")
    high_frequency = np.abs(np.diff(values, prepend=values[:1]))
    power = _sliding_mean(high_frequency, fast_window_seconds * sample_frequency)
    power = _sliding_mean(power, smooth_window_seconds * sample_frequency)
    return power


def estimate_offset_from_interference(
    emg_values: np.ndarray,
    sample_frequency: float,
    reference_duration_seconds: float,
    *,
    emg_time: np.ndarray | None = None,
    detection_rate_hz: float = 10.0,
    search_window_seconds: float = 300.0,
    plateau_guard_seconds: float = 180.0,
    tail_seconds: float = 45.0,
    step_window_seconds: float = 20.0,
    min_power_ratio: float = 1.25,
    min_drop_fraction: float = 0.25,
    fast_window_seconds: float = 0.5,
    smooth_window_seconds: float = 2.0,
) -> InterferenceOffsetResult:
    """Estimate the offset from a protocol-specific sEMG artifact drop.

    In recordings where the acquisition protocol leaves an EIT-related
    high-frequency artifact in the raw sEMG that disappears when EIT stops, this
    finds the strongest sustained *downward* step in the sEMG interference power
    within the last ``search_window_seconds`` of the recording, treats that as
    the EIT-off instant on the reference clock, and returns
    ``edge_time - reference_duration_seconds`` as the offset.

    :param reference_duration_seconds: Duration of the target (EIT) recording,
        used to back out its start from the detected end edge.
    :param emg_time: Optional explicit time axis for ``emg_values``; if omitted,
        ``numpy.arange(n) / sample_frequency`` is assumed (recording starts at 0).
    :param plateau_guard_seconds: The "interference present" plateau level is the
        median power before ``end - plateau_guard_seconds``.
    :param tail_seconds: The "interference absent" level is the median power over
        the final ``tail_seconds`` (assumed to be inside the artifact-off
        region).
    :param min_power_ratio: Require ``high >= min_power_ratio * low`` for an edge
        to be considered present at all.
    :param min_drop_fraction: Require the detected step to drop by at least
        ``min_drop_fraction * (high - low)`` to be accepted.

    Detection thresholds are heuristics; always confirm the returned
    ``edge_time_seconds`` against the ``power_time``/``power_values`` trace.
    """

    values = np.asarray(emg_values, dtype=float)
    power = interference_power(
        values,
        sample_frequency,
        fast_window_seconds=fast_window_seconds,
        smooth_window_seconds=smooth_window_seconds,
    )
    if emg_time is None:
        time = np.arange(values.size, dtype=float) / sample_frequency
    else:
        time = np.asarray(emg_time, dtype=float)
        if time.size != values.size:
            raise ValueError("emg_time must match emg_values length")

    step = max(1, round(sample_frequency / detection_rate_hz))
    time_ds = time[::step]
    power_ds = power[::step]
    rate_ds = sample_frequency / step

    end = float(time_ds[-1])
    plateau_mask = time_ds < end - plateau_guard_seconds
    high = (
        float(np.nanmedian(power_ds[plateau_mask])) if np.any(plateau_mask) else np.nan
    )
    tail_mask = time_ds > end - tail_seconds
    low = float(np.nanmedian(power_ds[tail_mask])) if np.any(tail_mask) else np.nan

    edge_time: float | None = None
    offset: float | None = None
    threshold: float | None = None
    best_drop = float("nan")
    detected = False

    if np.isfinite(high) and np.isfinite(low) and high > low * min_power_ratio:
        threshold = 0.5 * (high + low)
        window = max(1, int(step_window_seconds * rate_ds))
        search_idx = np.where(time_ds > end - search_window_seconds)[0]
        best = -np.inf
        best_i: int | None = None
        for i in search_idx:
            before = power_ds[max(0, i - window) : i]
            after = power_ds[i : i + window]
            if before.size > 10 and after.size > 10:
                drop = float(before.mean() - after.mean())
                if drop > best:
                    best = drop
                    best_i = int(i)
        best_drop = best
        if best_i is not None and best > min_drop_fraction * (high - low):
            edge_time = float(time_ds[best_i])
            offset = edge_time - float(reference_duration_seconds)
            detected = True

    return InterferenceOffsetResult(
        detected=detected,
        offset_seconds=offset,
        edge_time_seconds=edge_time,
        reference_duration_seconds=float(reference_duration_seconds),
        threshold=threshold,
        high_power=high,
        low_power=low,
        drop=best_drop,
        power_time=time_ds,
        power_values=power_ds,
    )


def refine_offset_by_crosscorrelation(
    target_time: np.ndarray,
    target_values: np.ndarray,
    reference_time: np.ndarray,
    reference_values: np.ndarray,
    base_offset_seconds: float,
    *,
    max_lag_seconds: float = 5.0,
    grid_rate_hz: float = 50.0,
    stretch: float = 1.0,
    window: tuple[float, float] | None = None,
    min_overlap_seconds: float = 10.0,
) -> CrossCorrelationOffsetResult:
    """Refine ``base_offset_seconds`` by cross-correlating two breathing signals.

    ``target`` (e.g. EIT global impedance, in its own ``t~0`` clock) and
    ``reference`` (e.g. airway pressure, in the reference clock) are both mapped
    onto a shared, uniformly-sampled grid over their overlap, z-scored, and
    cross-correlated within +/- ``max_lag_seconds``. The lag of the peak absolute
    correlation is returned as a correction to the base offset.

    :param base_offset_seconds: Current best offset (e.g. from the interference
        anchor) used to bring ``reference`` into the target's relative clock.
    :param stretch: Optional clock-drift factor mapping target seconds to
        reference seconds (``reference_t = base_offset + target_t * stretch``).
    :param window: Optional ``(t0, t1)`` in target-relative seconds restricting
        the correlation to a sub-window of the target.

    Returns a zero-lag result (with an empty curve) when the overlap is shorter
    than ``min_overlap_seconds`` or either signal is too short to correlate.
    """

    target_time = np.asarray(target_time, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    reference_time = np.asarray(reference_time, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    stretch = float(stretch)

    # A single NaN sample (e.g. a masked-invalid breath value) would otherwise
    # propagate through np.interp's neighboring grid points, then through
    # a.mean()/a.std(), turning the *entire* z-scored array into NaN and
    # silently collapsing np.correlate to an all-NaN result - argmax on that
    # picks index 0 with no error, i.e. a bogus lag with a NaN
    # peak_correlation that no caller checks for. Drop non-finite samples
    # before interpolating instead.
    target_finite = np.isfinite(target_values)
    if not np.all(target_finite):
        target_time = target_time[target_finite]
        target_values = target_values[target_finite]
    reference_finite = np.isfinite(reference_values)
    if not np.all(reference_finite):
        reference_time = reference_time[reference_finite]
        reference_values = reference_values[reference_finite]

    empty = CrossCorrelationOffsetResult(
        lag_seconds=0.0,
        refined_offset_seconds=float(base_offset_seconds),
        peak_correlation=float("nan"),
        lags_seconds=np.empty(0),
        correlation=np.empty(0),
    )

    if window is not None:
        t0, t1 = float(window[0]), float(window[1])
        mask = (target_time >= t0) & (target_time <= t1)
        target_time = target_time[mask]
        target_values = target_values[mask]

    if target_time.size < 20 or reference_time.size < 20:
        return empty

    # Bring the reference signal into the target's relative clock.
    reference_relative = (reference_time - float(base_offset_seconds)) / stretch

    low = max(target_time[0], reference_relative[0])
    high = min(target_time[-1], reference_relative[-1])
    if high - low < min_overlap_seconds:
        return empty

    grid = np.arange(low, high, 1.0 / grid_rate_hz)
    if grid.size < 2:
        return empty
    a = np.interp(grid, target_time, target_values)
    b = np.interp(grid, reference_relative, reference_values)
    a = a - a.mean()
    b = b - b.mean()
    a = a / (a.std() + 1e-12)
    b = b / (b.std() + 1e-12)

    n = a.size
    correlation = np.correlate(a, b, mode="full") / n
    lags = np.arange(-(n - 1), n) / grid_rate_hz
    keep = np.abs(lags) <= max_lag_seconds
    lags = lags[keep]
    correlation = correlation[keep]
    peak_index = int(np.argmax(np.abs(correlation)))
    best_lag = float(lags[peak_index])
    # A positive peak lag means the target trails the reference under the current
    # mapping, so reducing the offset re-aligns them: correction = -best_lag.
    lag_seconds = -best_lag

    return CrossCorrelationOffsetResult(
        lag_seconds=lag_seconds,
        refined_offset_seconds=float(base_offset_seconds) + lag_seconds,
        peak_correlation=float(correlation[peak_index]),
        lags_seconds=lags,
        correlation=correlation,
    )


def _frequency(signal: TimeSeries) -> float:
    """Return a positive sampling frequency for ``signal``, inferring if needed."""

    if signal.sample_frequency:
        return float(signal.sample_frequency)
    if signal.time.size >= 2:
        step = float(np.median(np.diff(signal.time)))
        if step > 0:
            return 1.0 / step
    raise ValueError("Cannot determine sample frequency for signal")


def estimate_offset_from_interference_signal(
    emg: TimeSeries,
    reference_duration_seconds: float,
    **kwargs: float,
) -> InterferenceOffsetResult:
    """:func:`estimate_offset_from_interference` for a :class:`TimeSeries` sEMG."""

    return estimate_offset_from_interference(
        emg.values,
        _frequency(emg),
        reference_duration_seconds,
        emg_time=emg.time,
        **kwargs,
    )


def refine_offset_by_crosscorrelation_signals(
    target: TimeSeries,
    reference: TimeSeries,
    base_offset_seconds: float,
    **kwargs: Any,
) -> CrossCorrelationOffsetResult:
    """:func:`refine_offset_by_crosscorrelation` for two :class:`TimeSeries`."""

    return refine_offset_by_crosscorrelation(
        target.time,
        target.values,
        reference.time,
        reference.values,
        base_offset_seconds,
        **kwargs,
    )


def estimate_sync_offset(
    *,
    method: str = "interference",
    emg: TimeSeries | None = None,
    target: TimeSeries | None = None,
    reference: TimeSeries | None = None,
    reference_duration_seconds: float | None = None,
    manual_offset_seconds: float = 0.0,
    interference_kwargs: dict[str, float] | None = None,
    crosscorrelation_kwargs: dict[str, float] | None = None,
) -> SyncOffsetResult:
    """Estimate a sync offset using one of the supported ``method`` strategies.

    Methods:

    * ``"manual"`` -- return ``manual_offset_seconds`` unchanged.
    * ``"interference"`` -- interference anchor only (falls back to the manual
      offset if no edge is found).
    * ``"crosscorrelation"`` -- cross-correlation refinement of the manual offset.
    * ``"interference+crosscorrelation"`` -- interference anchor, then refine.

    :param emg: sEMG signal (required for interference-based methods).
    :param target: Target breathing signal, e.g. EIT global impedance (required
        for cross-correlation; also supplies ``reference_duration_seconds`` when
        that argument is omitted).
    :param reference: Reference breathing signal, e.g. airway pressure (required
        for cross-correlation).
    """

    if method not in _METHODS:
        raise ValueError(
            f"Unknown method {method!r}; expected one of {sorted(_METHODS)}"
        )

    interference_kwargs = dict(interference_kwargs or {})
    crosscorrelation_kwargs = dict(crosscorrelation_kwargs or {})

    interference_result: InterferenceOffsetResult | None = None
    crosscorrelation_result: CrossCorrelationOffsetResult | None = None
    offset = float(manual_offset_seconds)
    source = "manual offset"

    if method in ("interference", "interference+crosscorrelation"):
        if emg is None:
            raise ValueError(f"method={method!r} requires an emg signal")
        duration = reference_duration_seconds
        if duration is None:
            if target is None:
                raise ValueError(
                    "reference_duration_seconds or a target signal is required "
                    "for interference-based estimation"
                )
            duration = float(target.duration)
        interference_result = estimate_offset_from_interference_signal(
            emg, duration, **interference_kwargs
        )
        if (
            interference_result.detected
            and interference_result.offset_seconds is not None
        ):
            offset = interference_result.offset_seconds
            source = "interference edge"
        else:
            source = "manual offset (interference edge not found)"

    if method in ("crosscorrelation", "interference+crosscorrelation"):
        if target is None or reference is None:
            raise ValueError(
                f"method={method!r} requires both target and reference signals"
            )
        crosscorrelation_result = refine_offset_by_crosscorrelation_signals(
            target, reference, offset, **crosscorrelation_kwargs
        )
        offset = crosscorrelation_result.refined_offset_seconds
        source = f"{source} + cross-correlation"

    return SyncOffsetResult(
        offset_seconds=float(offset),
        method=method,
        source=source,
        interference=interference_result,
        crosscorrelation=crosscorrelation_result,
    )
