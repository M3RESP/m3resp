"""Session-level visualization helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from m3resp.core.events import BreathEvent
from m3resp.core.session import M3Session


def plot_session_overview(
    session: M3Session,
    *,
    max_seconds: float | None = 120.0,
    emg_channel: int | None = None,
    eit_waveform: str = "global_impedance_(raw)",
):
    """Plot loaded EIT and processed EMG signals from an ``M3Session``.

    Matplotlib is imported lazily so visualization remains optional for the
    core package.
    """

    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise ImportError(
            "Session visualization requires matplotlib. Install the EIT or EMG "
            "optional dependencies, or install matplotlib directly."
        ) from exc

    rows = []
    eit_series = _get_eit_waveform(session, eit_waveform)
    if eit_series is not None:
        rows.append(("EIT", *eit_series))

    emg_rows = _get_emg_rows(session, emg_channel)
    rows.extend(emg_rows)

    if not rows:
        raise ValueError(
            "No plottable EIT or EMG data found. Load data and run EMG "
            "preprocessing before calling plot_session_overview."
        )

    fig, axes = plt.subplots(
        len(rows),
        1,
        figsize=(11, max(2.6, 2.4 * len(rows))),
        sharex=False,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (title, time, values, ylabel) in zip(axes, rows, strict=True):
        time, values = _limit_time(time, values, max_seconds)
        ax.plot(time, values, linewidth=1)
        ax.set(title=title, ylabel=ylabel)
        ax.grid(True, alpha=0.25)

    _plot_events(axes, session.events.get("emg_breaths", []), color="tab:red", label="EMG breath")
    _plot_events(axes, session.events.get("eit_breaths", []), color="tab:green", label="EIT breath")

    axes[-1].set_xlabel("Time (s)")
    _deduplicate_legends(axes)
    return fig


def _get_eit_waveform(
    session: M3Session, waveform: str
) -> tuple[np.ndarray, np.ndarray, str] | None:
    recording = session.raw.get("eit")
    if recording is None:
        return None

    sequence = recording.data
    continuous_data = getattr(sequence, "continuous_data", {})
    if waveform not in continuous_data:
        return None

    data = continuous_data[waveform]
    time = np.asarray(getattr(data, "time", getattr(sequence, "time", [])), dtype=float)
    values = np.asarray(getattr(data, "values", data), dtype=float)
    return time, values, waveform


def _get_emg_rows(
    session: M3Session, channel: int | None
) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    processed = session.processed.get("emg")
    if not isinstance(processed, dict):
        return []

    fs = float(processed.get("fs", processed.get("metadata", {}).get("fs", 0)))
    if fs <= 0:
        return []

    channel = int(processed.get("channel", 0) if channel is None else channel)
    metadata = processed.get("metadata", {})
    labels = metadata.get("labels") or []
    units = metadata.get("units") or []
    label = labels[channel] if channel < len(labels) else f"channel {channel}"
    unit = units[channel] if channel < len(units) else "a.u."

    rows = []
    raw = processed.get("raw_channel")
    filtered = processed.get("filtered")
    envelope = processed.get("envelope")

    if raw is not None:
        rows.append((f"EMG raw ({label})", _time_for(raw, fs), np.asarray(raw), unit))
    if filtered is not None:
        rows.append((f"EMG filtered ({label})", _time_for(filtered, fs), np.asarray(filtered), unit))
    if envelope is not None:
        rows.append((f"EMG envelope ({label})", _time_for(envelope, fs), np.asarray(envelope), unit))

    return rows


def _time_for(values: Any, sample_rate: float) -> np.ndarray:
    return np.arange(len(values), dtype=float) / sample_rate


def _limit_time(
    time: np.ndarray, values: np.ndarray, max_seconds: float | None
) -> tuple[np.ndarray, np.ndarray]:
    if max_seconds is None:
        return time, values
    keep = time <= max_seconds
    return time[keep], values[keep]


def _plot_events(
    axes: Iterable[Any],
    events: Iterable[BreathEvent],
    *,
    color: str,
    label: str,
) -> None:
    for event in events:
        for ax in axes:
            ax.axvspan(event.start_time, event.end_time, color=color, alpha=0.08)
            if event.peak_time is not None:
                ax.axvline(event.peak_time, color=color, alpha=0.45, linewidth=1, label=label)


def _deduplicate_legends(axes: Iterable[Any]) -> None:
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=False))
        if unique:
            ax.legend(unique.values(), unique.keys(), loc="upper right")
