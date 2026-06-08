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

    rows: list[tuple[str, str, np.ndarray, np.ndarray, str]] = []
    rows.extend(_get_eit_rows(session, eit_waveform))

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

    for ax, (_, title, time, values, ylabel) in zip(axes, rows, strict=True):
        time, values = _limit_time(time, values, max_seconds)
        ax.plot(time, values, linewidth=1)
        ax.set(title=title, ylabel=ylabel)
        ax.grid(True, alpha=0.25)

    eit_axes = [ax for ax, row in zip(axes, rows, strict=True) if row[0] == "eit"]
    emg_axes = [ax for ax, row in zip(axes, rows, strict=True) if row[0] == "emg"]
    _plot_events(emg_axes, session.events.get("emg_breaths", []), color="tab:red", label="EMG breath")
    _plot_events(eit_axes, session.events.get("eit_breaths", []), color="tab:green", label="EIT breath")

    axes[-1].set_xlabel("Time (s)")
    _deduplicate_legends(axes)
    return fig


def plot_eit_processing_summary(session: M3Session):
    """Plot EIT processing outputs stored by the multimodal notebook."""

    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise ImportError(
            "EIT visualization requires matplotlib. Install the EIT optional "
            "dependencies, or install matplotlib directly."
        ) from exc

    processed = session.processed.get("eit")
    if not isinstance(processed, dict):
        raise ValueError("No processed EIT dictionary found on the session.")

    signal = processed.get("filtered_global_impedance") or processed.get("raw_global_impedance")
    if signal is None:
        raise ValueError("No EIT global impedance signal found in processed EIT data.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax_signal, ax_tiv, ax_eeli, ax_map = axes.ravel()

    ax_signal.plot(signal.time, signal.values, linewidth=1)
    ax_signal.set(title="EIT global impedance", xlabel="Time (s)", ylabel=signal.label)
    ax_signal.grid(True, alpha=0.25)
    _plot_events([ax_signal], session.events.get("eit_breaths", []), color="tab:green", label="EIT breath")
    _deduplicate_legends([ax_signal])

    _plot_sparse(ax_tiv, processed.get("continuous_tiv"), "Continuous TIV", "TIV")
    _plot_sparse(ax_eeli, processed.get("eeli"), "EELI", "EELI")

    pixel_tiv = processed.get("pixel_tiv")
    if pixel_tiv is not None and len(pixel_tiv.values):
        mean_pixel_tiv = np.nanmean(np.stack(pixel_tiv.values), axis=0)
        image = ax_map.imshow(mean_pixel_tiv, cmap="viridis")
        ax_map.set(title="Mean pixel TIV")
        fig.colorbar(image, ax=ax_map, fraction=0.046, pad=0.04)
    else:
        ax_map.set_axis_off()

    return fig


def _get_eit_rows(
    session: M3Session, waveform: str
) -> list[tuple[str, str, np.ndarray, np.ndarray, str]]:
    recording = session.raw.get("eit")
    if recording is None:
        return []

    sequence = recording.data
    rows = []
    continuous_data = getattr(sequence, "continuous_data", {})
    if waveform not in continuous_data:
        return rows

    rows.append(("eit", "EIT raw global impedance", *_continuous_series(continuous_data[waveform])))

    processed = session.processed.get("eit")
    if isinstance(processed, dict):
        filtered = processed.get("filtered_global_impedance")
        raw = processed.get("raw_global_impedance")
        if filtered is not None and filtered is not raw:
            rows.append(("eit", "EIT filtered global impedance", *_continuous_series(filtered)))

    return rows


def _continuous_series(data: Any) -> tuple[np.ndarray, np.ndarray, str]:
    time = np.asarray(data.time, dtype=float)
    values = np.asarray(data.values, dtype=float)
    ylabel = data.label
    return time, values, ylabel


def _get_emg_rows(
    session: M3Session, channel: int | None
) -> list[tuple[str, str, np.ndarray, np.ndarray, str]]:
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
        rows.append(("emg", f"EMG raw ({label})", _time_for(raw, fs), np.asarray(raw), unit))
    if filtered is not None:
        rows.append(("emg", f"EMG filtered ({label})", _time_for(filtered, fs), np.asarray(filtered), unit))
    if envelope is not None:
        rows.append(("emg", f"EMG envelope ({label})", _time_for(envelope, fs), np.asarray(envelope), unit))

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


def _plot_sparse(ax: Any, sparse: Any, title: str, ylabel: str) -> None:
    if sparse is None or not len(sparse):
        ax.set(title=title)
        ax.set_axis_off()
        return

    ax.plot(sparse.time, sparse.values, marker="o", linestyle="-", linewidth=1)
    ax.set(title=title, xlabel="Time (s)", ylabel=ylabel)
    ax.grid(True, alpha=0.25)


def _deduplicate_legends(axes: Iterable[Any]) -> None:
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=False))
        if unique:
            ax.legend(unique.values(), unique.keys(), loc="upper right")
