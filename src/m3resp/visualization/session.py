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
    _plot_events(
        emg_axes,
        session.events.get("emg_breaths", []),
        color="tab:red",
        label="EMG breath",
    )
    _plot_events(
        eit_axes,
        session.events.get("eit_breaths", []),
        color="tab:green",
        label="EIT breath",
    )

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
        raise TypeError("No processed EIT dictionary found on the session.")

    signal = processed.get("filtered_global_impedance") or processed.get(
        "raw_global_impedance"
    )
    if signal is None:
        raise ValueError("No EIT global impedance signal found in processed EIT data.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax_signal, ax_tiv, ax_eeli, ax_map = axes.ravel()

    ax_signal.plot(signal.time, signal.values, linewidth=1)
    ax_signal.set(title="EIT global impedance", xlabel="Time (s)", ylabel=signal.label)
    ax_signal.grid(True, alpha=0.25)
    _plot_events(
        [ax_signal],
        session.events.get("eit_breaths", []),
        color="tab:green",
        label="EIT breath",
    )
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


def plot_synchronization_comparison(
    session: M3Session,
    *,
    max_seconds: float | None = 120.0,
    emg_channel: int | None = None,
    eit_waveform: str = "global_impedance_(raw)",
):
    """Plot plottable signals before and after manual synchronization."""

    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise ImportError(
            "Synchronization visualization requires matplotlib. Install the EIT "
            "or EMG optional dependencies, or install matplotlib directly."
        ) from exc

    synchronized = session.processed.get("synchronized")
    if not isinstance(synchronized, dict):
        synchronized = {}
    raw_sync = session.processed.get("raw_synchronization")
    if not synchronized and not isinstance(raw_sync, dict):
        raise ValueError(
            "No synchronization data found. Run synchronize_raw_modalities before "
            "calling plot_synchronization_comparison."
        )

    offsets = session.parameters.get("raw_alignment", {}).get(
        "offset_seconds",
        session.parameters.get("alignment", {}).get("offset_seconds", {}),
    )
    if not isinstance(offsets, dict):
        offsets = {}

    rows = _get_synchronization_plot_rows(session, emg_channel, eit_waveform, offsets)
    if not rows:
        raise ValueError(
            "No plottable EIT or EMG signals found for synchronization comparison."
        )

    fig, axes = plt.subplots(
        len(rows),
        2,
        figsize=(12, max(3.0, 2.6 * len(rows))),
        sharex=False,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(rows), 2)

    for row_axes, row in zip(axes, rows, strict=True):
        (
            modality,
            event_key,
            title,
            before_time,
            before_values,
            after_time,
            after_values,
            ylabel,
        ) = row
        before_time, before_values = _limit_time(
            before_time,
            before_values,
            max_seconds,
        )
        after_time, after_values = _limit_time(after_time, after_values, max_seconds)

        before_ax, after_ax = row_axes
        before_ax.plot(before_time, before_values, linewidth=1)
        before_ax.set(title=f"{title} before synchronization", ylabel=ylabel)
        before_ax.grid(True, alpha=0.25)

        after_ax.plot(after_time, after_values, linewidth=1)
        after_ax.set(title=f"{title} after synchronization", ylabel=ylabel)
        after_ax.grid(True, alpha=0.25)

        color, label = _event_style(modality)
        _plot_events(
            [before_ax],
            session.events.get(event_key, []),
            color=color,
            label=f"{label} original",
        )
        _plot_events(
            [after_ax],
            synchronized.get(event_key, []),
            color=color,
            label=f"{label} synchronized",
        )

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    _deduplicate_legends(axes.ravel())
    return fig


def _get_eit_rows(
    session: M3Session, waveform: str
) -> list[tuple[str, str, np.ndarray, np.ndarray, str]]:
    recording = session.raw.get("eit")
    if recording is None:
        return []

    sequence = recording.data
    rows: list[tuple[str, str, np.ndarray, np.ndarray, str]] = []
    continuous_data = getattr(sequence, "continuous_data", {})
    if waveform not in continuous_data:
        return rows

    rows.append(
        (
            "eit",
            "EIT raw global impedance",
            *_continuous_series(continuous_data[waveform]),
        )
    )

    processed = session.processed.get("eit")
    if isinstance(processed, dict):
        filtered = processed.get("filtered_global_impedance")
        raw = processed.get("raw_global_impedance")
        if filtered is not None and filtered is not raw:
            rows.append(
                ("eit", "EIT filtered global impedance", *_continuous_series(filtered))
            )

    return rows


def _get_synchronization_rows(
    session: M3Session,
    emg_channel: int | None,
    eit_waveform: str,
) -> list[tuple[str, str, str, np.ndarray, np.ndarray, str]]:
    rows: list[tuple[str, str, str, np.ndarray, np.ndarray, str]] = []

    eit_rows = _get_eit_rows(session, eit_waveform)
    if eit_rows:
        # Synchronization views should compare timing, not filtering effects.
        _, title, time, values, ylabel = eit_rows[0]
        rows.append(("eit", "eit_breaths", title, time, values, ylabel))

    emg_rows = _get_emg_rows(session, emg_channel)
    if emg_rows:
        _, title, time, values, ylabel = emg_rows[-1]
        rows.append(("emg", "emg_breaths", title, time, values, ylabel))

    return rows


def _get_synchronization_plot_rows(
    session: M3Session,
    emg_channel: int | None,
    eit_waveform: str,
    offsets: dict[str, float],
) -> list[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    rows: list[
        tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]
    ] = []
    raw_sync = session.processed.get("raw_synchronization")
    if isinstance(raw_sync, dict):
        for modality, event_key in (
            ("eit", "eit_breaths"),
            ("emg", "emg_breaths"),
            ("vent_pressure", "ventilator_breaths"),
            ("vent_flow", "ventilator_breaths"),
            ("vent_volume", "ventilator_breaths"),
        ):
            record = raw_sync.get(modality)
            if not isinstance(record, dict):
                continue
            before = record.get("before")
            after = record.get("after")
            if not isinstance(before, dict) or not isinstance(after, dict):
                continue
            rows.append(
                (
                    modality,
                    event_key,
                    str(before["title"]),
                    np.asarray(before["time"], dtype=float),
                    np.asarray(before["values"], dtype=float),
                    np.asarray(after["time"], dtype=float),
                    np.asarray(after["values"], dtype=float),
                    str(before["ylabel"]),
                )
            )
        if rows:
            return rows

    for modality, event_key, title, time, values, ylabel in _get_synchronization_rows(
        session,
        emg_channel,
        eit_waveform,
    ):
        offset = float(offsets.get(modality, 0.0))
        after_time, after_values = _shift_and_crop_series(time, values, offset)
        rows.append(
            (
                modality,
                event_key,
                title,
                time,
                values,
                after_time,
                after_values,
                ylabel,
            )
        )
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
    ylabel = _emg_amplitude_label(unit)

    rows: list[tuple[str, str, np.ndarray, np.ndarray, str]] = []
    raw = processed.get("raw_channel")
    filtered = processed.get("filtered")
    envelope = processed.get("envelope")

    if raw is not None:
        rows.append(
            (
                "emg",
                f"EMG raw ({label})",
                _time_for(raw, fs),
                np.asarray(raw),
                ylabel,
            )
        )
    if filtered is not None:
        rows.append(
            (
                "emg",
                f"EMG filtered ({label})",
                _time_for(filtered, fs),
                np.asarray(filtered),
                ylabel,
            )
        )
    if envelope is not None:
        rows.append(
            (
                "emg",
                f"EMG envelope ({label})",
                _time_for(envelope, fs),
                np.asarray(envelope),
                ylabel,
            )
        )

    return rows


def _emg_amplitude_label(unit: str) -> str:
    return f"EMG amplitude ({unit})" if unit else "EMG amplitude"


def _time_for(values: Any, sample_rate: float) -> np.ndarray:
    return np.arange(len(values), dtype=float) / sample_rate


def _limit_time(
    time: np.ndarray, values: np.ndarray, max_seconds: float | None
) -> tuple[np.ndarray, np.ndarray]:
    if max_seconds is None:
        return time, values
    keep = time <= max_seconds
    return time[keep], values[keep]


def _shift_and_crop_series(
    time: np.ndarray,
    values: np.ndarray,
    offset_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(time) == 0 or offset_seconds == 0:
        return time, values
    shifted_time = time + offset_seconds
    keep = (shifted_time >= time[0]) & (shifted_time <= time[-1])
    return shifted_time[keep], values[keep]


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
                ax.axvline(
                    event.peak_time, color=color, alpha=0.45, linewidth=1, label=label
                )


def _plot_sparse(ax: Any, sparse: Any, title: str, ylabel: str) -> None:
    if sparse is None or not len(sparse):
        ax.set(title=title)
        ax.set_axis_off()
        return

    ax.plot(sparse.time, sparse.values, marker="o", linestyle="-", linewidth=1)
    ax.set(title=title, xlabel="Time (s)", ylabel=ylabel)
    ax.grid(True, alpha=0.25)


def _event_style(modality: str) -> tuple[str, str]:
    if modality == "eit":
        return "tab:green", "EIT breath"
    if modality == "emg":
        return "tab:red", "EMG breath"
    if modality.startswith("vent"):
        return "tab:blue", "Ventilator breath"
    return "tab:blue", f"{modality.upper()} event"


def _deduplicate_legends(axes: Iterable[Any]) -> None:
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=False))
        if unique:
            ax.legend(unique.values(), unique.keys(), loc="upper right")
