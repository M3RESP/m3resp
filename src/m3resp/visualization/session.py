"""Session-level visualization helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.data.event_data import IntervalData
from m3resp.data.events import BreathEvent
from m3resp.synchronization.alignment import align_events_by_modality_offset
from m3resp.synchronization.start_times import on_clock_of, shared_clock_shifts
from m3resp.visualization import plotly_figures
from m3resp.visualization.plotly_figures import EventOverlay, LinePanel

BACKENDS = ("matplotlib", "plotly")


def plot_session_overview(
    session: M3Session,
    *,
    max_seconds: float | None = 120.0,
    emg_channel: int | None = None,
    eit_waveform: str = "global_impedance_(raw)",
    backend: str = "matplotlib",
    max_points: int | None = 5000,
):
    """Plot the loaded EIT signal and the processed EMG signals of a session.

    Each signal gets its own panel. Green lines mark the EIT breaths and red
    lines mark the EMG breaths, at the extremum of each breath.

    Args:
        session (M3Session): The session with the loaded EIT data, the
            processed EMG data, or both.
        max_seconds (float | None): Show the first ``max_seconds`` seconds of
            each signal. ``None`` shows the full signal.
        emg_channel (int | None): The EMG channel to show. It must be the
            channel that the EMG processing used. ``None`` shows that channel.
        eit_waveform (str): The name of the EIT signal in the first panel.
        backend (str): ``"matplotlib"`` gives a static figure. ``"plotly"``
            gives an interactive figure for a web page or a notebook.
        max_points (int | None): For ``backend="plotly"`` only. The largest
            number of points drawn for one signal. A longer signal keeps the
            lowest and the highest value of each short time bin, so the peaks
            stay visible. ``None`` draws every sample.

    Returns:
        matplotlib.figure.Figure | plotly.graph_objects.Figure: The figure.

    Raises:
        ValueError: If ``backend`` is unknown, if the session has no signal to
            plot, or if ``emg_channel`` is not the processed channel.
        ImportError: If the plot library of ``backend`` is not installed.
    """

    _check_backend(backend)
    rows: list[tuple[str, str, np.ndarray, np.ndarray, str]] = []
    rows.extend(_get_eit_rows(session, eit_waveform))

    emg_rows = _get_emg_rows(session, emg_channel)
    rows.extend(emg_rows)

    if not rows:
        raise ValueError(
            "No plottable EIT or EMG data found. Load data and run EMG "
            "preprocessing before calling plot_session_overview."
        )

    if backend == "plotly":
        panels = []
        for modality, title, time, values, ylabel in rows:
            time, values = _limit_time(time, values, max_seconds)
            event_key = f"{modality}_breaths"
            color, label = _event_style(modality)
            overlay = EventOverlay(session.events.get(event_key, []), color, label)
            panels.append([LinePanel(title, ylabel, time, values, [overlay])])
        return plotly_figures.line_grid_figure(panels, max_points=max_points)

    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise ImportError(
            "Session visualization requires matplotlib. Install the EIT or EMG "
            "optional dependencies, or install matplotlib directly."
        ) from exc

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


def plot_eit_processing_summary(
    session: M3Session,
    *,
    backend: str = "matplotlib",
    max_points: int | None = 5000,
):
    """Plot the results of the EIT processing of a session.

    The four panels show the global impedance with the EIT breaths, the TIV
    and the EELI of each breath, and the mean TIV of each pixel.

    Args:
        session (M3Session): The session after the EIT processing.
        backend (str): ``"matplotlib"`` gives a static figure. ``"plotly"``
            gives an interactive figure for a web page or a notebook.
        max_points (int | None): For ``backend="plotly"`` only. The largest
            number of points drawn for the global impedance. See
            :func:`plot_session_overview`.

    Returns:
        matplotlib.figure.Figure | plotly.graph_objects.Figure: The figure.

    Raises:
        TypeError: If the session has no processed EIT data.
        ValueError: If ``backend`` is unknown, or if the processed EIT data
            has no global impedance.
        ImportError: If the plot library of ``backend`` is not installed.
    """

    _check_backend(backend)
    processed = session.processed.get("eit")
    if not isinstance(processed, dict):
        raise TypeError("No processed EIT dictionary found on the session.")

    signal = processed.get("filtered_global_impedance") or processed.get(
        "raw_global_impedance"
    )
    if signal is None:
        raise ValueError("No EIT global impedance signal found in processed EIT data.")
    # An eitprocessing waveform has a label; an m3resp Signal has a name.
    signal_label = getattr(signal, "label", None) or signal.name

    tiv = _breath_values(processed.get("continuous_tiv"), "extremum_time", session)
    eeli = _breath_values(processed.get("eeli"), "end_time", session)
    pixel_tiv = processed.get("pixel_tiv")
    mean_pixel_tiv = None
    if pixel_tiv is not None and len(pixel_tiv.values):
        mean_pixel_tiv = np.nanmean(np.stack(pixel_tiv.values), axis=0)

    if backend == "plotly":
        color, label = _event_style("eit")
        panel = LinePanel(
            "EIT global impedance",
            signal_label,
            np.asarray(signal.time, dtype=float),
            np.asarray(signal.values, dtype=float),
            [EventOverlay(session.events.get("eit_breaths", []), color, label)],
        )
        return plotly_figures.eit_summary_figure(
            panel,
            tiv,
            eeli,
            mean_pixel_tiv,
            max_points=max_points,
        )

    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise ImportError(
            "EIT visualization requires matplotlib. Install the EIT optional "
            "dependencies, or install matplotlib directly."
        ) from exc

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax_signal, ax_tiv, ax_eeli, ax_map = axes.ravel()

    ax_signal.plot(signal.time, signal.values, linewidth=1)
    ax_signal.set(title="EIT global impedance", xlabel="Time (s)", ylabel=signal_label)
    ax_signal.grid(True, alpha=0.25)
    _plot_events(
        [ax_signal],
        session.events.get("eit_breaths", []),
        color="tab:green",
        label="EIT breath",
    )
    _deduplicate_legends([ax_signal])

    _plot_sparse(ax_tiv, tiv, "Continuous TIV", "TIV")
    _plot_sparse(ax_eeli, eeli, "EELI", "EELI")

    if mean_pixel_tiv is not None:
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
    backend: str = "matplotlib",
    max_points: int | None = 5000,
):
    """Plot each signal before and after synchronization.

    Each row shows one signal: on its own clock at the left, and on the shared
    clock at the right. The lines mark the breaths of each modality.

    Args:
        session (M3Session): The session after
            ``synchronize_raw_modalities`` or after
            ``synchronize_multimodal_breaths``.
        max_seconds (float | None): Show the first ``max_seconds`` seconds of
            each signal. ``None`` shows the full signal.
        emg_channel (int | None): The EMG channel to show. It must be the
            channel that the EMG processing used. ``None`` shows that channel.
        eit_waveform (str): The name of the EIT signal to show.
        backend (str): ``"matplotlib"`` gives a static figure. ``"plotly"``
            gives an interactive figure for a web page or a notebook.
        max_points (int | None): For ``backend="plotly"`` only. The largest
            number of points drawn for one signal. See
            :func:`plot_session_overview`.

    Returns:
        matplotlib.figure.Figure | plotly.graph_objects.Figure: The figure.

    Raises:
        ValueError: If ``backend`` is unknown, or if the session has no
            synchronization data or no signal to plot.
        ImportError: If the plot library of ``backend`` is not installed.
    """

    _check_backend(backend)
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
    after_events = _after_synchronization_events(session, synchronized)

    rows = _get_synchronization_plot_rows(session, emg_channel, eit_waveform, offsets)
    if not rows:
        raise ValueError(
            "No plottable EIT or EMG signals found for synchronization comparison."
        )

    if backend == "plotly":
        panels = []
        for (
            modality,
            event_key,
            title,
            before_time,
            before_values,
            after_time,
            after_values,
            ylabel,
        ) in rows:
            color, label = _event_style(modality)
            before_time, before_values = _limit_time(
                before_time, before_values, max_seconds
            )
            after_time, after_values = _limit_time(
                after_time, after_values, max_seconds
            )
            before_events = session.events.get(event_key, [])
            panels.append(
                [
                    LinePanel(
                        f"{title} before synchronization",
                        ylabel,
                        before_time,
                        before_values,
                        [EventOverlay(before_events, color, f"{label} original")],
                    ),
                    LinePanel(
                        f"{title} after synchronization",
                        ylabel,
                        after_time,
                        after_values,
                        [
                            EventOverlay(
                                after_events.get(event_key, []),
                                color,
                                f"{label} synchronized",
                            )
                        ],
                    ),
                ]
            )
        return plotly_figures.line_grid_figure(panels, max_points=max_points)

    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise ImportError(
            "Synchronization visualization requires matplotlib. Install the EIT "
            "or EMG optional dependencies, or install matplotlib directly."
        ) from exc

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
            after_events.get(event_key, []),
            color=color,
            label=f"{label} synchronized",
        )

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    _deduplicate_legends(axes.ravel())
    return fig


def _check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")


def _after_synchronization_events(
    session: M3Session,
    synchronized: dict[str, Any],
) -> dict[str, list[Any]]:
    """Return event overlays on the synchronized time base.

    ``synchronize_multimodal_breaths`` stores shifted event copies in
    ``synchronized``. Raw synchronization only sets each recording's start
    time, so the original events are moved onto the shared clock here, the
    same way the "after" traces are.
    """

    if synchronized:
        return synchronized
    shifts = shared_clock_shifts(session)
    return {
        name: align_events_by_modality_offset(events, shifts)
        for name, events in session.events.items()
        if isinstance(events, list)
    }


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

    processed_channel = int(processed.get("channel", 0))
    if channel is not None and channel != processed_channel:
        raise ValueError(
            f"EMG channel {channel} was requested for plotting, but the available "
            f"processed data is for channel {processed_channel}. Preprocess the "
            "requested channel before plotting it."
        )
    metadata = processed.get("metadata", {})
    labels = metadata.get("labels") or []
    units = metadata.get("units") or []
    label = (
        labels[processed_channel]
        if processed_channel < len(labels)
        else f"channel {processed_channel}"
    )
    unit = units[processed_channel] if processed_channel < len(units) else "a.u."
    ylabel = _emg_amplitude_label(unit)

    rows: list[tuple[str, str, np.ndarray, np.ndarray, str]] = []
    raw = processed.get("raw_channel")
    filtered = processed.get("filtered")
    ecg_cleaned = processed.get("ecg_cleaned")
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
    if ecg_cleaned is not None:
        rows.append(
            (
                "emg",
                f"EMG ECG-removed ({label})",
                _time_for(ecg_cleaned, fs),
                np.asarray(ecg_cleaned),
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
    """Shade each breath interval and mark available extrema on each axis.

    Times are in seconds. Adds artists to the supplied axes using color,
    with label attached to each extremum line.
    """

    for event in events:
        for ax in axes:
            ax.axvspan(event.start_time, event.end_time, color=color, alpha=0.08)
            if event.extremum_time is not None:
                ax.axvline(
                    event.extremum_time,
                    color=color,
                    alpha=0.45,
                    linewidth=1,
                    label=label,
                )


@dataclass(frozen=True)
class _BreathValues:
    """Times (s) and values of a per-breath result, for the TIV and EELI panels."""

    time: np.ndarray
    values: np.ndarray

    def __len__(self) -> int:
        return len(self.values)


def _breath_values(result: Any, point: str, session: M3Session) -> Any:
    """Return a per-breath result as times and values for a figure.

    Args:
        result (Any): An ``IntervalData`` of breaths, an object with ``time``
            and ``values``, or None.
        point (str): For an ``IntervalData``, the breath time of each value:
            ``"extremum_time"`` for TIV, ``"end_time"`` for EELI. A breath
            with no time at that point gives NaN.
        session (M3Session): Session with the start time of each recording.
            Breaths of another modality are moved to the EIT clock with
            these start times.

    Returns:
        Any: A ``_BreathValues`` for an ``IntervalData``, with times on the
            EIT clock; any other result as it is.
    """

    if not isinstance(result, IntervalData):
        return result
    breaths = on_clock_of(session, result.intervals, "eit")
    times = [getattr(breath, point, None) for breath in breaths]
    return _BreathValues(
        time=np.array([np.nan if t is None else t for t in times], dtype=float),
        values=np.asarray(result.values, dtype=float),
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
