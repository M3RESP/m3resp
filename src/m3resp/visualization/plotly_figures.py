"""Interactive plotly versions of the session figures.

The plot functions in :mod:`m3resp.visualization.session` use these functions
when they get ``backend="plotly"``. Each function returns a
``plotly.graph_objects.Figure``. All times are in seconds.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np

from m3resp.data.events import BreathEvent

# Hex codes of the matplotlib "tab" colors, so both versions of a figure use
# the same colors.
_COLORS = {
    "tab:blue": "#1f77b4",
    "tab:green": "#2ca02c",
    "tab:red": "#d62728",
}
_SIGNAL_COLOR = _COLORS["tab:blue"]
# Height of one row of panels, in pixels.
_ROW_HEIGHT = 230
# Space above the first row for the legend, in pixels.
_TOP_MARGIN = 90


@dataclass
class EventOverlay:
    """Breaths to draw on top of one signal panel.

    Attributes:
        events (Iterable[BreathEvent]): The breaths. Times are in seconds.
        color (str): A matplotlib "tab" color name, for example ``"tab:red"``.
        label (str): The name of the breaths in the legend.
    """

    events: Iterable[BreathEvent]
    color: str
    label: str


@dataclass
class LinePanel:
    """One signal panel of a figure.

    Attributes:
        title (str): The title above the panel.
        ylabel (str): The label of the y axis, with the unit.
        time (np.ndarray): The time of each sample, in seconds.
        values (np.ndarray): The value of each sample.
        overlays (list[EventOverlay]): The breaths to draw on the panel.
    """

    title: str
    ylabel: str
    time: np.ndarray
    values: np.ndarray
    overlays: list[EventOverlay] = field(default_factory=list)


def line_grid_figure(
    panels: Sequence[Sequence[LinePanel]],
    *,
    max_points: int | None,
) -> Any:
    """Draw signal panels in a grid of rows and columns.

    The panels in one column share the time axis, so a zoom in one panel
    zooms all panels of that column.

    Args:
        panels (Sequence[Sequence[LinePanel]]): One list of panels per row.
            All rows have the same number of panels.
        max_points (int | None): The largest number of points drawn for one
            signal. See :func:`reduce_points`. ``None`` draws every sample.

    Returns:
        plotly.graph_objects.Figure: The figure.

    Raises:
        ImportError: If plotly is not installed.
    """

    go, make_subplots = _import_plotly()
    n_rows = len(panels)
    n_cols = len(panels[0])
    height = _ROW_HEIGHT * n_rows + _TOP_MARGIN
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes="columns",
        subplot_titles=[panel.title for row in panels for panel in row],
        vertical_spacing=70 / height,
        horizontal_spacing=0.08,
    )
    legend_labels: set[str] = set()

    for row_number, row in enumerate(panels, start=1):
        for col_number, panel in enumerate(row, start=1):
            time, values = reduce_points(panel.time, panel.values, max_points)
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=values,
                    mode="lines",
                    line={"color": _SIGNAL_COLOR, "width": 1},
                    name=panel.title,
                    showlegend=False,
                    hovertemplate="%{x:.3f} s<br>%{y:.4g}<extra></extra>",
                ),
                row=row_number,
                col=col_number,
            )
            for overlay in panel.overlays:
                _add_events(
                    fig,
                    go,
                    overlay,
                    time,
                    values,
                    row=row_number,
                    col=col_number,
                    legend_labels=legend_labels,
                )
            fig.update_yaxes(title_text=panel.ylabel, row=row_number, col=col_number)

    fig.update_xaxes(showticklabels=True, showgrid=True)
    fig.update_xaxes(title_text="Time (s)", row=n_rows)
    _apply_layout(fig, height)
    return fig


def eit_summary_figure(
    signal: LinePanel,
    tiv: Any,
    eeli: Any,
    mean_pixel_tiv: np.ndarray | None,
    *,
    max_points: int | None,
) -> Any:
    """Draw the four panels of the EIT summary.

    Args:
        signal (LinePanel): The global impedance with the EIT breaths.
        tiv (Any): The TIV of each breath, with ``time`` (s) and ``values``,
            or ``None``.
        eeli (Any): The EELI of each breath, with ``time`` (s) and
            ``values``, or ``None``.
        mean_pixel_tiv (np.ndarray | None): The mean TIV of each pixel, as a
            2D image, or ``None``.
        max_points (int | None): The largest number of points drawn for the
            global impedance. See :func:`reduce_points`.

    Returns:
        plotly.graph_objects.Figure: The figure.

    Raises:
        ImportError: If plotly is not installed.
    """

    go, make_subplots = _import_plotly()
    height = 2 * _ROW_HEIGHT + 120 + _TOP_MARGIN
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[signal.title, "Continuous TIV", "EELI", "Mean pixel TIV"],
        vertical_spacing=0.16,
        horizontal_spacing=0.1,
    )

    time, values = reduce_points(signal.time, signal.values, max_points)
    fig.add_trace(
        go.Scatter(
            x=time,
            y=values,
            mode="lines",
            line={"color": _SIGNAL_COLOR, "width": 1},
            name=signal.title,
            showlegend=False,
            hovertemplate="%{x:.3f} s<br>%{y:.4g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    legend_labels: set[str] = set()
    for overlay in signal.overlays:
        _add_events(
            fig, go, overlay, time, values, row=1, col=1, legend_labels=legend_labels
        )
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text=signal.ylabel, row=1, col=1)

    for sparse, ylabel, row, col in ((tiv, "TIV", 1, 2), (eeli, "EELI", 2, 1)):
        if sparse is None or not len(sparse):
            _hide_axes(fig, row=row, col=col)
            continue
        fig.add_trace(
            go.Scatter(
                x=np.asarray(sparse.time, dtype=float),
                y=np.asarray(sparse.values, dtype=float),
                mode="lines+markers",
                line={"color": _SIGNAL_COLOR, "width": 1},
                marker={"size": 5},
                name=ylabel,
                showlegend=False,
                hovertemplate="%{x:.3f} s<br>%{y:.4g}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Time (s)", row=row, col=col)
        fig.update_yaxes(title_text=ylabel, row=row, col=col)

    if mean_pixel_tiv is None:
        _hide_axes(fig, row=2, col=2)
    else:
        fig.add_trace(
            go.Heatmap(
                z=mean_pixel_tiv,
                colorscale="Viridis",
                colorbar={"len": 0.42, "y": 0.21, "thickness": 14},
                hovertemplate="column %{x}, row %{y}<br>%{z:.4g}<extra></extra>",
            ),
            row=2,
            col=2,
        )
        # Row 0 of the image is at the top and the pixels are square.
        fig.update_yaxes(
            autorange="reversed", scaleanchor="x4", constrain="domain", row=2, col=2
        )
        fig.update_xaxes(constrain="domain", row=2, col=2)

    _apply_layout(fig, height)
    return fig


def reduce_points(
    time: np.ndarray,
    values: np.ndarray,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep at most ``max_points`` samples of a signal for display.

    The function cuts the signal into ``max_points // 2`` bins with the same
    number of samples. From each bin it keeps the sample with the lowest value
    and the sample with the highest value, in time order. The peaks and the
    troughs of the signal stay visible. A bin with only NaN values keeps its
    first sample, so the gap stays visible.

    Args:
        time (np.ndarray): The time of each sample, in seconds.
        values (np.ndarray): The value of each sample.
        max_points (int | None): The largest number of samples to keep.
            ``None`` keeps every sample.

    Returns:
        tuple[np.ndarray, np.ndarray]: The time and the values of the kept
        samples.

    Raises:
        ValueError: If ``max_points`` is smaller than 2.
    """

    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if max_points is None or len(values) <= max_points:
        return time, values
    if max_points < 2:
        raise ValueError(f"max_points must be 2 or more, got {max_points}.")

    edges = np.linspace(0, len(values), max_points // 2 + 1).astype(int)
    keep: list[int] = []
    for start, stop in pairwise(edges):
        if stop <= start:
            continue
        chunk = values[start:stop]
        finite = np.isfinite(chunk)
        if not finite.any():
            keep.append(int(start))
            continue
        low = int(np.argmin(np.where(finite, chunk, np.inf)))
        high = int(np.argmax(np.where(finite, chunk, -np.inf)))
        keep.extend(start + index for index in sorted({low, high}))
    indices = np.asarray(keep, dtype=int)
    return time[indices], values[indices]


def _import_plotly() -> tuple[Any, Any]:
    try:
        from plotly import graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise ImportError(
            "Interactive figures require plotly. Install the EIT or EMG optional "
            "dependencies, or install plotly directly."
        ) from exc
    return go, make_subplots


def _add_events(
    fig: Any,
    go: Any,
    overlay: EventOverlay,
    time: np.ndarray,
    values: np.ndarray,
    *,
    row: int,
    col: int,
    legend_labels: set[str],
) -> None:
    """Shade each breath and draw a line at each extremum on one panel.

    Only breaths inside the time range of the drawn signal are shown. The
    extremum lines go from the lowest to the highest value of the signal.
    The legend shows each label once per figure.
    """

    if len(time) == 0:
        return
    t_min, t_max = float(time[0]), float(time[-1])
    color = _COLORS.get(overlay.color, overlay.color)
    finite = values[np.isfinite(values)]

    line_x: list[float | None] = []
    line_y: list[float | None] = []
    for event in overlay.events:
        if event.end_time < t_min or event.start_time > t_max:
            continue
        if event.end_time > event.start_time:
            fig.add_vrect(
                x0=event.start_time,
                x1=event.end_time,
                fillcolor=color,
                opacity=0.08,
                line_width=0,
                layer="below",
                row=row,
                col=col,
            )
        if event.extremum_time is not None and len(finite):
            line_x.extend([event.extremum_time, event.extremum_time, None])
            line_y.extend([float(finite.min()), float(finite.max()), None])

    if not line_x:
        return
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line={"color": color, "width": 1},
            opacity=0.6,
            name=overlay.label,
            legendgroup=overlay.label,
            showlegend=overlay.label not in legend_labels,
            hovertemplate=f"{overlay.label}<br>%{{x:.3f}} s<extra></extra>",
        ),
        row=row,
        col=col,
    )
    legend_labels.add(overlay.label)


def _hide_axes(fig: Any, *, row: int, col: int) -> None:
    fig.update_xaxes(visible=False, row=row, col=col)
    fig.update_yaxes(visible=False, row=row, col=col)


def _apply_layout(fig: Any, height: int) -> None:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 70, "r": 30, "t": _TOP_MARGIN, "b": 50},
        # The legend is at the top left. The plotly tool buttons are at the
        # top right.
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "xref": "container",
            "y": 1,
            "yanchor": "top",
            "yref": "container",
        },
        hovermode="closest",
    )
