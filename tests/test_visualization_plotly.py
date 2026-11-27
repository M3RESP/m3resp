from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from m3resp import BreathEvent, M3Session
from m3resp.visualization import (
    plot_eit_processing_summary,
    plot_session_overview,
    plot_synchronization_comparison,
)
from m3resp.visualization._plotly_figures import pixel_map_traces, reduce_points
from m3resp.visualization.pixel_map_style import (
    pixel_colors,
    pixel_map_style,
    scale_limits,
)

pytest.importorskip("plotly")


class FakeSparse:
    """One value per breath, with ``time`` in seconds and ``values``."""

    def __init__(self, time: list[float], values: list[float]) -> None:
        self.time = time
        self.values = values

    def __len__(self) -> int:
        return len(self.values)


def emg_session() -> M3Session:
    session = M3Session()
    session.processed["emg"] = {
        "channel": 1,
        "fs": 1000.0,
        "metadata": {"labels": ["unused", "diaphragm"], "units": ["mV", "uV"]},
        "raw_channel": [0.0, 1.0, 0.0, -1.0, 0.0],
        "envelope": [0.0, 0.25, 0.5, 0.25, 0.0],
    }
    session.add_events(
        "emg_breaths",
        [BreathEvent("emg", 0.002, 0.002, extremum_time=0.002)],
    )
    return session


def titles(fig) -> list[str]:
    return [annotation.text for annotation in fig.layout.annotations]


def test_plotly_overview_has_one_panel_per_emg_signal():
    fig = plot_session_overview(emg_session(), max_seconds=None, backend="plotly")

    assert titles(fig) == ["EMG raw (diaphragm)", "EMG envelope (diaphragm)"]
    assert fig.layout.yaxis.title.text == "EMG amplitude (uV)"
    assert fig.layout.yaxis2.title.text == "EMG amplitude (uV)"
    assert fig.layout.xaxis2.title.text == "Time (s)"
    raw, envelope = (trace for trace in fig.data if trace.name != "EMG breath")
    assert list(raw.x) == [0.0, 0.001, 0.002, 0.003, 0.004]
    assert list(raw.y) == [0.0, 1.0, 0.0, -1.0, 0.0]
    assert list(envelope.y) == [0.0, 0.25, 0.5, 0.25, 0.0]


def test_plotly_overview_draws_the_breath_extremum_from_lowest_to_highest_value():
    fig = plot_session_overview(emg_session(), max_seconds=None, backend="plotly")

    breath_traces = [trace for trace in fig.data if trace.name == "EMG breath"]
    assert len(breath_traces) == 2
    assert list(breath_traces[0].x) == [0.002, 0.002, None]
    assert list(breath_traces[0].y) == [-1.0, 1.0, None]
    assert list(breath_traces[1].y) == [0.0, 0.5, None]
    # The legend shows the label once.
    assert [trace.showlegend for trace in breath_traces] == [True, False]


def test_plotly_overview_limits_the_time_to_max_seconds():
    fig = plot_session_overview(emg_session(), max_seconds=0.002, backend="plotly")

    assert list(fig.data[0].x) == [0.0, 0.001, 0.002]


def test_unknown_backend_raises_a_clear_error():
    with pytest.raises(ValueError, match="backend must be one of"):
        plot_session_overview(emg_session(), backend="bokeh")


def test_plotly_synchronization_comparison_shifts_signal_time_by_alignment_offset():
    session = M3Session()
    session.processed["emg"] = {
        "channel": 0,
        "fs": 1000.0,
        "metadata": {"labels": ["EMG"], "units": ["uV"]},
        "envelope": [0.0, 0.0, 1.0, 2.0, 3.0],
    }
    session.add_events(
        "emg_breaths",
        [BreathEvent("emg", 0.002, 0.004, extremum_time=0.003)],
    )
    session.synchronize_multimodal_breaths(offset_seconds={"emg": -0.002})

    fig = plot_synchronization_comparison(session, max_seconds=None, backend="plotly")

    assert titles(fig) == [
        "EMG envelope (EMG) before synchronization",
        "EMG envelope (EMG) after synchronization",
    ]
    before, after = (t for t in fig.data if t.name.startswith("EMG envelope"))
    assert list(before.x) == [0.0, 0.001, 0.002, 0.003, 0.004]
    assert list(after.x) == [0.0, 0.001, 0.002]
    assert list(after.y) == [1.0, 2.0, 3.0]
    original = next(t for t in fig.data if t.name == "EMG breath original")
    synchronized = next(t for t in fig.data if t.name == "EMG breath synchronized")
    assert list(original.x) == [0.003, 0.003, None]
    assert list(synchronized.x) == pytest.approx([0.001, 0.001, None])
    # The breath from 0.002 s to 0.004 s is shaded in the "before" panel.
    first_shape = fig.layout.shapes[0]
    assert (first_shape.x0, first_shape.x1) == (0.002, 0.004)


def test_plotly_eit_summary_has_signal_tiv_eeli_and_pixel_map():
    session = M3Session()
    session.processed["eit"] = {
        "filtered_global_impedance": SimpleNamespace(
            time=[0.0, 0.05, 0.1, 0.15],
            values=[1.0, 3.0, 2.0, 1.0],
            name="impedance (a.u.)",
        ),
        "continuous_tiv": FakeSparse([0.05], [2.0]),
        "eeli": FakeSparse([], []),
        "pixel_tiv": SimpleNamespace(
            values=[np.ones((2, 3)), 3 * np.ones((2, 3))],
        ),
    }

    fig = plot_eit_processing_summary(session, backend="plotly")

    assert titles(fig) == [
        "EIT global impedance",
        "Continuous TIV",
        "EELI",
        "Mean pixel TIV",
    ]
    signal, tiv, background, pixel_map = fig.data
    assert list(signal.y) == [1.0, 3.0, 2.0, 1.0]
    assert list(tiv.y) == [2.0]
    assert np.array_equal(np.asarray(background.z), np.zeros((2, 3)))
    assert np.array_equal(np.asarray(pixel_map.z), 2 * np.ones((2, 3)))
    assert pixel_map.colorbar.title.text == "TIV (a.u.)"
    assert fig.layout.yaxis.title.text == "impedance (a.u.)"
    # EELI has no values, so its panel has no axes.
    assert fig.layout.xaxis3.visible is False


def rgba_of(text: str) -> tuple[float, ...]:
    """Red, green and blue (0 to 255) and alpha of a plotly colour text."""

    return tuple(float(part) for part in text.removeprefix("rgba(")[:-1].split(","))


def plotly_pixel_color(trace, value: float) -> tuple[float, ...]:
    """The colour plotly gives a value in a layer with a step colour scale."""

    position = (value - trace.zmin) / (trace.zmax - trace.zmin)
    steps = len(trace.colorscale) // 2
    step = min(int(position * steps), steps - 1)
    return rgba_of(trace.colorscale[2 * step][1])


def test_plotly_pixel_map_has_the_matplotlib_colours_of_the_tiv_style():
    style = pixel_map_style("tiv")
    values = np.array([[0.0, 0.5, 1.0], [2.0, -0.5, np.nan]])

    background, pixel_map, under = pixel_map_traces(values, style=style)

    expected = np.round(pixel_colors(style, values=values) * 255)
    assert (pixel_map.zmin, pixel_map.zmax) == scale_limits(style, values=values)
    for row, column in [(0, 0), (0, 1), (0, 2), (1, 0)]:
        color = plotly_pixel_color(pixel_map, values[row, column])
        assert color[:3] == tuple(expected[row, column, :3])
    # The negative pixel is purple, on top of the colour scale.
    assert np.isnan(pixel_map.z[1][1])
    assert under.z[1][1] == -0.5
    assert rgba_of(under.colorscale[0][1]) == (128.0, 0.0, 128.0, 1.0)
    # The NaN pixel shows the grey background.
    assert rgba_of(background.colorscale[0][1]) == (169.0, 169.0, 169.0, 1.0)
    assert np.isnan(pixel_map.z[1][2]) and np.isnan(under.z[1][2])
    assert pixel_map.colorbar.title.text == "TIV (a.u.)"


def test_plotly_pixel_map_of_a_mask_style_is_an_error():
    with pytest.raises(ValueError, match="no plotly version"):
        pixel_map_traces(np.ones((2, 2)), style=pixel_map_style("mask"))


def test_matplotlib_eit_summary_has_the_tiv_colours():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    session = M3Session()
    session.processed["eit"] = {
        "filtered_global_impedance": SimpleNamespace(
            time=[0.0, 0.05, 0.1], values=[1.0, 3.0, 1.0], name="impedance (a.u.)"
        ),
        "pixel_tiv": SimpleNamespace(
            values=[np.array([[0.0, 1.0], [-1.0, np.nan]]), np.ones((2, 2))],
        ),
    }

    fig = plot_eit_processing_summary(session)

    image = fig.axes[3].images[0]
    mean = np.array([[0.5, 1.0], [0.0, 1.0]])
    drawn = image.cmap(image.norm(np.ma.masked_invalid(mean)))
    expected = pixel_colors(pixel_map_style("tiv"), values=mean)
    assert np.allclose(drawn, expected)
    assert image.colorbar.ax.get_ylabel() == "TIV (a.u.)"
    plt.close(fig)


def test_reduce_points_keeps_the_lowest_and_highest_value_of_each_bin():
    time = np.arange(8, dtype=float)
    values = np.array([0.0, 5.0, 1.0, 2.0, 9.0, 3.0, -4.0, 1.0])

    kept_time, kept_values = reduce_points(time, values, max_points=4)

    assert list(kept_time) == [0.0, 1.0, 4.0, 6.0]
    assert list(kept_values) == [0.0, 5.0, 9.0, -4.0]


def test_reduce_points_keeps_short_signals_and_none_unchanged():
    time = np.arange(3, dtype=float)
    values = np.array([1.0, 2.0, 3.0])

    assert list(reduce_points(time, values, max_points=5)[1]) == [1.0, 2.0, 3.0]
    assert list(reduce_points(time, values, max_points=None)[1]) == [1.0, 2.0, 3.0]


def test_reduce_points_keeps_the_first_sample_of_a_bin_with_only_nan():
    time = np.arange(8, dtype=float)
    values = np.array([np.nan, np.nan, np.nan, np.nan, 1.0, 5.0, 2.0, 3.0])

    kept_time, kept_values = reduce_points(time, values, max_points=4)

    assert list(kept_time) == [0.0, 4.0, 5.0]
    assert np.isnan(kept_values[0])
    assert list(kept_values[1:]) == [1.0, 5.0]


def test_reduce_points_rejects_max_points_below_two():
    with pytest.raises(ValueError, match="max_points must be 2 or more"):
        reduce_points(np.arange(5.0), np.arange(5.0), max_points=1)
