import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full", app_title="M3Resp Annemijn multimodal viewer")


@app.cell
def _imports():
    from functools import lru_cache
    from pathlib import Path
    import sys

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ROOT = Path(__file__).resolve().parents[2]
    SRC = ROOT / "src"
    EITPROCESSING = ROOT.parent / "eitprocessing"
    RESURFEMG = ROOT.parent / "ReSurfEMG"
    for _path in (SRC, EITPROCESSING, RESURFEMG):
        if _path.exists() and str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

    from m3resp.adapters import EITProcessingAdapter

    _data_candidates = [
        ROOT / "data" / "source" / "eit_emg_annemijn",
        ROOT.parent / "m3resp" / "data" / "source" / "eit_emg_annemijn",
    ]
    DATA_DIR = next(
        (_path for _path in _data_candidates if _path.exists()), _data_candidates[0]
    )
    BIOPAC_FILE = DATA_DIR / "Paw_EMG_ajM3Resp_test.txt"
    EIT_FILE = DATA_DIR / "ajM3resp_03_001_01.bin"
    BIOPAC_SAMPLE_FREQUENCY = 2000.0

    return (
        BIOPAC_FILE,
        BIOPAC_SAMPLE_FREQUENCY,
        DATA_DIR,
        EITProcessingAdapter,
        EIT_FILE,
        Path,
        go,
        lru_cache,
        make_subplots,
        mo,
        np,
        pd,
    )


@app.cell
def _loaders(BIOPAC_SAMPLE_FREQUENCY, EITProcessingAdapter, lru_cache, np, pd):
    @lru_cache(maxsize=2)
    def load_biopac(path):
        data = pd.read_csv(
            path,
            sep="\t",
            skiprows=11,
            names=["paw", "emg_di", "emg_ps", "_empty"],
            usecols=[0, 1, 2],
            engine="c",
        )
        data.insert(
            0,
            "time_seconds",
            np.arange(len(data), dtype=float) / BIOPAC_SAMPLE_FREQUENCY,
        )
        return data

    @lru_cache(maxsize=2)
    def load_eit(path):
        sequence = EITProcessingAdapter().load(str(path), vendor="draeger")
        raw = sequence.eit_data["raw"]
        time = np.asarray(raw.time, dtype=float)
        time = time - time[0]
        pixel_impedance = np.asarray(raw.pixel_impedance, dtype=float)
        global_impedance = np.nansum(pixel_impedance, axis=(1, 2))
        mean_image = np.nanmean(pixel_impedance, axis=0)
        return {
            "time_seconds": time,
            "global_impedance": global_impedance,
            "sample_frequency": float(raw.sample_frequency),
            "n_frames": int(pixel_impedance.shape[0]),
            "pixel_shape": tuple(pixel_impedance.shape[1:]),
            "mean_image": mean_image,
        }

    def center(values):
        values = np.asarray(values, dtype=float)
        return values - np.nanmedian(values)

    def robust_range(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return None
        low, high = np.nanpercentile(values, [1, 99])
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            return None
        pad = 0.08 * (high - low)
        return [low - pad, high + pad]

    return center, load_biopac, load_eit, robust_range


@app.cell
def _load_data(BIOPAC_FILE, EIT_FILE, center, load_biopac, load_eit, np):
    biopac = load_biopac(str(BIOPAC_FILE))
    eit = load_eit(str(EIT_FILE))
    biopac_duration = float(biopac["time_seconds"].iloc[-1])
    eit_duration = float(eit["time_seconds"][-1])
    default_biopac_offset = max(0.0, biopac_duration - 120.0 - eit_duration)
    eit_centered = center(eit["global_impedance"])
    metadata = {
        "EIT file": EIT_FILE.name,
        "EIT frames": eit["n_frames"],
        "EIT Fs": eit["sample_frequency"],
        "EIT duration": round(eit_duration, 2),
        "Biopac file": BIOPAC_FILE.name,
        "Biopac samples": len(biopac),
        "Biopac Fs": 2000.0,
        "Biopac duration": round(biopac_duration, 2),
        "Default Biopac offset": round(default_biopac_offset, 2),
    }
    return (
        biopac,
        biopac_duration,
        default_biopac_offset,
        eit,
        eit_centered,
        eit_duration,
        metadata,
    )


@app.cell
def _controls(biopac_duration, default_biopac_offset, eit_duration, mo):
    time_range = mo.ui.range_slider(
        start=0.0,
        stop=float(eit_duration),
        step=1.0,
        value=[0.0, min(180.0, float(eit_duration))],
        label="EIT time window (s)",
        full_width=True,
    )
    biopac_offset = mo.ui.slider(
        start=0.0,
        stop=float(max(biopac_duration - eit_duration, 0.0)),
        step=0.5,
        value=float(default_biopac_offset),
        label="Biopac offset at EIT t=0 (s)",
        full_width=True,
    )
    downsample = mo.ui.slider(
        start=1,
        stop=200,
        step=1,
        value=20,
        label="Biopac downsample",
        full_width=True,
    )
    normalize = mo.ui.checkbox(value=True, label="Center EIT and EMG")
    show_emg_ps = mo.ui.checkbox(value=False, label="Show EMGps")
    mo.vstack(
        [
            mo.md("# Annemijn EIT + EMG + Paw"),
            time_range,
            biopac_offset,
            mo.hstack([downsample, normalize, show_emg_ps], justify="start", gap=2),
        ],
        gap=1,
    )
    return biopac_offset, downsample, normalize, show_emg_ps, time_range


@app.cell
def _windowed_data(
    biopac,
    biopac_offset,
    downsample,
    eit,
    eit_centered,
    normalize,
    np,
    time_range,
):
    t0, t1 = [float(value) for value in time_range.value]
    eit_time = np.asarray(eit["time_seconds"], dtype=float)
    eit_values = (
        eit_centered
        if normalize.value
        else np.asarray(eit["global_impedance"], dtype=float)
    )
    eit_mask = (eit_time >= t0) & (eit_time <= t1)
    eit_plot = {
        "time": eit_time[eit_mask],
        "values": eit_values[eit_mask],
    }

    biopac_t0 = t0 + float(biopac_offset.value)
    biopac_t1 = t1 + float(biopac_offset.value)
    biopac_mask = (biopac["time_seconds"] >= biopac_t0) & (
        biopac["time_seconds"] <= biopac_t1
    )
    biopac_window = biopac.loc[biopac_mask].iloc[:: int(downsample.value)].copy()
    biopac_plot_time = biopac_window["time_seconds"].to_numpy() - float(
        biopac_offset.value
    )
    emg_values = biopac_window["emg_di"].to_numpy(dtype=float)
    if normalize.value:
        emg_values = emg_values - np.nanmedian(emg_values)

    paw_plot = {
        "time": biopac_plot_time,
        "values": biopac_window["paw"].to_numpy(dtype=float),
    }
    emg_plot = {
        "time": biopac_plot_time,
        "values": emg_values,
        "emg_ps": biopac_window["emg_ps"].to_numpy(dtype=float),
    }
    return emg_plot, eit_plot, paw_plot


@app.cell
def _stacked_plot(
    emg_plot,
    eit_plot,
    go,
    make_subplots,
    normalize,
    paw_plot,
    robust_range,
    show_emg_ps,
):
    _fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=[
            "EIT global impedance",
            "Paw / airway pressure",
            "EMGdi",
        ],
    )
    _fig.add_trace(
        go.Scatter(
            x=eit_plot["time"],
            y=eit_plot["values"],
            mode="lines",
            name="EIT",
            line=dict(color="#185FA5", width=1.4),
            hovertemplate="t=%{x:.2f}s<br>EIT=%{y:.4g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=paw_plot["time"],
            y=paw_plot["values"],
            mode="lines",
            name="Paw",
            line=dict(color="#D1495B", width=1.1),
            hovertemplate="t=%{x:.2f}s<br>Paw=%{y:.3f} cmH2O<extra></extra>",
        ),
        row=2,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=emg_plot["time"],
            y=emg_plot["values"],
            mode="lines",
            name="EMGdi",
            line=dict(color="#2A9D8F", width=0.8),
            hovertemplate="t=%{x:.2f}s<br>EMGdi=%{y:.3f} mV<extra></extra>",
        ),
        row=3,
        col=1,
    )
    if show_emg_ps.value:
        _fig.add_trace(
            go.Scatter(
                x=emg_plot["time"],
                y=emg_plot["emg_ps"],
                mode="lines",
                name="EMGps",
                line=dict(color="#7A5195", width=0.7),
                opacity=0.65,
                hovertemplate="t=%{x:.2f}s<br>EMGps=%{y:.3f} mV<extra></extra>",
            ),
            row=3,
            col=1,
        )

    _fig.update_yaxes(
        title_text="a.u. centered" if normalize.value else "a.u.", row=1, col=1
    )
    _fig.update_yaxes(title_text="cmH2O", row=2, col=1)
    _fig.update_yaxes(
        title_text="mV centered" if normalize.value else "mV", row=3, col=1
    )
    _fig.update_xaxes(title_text="EIT-relative time (s)", row=3, col=1)

    for row, values in (
        (1, eit_plot["values"]),
        (2, paw_plot["values"]),
        (3, emg_plot["values"]),
    ):
        yrange = robust_range(values)
        if yrange is not None:
            _fig.update_yaxes(range=yrange, row=row, col=1)

    _fig.update_layout(
        template="plotly_white",
        height=760,
        showlegend=True,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=74, r=24, b=44, l=72),
    )
    _fig
    return


@app.cell
def _eit_image(eit, go):
    _fig = go.Figure(
        go.Heatmap(
            z=eit["mean_image"],
            colorscale="Viridis",
            colorbar=dict(title="mean"),
        )
    )
    _fig.update_layout(
        title="EIT mean image",
        template="plotly_white",
        height=360,
        width=420,
        margin=dict(t=48, r=20, b=20, l=20),
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(autorange="reversed"),
    )
    _fig
    return


@app.cell
def _metadata(metadata, mo):
    mo.ui.table([{"Field": key, "Value": value} for key, value in metadata.items()])
    return


if __name__ == "__main__":
    app.run()
