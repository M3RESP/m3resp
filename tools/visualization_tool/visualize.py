import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full", app_title="Pendelluft — raw data explorer")


@app.cell
def _imports():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return Path, go, json, make_subplots, mo, np, pd


@app.cell
def _discover(Path):
    _data_root = Path(__file__).parent.parent / "Data"
    _dir_sessions = [
        f"{_sub.name}/{_sess.name}"
        for _sub in sorted(_data_root.iterdir())
        if _sub.is_dir()
        for _sess in sorted(_sub.iterdir())
        if _sess.is_dir()
        if any(_sess.glob("*_metadata.json"))
    ]
    _pickle_sessions = [
        f"{_sub.name}/{_f.stem}"
        for _sub in sorted(_data_root.iterdir())
        if _sub.is_dir()
        for _f in sorted(_sub.glob("*.pickle"))
    ]
    sessions = sorted(_dir_sessions + _pickle_sessions)
    data_root = _data_root
    return data_root, sessions


@app.cell
def _ui_header(mo):
    mo.md("""
    # Pendelluft — raw data explorer
    """)
    return


@app.cell
def _ui_session(mo, sessions):
    session_dropdown = mo.ui.dropdown(
        options=sessions,
        value=sessions[0] if sessions else None,
        label="Session",
    )
    mo.hstack([mo.md("**Dataset**"), session_dropdown], justify="start", gap=1)
    return (session_dropdown,)


@app.cell
def _load(data_root, json, np, pd, session_dropdown):
    import pickle as _pickle

    _session_path = data_root / session_dropdown.value
    _pickle_path = data_root / f"{session_dropdown.value}.pickle"

    if _pickle_path.exists():

        class _PathShim:
            def __new__(cls, *args, **kwargs):
                from pathlib import Path as _P

                parts = [
                    str(a.decode("utf-8") if isinstance(a, bytes) else a) for a in args
                ]
                return _P(*parts) if parts else _P()

        class _Unpickler(_pickle.Unpickler):
            def find_class(self, module, name):
                if module in ("pathlib", "pathlib._local") and name in (
                    "PosixPath",
                    "PurePosixPath",
                    "WindowsPath",
                    "PureWindowsPath",
                ):
                    return _PathShim
                return super().find_class(module, name)

        with _pickle_path.open("rb") as _fh:
            _seq = _Unpickler(_fh).load()

        _raw = _seq.eit_data["raw"]
        _t = _raw.time - _raw.time[0]
        _cd = _seq.continuous_data

        eit_pixels = _raw.pixel_impedance
        eit_df = pd.DataFrame(
            {
                "time_seconds": _t,
                "global_impedance": _cd["global_impedance_(raw)"].values,
            }
        )
        # 'airway pressure', 'flow', 'volume' are already resampled to EIT rate (20 Hz)
        vent_df = pd.DataFrame(
            {
                "time_seconds": _t,
                "pressure": _cd["airway pressure"].values,
                "flow": _cd["flow"].values,
                "volume": _cd["volume"].values,
            }
        )
        # Pes/Pga are at 256 Hz — use their own time axis so the EMG downsample slider works
        if "synchronized_pes" in _cd:
            _t_pes = _cd["synchronized_pes"].time - _raw.time[0]
            _pes = _cd["synchronized_pes"].values
            _pga = (
                _cd["synchronized_pga"].values
                if "synchronized_pga" in _cd
                else np.zeros(len(_t_pes))
            )
        else:
            _t_pes, _pes, _pga = _t, np.zeros(len(_t)), np.zeros(len(_t))
        emg_df = pd.DataFrame({"time_seconds": _t_pes, "emg_0": _pes, "emg_1": _pga})
        eit_components = {
            k: np.zeros(len(_t))
            for k in ("breathing", "cardiac", "baseline", "drift", "noise")
        }
        p_mus = np.array([])
        metadata = {
            "source": "pickle",
            "file": _pickle_path.name,
            "n_frames": len(_t),
            "sample_frequency": float(_raw.sample_frequency),
        }
    else:
        _meta_files = list(_session_path.glob("*_metadata.json"))
        with _meta_files[0].open() as _f:
            metadata = json.load(_f)
        _basename = _meta_files[0].stem.replace("_metadata", "")

        vent_df = pd.read_csv(_session_path / f"{_basename}_ventilator.csv")
        eit_df = pd.read_csv(_session_path / f"{_basename}_eit_global.csv")
        emg_df = pd.read_csv(_session_path / f"{_basename}_emg.csv")
        eit_pixels = np.load(_session_path / f"{_basename}_eit_pixels.npy")
        eit_components = np.load(_session_path / f"{_basename}_eit_components.npz")
        p_mus = np.load(_session_path / f"{_basename}_ventilator_p_mus.npy")

    return eit_components, eit_df, eit_pixels, emg_df, metadata, p_mus, vent_df


@app.cell
def _ui_time_range(mo, vent_df):
    _t_max = float(vent_df["time_seconds"].max())
    time_range_slider = mo.ui.range_slider(
        start=0.0,
        stop=_t_max,
        step=0.5,
        value=[0.0, _t_max],
        label="Time range (s)",
        full_width=True,
    )
    mo.vstack([mo.md("---\n### Time range"), time_range_slider])
    return (time_range_slider,)


@app.cell
def _section_ventilator(mo):
    mo.md("""
    ## Ventilator
    """)
    return


@app.cell
def _plot_ventilator(go, make_subplots, np, p_mus, time_range_slider, vent_df):
    _t0, _t1 = time_range_slider.value
    _mask = (vent_df["time_seconds"] >= _t0) & (vent_df["time_seconds"] <= _t1)
    _df = vent_df[_mask]
    _t_pmus = np.arange(len(p_mus)) / 100.0
    _pmus_mask = (_t_pmus >= _t0) & (_t_pmus <= _t1)

    _fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Pressure (cmH₂O)",
            "Flow (L/s)",
            "Volume (L)",
            "P_mus (cmH₂O)",
        ],
    )
    _fig.add_trace(
        go.Scatter(
            x=_df["time_seconds"],
            y=_df["pressure"],
            name="Pressure",
            line=dict(color="steelblue"),
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_df["time_seconds"],
            y=_df["flow"],
            name="Flow",
            line=dict(color="darkorange"),
        ),
        row=2,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_df["time_seconds"],
            y=_df["volume"],
            name="Volume",
            line=dict(color="seagreen"),
        ),
        row=3,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_t_pmus[_pmus_mask],
            y=p_mus[_pmus_mask],
            name="P_mus",
            line=dict(color="crimson"),
        ),
        row=4,
        col=1,
    )
    _fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    _fig.update_layout(height=600, showlegend=False, margin=dict(t=40, b=20))
    _fig
    return


@app.cell
def _section_eit_global(mo):
    mo.md("""
    ## EIT — global signal & components
    """)
    return


@app.cell
def _plot_eit_global(
    eit_components,
    eit_df,
    go,
    make_subplots,
    np,
    time_range_slider,
):
    _t0, _t1 = time_range_slider.value
    _mask = (eit_df["time_seconds"] >= _t0) & (eit_df["time_seconds"] <= _t1)
    _df = eit_df[_mask]
    _t_eit = np.arange(len(eit_components["breathing"])) / 20.0
    _cmask = (_t_eit >= _t0) & (_t_eit <= _t1)

    _fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=["Global impedance (a.u.)", "Components"],
    )
    _fig.add_trace(
        go.Scatter(
            x=_df["time_seconds"],
            y=_df["global_impedance"],
            name="Global",
            line=dict(color="teal"),
        ),
        row=1,
        col=1,
    )
    for _comp, _color in {
        "breathing": "steelblue",
        "cardiac": "crimson",
        "baseline": "gray",
        "drift": "darkorange",
        "noise": "lightgray",
    }.items():
        _fig.add_trace(
            go.Scatter(
                x=_t_eit[_cmask],
                y=eit_components[_comp][_cmask],
                name=_comp,
                line=dict(color=_color),
            ),
            row=2,
            col=1,
        )
    _fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    _fig.update_layout(height=500, margin=dict(t=40, b=20))
    _fig
    return


@app.cell
def _section_eit_heatmap(mo):
    mo.md("""
    ## EIT — pixel heatmap
    """)
    return


@app.cell
def _ui_frame(eit_pixels, mo):
    frame_slider = mo.ui.slider(
        start=0,
        stop=eit_pixels.shape[0] - 1,
        step=1,
        value=0,
        label="Frame index",
        full_width=True,
    )
    frame_slider
    return (frame_slider,)


@app.cell
def _plot_eit_heatmap(eit_pixels, frame_slider, go, np):
    _frame = eit_pixels[frame_slider.value]
    _fig = go.Figure(
        go.Heatmap(
            z=_frame,
            colorscale="Viridis",
            zmin=0,
            zmax=float(np.max(eit_pixels)),
            colorbar=dict(title="Impedance (a.u.)"),
        )
    )
    _fig.update_layout(
        title=f"Frame {frame_slider.value} — t = {frame_slider.value / 20.0:.2f} s",
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(autorange="reversed"),
        height=460,
        width=460,
        margin=dict(t=40, b=20),
    )
    _fig
    return


@app.cell
def _section_emg(mo):
    mo.md("""
    ## EMG
    """)
    return


@app.cell
def _ui_emg_downsample(mo):
    downsample = mo.ui.slider(
        start=1, stop=32, step=1, value=8, label="Downsample factor"
    )
    mo.hstack(
        [mo.md("EMG is 2048 Hz — downsample to keep rendering fast:"), downsample],
        justify="start",
        gap=1,
    )
    return (downsample,)


@app.cell
def _plot_emg(downsample, emg_df, go, make_subplots, time_range_slider):
    _t0, _t1 = time_range_slider.value
    _mask = (emg_df["time_seconds"] >= _t0) & (emg_df["time_seconds"] <= _t1)
    _df = emg_df[_mask].iloc[:: downsample.value]
    _fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=["EMG channel 0 (µV)", "EMG channel 1 (µV)"],
    )
    _fig.add_trace(
        go.Scatter(
            x=_df["time_seconds"],
            y=_df["emg_0"],
            name="EMG 0",
            line=dict(color="mediumorchid", width=0.8),
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_df["time_seconds"],
            y=_df["emg_1"],
            name="EMG 1",
            line=dict(color="darkcyan", width=0.8),
        ),
        row=2,
        col=1,
    )
    _fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    _fig.update_layout(height=420, showlegend=False, margin=dict(t=40, b=20))
    _fig
    return


@app.cell
def _section_metadata(mo):
    mo.md("""
    ## Session metadata
    """)
    return


@app.cell
def _display_metadata(metadata, mo):
    if "records" in metadata:
        mo.ui.table(
            [
                {
                    "Modality": mod,
                    "Fs (Hz)": info["sample_frequency"],
                    "Samples": info["n_samples"],
                    "Shape": str(info["array_shape"]),
                    "Labels": ", ".join(info["labels"]),
                    "Units": ", ".join(info["units"]),
                }
                for mod, info in metadata["records"].items()
            ]
        )
    else:
        mo.ui.table([{"Key": k, "Value": str(v)} for k, v in metadata.items()])
    return


if __name__ == "__main__":
    app.run()
