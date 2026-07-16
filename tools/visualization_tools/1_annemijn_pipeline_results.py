import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full", app_title="M3Resp — annemijn pipeline results")


@app.cell
def _imports():
    import os
    import sys
    from functools import lru_cache
    from importlib.util import find_spec
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # matplotlib is used only to receive the *native* figures produced by the
    # eitprocessing / ReSurfEMG plotting helpers; marimo renders a returned
    # Figure directly.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _ensure_importable(module_name, local_path):
        # Prefer a properly installed package; only fall back to a sibling
        # checkout (appended, so it can't shadow an install) when missing.
        if find_spec(module_name) is not None:
            return
        if local_path.exists() and str(local_path) not in sys.path:
            sys.path.append(str(local_path))

    # tools/visualization_tool/<this file>  ->  repo root is parents[2].
    ROOT = Path(__file__).resolve().parents[2]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    _ensure_importable("eitprocessing", ROOT.parent / "eitprocessing")
    _ensure_importable("resurfemg", ROOT.parent / "ReSurfEMG")

    DEFAULT_SPEC = ROOT / "examples" / "annemijn_multimodal" / "annemijn.pipeline.yaml"

    return (
        DEFAULT_SPEC,
        Path,
        ROOT,
        go,
        lru_cache,
        make_subplots,
        mo,
        np,
        os,
        plt,
    )


@app.cell
def _intro(DEFAULT_SPEC, mo):
    mo.md(
        f"""
        # M3Resp — annemijn multimodal pipeline results

        This notebook **runs the declarative pipeline**
        [`{DEFAULT_SPEC.name}`]() end-to-end (EIT + diaphragm sEMG + airway
        pressure), synchronizes the raw signals with the interference and
        cross-correlation estimate, and visualizes the results. Native plotting
        functions from `eitprocessing` are reused wherever they exist:

        * `eitprocessing` — `RateDetection.plotting.plot(...)` (frequency
          analysis) and `PixelMap.plotting.imshow(...)` (per-pixel TIV map).
        Time-series overlays (impedance + detected breaths, sEMG envelope) use
        Plotly, matching the other tools in this folder.
        """
    )
    return


@app.cell
def _run_controls(DEFAULT_SPEC, mo):
    spec_path = mo.ui.text(
        value=str(DEFAULT_SPEC),
        label="Pipeline spec (.yaml)",
        full_width=True,
    )
    run_button = mo.ui.run_button(label="Run / re-run pipeline")
    mo.vstack(
        [
            spec_path,
            mo.hstack(
                [run_button, mo.md("*Runs once on load; click to re-run.*")],
                justify="start",
                gap=1,
            ),
        ]
    )
    return run_button, spec_path


@app.cell
def _run_pipeline(ROOT, mo, os, run_button, spec_path):
    # Referencing `run_button.value` makes this cell reactive to the button
    # (it also runs once on load, when the value is still False). It does NOT
    # depend on any slider, so scrubbing the plots never re-runs the pipeline.
    _ = run_button.value

    from m3resp.workflows.engine import run_spec

    # `inputs:` paths in the spec are resolved relative to the working
    # directory (the repo root); `outputs.dir` relative to the spec file.
    os.chdir(ROOT)
    _result = run_spec(spec_path.value)
    ctx = dict(_result.context.values)
    session = _result.session

    mo.output.replace(
        mo.callout(
            mo.md(f"Pipeline **{_result.name}** ran successfully."),
            kind="success",
        )
    )
    return ctx, session


@app.cell
def _section_synced_raw(mo):
    mo.vstack([mo.md("## Synchronized raw EIT, EMG, and ventilation data")])
    return


@app.cell
def _raw_sync_data(np, session):
    _eit_signal = session.eit.global_impedance if session.eit is not None else None
    _eit_time = np.asarray(getattr(_eit_signal, "time", []), dtype=float)
    _eit_values = np.asarray(getattr(_eit_signal, "values", []), dtype=float)
    if _eit_time.size:
        _eit_time = _eit_time - _eit_time[0]

    _emg_payload = (
        session.emg.data
        if session.emg is not None and isinstance(session.emg.data, dict)
        else {}
    )
    _emg_array = np.asarray(_emg_payload.get("array", []), dtype=float)
    _emg_metadata = _emg_payload.get("metadata") or {}
    _emg_fs = float(_emg_metadata.get("fs", 0.0) or 0.0)
    _emg_time = (
        np.arange(_emg_array.shape[-1], dtype=float) / _emg_fs
        if _emg_array.ndim == 2 and _emg_array.shape[0] >= 2 and _emg_fs > 0
        else np.asarray([], dtype=float)
    )
    _labels = _emg_metadata.get("labels") or []
    _units = _emg_metadata.get("units") or []

    raw_sync_data = {
        "eit": {
            "time": _eit_time,
            "values": _eit_values,
            "title": "EIT raw global impedance",
            "ylabel": getattr(_eit_signal, "label", "impedance"),
            "color": "#185FA5",
        },
        "emg": {
            "time": _emg_time,
            "values": _emg_array[1] if _emg_time.size else np.asarray([]),
            "title": _labels[1] if len(_labels) > 1 else "Diaphragm sEMG",
            "ylabel": _units[1] if len(_units) > 1 else "mV",
            "color": "#2A9D8F",
        },
        "vent": {
            "time": _emg_time,
            "values": _emg_array[0] if _emg_time.size else np.asarray([]),
            "title": _labels[0] if _labels else "Airway pressure (Paw)",
            "ylabel": _units[0] if _units else "pressure",
            "color": "#D1495B",
        },
    }
    return (raw_sync_data,)


@app.cell
def _ui_raw_sync_range(mo, raw_sync_data):
    _durations = [
        float(_trace["time"][-1])
        for _trace in raw_sync_data.values()
        if _trace["time"].size
    ]
    _duration = min(_durations) if _durations else 1.0
    raw_sync_range = mo.ui.range_slider(
        start=0.0,
        stop=max(1.0, round(_duration, 1)),
        step=1.0,
        value=[0.0, round(min(120.0, _duration), 1)],
        label="Synchronized raw time window (s)",
        full_width=True,
    )
    raw_sync_range
    return (raw_sync_range,)


@app.cell
def _plot_synced_raw(go, make_subplots, mo, np, raw_sync_data, raw_sync_range, session):
    _missing = [
        _name for _name, _trace in raw_sync_data.items() if not _trace["time"].size
    ]
    if _missing:
        mo.output.replace(
            mo.callout(
                mo.md(f"Missing synchronized raw data: {', '.join(_missing)}."),
                kind="warn",
            )
        )
    else:
        _lo, _hi = [float(value) for value in raw_sync_range.value]
        _fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=[
                raw_sync_data["vent"]["title"],
                raw_sync_data["eit"]["title"],
                raw_sync_data["emg"]["title"],
            ],
        )
        for _row, _name in enumerate(("vent", "eit", "emg"), start=1):
            _trace = raw_sync_data[_name]
            _time = _trace["time"]
            _values = _trace["values"]
            _indices = np.flatnonzero((_time >= _lo) & (_time <= _hi))
            _stride = max(1, int(np.ceil(_indices.size / 20_000)))
            _indices = _indices[::_stride]
            _fig.add_trace(
                go.Scattergl(
                    x=_time[_indices],
                    y=_values[_indices],
                    mode="lines",
                    name=_name.upper(),
                    line=dict(color=_trace["color"], width=1.0),
                ),
                row=_row,
                col=1,
            )
            _fig.update_yaxes(title_text=_trace["ylabel"], row=_row, col=1)

        _alignment = session.parameters.get("raw_alignment", {}) or {}
        _estimate = _alignment.get("estimated_offset_seconds")
        _subtitle = (
            f"Estimated EIT-to-Biopac offset: {_estimate:.2f} s"
            if _estimate is not None
            else "Estimated EIT-to-Biopac offset unavailable"
        )
        _fig.update_xaxes(title_text="synchronized time (s)", row=3, col=1)
        _fig.update_layout(
            title=dict(text=_subtitle, font=dict(size=14)),
            template="plotly_white",
            height=680,
            hovermode="x unified",
            showlegend=False,
            margin=dict(t=70, r=20, b=44, l=76),
        )
        mo.output.replace(_fig)
    return


@app.cell
def _summary(ctx, mo, np):
    _oe = ctx.get("offset_estimation", {}) or {}
    _interf = _oe.get("interference", {}) or {}
    _xcorr = _oe.get("crosscorrelation", {}) or {}
    _rr = ctx.get("respiratory_rate_hz")
    _hr = ctx.get("heart_rate_hz")
    _n_eit = len(ctx["breath_intervals"].values)
    _n_emg = len(ctx["emg_breath_events"])
    _gi = ctx["global_impedance"]
    _eit_dur = float(np.asarray(_gi.time)[-1] - np.asarray(_gi.time)[0])
    _emg_dur = float(np.asarray(ctx["processed_emg"]["raw_channel"]).size) / float(
        ctx["processed_emg"]["fs"]
    )

    _rows = [
        (
            "EIT–Biopac offset (interference + cross-correlation)",
            f"{_oe.get('offset_seconds', float('nan')):.2f} s",
        ),
        (
            "Cross-correlation refinement",
            f"{_xcorr.get('lag_seconds', float('nan')):+.2f} s",
        ),
        (
            "EIT-off edge in Biopac time",
            f"{_interf.get('edge_time_seconds', float('nan')):.1f} s",
        ),
        ("EIT duration", f"{_eit_dur:.1f} s"),
        ("Biopac duration", f"{_emg_dur:.1f} s"),
        ("EIT respiratory rate", f"{_rr * 60:.1f} /min" if _rr else "n/a"),
        ("EIT heart rate", f"{_hr * 60:.1f} /min" if _hr else "n/a"),
        ("EIT breaths detected", str(_n_eit)),
        ("sEMG breaths detected", str(_n_emg)),
    ]
    mo.vstack(
        [
            mo.md("## Run summary"),
            mo.ui.table([{"Metric": k, "Value": v} for k, v in _rows], selection=None),
        ]
    )
    return


@app.cell
def _section_rate(mo):
    mo.md(
        """
        ## EIT — rate detection (native `eitprocessing` figure)

        Produced by `RateDetection.plotting.plot(...)` from the capture the
        `eit.detect_rates` step recorded. Vertical markers are the estimated
        respiratory and heart rates (shown in breaths/beats per minute).
        """
    )
    return


@app.cell
def _plot_rate_detection(ctx, mo, plt):
    _detector = ctx["rate_detector"]
    _captures = ctx["rate_captures"]
    if not _captures:
        mo.output.replace(
            mo.callout(
                mo.md("No rate-detection capture found (run with `capture: true`)."),
                kind="warn",
            )
        )
    else:
        plt.close("all")
        _fig = _detector.plotting.plot(**_captures)
        _fig.set_size_inches(11, 4.5)
        mo.output.replace(_fig)
    return


@app.cell
def _section_global(mo):
    mo.md(
        """
        ## EIT — global impedance, detected breaths, TIV & EELI

        The continuous global impedance signal with each detected breath's
        **TIV** (tidal impedance variation) drawn as a vertical green segment
        from the start of inspiration (`breath.start_time`) up to the peak
        (`breath.middle_time`) — matching `eitprocessing`'s own "inspiratory"
        TIV definition (`value_at_middle - value_at_start`) and its TIV
        example plot. **EELI** (end-expiratory lung impedance) is plotted the
        way `eitprocessing` defines it: one black dot per breath at its true
        end-expiratory sample (`eeli.time` / `eeli.values`, i.e. the impedance
        value at each breath's `end_time`), with the session mean/median and a
        ±1 SD band as reference lines — matching `eitprocessing`'s own EELI
        example.
        """
    )
    return


@app.cell
def _ui_eit_range(ctx, mo, np):
    _t = np.asarray(ctx["global_impedance"].time)
    _dur = float(_t[-1] - _t[0])
    eit_range = mo.ui.range_slider(
        start=0.0,
        stop=round(_dur, 1),
        step=1.0,
        value=[0.0, round(min(180.0, _dur), 1)],
        label="EIT time window (s, from recording start)",
        full_width=True,
    )
    eit_range
    return (eit_range,)


@app.cell
def _plot_global_impedance(ctx, eit_range, go, np):
    _gi = ctx["global_impedance"]
    _t = np.asarray(_gi.time, dtype=float)
    _t0abs = _t[0]
    _t = _t - _t0abs
    _y = np.asarray(_gi.values, dtype=float)
    _lo, _hi = [float(v) for v in eit_range.value]
    _m = (_t >= _lo) & (_t <= _hi)

    # Detected breaths (Breath objects carry absolute start/middle/end times).
    _breaths = ctx["breath_intervals"].values
    _mid = np.array([b.middle_time - _t0abs for b in _breaths], dtype=float)
    _start = np.array([b.start_time - _t0abs for b in _breaths], dtype=float)
    _bmask = (_mid >= _lo) & (_mid <= _hi)

    # EELI (`eitprocessing.parameters.eeli.EELI`) is, per breath, the global
    # impedance sample *at that breath's end-expiratory time* -- i.e.
    # `eeli.time`/`eeli.values` already sit exactly on the impedance curve's
    # troughs, one point per breath (same order as `_breaths`). TIV
    # (`continuous_tiv`) is stamped at each breath's peak (middle_time); pair
    # the two by breath index (not timestamp) to draw each breath's tidal
    # excursion.
    _eeli = ctx["eeli"]
    _eeli_t_all = np.asarray(_eeli.time, dtype=float) - _t0abs
    _eeli_v_all = np.asarray(_eeli.values, dtype=float)
    _eeli_mask = (_eeli_t_all >= _lo) & (_eeli_t_all <= _hi)

    _fig = go.Figure()
    _fig.add_trace(
        go.Scattergl(
            x=_t[_m],
            y=_y[_m],
            mode="lines",
            name="global impedance",
            line=dict(color="#185FA5", width=1.3),
        )
    )
    # Mark each detected breath's inspiration peak via nearest sample.
    if _bmask.any():
        _idx = np.searchsorted(_t, _mid[_bmask])
        _idx = np.clip(_idx, 0, _t.size - 1)
        _peak_x = _t[_idx]
        _peak_y = _y[_idx]
        _fig.add_trace(
            go.Scattergl(
                x=_peak_x,
                y=_peak_y,
                mode="markers",
                name="detected breaths",
                marker=dict(color="#D1495B", size=8, symbol="triangle-down"),
            )
        )
        # TIV ("inspiratory" method, eitprocessing's default) is the impedance
        # rise from the start of inspiration to the peak: value_at_middle -
        # value_at_start. Draw that as a vertical segment per breath, from the
        # impedance at the breath's start_time up to its peak; NaN entries
        # break the line between breaths so a single trace draws every
        # segment.
        _start_idx = np.clip(np.searchsorted(_t, _start[_bmask]), 0, _t.size - 1)
        _base_y = _y[_start_idx]
        _seg_x = np.repeat(_peak_x, 3)
        _seg_x[2::3] = np.nan
        _seg_y = np.empty(_base_y.size * 3, dtype=float)
        _seg_y[0::3] = _base_y
        _seg_y[1::3] = _peak_y
        _seg_y[2::3] = np.nan
        _fig.add_trace(
            go.Scattergl(
                x=_seg_x,
                y=_seg_y,
                mode="lines",
                name="TIV",
                line=dict(color="#2A9D8F", width=2.2),
                connectgaps=False,
            )
        )
    # EELIs -- plotted at their own correct (end-expiratory) time/value, as in
    # eitprocessing's own EELI example.
    _fig.add_trace(
        go.Scattergl(
            x=_eeli_t_all[_eeli_mask],
            y=_eeli_v_all[_eeli_mask],
            mode="markers",
            name="EELIs",
            marker=dict(color="black", size=6),
        )
    )
    # Session-wide mean/median + ±1 SD band (computed over all breaths, not
    # just the visible window, so the reference doesn't jump as you scroll).
    _mean_eeli = float(np.mean(_eeli_v_all))
    _median_eeli = float(np.median(_eeli_v_all))
    _std_eeli = float(np.std(_eeli_v_all))
    _fig.add_hrect(
        y0=_mean_eeli - _std_eeli,
        y1=_mean_eeli + _std_eeli,
        fillcolor="blue",
        opacity=0.15,
        line_width=0,
        layer="below",
        annotation_text="EELI ±1 SD",
        annotation_position="bottom right",
    )
    _fig.add_hline(
        y=_mean_eeli,
        line=dict(color="red", width=1.5),
        annotation_text="EELI mean",
        annotation_position="top left",
    )
    _fig.add_hline(
        y=_median_eeli,
        line=dict(color="green", width=1.5),
        annotation_text="EELI median",
        annotation_position="bottom left",
    )
    _fig.update_layout(
        template="plotly_white",
        height=380,
        hovermode="x unified",
        xaxis_title="time (s)",
        yaxis_title="impedance (a.u.)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, r=20, b=44, l=64),
    )
    _fig
    return


@app.cell
def _section_pixelmap(mo):
    mo.md(
        """
        ## EIT — per-pixel TIV map (native `eitprocessing` `PixelMap`)

        Mean of the per-breath pixel-TIV maps produced by `eit.pixel_tiv`,
        rendered with `PixelMap.plotting.imshow(...)`.
        """
    )
    return


@app.cell
def _plot_pixel_tiv(ctx, mo, np, plt):
    _pt = np.asarray(ctx["pixel_tiv"].values, dtype=float)  # (n_breaths, 32, 32)
    _mean_map = np.nanmean(_pt, axis=0)

    from eitprocessing.plotting.pixelmap import PixelMap

    _pm = PixelMap(_mean_map, label="mean tidal impedance variation")
    plt.close("all")
    _fig, _ax = plt.subplots(figsize=(5.0, 4.6))
    plt.sca(_ax)
    _pm.plotting.imshow(colorbar=True)
    _ax.set_title(f"Mean pixel TIV over {_pt.shape[0]} breaths")
    mo.output.replace(_fig)
    return


# @app.cell
# def _section_emg(mo):
#     mo.md("## sEMG (diaphragm) — filtered signal, envelope & detected breaths")
#     return


# @app.cell
# def _ui_emg_range(ctx, mo, np):
#     _fs = float(ctx["processed_emg"]["fs"])
#     _dur = float(np.asarray(ctx["processed_emg"]["raw_channel"]).size) / _fs
#     emg_range = mo.ui.range_slider(
#         start=0.0,
#         stop=round(_dur, 1),
#         step=1.0,
#         value=[0.0, round(min(120.0, _dur), 1)],
#         label="sEMG time window (s)",
#         full_width=True,
#     )
#     emg_downsample = mo.ui.slider(
#         start=1, stop=50, step=1, value=10, label="downsample"
#     )
#     mo.vstack(
#         [
#             emg_range,
#             mo.hstack([mo.md("**Downsample**"), emg_downsample], justify="start"),
#         ]
#     )
#     return emg_downsample, emg_range


# @app.cell
# def _plot_emg(ctx, emg_downsample, emg_range, go, make_subplots, np):
#     _pe = ctx["processed_emg"]
#     _fs = float(_pe["fs"])
#     _raw = np.asarray(_pe["raw_channel"], dtype=float)
#     _filt = np.asarray(_pe["filtered"], dtype=float)
#     _env = np.asarray(_pe["envelope"], dtype=float)
#     _base = np.asarray(ctx["baseline"], dtype=float)
#     _t = np.arange(_raw.size, dtype=float) / _fs

#     _lo, _hi = [float(v) for v in emg_range.value]
#     _step = int(emg_downsample.value)
#     _m = np.where((_t >= _lo) & (_t <= _hi))[0][::_step]

#     _events = ctx["emg_breath_events"]
#     _peak_t = np.array([e.peak_time for e in _events], dtype=float)
#     _emask = (_peak_t >= _lo) & (_peak_t <= _hi)

#     _fig = make_subplots(
#         rows=2,
#         cols=1,
#         shared_xaxes=True,
#         vertical_spacing=0.08,
#         subplot_titles=[
#             "Filtered sEMGdi (band-pass + 50 Hz interference notch)",
#             "ARV envelope, baseline & detected breaths",
#         ],
#     )
#     _fig.add_trace(
#         go.Scattergl(
#             x=_t[_m],
#             y=_filt[_m],
#             mode="lines",
#             name="filtered sEMG",
#             line=dict(color="#2A9D8F", width=0.7),
#         ),
#         row=1,
#         col=1,
#     )
#     _fig.add_trace(
#         go.Scattergl(
#             x=_t[_m],
#             y=_env[_m],
#             mode="lines",
#             name="envelope (ARV)",
#             line=dict(color="#E07A00", width=1.4),
#         ),
#         row=2,
#         col=1,
#     )
#     _fig.add_trace(
#         go.Scattergl(
#             x=_t[_m],
#             y=_base[_m],
#             mode="lines",
#             name="baseline",
#             line=dict(color="#888888", width=1.0, dash="dot"),
#         ),
#         row=2,
#         col=1,
#     )
#     if _emask.any():
#         _pk = np.clip((_peak_t[_emask] * _fs).astype(int), 0, _env.size - 1)
#         _fig.add_trace(
#             go.Scattergl(
#                 x=_peak_t[_emask],
#                 y=_env[_pk],
#                 mode="markers",
#                 name="detected breaths",
#                 marker=dict(color="#D1495B", size=9, symbol="triangle-down"),
#             ),
#             row=2,
#             col=1,
#         )
#     _fig.update_xaxes(title_text="time (s)", row=2, col=1)
#     _fig.update_yaxes(title_text="mV", row=1, col=1)
#     _fig.update_yaxes(title_text="a.u.", row=2, col=1)
#     _fig.update_layout(
#         template="plotly_white",
#         height=540,
#         hovermode="x unified",
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#         margin=dict(t=54, r=20, b=44, l=64),
#     )
#     _fig
#     return


# @app.cell
# def _section_psd(mo):
#     mo.md(
#         """
#         ## sEMG — power spectrum (native `ReSurfEMG` figure)

#         `ReSurfEMG`'s `show_psd_welch(...)` on the raw diaphragm sEMG. With the
#         EIT device recording simultaneously, its 50 Hz frame-rate injects a
#         harmonic **interference comb** (50, 100, 150 … Hz) that is clearly
#         visible here — exactly what the pipeline's 50 Hz notch removes before
#         computing the envelope.
#         """
#     )
#     return


# @app.cell
# def _ui_psd(ctx, mo, np):
#     _fs = float(ctx["processed_emg"]["fs"])
#     _dur = float(np.asarray(ctx["processed_emg"]["raw_channel"]).size) / _fs
#     psd_range = mo.ui.range_slider(
#         start=0.0,
#         stop=round(_dur, 1),
#         step=1.0,
#         # Default to a window inside the EIT-on part (interference present).
#         value=[0.0, round(min(120.0, _dur), 1)],
#         label="sEMG window for the spectrum (s)",
#         full_width=True,
#     )
#     psd_range
#     return (psd_range,)


# @app.cell
# def _plot_psd(ctx, mo, np, plt, psd_range):
#     from resurfemg.helper_functions.visualization import show_psd_welch

#     _pe = ctx["processed_emg"]
#     _fs = float(_pe["fs"])
#     _raw = np.asarray(_pe["raw_channel"], dtype=float)
#     _lo, _hi = [float(v) for v in psd_range.value]
#     _seg = _raw[int(_lo * _fs) : int(_hi * _fs)]

#     plt.close("all")
#     _fig = plt.figure(figsize=(11, 4.2))
#     # show_psd_welch draws on the current pyplot figure (and calls plt.show,
#     # which only warns under the Agg backend); grab the figure it drew on.
#     show_psd_welch(_seg, _fs, t_window_s=2, axis_spec=1, signal_unit="mV")
#     _fig = plt.gcf()
#     _fig.axes[0].set_title(f"sEMGdi Welch PSD — window {_lo:.0f}–{_hi:.0f} s")
#     mo.output.replace(_fig)
#     return


@app.cell
def _section_meta(mo):
    mo.md("## Offset-estimation provenance")
    return


@app.cell
def _meta_table(ctx, mo):
    _oe = ctx.get("offset_estimation", {}) or {}

    def _flatten(prefix, obj, out):
        for k, v in obj.items():
            if isinstance(v, dict):
                _flatten(f"{prefix}{k}.", v, out)
            else:
                out.append({"Field": f"{prefix}{k}", "Value": str(v)})
        return out

    mo.ui.table(_flatten("", _oe, []), selection=None)
    return


if __name__ == "__main__":
    app.run()
