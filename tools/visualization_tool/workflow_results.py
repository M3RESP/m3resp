import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full", app_title="Pendelluft — workflow results")


@app.cell
def _imports():
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return Path, go, make_subplots, mo, pd


@app.cell
def _wf_constants():
    # Colors match the workflow matplotlib/seaborn figures exactly:
    # PC uses the neutral gray, PS uses the deep institutional blue.
    DATASET_COLORS = {
        "SWITCH_SAFE_PC": "#B4B2A9",
        "SWITCH_SAFE_PS": "#185FA5",
    }
    DATASET_LABELS = {
        "SWITCH_SAFE_PC": "SWITCH SAFE PC",
        "SWITCH_SAFE_PS": "SWITCH SAFE PS",
    }
    METHOD_COLORS = {
        "adler": "#1f77b4",
        "coppadoro": "#ff7f0e",
        "cornejo": "#2ca02c",
        "menga": "#9467bd",
        "sang": "#d62728",
    }
    METHOD_LABELS = {
        "adler": "Adler",
        "coppadoro": "Coppadoro",
        "cornejo": "Cornejo",
        "menga": "Menga",
        "sang": "Sang",
    }
    METHOD_ORDER = ["adler", "coppadoro", "cornejo", "menga", "sang"]
    TIMESTAMP_ORDER = [
        "preswitch",
        "t0",
        "t30",
        "t60",
        "t90",
        "t120",
        "t150",
        "t180",
        "t210",
        "followup1",
        "followup2",
        "followup3",
        "followup4",
        "followup5",
        "followup6",
    ]
    return (
        DATASET_COLORS,
        DATASET_LABELS,
        METHOD_COLORS,
        METHOD_LABELS,
        METHOD_ORDER,
        TIMESTAMP_ORDER,
    )


@app.cell
def _ui_header(mo):
    mo.md("""
    # Pendelluft — workflow results explorer
    """)
    return


@app.cell
def _discover_runs(Path):
    _output_root = Path(__file__).parent.parent / "pendelluft_workflow" / "output"
    workflow_runs = sorted(
        [d.name for d in _output_root.iterdir() if d.is_dir()], reverse=True
    )
    output_root = _output_root
    return output_root, workflow_runs


@app.cell
def _ui_run_selector(mo, workflow_runs):
    run_dropdown = mo.ui.dropdown(
        options=workflow_runs,
        value=workflow_runs[0] if workflow_runs else None,
        label="Run",
    )
    preprocessing_radio = mo.ui.radio(
        options=["emc", "individual"],
        value="emc",
        label="Preprocessing",
    )
    mo.hstack(
        [
            mo.md("**Run**"),
            run_dropdown,
            mo.md("**Preprocessing**"),
            preprocessing_radio,
        ],
        justify="start",
        gap=2,
    )
    return preprocessing_radio, run_dropdown


@app.cell
def _load_workflow_data(output_root, pd, preprocessing_radio, run_dropdown):
    _run_dir = output_root / run_dropdown.value
    _prep = preprocessing_radio.value
    _csv = _run_dir / "csv_results" / "SWITCH-SAFE"

    def _safe_read(path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    wf_bxb = _safe_read(
        _csv
        / _prep
        / "breath_by_breath"
        / f"pendelluft_results_{_prep}_breath_by_breath.csv"
    )
    wf_desc_stats = _safe_read(
        _run_dir / "csv_results" / "statistics" / f"descriptive_statistics_{_prep}.csv"
    )
    wf_prev_patient_95 = _safe_read(
        _run_dir / "pendelluft_threshold_prevalence_by_dataset_95th.csv"
    )
    wf_prev_patient_youden = _safe_read(
        _run_dir / "pendelluft_threshold_prevalence_by_dataset_youden.csv"
    )
    wf_prev_breath_95 = _safe_read(
        _run_dir / "pendelluft_breath_percentage_by_dataset_95th.csv"
    )
    wf_prev_breath_youden = _safe_read(
        _run_dir / "pendelluft_breath_percentage_by_dataset_youden.csv"
    )
    _pes_dir = _run_dir / "csv_results" / "pes_results" / "SWITCH-SAFE"
    _pes_files = list(_pes_dir.glob("pes_swings_*.csv")) if _pes_dir.exists() else []
    wf_pes = (
        pd.concat([pd.read_csv(f) for f in _pes_files], ignore_index=True)
        if _pes_files
        else pd.DataFrame()
    )
    wf_prep_label = _prep.upper()
    return (
        wf_bxb,
        wf_desc_stats,
        wf_pes,
        wf_prep_label,
        wf_prev_breath_95,
        wf_prev_breath_youden,
        wf_prev_patient_95,
        wf_prev_patient_youden,
    )


@app.cell
def _section_a(mo):
    mo.md("""
    ## A — Summary: patient median pendelluft per method
    """)
    return


@app.cell
def _plot_summary_boxplot(
    DATASET_COLORS,
    DATASET_LABELS,
    METHOD_LABELS,
    METHOD_ORDER,
    go,
    mo,
    wf_desc_stats,
    wf_prep_label,
):
    if wf_desc_stats.empty:
        mo.output.replace(
            mo.callout(
                mo.md("No descriptive statistics found for this run / preprocessing."),
                kind="warn",
            )
        )
    else:
        _fig = go.Figure()
        for _ds in sorted(wf_desc_stats["dataset"].unique().tolist()):
            _color = DATASET_COLORS.get(_ds, "gray")
            _s = wf_desc_stats[wf_desc_stats["dataset"] == _ds]
            _x, _q1, _med, _q3, _lo, _hi = [], [], [], [], [], []
            for _m in METHOD_ORDER:
                _row = _s[_s["method"] == _m]
                if _row.empty:
                    continue
                _r = _row.iloc[0]
                _x.append(METHOD_LABELS.get(_m, _m))
                _q1.append(_r["q25"])
                _med.append(_r["median"])
                _q3.append(_r["q75"])
                _lo.append(_r["min"])
                _hi.append(_r["max"])
            _fig.add_trace(
                go.Box(
                    x=_x,
                    q1=_q1,
                    median=_med,
                    q3=_q3,
                    lowerfence=_lo,
                    upperfence=_hi,
                    name=DATASET_LABELS.get(_ds, _ds),
                    marker_color=_color,
                    line_color=_color,
                    fillcolor=_color,
                    opacity=0.75,
                )
            )
        _fig.update_layout(
            template="plotly_white",
            title=f"{wf_prep_label} — patient median pendelluft per method",
            xaxis_title="Method",
            yaxis_title="Patient Median Pendelluft Value",
            boxmode="group",
            height=480,
            legend=dict(title="Dataset"),
            margin=dict(t=50, b=40),
        )
        mo.output.replace(_fig)
    return


@app.cell
def _plot_strip_patient_medians(
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    go,
    mo,
    wf_bxb,
    wf_prep_label,
):
    """Violin + strip of per-patient × timestamp medians — shows individual spread hidden behind the pre-aggregated box above."""
    if wf_bxb.empty:
        mo.output.replace(
            mo.callout(
                mo.md("No breath-by-breath data to compute patient medians."),
                kind="warn",
            )
        )
    else:
        _pt_med = (
            wf_bxb.groupby(["patient", "timestamp", "method"])["pendelluft_value"]
            .median()
            .reset_index()
        )
        _fig = go.Figure()
        for _m in METHOD_ORDER:
            _s = _pt_med[_pt_med["method"] == _m]
            if _s.empty:
                continue
            _clr = METHOD_COLORS[_m]
            _label = METHOD_LABELS.get(_m, _m)
            _fig.add_trace(
                go.Violin(
                    x=[_label] * len(_s),
                    y=_s["pendelluft_value"],
                    name=_label,
                    line_color=_clr,
                    fillcolor=_clr,
                    opacity=0.3,
                    box_visible=True,
                    meanline_visible=False,
                    points=False,
                    showlegend=False,
                )
            )
            _fig.add_trace(
                go.Scatter(
                    x=[_label] * len(_s),
                    y=_s["pendelluft_value"],
                    mode="markers",
                    name=_label,
                    marker=dict(
                        color=_clr,
                        size=8,
                        opacity=0.75,
                        line=dict(color="white", width=0.8),
                    ),
                    text="pt "
                    + _s["patient"].astype(str)
                    + " / "
                    + _s["timestamp"].astype(str),
                    hovertemplate="%{text}<br>%{y:.4f}<extra></extra>",
                )
            )
        _fig.update_layout(
            template="plotly_white",
            title=f"{wf_prep_label} — patient × timestamp median pendelluft per method (violin + strip)",
            xaxis_title="Method",
            yaxis_title="Median pendelluft value",
            height=460,
            violinmode="overlay",
            legend=dict(title="Method"),
            margin=dict(t=50, b=40),
        )
        mo.output.replace(_fig)
    return


@app.cell
def _section_b(mo):
    mo.md("""
    ## B — Breath-by-breath pendelluft per patient
    """)
    return


@app.cell
def _ui_bxb_selector(mo, wf_bxb):
    _patients = (
        sorted(wf_bxb["patient"].astype(str).unique().tolist())
        if not wf_bxb.empty
        else []
    )
    bxb_patient_picker = mo.ui.dropdown(
        options=_patients,
        value=_patients[0] if _patients else None,
        label="Patient",
    )
    bxb_patient_picker
    return (bxb_patient_picker,)


@app.cell
def _plot_bxb_timeseries(
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    bxb_patient_picker,
    go,
    make_subplots,
    mo,
    wf_bxb,
):
    if wf_bxb.empty:
        mo.output.replace(
            mo.callout(
                mo.md(
                    "No breath-by-breath data available for this run / preprocessing."
                ),
                kind="warn",
            )
        )
    elif bxb_patient_picker.value is None:
        mo.output.replace(mo.callout(mo.md("Select a patient above."), kind="warn"))
    else:
        _all_pts = wf_bxb["patient"].astype(str).unique().tolist()
        _pt = (
            bxb_patient_picker.value
            if bxb_patient_picker.value in _all_pts
            else _all_pts[0]
        )
        _df = wf_bxb[wf_bxb["patient"].astype(str) == _pt].copy()
        _timestamps = sorted(_df["timestamp"].unique().tolist())
        _fig = make_subplots(
            rows=len(_timestamps),
            cols=1,
            shared_xaxes=False,
            subplot_titles=[f"Timestamp: {t}" for t in _timestamps],
            vertical_spacing=0.03,
        )
        _shown = set()
        for _i, _ts in enumerate(_timestamps, start=1):
            _ts_df = _df[_df["timestamp"] == _ts]
            for _m in METHOD_ORDER:
                _r = _ts_df[_ts_df["method"] == _m]
                if _r.empty:
                    continue
                _fig.add_trace(
                    go.Scatter(
                        x=_r["middle_time"],
                        y=_r["pendelluft_value"],
                        mode="lines+markers",
                        name=METHOD_LABELS.get(_m, _m),
                        marker_color=METHOD_COLORS[_m],
                        showlegend=(_m not in _shown),
                    ),
                    row=_i,
                    col=1,
                )
                _shown.add(_m)
        _fig.update_layout(
            template="plotly_white",
            title=f"Breath-by-breath pendelluft — patient {_pt}",
            yaxis_title="Pendelluft value",
            height=max(500, 400 * len(_timestamps)),
            margin=dict(t=50, b=20),
        )
        mo.output.replace(_fig)
    return


@app.cell
def _section_c(mo):
    mo.md("""
    ## C — Prevalence analysis
    """)
    return


@app.cell
def _ui_prevalence_controls(mo):
    threshold_radio = mo.ui.radio(
        options=["95th percentile", "Youden"],
        value="95th percentile",
        label="Threshold strategy",
    )
    threshold_radio
    return (threshold_radio,)


@app.cell
def _plot_patient_prevalence(
    DATASET_COLORS,
    DATASET_LABELS,
    METHOD_LABELS,
    METHOD_ORDER,
    go,
    mo,
    threshold_radio,
    wf_prev_patient_95,
    wf_prev_patient_youden,
):
    _df = (
        wf_prev_patient_95
        if threshold_radio.value == "95th percentile"
        else wf_prev_patient_youden
    )
    if _df.empty:
        mo.output.replace(
            mo.callout(
                mo.md("No patient-level prevalence data found for this run."),
                kind="warn",
            )
        )
    else:
        _df = _df.copy()
        _df.columns = [c.lower() for c in _df.columns]
        _fig = go.Figure()
        for _ds in sorted(_df["dataset"].unique().tolist()):
            _color = DATASET_COLORS.get(_ds, "gray")
            _s = _df[_df["dataset"] == _ds]
            _x, _y = [], []
            for _m in METHOD_ORDER:
                _row = _s[_s["method"] == _m]
                if _row.empty:
                    continue
                _x.append(METHOD_LABELS.get(_m, _m))
                _y.append(float(_row.iloc[0]["prevalence_percentage"]))
            _fig.add_trace(
                go.Bar(
                    x=_x,
                    y=_y,
                    name=DATASET_LABELS.get(_ds, _ds),
                    marker_color=_color,
                    opacity=0.85,
                )
            )
        _fig.update_layout(
            template="plotly_white",
            title=f"Patient-level pendelluft prevalence ({threshold_radio.value})",
            xaxis_title="Method",
            yaxis_title="Prevalence (%)",
            barmode="group",
            height=430,
            legend=dict(title="Dataset"),
            margin=dict(t=50, b=40),
        )
        mo.output.replace(_fig)
    return


@app.cell
def _plot_breath_prevalence(
    DATASET_COLORS,
    DATASET_LABELS,
    METHOD_LABELS,
    METHOD_ORDER,
    go,
    mo,
    threshold_radio,
    wf_prev_breath_95,
    wf_prev_breath_youden,
):
    _df = (
        wf_prev_breath_95
        if threshold_radio.value == "95th percentile"
        else wf_prev_breath_youden
    )
    if _df.empty:
        mo.output.replace(
            mo.callout(
                mo.md("No breath-level prevalence data found for this run."),
                kind="warn",
            )
        )
    else:
        _df = _df.copy()
        _df.columns = [c.lower() for c in _df.columns]
        _fig = go.Figure()
        for _ds in sorted(_df["dataset"].unique().tolist()):
            _color = DATASET_COLORS.get(_ds, "gray")
            _s = _df[_df["dataset"] == _ds]
            _x, _med, _err_plus, _err_minus = [], [], [], []
            for _m in METHOD_ORDER:
                _row = _s[_s["method"] == _m]
                if _row.empty:
                    continue
                _r = _row.iloc[0]
                _x.append(METHOD_LABELS.get(_m, _m))
                _mv = float(_r["median_pct_above"])
                _med.append(_mv)
                _err_plus.append(float(_r["q3_pct_above"]) - _mv)
                _err_minus.append(_mv - float(_r["q1_pct_above"]))
            _fig.add_trace(
                go.Bar(
                    x=_x,
                    y=_med,
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=_err_plus,
                        arrayminus=_err_minus,
                    ),
                    name=DATASET_LABELS.get(_ds, _ds),
                    marker_color=_color,
                    opacity=0.85,
                )
            )
        _fig.update_layout(
            template="plotly_white",
            title=f"Breath-level prevalence — median % above threshold ({threshold_radio.value}), IQR error bars",
            xaxis_title="Method",
            yaxis_title="Median % breaths above threshold",
            barmode="group",
            height=430,
            legend=dict(title="Dataset"),
            margin=dict(t=50, b=40),
        )
        mo.output.replace(_fig)
    return


@app.cell
def _section_c_threshold(mo):
    mo.md("""
    ### C3 — Pendelluft distribution per method with threshold lines
    """)
    return


@app.cell
def _plot_threshold_distribution(
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    go,
    make_subplots,
    mo,
    wf_bxb,
    wf_prep_label,
    wf_prev_patient_95,
    wf_prev_patient_youden,
):
    if wf_bxb.empty:
        mo.output.replace(
            mo.callout(mo.md("No breath-by-breath data available."), kind="warn")
        )
    else:
        _prep = wf_prep_label.lower()

        def _get_threshold(df, method):
            if df.empty:
                return None
            _df = df.copy()
            _df.columns = [c.lower() for c in _df.columns]
            if "preprocessing" in _df.columns:
                _r = _df[(_df["method"] == method) & (_df["preprocessing"] == _prep)]
                if _r.empty:
                    _r = _df[_df["method"] == method]
            else:
                _r = _df[_df["method"] == method]
            return float(_r.iloc[0]["threshold_value"]) if not _r.empty else None

        _n = len(METHOD_ORDER)
        _fig = make_subplots(
            rows=_n,
            cols=1,
            subplot_titles=[METHOD_LABELS.get(_m, _m) for _m in METHOD_ORDER],
            vertical_spacing=0.09,
        )
        for _i, _m in enumerate(METHOD_ORDER, start=1):
            _vals = wf_bxb[wf_bxb["method"] == _m]["pendelluft_value"].dropna()
            if _vals.empty:
                continue
            _clr = METHOD_COLORS[_m]
            _fig.add_trace(
                go.Histogram(
                    x=_vals,
                    name=METHOD_LABELS.get(_m, _m),
                    marker_color=_clr,
                    opacity=0.7,
                    nbinsx=40,
                    showlegend=False,
                ),
                row=_i,
                col=1,
            )

            _thr_95 = _get_threshold(wf_prev_patient_95, _m)
            _thr_y = _get_threshold(wf_prev_patient_youden, _m)
            if _thr_95 is not None:
                _fig.add_vline(
                    x=_thr_95,
                    row=_i,
                    col=1,
                    line_dash="dot",
                    line_color="black",
                    line_width=1.5,
                    annotation_text=f"95th: {_thr_95:.3f}",
                    annotation_position="top right",
                    annotation_font_size=10,
                )
            if _thr_y is not None:
                _fig.add_vline(
                    x=_thr_y,
                    row=_i,
                    col=1,
                    line_dash="dash",
                    line_color="black",
                    line_width=1.5,
                    annotation_text=f"Youden: {_thr_y:.3f}",
                    annotation_position="top left",
                    annotation_font_size=10,
                )

        _fig.update_xaxes(title_text="Pendelluft value")
        _fig.update_yaxes(title_text="Count")
        _fig.update_layout(
            template="plotly_white",
            title=f"{wf_prep_label} — breath-level pendelluft distribution per method",
            height=400 * _n,
            margin=dict(t=60, b=40),
        )
        mo.output.replace(_fig)
    return


@app.cell
def _section_d(mo):
    mo.md("""
    ## D — PES swings
    """)
    return


@app.cell
def _plot_pes_boxplot(go, mo, wf_pes):
    if wf_pes.empty:
        mo.output.replace(
            mo.callout(mo.md("No PES swing data found for this run."), kind="warn")
        )
    else:
        _fig = go.Figure()
        for _pt in sorted(wf_pes["patient"].astype(str).unique()):
            _s = wf_pes[wf_pes["patient"].astype(str) == _pt]
            _fig.add_trace(
                go.Box(
                    y=_s["pes_swing"].abs(),
                    name=_pt,
                    boxpoints="all",
                    jitter=0.35,
                    marker=dict(size=5, opacity=0.6),
                )
            )
        _fig.update_layout(
            template="plotly_white",
            title="PES swing magnitude per patient (all timestamps combined)",
            yaxis_title="|PES swing| (cmH₂O)",
            xaxis_title="Patient",
            height=440,
            showlegend=False,
            margin=dict(t=50, b=40),
        )
        mo.output.replace(_fig)
    return


@app.cell
def _plot_pes_vs_pendelluft(
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    go,
    mo,
    wf_bxb,
    wf_pes,
):
    if wf_pes.empty or wf_bxb.empty:
        mo.output.replace(
            mo.callout(
                mo.md(
                    "PES or breath-by-breath data not available — cannot plot PES vs pendelluft."
                ),
                kind="warn",
            )
        )
    else:
        _pes = wf_pes.copy()
        _pes["patient_key"] = (
            _pes["patient"].astype(str).str.extract(r"(\d+)$")[0].str.lstrip("0")
        )
        _pes_med = (
            _pes.assign(pes_abs=_pes["pes_swing"].abs())
            .groupby(["patient_key", "timestamp"])["pes_abs"]
            .median()
            .reset_index()
            .rename(columns={"pes_abs": "pes_median"})
        )
        _bxb = wf_bxb.copy()
        _bxb["patient_key"] = (
            _bxb["patient"].astype(str).str.extract(r"(\d+)$")[0].str.lstrip("0")
        )
        _bxb_med = (
            _bxb.groupby(["patient_key", "timestamp", "method"])["pendelluft_value"]
            .median()
            .reset_index()
        )
        _merged = _bxb_med.merge(_pes_med, on=["patient_key", "timestamp"], how="inner")
        if _merged.empty:
            mo.output.replace(
                mo.callout(
                    mo.md(
                        "Could not align PES and pendelluft data — patient/timestamp keys did not match."
                    ),
                    kind="warn",
                )
            )
        else:
            _fig = go.Figure()
            for _m in METHOD_ORDER:
                _s = _merged[_merged["method"] == _m]
                if _s.empty:
                    continue
                _fig.add_trace(
                    go.Scatter(
                        x=_s["pes_median"],
                        y=_s["pendelluft_value"],
                        mode="markers",
                        name=METHOD_LABELS.get(_m, _m),
                        marker=dict(color=METHOD_COLORS[_m], size=10, opacity=0.8),
                        text="pt " + _s["patient_key"] + " / " + _s["timestamp"],
                        hovertemplate="%{text}<br>PES: %{x:.2f} cmH₂O<br>Pendelluft: %{y:.4f}<extra></extra>",
                    )
                )
            _fig.update_layout(
                template="plotly_white",
                title="Median |PES swing| vs median pendelluft per patient × timestamp",
                xaxis_title="Median |PES swing| (cmH₂O)",
                yaxis_title="Median pendelluft value",
                height=460,
                legend=dict(title="Method"),
                margin=dict(t=50, b=40),
            )
            mo.output.replace(_fig)
    return


@app.cell
def _section_d_per_timestamp(mo):
    mo.md("""
    ### D3 — PES per timestamp
    """)
    return


@app.cell
def _plot_pes_per_timestamp(TIMESTAMP_ORDER, go, mo, wf_pes):
    if wf_pes.empty:
        mo.output.replace(
            mo.callout(mo.md("No PES swing data found for this run."), kind="warn")
        )
    else:
        _patients = sorted(wf_pes["patient"].astype(str).unique())
        _all_ts = wf_pes["timestamp"].unique().tolist()
        _ts_ordered = [t for t in TIMESTAMP_ORDER if t in _all_ts] + [
            t for t in _all_ts if t not in TIMESTAMP_ORDER
        ]
        # One box trace per patient across the timestamp x-axis
        _palette = ["#185FA5", "#7F77DD", "#B4B2A9", "#ff7f0e", "#2ca02c"]
        _fig = go.Figure()
        for _pi, _pt in enumerate(_patients):
            _pdf = wf_pes[wf_pes["patient"].astype(str) == _pt]
            _clr = _palette[_pi % len(_palette)]
            _x_vals, _y_vals = [], []
            for _ts in _ts_ordered:
                _swings = _pdf[_pdf["timestamp"] == _ts]["pes_swing"].abs().tolist()
                if _swings:
                    _x_vals.extend([_ts] * len(_swings))
                    _y_vals.extend(_swings)
            _fig.add_trace(
                go.Box(
                    x=_x_vals,
                    y=_y_vals,
                    name=f"pt {_pt}",
                    marker_color=_clr,
                    boxpoints="all",
                    jitter=0.3,
                    marker=dict(size=5, opacity=0.65),
                )
            )
        _fig.update_layout(
            template="plotly_white",
            title="|PES swing| per timestamp — grouped by patient",
            xaxis_title="Timestamp",
            yaxis_title="|PES swing| (cmH₂O)",
            xaxis=dict(categoryorder="array", categoryarray=_ts_ordered),
            boxmode="group",
            height=460,
            legend=dict(title="Patient"),
            margin=dict(t=50, b=40),
        )
        mo.output.replace(_fig)
    return


@app.cell
def _section_e(mo):
    mo.md("""
    ## E — Method agreement
    """)
    return


@app.cell
def _ui_method_agreement(METHOD_LABELS, METHOD_ORDER, mo):
    _opts = {METHOD_LABELS[m]: m for m in METHOD_ORDER}
    _labels = list(_opts.keys())
    method_a_picker = mo.ui.dropdown(
        options=_opts,
        value=_labels[0],
        label="Method A",
    )
    method_b_picker = mo.ui.dropdown(
        options=_opts,
        value=_labels[1],
        label="Method B",
    )
    mo.hstack(
        [
            mo.md("**Method A**"),
            method_a_picker,
            mo.md("**Method B**"),
            method_b_picker,
        ],
        justify="start",
        gap=2,
    )
    return method_a_picker, method_b_picker


@app.cell
def _plot_method_agreement(
    METHOD_LABELS,
    METHOD_ORDER,
    go,
    method_a_picker,
    method_b_picker,
    mo,
    pd,
    wf_bxb,
):
    if wf_bxb.empty:
        mo.output.replace(
            mo.callout(mo.md("No breath-by-breath data available."), kind="warn")
        )
    else:
        # Resolve picker value: marimo may return the label key or the internal value
        # depending on version; _opts.get() handles both safely.
        _opts = {METHOD_LABELS[m]: m for m in METHOD_ORDER}
        _ma = _opts.get(method_a_picker.value, method_a_picker.value)
        _mb = _opts.get(method_b_picker.value, method_b_picker.value)
        _med = (
            wf_bxb.groupby(["patient", "timestamp", "method"])["pendelluft_value"]
            .median()
            .reset_index()
        )
        _wide = _med.pivot_table(
            index=["patient", "timestamp"], columns="method", values="pendelluft_value"
        ).reset_index()

        if _ma not in _wide.columns or _mb not in _wide.columns:
            mo.output.replace(
                mo.callout(
                    mo.md(f"Could not find columns for {_ma} or {_mb}."), kind="warn"
                )
            )
        else:
            _wide = _wide.dropna(subset=[_ma, _mb])
            _patients = sorted(_wide["patient"].astype(str).unique())
            _palette = [
                "#185FA5",
                "#7F77DD",
                "#B4B2A9",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#8c564b",
            ]
            _pt_colors = {
                pt: _palette[i % len(_palette)] for i, pt in enumerate(_patients)
            }

            _fig = go.Figure()
            for _pt in _patients:
                _s = _wide[_wide["patient"].astype(str) == _pt]
                _fig.add_trace(
                    go.Scatter(
                        x=_s[_ma],
                        y=_s[_mb],
                        mode="markers",
                        name=f"pt {_pt}",
                        marker=dict(color=_pt_colors[_pt], size=10, opacity=0.8),
                        text=_s["timestamp"],
                        hovertemplate=(
                            f"pt {_pt} / %{{text}}<br>"
                            f"{METHOD_LABELS.get(_ma, _ma)}: %{{x:.4f}}<br>"
                            f"{METHOD_LABELS.get(_mb, _mb)}: %{{y:.4f}}<extra></extra>"
                        ),
                    )
                )

            # Perfect agreement diagonal
            _all_vals = pd.concat([_wide[_ma], _wide[_mb]]).dropna()
            _mn, _mx = float(_all_vals.min()), float(_all_vals.max())
            _fig.add_trace(
                go.Scatter(
                    x=[_mn, _mx],
                    y=[_mn, _mx],
                    mode="lines",
                    line=dict(color="gray", dash="dash", width=1.5),
                    name="Perfect agreement",
                )
            )

            _la = METHOD_LABELS.get(_ma, _ma)
            _lb = METHOD_LABELS.get(_mb, _mb)
            _fig.update_layout(
                template="plotly_white",
                title=f"Method agreement: {_la} vs {_lb} — per patient × timestamp median",
                xaxis_title=f"{_la} — median pendelluft",
                yaxis_title=f"{_lb} — median pendelluft",
                height=480,
                legend=dict(title="Patient"),
                margin=dict(t=50, b=40),
            )
            mo.output.replace(_fig)
    return


@app.cell
def _section_f(mo):
    mo.md("""
    ## F — LME model results
    """)
    return


@app.cell
def _plot_lme_figures(mo, output_root, run_dropdown):
    import base64

    _figs_dir = output_root / run_dropdown.value / "figures" / "SWITCH-SAFE"

    def _img(path):
        if not path.exists():
            return mo.callout(
                mo.md(f"Figure not found for this run: `{path.name}`"), kind="warn"
            )
        data = base64.b64encode(path.read_bytes()).decode()
        return mo.image(src=f"data:image/png;base64,{data}", width="100%")

    _figures = [
        (
            "### F1 — Fixed effects & variance decomposition",
            "pendelluft_model_results.png",
        ),
        (
            "### F2 — Method-specific slopes: respiratory rate & inspiratory time ratio",
            "lme_slopes_timing.png",
        ),
        (
            "### F3 — Residual diagnostics",
            "residuals_Breath-by-breath model with 11 patients.png",
        ),
    ]

    mo.output.replace(
        mo.vstack(
            [
                item
                for header, fname in _figures
                for item in [mo.md(header), _img(_figs_dir / fname)]
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
