import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full", app_title="PES preprocessing comparison")


@app.cell
def _imports():
    import copy
    import pickle
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from scipy.signal import butter, sosfiltfilt
    from plotly.subplots import make_subplots

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    from eitprocessing.features.rate_detection import RateDetection
    from preprocessing.pes import PreprocessPes, PreprocessPesExperimental

    return (
        Path,
        PreprocessPes,
        PreprocessPesExperimental,
        ROOT,
        RateDetection,
        butter,
        copy,
        go,
        make_subplots,
        mo,
        np,
        pd,
        pickle,
        sosfiltfilt,
    )


@app.cell
def _helpers(np, pd, pickle):
    class _PathShim:
        def __new__(cls, *args, **kwargs):
            from pathlib import Path as _Path

            parts = [
                str(arg.decode("utf-8") if isinstance(arg, bytes) else arg)
                for arg in args
            ]
            return _Path(*parts) if parts else _Path()

    class _PathConvertingUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in ("pathlib", "pathlib._local") and name in (
                "PosixPath",
                "PurePosixPath",
                "WindowsPath",
                "PureWindowsPath",
            ):
                return _PathShim
            return super().find_class(module, name)

    def load_sequence(path):
        with path.open("rb") as handle:
            return _PathConvertingUnpickler(handle).load()

    def first_existing(mapping, preferred):
        for label in preferred:
            if label in mapping:
                return label
        return next(iter(mapping.keys()), None)

    def normalize_time(time):
        time = np.asarray(time, dtype=float)
        return time - time[0] if len(time) else time

    def summarize_rate_detection(sequence, RateDetection):
        try:
            rr, hr = RateDetection(subject_type="adult").apply(sequence.eit_data["raw"])
            return float(rr), float(hr), None
        except Exception as exc:
            return 0.25, 1.2, str(exc)

    def method_result(name, output, captures, error):
        if error is not None:
            return {
                "name": name,
                "ok": False,
                "error": error,
                "filtered": None,
                "peaks": np.array([], dtype=int),
                "starts": np.array([], dtype=int),
                "peak_times": np.array([], dtype=float),
                "start_times": np.array([], dtype=float),
                "captures": captures,
            }

        filtered, peaks, starts, peak_times, start_times = output
        return {
            "name": name,
            "ok": True,
            "error": None,
            "filtered": filtered,
            "peaks": np.asarray(peaks, dtype=int),
            "starts": np.asarray(starts, dtype=int),
            "peak_times": np.asarray(peak_times, dtype=float),
            "start_times": np.asarray(start_times, dtype=float),
            "captures": captures,
        }

    def run_processor(
        processor, sequence, pes_label, rr, hr, notch_distance, sequence_label, **kwargs
    ):
        captures = {}
        try:
            output = processor.apply(
                sequence,
                pes_label,
                rr,
                hr,
                notch_distance=notch_distance,
                sequence_label=sequence_label,
                captures=captures,
                **kwargs,
            )
            return output, captures, None
        except Exception as exc:
            return None, captures, str(exc)

    def breath_table(result):
        if not result["ok"]:
            return pd.DataFrame()

        filtered = result["filtered"]
        values = np.asarray(filtered.values, dtype=float)
        time = np.asarray(filtered.time, dtype=float)
        rows = []

        for breath_idx, (start, peak) in enumerate(
            zip(result["starts"], result["peaks"]), start=1
        ):
            if start < 0 or peak < 0 or start >= len(values) or peak >= len(values):
                continue
            signed_swing = values[start] - values[peak]
            rows.append(
                {
                    "method": result["name"],
                    "breath": breath_idx,
                    "start_idx": int(start),
                    "peak_idx": int(peak),
                    "start_time": float(time[start]),
                    "peak_time": float(time[peak]),
                    "inspiratory_time_s": float(time[peak] - time[start]),
                    "pes_start": float(values[start]),
                    "pes_peak": float(values[peak]),
                    "pes_swing_signed": float(signed_swing),
                    "pes_swing_abs": float(abs(signed_swing)),
                }
            )

        return pd.DataFrame(rows)

    def flow_windows_from_captures(captures, pes_time):
        starts = np.asarray(captures.get("flow_inspiration_starts", []), dtype=int)
        ends = np.asarray(captures.get("flow_inspiration_ends", []), dtype=int)
        if len(starts) == 0 or len(ends) == 0:
            return pd.DataFrame()

        n = min(len(starts), len(ends))
        starts = np.clip(starts[:n], 0, len(pes_time) - 1)
        ends = np.clip(ends[:n], 0, len(pes_time) - 1)
        valid = ends > starts

        return pd.DataFrame(
            {
                "start_idx": starts[valid],
                "end_idx": ends[valid],
                "start_time": np.asarray(pes_time)[starts[valid]],
                "end_time": np.asarray(pes_time)[ends[valid]],
            }
        )

    def nearest_abs_distance(values, reference):
        values = np.asarray(values, dtype=float)
        reference = np.asarray(reference, dtype=float)
        if len(values) == 0 or len(reference) == 0:
            return np.full(len(values), np.nan)
        distances = np.abs(values[:, None] - reference[None, :])
        return np.min(distances, axis=1)

    def inside_any_window(times, windows):
        if len(times) == 0 or windows.empty:
            return np.full(len(times), False)
        starts = windows["start_time"].to_numpy()
        ends = windows["end_time"].to_numpy()
        return np.asarray(
            [np.any((starts <= t) & (t <= ends)) for t in times], dtype=bool
        )

    def detection_summary(result, breath_df, windows, rr):
        if not result["ok"]:
            return {
                "method": result["name"],
                "status": "failed",
                "error": result["error"],
                "detected_breaths": 0,
            }

        expected_breath = 1.0 / rr if rr and rr > 0 else np.nan
        min_ti = 0.05 * expected_breath
        max_ti = 0.85 * expected_breath
        plausible = (
            (
                (breath_df["inspiratory_time_s"] > min_ti)
                & (breath_df["inspiratory_time_s"] < max_ti)
                & (breath_df["pes_swing_abs"] > 0)
            )
            if not breath_df.empty
            else pd.Series(dtype=bool)
        )

        if windows.empty or breath_df.empty:
            start_match_rate = np.nan
            median_start_to_flow = np.nan
            peak_inside_rate = np.nan
        else:
            tolerance = 0.20 * expected_breath
            start_dist = nearest_abs_distance(
                breath_df["start_time"].to_numpy(), windows["start_time"].to_numpy()
            )
            start_match_rate = float(np.nanmean(start_dist <= tolerance))
            median_start_to_flow = float(np.nanmedian(start_dist))
            peak_inside_rate = float(
                np.mean(inside_any_window(breath_df["peak_time"].to_numpy(), windows))
            )

        captures = result["captures"]
        confidence = np.asarray(captures.get("confidence", []), dtype=float)
        return {
            "method": result["name"],
            "status": "ok",
            "error": "",
            "detected_breaths": int(len(breath_df)),
            "median_abs_pes_swing": float(breath_df["pes_swing_abs"].median())
            if not breath_df.empty
            else np.nan,
            "mean_abs_pes_swing": float(breath_df["pes_swing_abs"].mean())
            if not breath_df.empty
            else np.nan,
            "median_inspiratory_time_s": float(breath_df["inspiratory_time_s"].median())
            if not breath_df.empty
            else np.nan,
            "plausible_breath_rate": float(plausible.mean())
            if len(plausible)
            else np.nan,
            "flow_start_match_rate": start_match_rate,
            "flow_peak_inside_rate": peak_inside_rate,
            "median_start_to_flow_s": median_start_to_flow,
            "mean_confidence": float(np.nanmean(confidence))
            if len(confidence)
            else np.nan,
            "fallback_used": bool(captures.get("fallback_used", False)),
            "rejected_breaths": len(captures.get("rejected_breaths", [])),
        }

    def improvement_table(summary):
        if summary.empty or not {"Old", "Experimental"}.issubset(
            set(summary["method"])
        ):
            return pd.DataFrame()

        old = summary.set_index("method").loc["Old"]
        exp = summary.set_index("method").loc["Experimental"]

        rows = [
            (
                "Detected breaths",
                exp.get("detected_breaths", np.nan)
                - old.get("detected_breaths", np.nan),
                "higher is not automatically better",
            ),
            (
                "Plausible breath rate",
                exp.get("plausible_breath_rate", np.nan)
                - old.get("plausible_breath_rate", np.nan),
                "positive is better",
            ),
            (
                "Flow start match rate",
                exp.get("flow_start_match_rate", np.nan)
                - old.get("flow_start_match_rate", np.nan),
                "positive is better",
            ),
            (
                "Flow peak inside inspiration rate",
                exp.get("flow_peak_inside_rate", np.nan)
                - old.get("flow_peak_inside_rate", np.nan),
                "positive is better",
            ),
            (
                "Median start-to-flow distance",
                old.get("median_start_to_flow_s", np.nan)
                - exp.get("median_start_to_flow_s", np.nan),
                "positive means experimental is closer",
            ),
            (
                "Median |PES swing|",
                exp.get("median_abs_pes_swing", np.nan)
                - old.get("median_abs_pes_swing", np.nan),
                "physiology check, not a direct quality score",
            ),
        ]
        return pd.DataFrame(
            rows, columns=["metric", "experimental_minus_old", "interpretation"]
        )

    def matched_peak_table(old_df, exp_df, tolerance_s):
        if old_df.empty or exp_df.empty:
            return pd.DataFrame()

        rows = []
        exp_times = exp_df["peak_time"].to_numpy()
        used_exp = set()
        for _, old_row in old_df.iterrows():
            distances = np.abs(exp_times - old_row["peak_time"])
            if len(distances) == 0:
                continue
            exp_idx = int(np.argmin(distances))
            if distances[exp_idx] > tolerance_s or exp_idx in used_exp:
                continue
            used_exp.add(exp_idx)
            exp_row = exp_df.iloc[exp_idx]
            rows.append(
                {
                    "old_breath": int(old_row["breath"]),
                    "experimental_breath": int(exp_row["breath"]),
                    "peak_time_difference_s": float(
                        exp_row["peak_time"] - old_row["peak_time"]
                    ),
                    "start_time_difference_s": float(
                        exp_row["start_time"] - old_row["start_time"]
                    ),
                    "pes_swing_abs_difference": float(
                        exp_row["pes_swing_abs"] - old_row["pes_swing_abs"]
                    ),
                }
            )
        return pd.DataFrame(rows)

    return (
        breath_table,
        detection_summary,
        first_existing,
        flow_windows_from_captures,
        improvement_table,
        load_sequence,
        matched_peak_table,
        method_result,
        run_processor,
        summarize_rate_detection,
    )


@app.cell
def _discover_pickles(ROOT):
    data_root = ROOT / "Data"
    pickle_paths = []
    if data_root.exists():
        pickle_paths = sorted(
            [
                path.relative_to(ROOT).as_posix()
                for pattern in ("*.pickle", "*.pkl")
                for path in data_root.rglob(pattern)
            ]
        )
    return (pickle_paths,)


@app.cell
def _header(mo):
    mo.md("""
    # PES preprocessing comparison

    Run the original and experimental PES preprocessing on the same sequence,
    then compare detected breaths, inspiration starts, maximum deflections, and PES swings.
    """)
    return


@app.cell
def _source_controls(mo, pickle_paths):
    _options = ["Custom path"] + pickle_paths
    source_picker = mo.ui.dropdown(
        options=_options,
        value=pickle_paths[0] if pickle_paths else "Custom path",
        label="Sequence pickle",
        searchable=True,
    )
    custom_path = mo.ui.text(
        value="",
        placeholder="/absolute/path/to/segment.pickle",
        label="Custom path",
        full_width=True,
    )
    mo.vstack([source_picker, custom_path])
    return custom_path, source_picker


@app.cell
def _load_sequence(Path, ROOT, custom_path, load_sequence, mo, source_picker):
    if source_picker.value == "Custom path":
        _path_text = custom_path.value.strip()
        mo.stop(
            not _path_text,
            mo.callout(mo.md("Enter a custom pickle path."), kind="warn"),
        )
        sequence_path = Path(_path_text).expanduser()
    else:
        sequence_path = ROOT / source_picker.value

    mo.stop(
        not sequence_path.exists(),
        mo.callout(mo.md(f"Sequence file not found: `{sequence_path}`"), kind="danger"),
    )

    sequence = load_sequence(sequence_path)
    mo.md(f"Loaded `{sequence_path.name}`")
    return (sequence,)


@app.cell
def _rate_detection(RateDetection, sequence, summarize_rate_detection):
    auto_rr, auto_hr, rate_error = summarize_rate_detection(sequence, RateDetection)
    return auto_hr, auto_rr, rate_error


@app.cell
def _method_controls(
    auto_hr,
    auto_rr,
    first_existing,
    mo,
    rate_error,
    sequence,
):
    continuous_labels = list(sequence.continuous_data.keys())
    pes_default = first_existing(
        sequence.continuous_data,
        [
            "synchronized_pes",
            "esophageal pressure (pod)",
            "pes",
            "P_es",
        ],
    )
    sequence_label_default = getattr(sequence, "label", "pressure_support")

    pes_label_picker = mo.ui.dropdown(
        options=continuous_labels,
        value=pes_default,
        label="PES signal",
        searchable=True,
    )
    sequence_label = mo.ui.text(
        value=str(sequence_label_default),
        label="Sequence label",
        full_width=True,
    )
    rr_input = mo.ui.number(
        start=0.01, stop=2.0, step=0.01, value=auto_rr, label="Respiratory rate (Hz)"
    )
    hr_input = mo.ui.number(
        start=0.1, stop=5.0, step=0.01, value=auto_hr, label="Heart rate (Hz)"
    )
    notch_distance = mo.ui.number(
        start=0.01, stop=1.0, step=0.01, value=0.2, label="Notch distance (Hz)"
    )
    use_flow = mo.ui.checkbox(
        value=True, label="Use flow guidance in experimental method"
    )

    _rate_note = (
        mo.callout(
            mo.md(
                f"RateDetection failed; using editable defaults. Error: `{rate_error}`"
            ),
            kind="warn",
        )
        if rate_error
        else mo.md(f"Auto-detected RR `{auto_rr:.3f}` Hz and HR `{auto_hr:.3f}` Hz.")
    )

    mo.vstack(
        [
            _rate_note,
            mo.hstack([pes_label_picker, sequence_label], gap=2),
            mo.hstack([rr_input, hr_input, notch_distance, use_flow], gap=2),
        ]
    )
    return (
        hr_input,
        notch_distance,
        pes_label_picker,
        rr_input,
        sequence_label,
        use_flow,
    )


@app.cell
def _run_comparison(
    PreprocessPes,
    PreprocessPesExperimental,
    copy,
    hr_input,
    method_result,
    notch_distance,
    pes_label_picker,
    rr_input,
    run_processor,
    sequence,
    sequence_label,
    use_flow,
):
    rr = float(rr_input.value)
    hr = float(hr_input.value)
    pes_label = pes_label_picker.value
    seq_label = sequence_label.value
    notch = float(notch_distance.value)

    old_output, old_captures, old_error = run_processor(
        PreprocessPes(),
        copy.deepcopy(sequence),
        pes_label,
        rr,
        hr,
        notch,
        seq_label,
    )
    experimental_output, experimental_captures, experimental_error = run_processor(
        PreprocessPesExperimental(),
        copy.deepcopy(sequence),
        pes_label,
        rr,
        hr,
        notch,
        seq_label,
        use_flow=bool(use_flow.value),
    )

    old_result = method_result("Old", old_output, old_captures, old_error)
    experimental_result = method_result(
        "Experimental",
        experimental_output,
        experimental_captures,
        experimental_error,
    )
    return experimental_result, old_result, pes_label, rr


@app.cell
def _tables(
    breath_table,
    detection_summary,
    experimental_result,
    flow_windows_from_captures,
    improvement_table,
    matched_peak_table,
    np,
    old_result,
    pd,
    pes_label,
    rr,
    sequence,
):
    old_breaths = breath_table(old_result)
    experimental_breaths = breath_table(experimental_result)
    breath_results = pd.concat([old_breaths, experimental_breaths], ignore_index=True)

    _pes_time = np.asarray(sequence.continuous_data[pes_label].time, dtype=float)
    flow_windows = flow_windows_from_captures(
        experimental_result["captures"], _pes_time
    )

    summary = pd.DataFrame(
        [
            detection_summary(old_result, old_breaths, flow_windows, rr),
            detection_summary(
                experimental_result, experimental_breaths, flow_windows, rr
            ),
        ]
    )
    improvement = improvement_table(summary)
    matched_peaks = matched_peak_table(
        old_breaths, experimental_breaths, tolerance_s=0.25 / rr
    )
    return breath_results, flow_windows, improvement, matched_peaks, summary


@app.cell
def _summary_view(improvement, mo, summary):
    mo.vstack(
        [
            mo.md("## Summary"),
            summary,
            mo.md("## Experimental minus old"),
            improvement
            if not improvement.empty
            else mo.callout(mo.md("No paired summary available."), kind="warn"),
        ]
    )
    return


@app.cell
def _time_slider(mo, np, pes_label, sequence):
    pes_signal = sequence.continuous_data[pes_label]
    pes_time = np.asarray(pes_signal.time, dtype=float)
    _t0 = float(pes_time[0])
    _t1 = float(pes_time[-1])
    time_range = mo.ui.range_slider(
        start=_t0,
        stop=_t1,
        step=max((_t1 - _t0) / 500.0, 0.1),
        value=[_t0, min(_t1, _t0 + 60.0)],
        label="Plot time range (s)",
        full_width=True,
    )
    mo.vstack([mo.md("## Signal plots"), time_range])
    return pes_signal, pes_time, time_range


@app.cell
def _plot_overlay(
    butter,
    experimental_result,
    flow_windows,
    go,
    make_subplots,
    np,
    old_result,
    pes_signal,
    pes_time,
    sequence,
    sosfiltfilt,
    time_range,
):
    _t0, _t1 = time_range.value
    _mask = (pes_time >= _t0) & (pes_time <= _t1)
    _fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Old PES with detected starts and maximum deflections",
            "Experimental PES with detected starts and maximum deflections",
            "Flow, if available",
            "Global EIT raw and filtered",
        ],
    )

    for _row, _result, _color, _symbol in [
        (1, old_result, "#D1495B", "circle"),
        (2, experimental_result, "#185FA5", "diamond"),
    ]:
        _fig.add_trace(
            go.Scatter(
                x=pes_time[_mask],
                y=np.asarray(pes_signal.values)[_mask],
                name=f"Raw PES ({_result['name']})",
                line=dict(color="#8C8C8C", width=1),
                opacity=0.45,
            ),
            row=_row,
            col=1,
        )
        if not _result["ok"]:
            continue
        _filtered = _result["filtered"]
        _time = np.asarray(_filtered.time, dtype=float)
        _values = np.asarray(_filtered.values, dtype=float)
        _local_mask = (_time >= _t0) & (_time <= _t1)
        _fig.add_trace(
            go.Scatter(
                x=_time[_local_mask],
                y=_values[_local_mask],
                name=f"{_result['name']} filtered",
                line=dict(color=_color, width=1.5),
            ),
            row=_row,
            col=1,
        )
        _starts = _result["starts"]
        _peaks = _result["peaks"]
        _starts = (
            _starts[(_time[_starts] >= _t0) & (_time[_starts] <= _t1)]
            if len(_starts)
            else _starts
        )
        _peaks = (
            _peaks[(_time[_peaks] >= _t0) & (_time[_peaks] <= _t1)]
            if len(_peaks)
            else _peaks
        )

        _fig.add_trace(
            go.Scatter(
                x=_time[_starts],
                y=_values[_starts],
                name=f"{_result['name']} starts",
                mode="markers",
                marker=dict(color=_color, symbol="x", size=9),
            ),
            row=_row,
            col=1,
        )
        _fig.add_trace(
            go.Scatter(
                x=_time[_peaks],
                y=_values[_peaks],
                name=f"{_result['name']} max deflection",
                mode="markers",
                marker=dict(color=_color, symbol=_symbol, size=8),
            ),
            row=_row,
            col=1,
        )

    if "synchronized_flow" in sequence.continuous_data:
        _flow = sequence.continuous_data["synchronized_flow"]
        _flow_time = np.asarray(_flow.time, dtype=float)
        _flow_values = np.asarray(_flow.values, dtype=float)
        _flow_mask = (_flow_time >= _t0) & (_flow_time <= _t1)
        _fig.add_trace(
            go.Scatter(
                x=_flow_time[_flow_mask],
                y=_flow_values[_flow_mask],
                name="Flow",
                line=dict(color="#2A9D8F", width=1.5),
            ),
            row=3,
            col=1,
        )

    _eit_time = _eit_values = None
    if "global_impedance_(raw)" in sequence.continuous_data:
        _eit = sequence.continuous_data["global_impedance_(raw)"]
        _eit_time = np.asarray(_eit.time, dtype=float)
        _eit_values = np.asarray(_eit.values, dtype=float)
    elif "raw" in sequence.eit_data:
        _eit = sequence.eit_data["raw"]
        _eit_time = np.asarray(_eit.time, dtype=float)
        _pixels = np.asarray(_eit.pixel_impedance, dtype=float)
        _eit_values = np.nansum(_pixels, axis=tuple(range(1, _pixels.ndim)))

    if _eit_time is not None and _eit_values is not None:
        _eit_mask = (_eit_time >= _t0) & (_eit_time <= _t1)
        _fig.add_trace(
            go.Scatter(
                x=_eit_time[_eit_mask],
                y=_eit_values[_eit_mask],
                name="Raw global EIT",
                line=dict(color="#8C8C8C", width=1),
                opacity=0.45,
            ),
            row=4,
            col=1,
        )

        _filtered_eit = None
        for _label in [
            "preprocessed_functional_impedance_emc",
            "preprocessed_functional_impedance_sang",
            "preprocessed_functional_impedance_adler",
            "preprocessed_functional_impedance_cornejo",
        ]:
            if _label in sequence.continuous_data:
                _filtered_eit = sequence.continuous_data[_label]
                break

        if _filtered_eit is not None:
            _filtered_time = np.asarray(_filtered_eit.time, dtype=float)
            _filtered_values = np.asarray(_filtered_eit.values, dtype=float)
        else:
            _filtered_time = _eit_time
            _filtered_values = _eit_values
            _fs = float(
                getattr(_eit, "sample_frequency", getattr(_eit, "framerate", 0.0))
                or 0.0
            )
            _cutoff = min(40 / 60, 0.45 * _fs) if _fs > 0 else 0.0
            if _cutoff > 0 and len(_eit_values) > 24:
                _sos = butter(4, _cutoff / (0.5 * _fs), btype="lowpass", output="sos")
                _filtered_values = sosfiltfilt(_sos, _eit_values)

        _filtered_mask = (_filtered_time >= _t0) & (_filtered_time <= _t1)
        _fig.add_trace(
            go.Scatter(
                x=_filtered_time[_filtered_mask],
                y=_filtered_values[_filtered_mask],
                name="Filtered global EIT",
                line=dict(color="#6A4C93", width=1.5),
            ),
            row=4,
            col=1,
        )

    if not flow_windows.empty:
        for _, _row in flow_windows.iterrows():
            if _row["end_time"] < _t0 or _row["start_time"] > _t1:
                continue
            for _pes_row in (1, 2):
                _fig.add_vrect(
                    x0=_row["start_time"],
                    x1=_row["end_time"],
                    fillcolor="#2A9D8F",
                    opacity=0.08,
                    line_width=0,
                    row=_pes_row,
                    col=1,
                )

    _fig.update_layout(
        template="plotly_white",
        height=1080,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=80, b=30),
    )
    _fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    _fig.update_yaxes(title_text="PES", row=1, col=1)
    _fig.update_yaxes(title_text="PES", row=2, col=1)
    _fig.update_yaxes(title_text="Flow", row=3, col=1)
    _fig.update_yaxes(title_text="EIT", row=4, col=1)
    _fig
    return


@app.cell
def _plot_ssf(
    experimental_result,
    go,
    make_subplots,
    np,
    old_result,
    time_range,
):
    _fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=["Old SSF", "Experimental SSF"],
    )

    for _row, _result, _color in [
        (1, old_result, "#D1495B"),
        (2, experimental_result, "#185FA5"),
    ]:
        _captures = _result["captures"]
        _ssf = np.asarray(_captures.get("ssf", []), dtype=float)
        if len(_ssf) == 0 or not _result["ok"]:
            continue
        _filtered = _result["filtered"]
        _ssf_time = np.asarray(_filtered.time, dtype=float)[: len(_ssf)]
        _t0, _t1 = time_range.value
        _mask = (_ssf_time >= _t0) & (_ssf_time <= _t1)
        _fig.add_trace(
            go.Scatter(
                x=_ssf_time[_mask],
                y=_ssf[_mask],
                name=f"{_result['name']} SSF",
                line=dict(color=_color),
            ),
            row=_row,
            col=1,
        )
        _maxima = np.asarray(_captures.get("ssf_maxima", []), dtype=int)
        _maxima = (
            _maxima[(_ssf_time[_maxima] >= _t0) & (_ssf_time[_maxima] <= _t1)]
            if len(_maxima)
            else _maxima
        )
        _fig.add_trace(
            go.Scatter(
                x=_ssf_time[_maxima],
                y=_ssf[_maxima],
                mode="markers",
                name=f"{_result['name']} SSF maxima",
                marker=dict(color=_color, size=7),
            ),
            row=_row,
            col=1,
        )

    _fig.update_layout(template="plotly_white", height=520, margin=dict(t=60, b=30))
    _fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    _fig
    return


@app.cell
def _plot_swings(breath_results, go, mo):
    mo.md("## PES swing comparison")
    if breath_results.empty:
        mo.output.replace(
            mo.callout(mo.md("No breath-level PES swings available."), kind="warn")
        )
    else:
        _fig = go.Figure()
        _method_colors = {"Old": "#D1495B", "Experimental": "#185FA5"}
        for _method, _df in breath_results.groupby("method"):
            _fig.add_trace(
                go.Box(
                    y=_df["pes_swing_abs"],
                    name=_method,
                    marker_color=_method_colors.get(_method),
                    line_color=_method_colors.get(_method),
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=0,
                    hovertemplate=(
                        f"{_method}<br>"
                        "Breath %{customdata}<br>"
                        "|PES swing| %{y:.3f}<extra></extra>"
                    ),
                    customdata=_df["breath"],
                )
            )
        _fig.update_layout(
            template="plotly_white",
            height=420,
            yaxis_title="|PES swing|",
            margin=dict(t=30, b=30),
        )
        mo.output.replace(_fig)
    return


@app.cell
def _details_view(breath_results, matched_peaks, mo):
    mo.vstack(
        [
            mo.md("## Breath-level details"),
            mo.md("### Matched old vs experimental deflection peaks"),
            matched_peaks
            if not matched_peaks.empty
            else mo.callout(mo.md("No matched peaks within tolerance."), kind="warn"),
            mo.md("### All detected breaths"),
            breath_results
            if not breath_results.empty
            else mo.callout(mo.md("No breath rows available."), kind="warn"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
