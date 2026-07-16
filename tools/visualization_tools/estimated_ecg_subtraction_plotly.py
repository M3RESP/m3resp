import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Estimated ECG Subtraction — interactive Plotly view

    Choose the deterministic regression-test signal, create a configurable
    synthetic EMG signal with ECG contamination, or upload a one-dimensional
    recording.

    The notebook then visualizes the EES method in the same
    sequence as Figure 1 of Jonkman et al. (2021).

    The composite figures contain the input, ECG-promoting filter,
    rectification, moving-average detection signal, dynamic threshold,
    threshold crossings, periodic rejection/restoration, Q/R/S localization,
    0.3 s windows, all four template-construction stages, reconstructed ECG,
    and final subtraction.

    Every graph supports hover readouts, zoom, pan, box selection, trace
    toggling through its legend, and reset through the Plotly mode bar. Default
    y-ranges are calculated from the complete recording, so changing the view
    does not rescale the signals automatically.
    """)
    return


@app.cell
def _():
    import io

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.signal import butter, sosfiltfilt

    from m3resp.processing.ecg import estimated_ecg_subtraction
    from m3resp.processing.filters import bandpass_filter
    from m3resp.processing.windows import moving_average

    return (
        bandpass_filter,
        butter,
        estimated_ecg_subtraction,
        go,
        io,
        make_subplots,
        moving_average,
        np,
        sosfiltfilt,
    )


@app.cell
def _(mo):
    source_mode = mo.ui.radio(
        options={
            "Paper/regression synthetic signal": "paper_synthetic",
            "Configurable synthetic EMG + ECG": "generated_synthetic",
            "Upload a recording": "uploaded_recording",
        },
        value="Paper/regression synthetic signal",
        label="Signal source",
        inline=True,
    )
    mo.vstack(
        [
            mo.md("## Signal source"),
            source_mode,
        ],
        gap=0.4,
    )
    return (source_mode,)


@app.cell
def _(mo, source_mode):
    uploaded_signal = mo.ui.file(
        filetypes=[".csv", ".txt", ".npy"],
        kind="area",
        label="Upload one numeric EMG column (.csv, .txt, or .npy)",
    )
    uploaded_sample_frequency = mo.ui.number(
        start=1.0,
        step=1.0,
        value=1000.0,
        label="Uploaded-recording sampling frequency (Hz)",
    )
    synthetic_duration_seconds = mo.ui.slider(
        4.0, 60.0, 1.0, 12.0, label="Synthetic duration (s)", show_value=True
    )
    synthetic_heart_rate_bpm = mo.ui.slider(
        40.0,
        150.0,
        1.0,
        75.0,
        label="Synthetic heart rate (beats/min)",
        show_value=True,
    )
    synthetic_emg_amplitude = mo.ui.slider(
        0.05, 1.00, 0.05, 0.40, label="Synthetic EMG amplitude", show_value=True
    )
    synthetic_emg_modulation_hz = mo.ui.slider(
        0.05,
        1.00,
        0.05,
        0.25,
        label="Synthetic EMG modulation (Hz)",
        show_value=True,
    )
    synthetic_ecg_amplitude = mo.ui.slider(
        0.1, 5.0, 0.1, 2.5, label="Synthetic ECG R-peak amplitude", show_value=True
    )
    synthetic_seed = mo.ui.number(
        start=0, step=1, value=7, label="Synthetic random seed"
    )
    if source_mode.value == "generated_synthetic":
        source_parameters = mo.vstack(
            [
                mo.md("### Generate synthetic EMG contaminated with ECG"),
                mo.hstack(
                    [
                        synthetic_duration_seconds,
                        synthetic_heart_rate_bpm,
                        synthetic_emg_amplitude,
                        synthetic_emg_modulation_hz,
                        synthetic_ecg_amplitude,
                        synthetic_seed,
                    ],
                    wrap=True,
                    widths="equal",
                ),
            ],
            gap=0.4,
        )
    elif source_mode.value == "uploaded_recording":
        source_parameters = mo.vstack(
            [
                mo.md("### Upload a recording"),
                mo.hstack(
                    [uploaded_signal, uploaded_sample_frequency],
                    wrap=True,
                    widths="equal",
                ),
                mo.md(
                    "Upload exactly one finite numeric signal column (no header). "
                    "An uploaded signal has no ECG-free reference, so the final "
                    "error-comparison plot is available only for synthetic signals."
                ),
            ],
            gap=0.4,
        )
    else:
        source_parameters = mo.md(
            "Using the deterministic paper/regression signal with its fixed "
            "sampling rate and known clean-EMG reference."
        )
    source_parameters
    return (
        synthetic_duration_seconds,
        synthetic_ecg_amplitude,
        synthetic_emg_amplitude,
        synthetic_emg_modulation_hz,
        synthetic_heart_rate_bpm,
        synthetic_seed,
        uploaded_sample_frequency,
        uploaded_signal,
    )


@app.cell
def _(
    butter,
    io,
    np,
    sosfiltfilt,
    source_mode,
    synthetic_duration_seconds,
    synthetic_ecg_amplitude,
    synthetic_emg_amplitude,
    synthetic_emg_modulation_hz,
    synthetic_heart_rate_bpm,
    synthetic_seed,
    uploaded_sample_frequency,
    uploaded_signal,
):
    """Load the selected input or generate a reproducible EMG-plus-ECG signal."""

    if source_mode.value == "uploaded_recording":
        if not uploaded_signal.value:
            raise ValueError("Choose a .csv, .txt, or .npy file before processing.")
        uploaded_file = uploaded_signal.value[0]
        uploaded_name = uploaded_file.name.lower()
        uploaded_contents = io.BytesIO(uploaded_file.contents)
        if uploaded_name.endswith(".npy"):
            contaminated_emg = np.load(uploaded_contents, allow_pickle=False)
        else:
            uploaded_delimiter = "," if uploaded_name.endswith(".csv") else None
            contaminated_emg = np.loadtxt(
                uploaded_contents,
                delimiter=uploaded_delimiter,
                dtype=float,
            )
        contaminated_emg = np.asarray(contaminated_emg, dtype=float)
        if contaminated_emg.ndim != 1:
            raise ValueError(
                "The uploaded recording must be one-dimensional: one numeric column."
            )
        if contaminated_emg.size < 3 or not np.all(np.isfinite(contaminated_emg)):
            raise ValueError(
                "The uploaded recording must contain at least three finite values."
            )
        sample_frequency = float(uploaded_sample_frequency.value)
        if not np.isfinite(sample_frequency) or sample_frequency <= 0:
            raise ValueError("Uploaded-recording sampling frequency must be positive.")
        time = np.arange(contaminated_emg.size) / sample_frequency
        clean_emg = None
        true_ecg = None
        true_beat_times = np.asarray([], dtype=float)
    else:
        sample_frequency = 1000.0
        if source_mode.value == "paper_synthetic":
            duration_seconds = 12.0
            heart_rate_bpm = 75.0
            emg_amplitude = 0.12
            ecg_amplitude = 2.5
            random_seed = 7
            modulation_frequency = 0.25
        else:
            duration_seconds = float(synthetic_duration_seconds.value)
            heart_rate_bpm = float(synthetic_heart_rate_bpm.value)
            emg_amplitude = float(synthetic_emg_amplitude.value)
            ecg_amplitude = float(synthetic_ecg_amplitude.value)
            random_seed = int(synthetic_seed.value)
            modulation_frequency = float(synthetic_emg_modulation_hz.value)
        time = (
            np.arange(int(round(duration_seconds * sample_frequency)))
            / sample_frequency
        )
        rng = np.random.default_rng(random_seed)
        emg_sos = butter(
            4,
            (80, 220),
            btype="bandpass",
            fs=sample_frequency,
            output="sos",
        )
        emg_noise = sosfiltfilt(emg_sos, rng.normal(size=time.size))
        if source_mode.value == "paper_synthetic":
            # Preserve the exact regression-test construction for this source.
            clean_emg = emg_amplitude * emg_noise
        else:
            # Respiratory modulation makes the synthetic EMG visibly distinct
            # from the repeating ECG while retaining high-frequency EMG content.
            emg_envelope = (
                0.20 + 0.80 * np.sin(2 * np.pi * modulation_frequency * time) ** 2
            )
            clean_emg = emg_amplitude * emg_envelope * emg_noise
        beat_interval_seconds = 60 / heart_rate_bpm
        true_beat_times = np.arange(
            0.8,
            duration_seconds - 0.6,
            beat_interval_seconds,
        )
        true_ecg = np.zeros_like(time)
        for beat_number, beat_time in enumerate(true_beat_times):
            beat_scale = 1 + 0.15 * np.sin(beat_number)
            true_ecg += beat_scale * (
                -0.24
                * ecg_amplitude
                * np.exp(-0.5 * ((time - (beat_time - 0.025)) / 0.008) ** 2)
                + ecg_amplitude * np.exp(-0.5 * ((time - beat_time) / 0.009) ** 2)
                - 0.36
                * ecg_amplitude
                * np.exp(-0.5 * ((time - (beat_time + 0.035)) / 0.012) ** 2)
                + 0.10
                * ecg_amplitude
                * np.exp(-0.5 * ((time - (beat_time + 0.16)) / 0.035) ** 2)
            )
        contaminated_emg = clean_emg + true_ecg

    return (
        clean_emg,
        contaminated_emg,
        sample_frequency,
        time,
        true_beat_times,
        true_ecg,
    )


@app.cell
def _(mo, time):
    """Expose every public EES parameter before the diagnostic figures."""

    detection_band_low = mo.ui.slider(
        start=1.0,
        stop=45.0,
        step=1.0,
        value=4.0,
        label="Low cutoff (Hz)",
        show_value=True,
    )
    detection_band_high = mo.ui.slider(
        start=50.0,
        stop=200.0,
        step=1.0,
        value=50.0,
        label="High cutoff (Hz)",
        show_value=True,
    )
    detection_filter_order = mo.ui.slider(
        start=1,
        stop=8,
        step=1,
        value=4,
        label="Filter order",
        show_value=True,
    )
    detection_smoothing_seconds = mo.ui.slider(
        start=0.001,
        stop=0.100,
        step=0.001,
        value=0.06,
        label="Rectified-signal smoothing (s)",
        show_value=True,
    )
    threshold_interval_seconds = mo.ui.slider(
        start=0.05,
        stop=2.0,
        step=0.05,
        value=1.5,
        label="Threshold interval (s)",
        show_value=True,
    )
    threshold_smoothing_seconds = mo.ui.slider(
        start=0.001,
        stop=0.100,
        step=0.001,
        value=0.0125,
        label="Threshold smoothing (s)",
        show_value=True,
    )
    qrs_window_seconds = mo.ui.slider(
        start=0.05,
        stop=0.60,
        step=0.01,
        value=0.30,
        label="Template window around R (s)",
        show_value=True,
    )
    inter_qrs_tolerance = mo.ui.slider(
        start=0.0,
        stop=0.95,
        step=0.01,
        value=0.66,
        label="Inter-QRS tolerance",
        show_value=True,
    )
    minimum_template_beats = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=3,
        label="Minimum complete template beats",
        show_value=True,
    )
    use_minimum_qrs_interval = mo.ui.checkbox(
        value=True,
        label="Enforce minimum QRS interval",
    )
    minimum_qrs_interval_seconds = mo.ui.slider(
        start=0.05,
        stop=1.0,
        step=0.05,
        value=0.25,
        label="Minimum QRS interval (s)",
        show_value=True,
    )
    use_maximum_qrs_interval = mo.ui.checkbox(
        value=True,
        label="Enforce maximum QRS interval",
    )
    maximum_qrs_interval_seconds = mo.ui.slider(
        start=1.1,
        stop=4.0,
        step=0.1,
        value=2.0,
        label="Maximum QRS interval (s)",
        show_value=True,
    )
    displayed_time = mo.ui.range_slider(
        start=float(time[0]),
        stop=float(time[-1]),
        step=0.1,
        value=[float(time[0]), float(time[-1])],
        label="Time shown in steps 1–7 and 10–11 (s)",
        full_width=True,
    )

    mo.vstack(
        [
            mo.md("## EES parameters"),
            mo.md("### ECG-promoting detection signal"),
            mo.hstack(
                [
                    detection_band_low,
                    detection_band_high,
                    detection_filter_order,
                    detection_smoothing_seconds,
                ],
                wrap=True,
                widths="equal",
            ),
            mo.md("### Dynamic threshold and periodicity correction"),
            mo.hstack(
                [
                    threshold_interval_seconds,
                    threshold_smoothing_seconds,
                    inter_qrs_tolerance,
                ],
                wrap=True,
                widths="equal",
            ),
            mo.md("### QRS template and physiological interval checks"),
            mo.hstack(
                [
                    qrs_window_seconds,
                    minimum_template_beats,
                    use_minimum_qrs_interval,
                    minimum_qrs_interval_seconds,
                    use_maximum_qrs_interval,
                    maximum_qrs_interval_seconds,
                ],
                wrap=True,
                widths="equal",
            ),
            mo.md("### Display"),
            displayed_time,
            mo.md(
                "The sliders start at the published/default EES values. "
                "Disable either interval check to pass `None` for that public parameter."
            ),
        ],
        gap=0.4,
    )
    return (
        detection_band_high,
        detection_band_low,
        detection_filter_order,
        detection_smoothing_seconds,
        displayed_time,
        inter_qrs_tolerance,
        maximum_qrs_interval_seconds,
        minimum_qrs_interval_seconds,
        minimum_template_beats,
        qrs_window_seconds,
        threshold_interval_seconds,
        threshold_smoothing_seconds,
        use_maximum_qrs_interval,
        use_minimum_qrs_interval,
    )


@app.cell
def _(
    bandpass_filter,
    contaminated_emg,
    detection_band_high,
    detection_band_low,
    detection_filter_order,
    detection_smoothing_seconds,
    estimated_ecg_subtraction,
    inter_qrs_tolerance,
    maximum_qrs_interval_seconds,
    minimum_qrs_interval_seconds,
    minimum_template_beats,
    moving_average,
    np,
    qrs_window_seconds,
    sample_frequency,
    threshold_interval_seconds,
    threshold_smoothing_seconds,
    use_maximum_qrs_interval,
    use_minimum_qrs_interval,
):
    detection_band = (
        float(detection_band_low.value),
        float(detection_band_high.value),
    )
    minimum_qrs_interval = (
        float(minimum_qrs_interval_seconds.value)
        if use_minimum_qrs_interval.value
        else None
    )
    maximum_qrs_interval = (
        float(maximum_qrs_interval_seconds.value)
        if use_maximum_qrs_interval.value
        else None
    )
    ees_result = estimated_ecg_subtraction(
        contaminated_emg,
        sample_frequency=sample_frequency,
        detection_band_hz=detection_band,
        filter_order=int(detection_filter_order.value),
        detection_smoothing_seconds=float(detection_smoothing_seconds.value),
        threshold_interval_seconds=float(threshold_interval_seconds.value),
        threshold_smoothing_seconds=float(threshold_smoothing_seconds.value),
        qrs_window_seconds=float(qrs_window_seconds.value),
        inter_qrs_tolerance=float(inter_qrs_tolerance.value),
        minimum_template_beats=int(minimum_template_beats.value),
        minimum_qrs_interval_seconds=minimum_qrs_interval,
        maximum_qrs_interval_seconds=maximum_qrs_interval,
    )

    # Steps 1–3, reproduced with the same configurable public primitives.
    promoted_ecg = bandpass_filter(
        contaminated_emg,
        cutoff_frequency=detection_band,
        sample_frequency=sample_frequency,
        order=int(detection_filter_order.value),
    )
    rectified_ecg = np.abs(promoted_ecg)
    detection_window_samples = max(
        1,
        int(round(float(detection_smoothing_seconds.value) * sample_frequency)),
    )
    smoothed_detection = moving_average(
        rectified_ecg,
        window_size=detection_window_samples,
    ) / np.ptp(contaminated_emg)

    # Step 5: starts and ends of every above-threshold segment.
    above_threshold = smoothed_detection > ees_result.dynamic_threshold
    padded_threshold_mask = np.pad(above_threshold.astype(np.int8), (1, 1))
    threshold_changes = np.diff(padded_threshold_mask)
    qrs_segment_starts = np.flatnonzero(threshold_changes == 1)
    qrs_segment_ends = np.flatnonzero(threshold_changes == -1) - 1

    # Steps 8–9: original, normalized, average, and denormalized beat windows.
    template_offsets = ees_result.template_sample_offsets
    template_length = len(template_offsets)
    half_template = template_length // 2
    original_qrs_segments = np.stack(
        [
            contaminated_emg[
                int(_r_index) - half_template : int(_r_index)
                - half_template
                + template_length
            ]
            for _r_index in ees_result.qrs_indices[:, 1]
        ]
    )
    denormalized_qrs_segments = []
    for _q_index, _r_index, _s_index in ees_result.qrs_indices:
        _q_value = contaminated_emg[_q_index]
        _r_value = contaminated_emg[_r_index]
        _s_value = contaminated_emg[_s_index]
        _denormalized = np.empty_like(ees_result.normalized_template)
        _denormalized[: half_template + 1] = (
            ees_result.normalized_template[: half_template + 1] * (_r_value - _q_value)
            + _q_value
        )
        _denormalized[half_template:] = (
            ees_result.normalized_template[half_template:] * (_r_value - _s_value)
            + _s_value
        )
        denormalized_qrs_segments.append(_denormalized)
    denormalized_qrs_segments = np.asarray(denormalized_qrs_segments)

    return (
        above_threshold,
        denormalized_qrs_segments,
        ees_result,
        original_qrs_segments,
        promoted_ecg,
        qrs_segment_ends,
        qrs_segment_starts,
        rectified_ecg,
        smoothed_detection,
        template_offsets,
    )


@app.cell
def _(
    clean_emg,
    contaminated_emg,
    ees_result,
    mo,
    np,
    sample_frequency,
    true_beat_times,
):
    if clean_emg is None:
        metrics_display = mo.md(
            f"""
            **Detected template beats:** {len(ees_result.qrs_indices)} &nbsp; · &nbsp;
            **Rejected candidates:** {len(ees_result.rejected_peak_indices)} &nbsp; · &nbsp;
            **Restored candidates:** {len(ees_result.restored_peak_indices)}

            This uploaded recording has no ECG-free reference, so error reduction
            cannot be calculated.
            """
        )
    else:
        qrs_error_mask = np.zeros(contaminated_emg.size, dtype=bool)
        metric_time = np.arange(contaminated_emg.size) / sample_frequency
        for true_beat_time in true_beat_times:
            qrs_error_mask |= np.abs(metric_time - true_beat_time) < 0.15
        mse_before = float(
            np.mean((contaminated_emg[qrs_error_mask] - clean_emg[qrs_error_mask]) ** 2)
        )
        mse_after = float(
            np.mean(
                (ees_result.cleaned[qrs_error_mask] - clean_emg[qrs_error_mask]) ** 2
            )
        )
        error_reduction = 100 * (1 - mse_after / mse_before)
        metrics_display = mo.md(
            f"""
            **Detected template beats:** {len(ees_result.qrs_indices)} &nbsp; · &nbsp;
            **Rejected candidates:** {len(ees_result.rejected_peak_indices)} &nbsp; · &nbsp;
            **Restored candidates:** {len(ees_result.restored_peak_indices)} &nbsp; · &nbsp;
            **Known-QRS error reduction:** {error_reduction:.1f}%
            """
        )
    metrics_display
    return


@app.cell
def _(
    contaminated_emg,
    displayed_time,
    ees_result,
    go,
    make_subplots,
    np,
    promoted_ecg,
    qrs_segment_ends,
    qrs_segment_starts,
    rectified_ecg,
    smoothed_detection,
    time,
):
    """Interactive steps 1–5: input through threshold crossings."""

    window_start, window_end = displayed_time.value
    visible = (time >= window_start) & (time <= window_end)

    paper_amplitude_limit = float(
        1.05
        * max(
            np.max(np.abs(contaminated_emg)),
            np.max(np.abs(promoted_ecg)),
            np.max(np.abs(ees_result.estimated_ecg)),
            np.max(np.abs(ees_result.cleaned)),
        )
    )
    paper_detection_limit = float(
        1.05
        * max(
            np.max(smoothed_detection),
            np.max(ees_result.dynamic_threshold),
        )
    )

    figure_1_to_5 = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Input: EMGdi-DS",
            "1. Bandpass filter to promote relative ECG amplitude",
            "2. Rectification",
            "3. Moving average filter\n4. Apply threshold level",
            "5. Detect crossings with threshold level",
            "",
        ),
        vertical_spacing=0.09,
    )

    def _add_signal(values, row, col, name, color, showlegend=False):
        figure_1_to_5.add_trace(
            go.Scattergl(
                x=time[visible],
                y=values[visible],
                name=name,
                legendgroup=name,
                showlegend=showlegend,
                line={"color": color, "width": 1},
                hovertemplate="Time: %{x:.3f} s<br>Amplitude: %{y:.4f}<extra>"
                + name
                + "</extra>",
            ),
            row=row,
            col=col,
        )

    _add_signal(contaminated_emg, 1, 1, "EMGdi-DS", "#3B7CCC", True)
    _add_signal(promoted_ecg, 1, 2, "Bandpass filtered", "#3B7CCC", True)
    _add_signal(rectified_ecg, 2, 1, "Rectified", "#3B7CCC", True)
    _add_signal(smoothed_detection, 2, 2, "Smoothed detection", "#3B7CCC", True)
    _add_signal(
        ees_result.dynamic_threshold, 2, 2, "Dynamic threshold", "#222222", True
    )
    _add_signal(smoothed_detection, 3, 1, "Smoothed detection", "#3B7CCC")
    _add_signal(ees_result.dynamic_threshold, 3, 1, "Dynamic threshold", "#222222")

    _start_visible = qrs_segment_starts[
        (time[qrs_segment_starts] >= window_start)
        & (time[qrs_segment_starts] <= window_end)
    ]
    _end_visible = qrs_segment_ends[
        (time[qrs_segment_ends] >= window_start)
        & (time[qrs_segment_ends] <= window_end)
    ]
    figure_1_to_5.add_trace(
        go.Scatter(
            x=time[_start_visible],
            y=smoothed_detection[_start_visible],
            mode="markers",
            name="start QRS segment",
            marker={"symbol": "x", "color": "#E64B35", "size": 8},
        ),
        row=3,
        col=1,
    )
    figure_1_to_5.add_trace(
        go.Scatter(
            x=time[_end_visible],
            y=smoothed_detection[_end_visible],
            mode="markers",
            name="end QRS segment",
            marker={"symbol": "x", "color": "#F28E2B", "size": 8},
        ),
        row=3,
        col=1,
    )

    figure_1_to_5.update_yaxes(
        range=[-paper_amplitude_limit, paper_amplitude_limit], row=1, col=1
    )
    figure_1_to_5.update_yaxes(
        range=[-paper_amplitude_limit, paper_amplitude_limit], row=1, col=2
    )
    figure_1_to_5.update_yaxes(range=[0, paper_amplitude_limit], row=2, col=1)
    figure_1_to_5.update_yaxes(range=[0, paper_detection_limit], row=2, col=2)
    figure_1_to_5.update_yaxes(range=[0, paper_detection_limit], row=3, col=1)
    figure_1_to_5.update_xaxes(range=[window_start, window_end])
    figure_1_to_5.update_xaxes(title_text="Time (s)")
    figure_1_to_5.update_layout(
        height=850,
        title="Steps 1–5: ECG-promoting detection signal and threshold crossings",
        hovermode="x unified",
        dragmode="pan",
        legend={"orientation": "h", "y": 1.05},
        margin={"l": 50, "r": 20, "t": 90, "b": 45},
    )
    figure_1_to_5
    return paper_amplitude_limit, visible, window_end, window_start


@app.cell
def _(
    ees_result,
    go,
    sample_frequency,
    smoothed_detection,
    time,
):
    """Step 6 (detail): periodicity correction — rejected and restored candidates."""

    if len(ees_result.rejected_peak_indices):
        _correction_center = int(ees_result.rejected_peak_indices[0])
    else:
        _correction_center = int(
            ees_result.candidate_peak_indices[
                len(ees_result.candidate_peak_indices) // 2
            ]
        )
    _correction_radius = int(round(0.45 * sample_frequency))
    _correction_start = max(0, _correction_center - _correction_radius)
    _correction_end = min(len(time), _correction_center + _correction_radius)
    _correction_slice = slice(_correction_start, _correction_end)

    figure_6 = go.Figure()
    figure_6.add_trace(
        go.Scatter(
            x=time[_correction_slice],
            y=smoothed_detection[_correction_slice],
            name="Smoothed detection",
            line={"color": "#3B7CCC"},
        )
    )
    figure_6.add_trace(
        go.Scatter(
            x=time[_correction_slice],
            y=ees_result.dynamic_threshold[_correction_slice],
            name="Dynamic threshold",
            line={"color": "#222222"},
        )
    )
    for _indices, _marker, _color, _label in (
        (ees_result.rejected_peak_indices, "x", "#E64B35", "deleted"),
        (ees_result.restored_peak_indices, "diamond", "#3B7CCC", "restored"),
    ):
        _nearby = _indices[
            (_indices >= _correction_start) & (_indices < _correction_end)
        ]
        if len(_nearby):
            figure_6.add_trace(
                go.Scatter(
                    x=time[_nearby],
                    y=smoothed_detection[_nearby],
                    mode="markers",
                    name=_label,
                    marker={"symbol": _marker, "color": _color, "size": 10},
                )
            )
    figure_6.update_layout(
        title=(
            "6. Periodicity correction (detail) — remove (and restore) wrongfully "
            f"detected (or deleted) QRS segments. Rejected: "
            f"{len(ees_result.rejected_peak_indices)}, Restored: "
            f"{len(ees_result.restored_peak_indices)}"
        ),
        hovermode="x unified",
        dragmode="pan",
        height=350,
        margin={"l": 50, "r": 20, "t": 70, "b": 40},
    )
    figure_6.update_xaxes(title="Time (s)")
    figure_6.update_yaxes(title="Normalized")
    figure_6
    return


@app.cell
def _(
    contaminated_emg,
    ees_result,
    go,
    paper_amplitude_limit,
    time,
    visible,
    window_end,
    window_start,
):
    """Step 7: Q, R, and S peaks on the unfiltered input."""

    figure_7 = go.Figure()
    figure_7.add_trace(
        go.Scattergl(
            x=time[visible],
            y=contaminated_emg[visible],
            name="EMGdi-DS",
            line={"color": "#3B7CCC", "width": 1},
        )
    )
    _qrs = ees_result.qrs_indices
    for _column, _color, _label in (
        (0, "#3F3F3F", "Q"),
        (1, "#E5A823", "R"),
        (2, "#E64B35", "S"),
    ):
        _indices = _qrs[:, _column]
        _inside = (time[_indices] >= window_start) & (time[_indices] <= window_end)
        figure_7.add_trace(
            go.Scatter(
                x=time[_indices[_inside]],
                y=contaminated_emg[_indices[_inside]],
                mode="markers",
                name=_label,
                marker={"color": _color, "size": 8},
            )
        )
    figure_7.update_layout(
        title="7. Detect QRS peaks in EMGdi-DS",
        hovermode="x unified",
        dragmode="pan",
        height=400,
        margin={"l": 50, "r": 20, "t": 55, "b": 40},
    )
    figure_7.update_xaxes(title="Time (s)", range=[window_start, window_end])
    figure_7.update_yaxes(
        title="Amplitude", range=[-paper_amplitude_limit, paper_amplitude_limit]
    )
    figure_7
    return


@app.cell
def _(
    contaminated_emg,
    ees_result,
    go,
    np,
    paper_amplitude_limit,
    qrs_window_seconds,
    sample_frequency,
    time,
):
    """Step 8: select the QRS template window around three example beats."""

    _qrs = ees_result.qrs_indices
    _middle_beat = len(_qrs) // 2
    _shown_beats = _qrs[max(0, _middle_beat - 1) : _middle_beat + 2]
    _window_left = max(0, int(_shown_beats[0, 1]) - int(0.35 * sample_frequency))
    _window_right = min(
        len(time), int(_shown_beats[-1, 1]) + int(0.35 * sample_frequency)
    )
    selected_template_window_seconds = float(qrs_window_seconds.value)
    selected_template_half_window_seconds = selected_template_window_seconds / 2

    figure_8 = go.Figure()
    figure_8.add_trace(
        go.Scattergl(
            x=time[_window_left:_window_right],
            y=contaminated_emg[_window_left:_window_right],
            name="EMGdi-DS",
            line={"color": "#3B7CCC"},
        )
    )
    for _column, _color, _label in (
        (0, "#3F3F3F", "Q"),
        (1, "#E5A823", "R"),
        (2, "#E64B35", "S"),
    ):
        figure_8.add_trace(
            go.Scatter(
                x=time[_shown_beats[:, _column]],
                y=contaminated_emg[_shown_beats[:, _column]],
                mode="markers",
                name=_label,
                marker={"color": _color, "size": 9},
            )
        )
    _selected_r_time = float(time[_shown_beats[-1, 1]])
    figure_8.add_vrect(
        x0=_selected_r_time - selected_template_half_window_seconds,
        x1=_selected_r_time + selected_template_half_window_seconds,
        fillcolor="#E5A823",
        opacity=0.18,
        line_width=0,
        annotation_text=f"{selected_template_window_seconds:.2f} s template window",
    )
    figure_8.update_layout(
        title=(
            "8. Select window length around R wave "
            f"({selected_template_window_seconds:.2f} s selected) to capture the "
            "full artifact"
        ),
        hovermode="x unified",
        dragmode="pan",
        height=400,
        margin={"l": 50, "r": 20, "t": 55, "b": 40},
    )
    figure_8.update_yaxes(
        range=[-paper_amplitude_limit, paper_amplitude_limit], title="Amplitude"
    )
    figure_8.update_xaxes(title="Time (s)")
    figure_8
    return


@app.cell
def _(
    denormalized_qrs_segments,
    ees_result,
    go,
    make_subplots,
    np,
    original_qrs_segments,
    sample_frequency,
    template_offsets,
):
    """Step 9: all four template-construction panels."""

    template_time = template_offsets / sample_frequency
    figure_9 = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=(
            "a. Original QRS segments",
            "b. Normalized QRS segments",
            "c. Average QRS segment",
            "d. Denormalized QRS segments",
        ),
    )
    colors = [
        f"hsl({hue}, 60%, 45%)"
        for hue in np.linspace(0, 330, len(original_qrs_segments))
    ]
    for original, normalized, denormalized, template_color in zip(
        original_qrs_segments,
        ees_result.normalized_segments,
        denormalized_qrs_segments,
        colors,
    ):
        figure_9.add_trace(
            go.Scatter(
                x=template_time,
                y=original,
                mode="lines",
                line={"color": template_color, "width": 1},
                showlegend=False,
                hovertemplate="Offset: %{x:.3f} s<br>Amplitude: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure_9.add_trace(
            go.Scatter(
                x=template_time,
                y=normalized,
                mode="lines",
                line={"color": template_color, "width": 1},
                showlegend=False,
                hovertemplate="Offset: %{x:.3f} s<br>Normalized: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        figure_9.add_trace(
            go.Scatter(
                x=template_time,
                y=denormalized,
                mode="lines",
                line={"color": template_color, "width": 1},
                showlegend=False,
                hovertemplate="Offset: %{x:.3f} s<br>Amplitude: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=4,
        )
    figure_9.add_trace(
        go.Scatter(
            x=template_time,
            y=ees_result.normalized_template,
            mode="lines",
            name="Average template",
            line={"color": "#F28E2B", "width": 2},
            hovertemplate="Offset: %{x:.3f} s<br>Normalized: %{y:.4f}<extra>Average template</extra>",
        ),
        row=1,
        col=3,
    )
    figure_9.update_layout(
        title="9. Create the QRS template",
        height=380,
        hovermode="x unified",
        dragmode="pan",
        margin={"l": 45, "r": 20, "t": 55, "b": 40},
    )
    figure_9.update_xaxes(title_text="Time around R (s)")
    figure_9
    return


@app.cell
def _(
    contaminated_emg,
    ees_result,
    go,
    make_subplots,
    paper_amplitude_limit,
    time,
    visible,
    window_end,
    window_start,
):
    """Steps 10–11: insert the reconstructed ECG and subtract it."""

    figure_10_to_11 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "10. Insert denormalized QRS segments at original location",
            "11. Subtract estimated ECG template from EMGdi-DS",
        ),
    )
    figure_10_to_11.add_trace(
        go.Scattergl(
            x=time[visible],
            y=ees_result.estimated_ecg[visible],
            name="Estimated ECG",
            line={"color": "#F28E2B", "width": 1},
            hovertemplate="Time: %{x:.3f} s<br>Estimated ECG: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure_10_to_11.add_trace(
        go.Scattergl(
            x=time[visible],
            y=contaminated_emg[visible],
            name="EMGdi-DS",
            line={"color": "#3B7CCC", "width": 1},
            visible="legendonly",
        ),
        row=1,
        col=1,
    )
    figure_10_to_11.add_trace(
        go.Scattergl(
            x=time[visible],
            y=ees_result.cleaned[visible],
            name="EES-subtracted EMG",
            line={"color": "#3B7CCC", "width": 1},
            hovertemplate="Time: %{x:.3f} s<br>Cleaned EMG: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure_10_to_11.update_yaxes(
        range=[-paper_amplitude_limit, paper_amplitude_limit],
        title="Amplitude",
    )
    figure_10_to_11.update_xaxes(
        title_text="Time (s)", range=[window_start, window_end]
    )
    figure_10_to_11.update_layout(
        title="Steps 10–11: reconstruction and subtraction",
        height=430,
        hovermode="x unified",
        dragmode="pan",
        margin={"l": 50, "r": 20, "t": 60, "b": 45},
    )
    figure_10_to_11
    return


@app.cell
def _(
    clean_emg,
    contaminated_emg,
    displayed_time,
    ees_result,
    go,
    make_subplots,
    mo,
    np,
    time,
    true_beat_times,
):
    """Validate EES against the known ECG-free synthetic reference signal."""

    def _reference_comparison(reference_emg):
        comparison_start, comparison_end = displayed_time.value
        comparison_visible = (time >= comparison_start) & (time <= comparison_end)
        reference_error = ees_result.cleaned - reference_emg
        baseline_error = contaminated_emg - reference_emg
        rmse_before = float(np.sqrt(np.mean(baseline_error**2)))
        rmse_after = float(np.sqrt(np.mean(reference_error**2)))
        error_reduction_percent = 100 * (1 - rmse_after / rmse_before)
        validation_amplitude_limit = float(
            1.05
            * max(
                np.max(np.abs(reference_emg)),
                np.max(np.abs(contaminated_emg)),
                np.max(np.abs(ees_result.cleaned)),
                np.max(np.abs(reference_error)),
            )
        )
        comparison_figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Validation: EES-subtracted EMG versus known ECG-free reference",
                "Residual error after subtraction (known ECG windows shaded)",
            ),
        )
        comparison_figure.add_trace(
            go.Scattergl(
                x=time[comparison_visible],
                y=reference_emg[comparison_visible],
                name="Reference EMG without ECG contamination",
                line={"color": "#222222", "width": 1},
            ),
            row=1,
            col=1,
        )
        comparison_figure.add_trace(
            go.Scattergl(
                x=time[comparison_visible],
                y=ees_result.cleaned[comparison_visible],
                name="EES-subtracted EMG",
                line={"color": "#3B7CCC", "width": 1},
            ),
            row=1,
            col=1,
        )
        comparison_figure.add_trace(
            go.Scattergl(
                x=time[comparison_visible],
                y=reference_error[comparison_visible],
                name="EES-subtracted EMG − reference EMG",
                line={"color": "#D1495B", "width": 1},
            ),
            row=2,
            col=1,
        )
        for reference_beat_time in true_beat_times:
            if comparison_start <= reference_beat_time <= comparison_end:
                comparison_figure.add_vrect(
                    x0=reference_beat_time - 0.15,
                    x1=reference_beat_time + 0.15,
                    fillcolor="#F59E0B",
                    opacity=0.12,
                    line_width=0,
                    row=2,
                    col=1,
                )
        comparison_figure.update_yaxes(
            range=[-validation_amplitude_limit, validation_amplitude_limit],
            title="Amplitude",
            row=1,
            col=1,
        )
        comparison_figure.update_yaxes(
            range=[-validation_amplitude_limit, validation_amplitude_limit],
            title="Error",
            row=2,
            col=1,
        )
        comparison_figure.update_xaxes(
            range=[comparison_start, comparison_end],
            title_text="Time (s)",
            row=2,
            col=1,
        )
        comparison_figure.update_layout(
            title=(
                f"Synthetic-reference validation — RMSE before: {rmse_before:.4f}, "
                f"after: {rmse_after:.4f}, reduction: {error_reduction_percent:.1f}%, "
                f"max |error|: {np.max(np.abs(reference_error)):.4f}"
            ),
            height=650,
            hovermode="x unified",
            dragmode="pan",
            margin={"l": 55, "r": 20, "t": 65, "b": 45},
        )
        return comparison_figure

    if clean_emg is None:
        comparison_display = mo.md(
            "### Reference comparison unavailable\n\nAn uploaded recording does not include a known ECG-free EMG reference. Select either synthetic source to calculate and plot subtraction error."
        )
    else:
        comparison_display = _reference_comparison(clean_emg)
    comparison_display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Run

    ```bash
    .venv/bin/marimo edit tools/visualization_tools/estimated_ecg_subtraction_plotly.py
    ```

    Select **Upload a recording** and provide its sampling frequency to process
    a `.csv`, `.txt`, or `.npy` one-dimensional signal. The configurable
    synthetic source provides known clean EMG and ECG components for testing.
    For a clinical recording, retain ECG frequency content before EES and
    review the QRS detections and estimated ECG before accepting the cleaned
    signal.
    """)
    return


if __name__ == "__main__":
    app.run()
