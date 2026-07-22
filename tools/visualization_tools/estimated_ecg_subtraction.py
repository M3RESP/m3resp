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
    # Estimated ECG Subtraction

    Choose the deterministic regression-test signal, create a configurable
    synthetic EMG signal with ECG contamination, or upload a one-dimensional
    recording. 
          
    The notebook then visualizes the EES method in the same
    sequence as Figure 1 of Jonkman et al. (2021).

    The composite figure contains the input, ECG-promoting filter,
    rectification, moving-average detection signal, dynamic threshold,
    threshold crossings, periodic rejection/restoration, Q/R/S localization,
    0.3 s windows, all four template-construction stages, reconstructed ECG,
    and final subtraction.
    """)
    return


@app.cell
def _():
    import io

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from m3resp.processing.ecg import estimated_ecg_subtraction
    from m3resp.processing.filters import bandpass_filter
    from m3resp.processing.windows import moving_average

    return (
        bandpass_filter,
        butter,
        estimated_ecg_subtraction,
        io,
        moving_average,
        np,
        plt,
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
        value="Configurable synthetic EMG + ECG",
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
        value=2000.0,
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
            contaminated_emg = np.asarray(contaminated_emg, dtype=float)
            sample_frequency = float(uploaded_sample_frequency.value)
        elif uploaded_name.endswith(".csv"):
            contaminated_emg = np.loadtxt(uploaded_contents, delimiter=",", dtype=float)
            sample_frequency = float(uploaded_sample_frequency.value)
        else:
            # A .txt upload may be a plain one-column signal or a BIOPAC
            # AcqKnowledge export (metadata header + tab-separated multi-channel
            # data, e.g. the slices produced by tools/signal_slicer). Reuse that
            # tool's tested auto-detecting parser instead of duplicating its
            # header rules here.
            import importlib.util
            import sys
            import tempfile
            from pathlib import Path as _Path

            _slicer_path = (
                _Path(__file__).resolve().parents[1]
                / "signal_slicer"
                / "slice_signal.py"
            )
            _slicer_spec = importlib.util.spec_from_file_location(
                "slice_signal", _slicer_path
            )
            _slicer_module = importlib.util.module_from_spec(_slicer_spec)
            # dataclass (with `from __future__ import annotations`) looks itself
            # up via sys.modules[cls.__module__] during class creation, so the
            # module must be registered before exec_module runs.
            sys.modules[_slicer_spec.name] = _slicer_module
            _slicer_spec.loader.exec_module(_slicer_module)

            with tempfile.NamedTemporaryFile(suffix=".txt") as _tmp:
                _tmp.write(uploaded_file.contents)
                _tmp.flush()
                uploaded_signal_file = _slicer_module.read_file(_tmp.name)

            if uploaded_signal_file.fmt == "plain":
                contaminated_emg = np.asarray(uploaded_signal_file.data, dtype=float)
                sample_frequency = float(uploaded_sample_frequency.value)
            else:
                _emg_channel_index = next(
                    (
                        _i
                        for _i, _label in enumerate(uploaded_signal_file.channels)
                        if _label.split(" - ")[0].strip().lower() == "emgdi"
                    ),
                    next(
                        (
                            _i
                            for _i, _label in enumerate(uploaded_signal_file.channels)
                            if "emg" in _label.lower()
                        ),
                        None,
                    ),
                )
                if _emg_channel_index is None:
                    raise ValueError(
                        "Could not find an EMG channel in this BIOPAC export. "
                        f"Channels found: {uploaded_signal_file.channels}."
                    )
                contaminated_emg = np.asarray(
                    uploaded_signal_file.channel(_emg_channel_index), dtype=float
                )
                sample_frequency = (
                    float(uploaded_signal_file.fs)
                    if uploaded_signal_file.fs
                    else float(uploaded_sample_frequency.value)
                )
        if contaminated_emg.ndim != 1:
            raise ValueError(
                "The uploaded recording must be one-dimensional: one numeric column."
            )
        if contaminated_emg.size < 3 or not np.all(np.isfinite(contaminated_emg)):
            raise ValueError(
                "The uploaded recording must contain at least three finite values."
            )
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
    use_output_bandpass = mo.ui.checkbox(
        value=False,
        label="Apply an extra output bandpass filter",
    )
    output_bandpass_low = mo.ui.slider(
        start=1.0,
        stop=45.0,
        step=1.0,
        value=4.0,
        label="Output bandpass low cutoff (Hz)",
        show_value=True,
    )
    output_bandpass_high = mo.ui.slider(
        start=50.0,
        stop=200.0,
        step=1.0,
        value=50.0,
        label="Output bandpass high cutoff (Hz)",
        show_value=True,
    )
    output_bandpass_order = mo.ui.slider(
        start=1,
        stop=8,
        step=1,
        value=4,
        label="Output bandpass filter order",
        show_value=True,
    )
    output_bandpass_stage = mo.ui.radio(
        options={
            "Before subtraction (filter the raw signal, then subtract the unfiltered estimated ECG)": "before_subtraction",
            "After subtraction (subtract first, then filter the cleaned result)": "after_subtraction",
        },
        value="After subtraction (subtract first, then filter the cleaned result)",
        label="Output bandpass stage",
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
            mo.md(
                "### Optional output bandpass\n"
                "A separate bandpass filter, independent of the detection band "
                "above. QRS detection and the template always use the raw "
                "signal; this filter only changes how the final `cleaned` "
                "output is computed from the raw signal and the (always "
                "unfiltered) estimated ECG template."
            ),
            mo.hstack(
                [
                    use_output_bandpass,
                    output_bandpass_low,
                    output_bandpass_high,
                    output_bandpass_order,
                ],
                wrap=True,
                widths="equal",
            ),
            output_bandpass_stage,
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
        output_bandpass_high,
        output_bandpass_low,
        output_bandpass_order,
        output_bandpass_stage,
        qrs_window_seconds,
        threshold_interval_seconds,
        threshold_smoothing_seconds,
        use_maximum_qrs_interval,
        use_minimum_qrs_interval,
        use_output_bandpass,
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
    output_bandpass_high,
    output_bandpass_low,
    output_bandpass_order,
    output_bandpass_stage,
    qrs_window_seconds,
    sample_frequency,
    threshold_interval_seconds,
    threshold_smoothing_seconds,
    use_maximum_qrs_interval,
    use_minimum_qrs_interval,
    use_output_bandpass,
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
    output_bandpass_hz = (
        (float(output_bandpass_low.value), float(output_bandpass_high.value))
        if use_output_bandpass.value
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
        output_bandpass_hz=output_bandpass_hz,
        output_bandpass_stage=output_bandpass_stage.value,
        output_bandpass_order=int(output_bandpass_order.value),
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
def _(contaminated_emg, displayed_time, np, sample_frequency, source_mode):
    """Locate the recording on the Annemijn Biopac clock and slice matching
    Paw / EIT context for the currently displayed time window. Only exact
    tail-slice files of Paw_EMG_ajM3Resp_test.txt can be located this way;
    anything else (including synthetic sources) reports why not."""

    from functools import lru_cache
    from importlib.util import find_spec
    from pathlib import Path

    import pandas as pd

    _eit_sync_available = (
        find_spec("eitprocessing") is not None
        and find_spec("m3resp.adapters") is not None
        and find_spec("m3resp.synchronization") is not None
    )

    _annemijn_dir_candidates = (
        Path(__file__).resolve().parents[2] / "data" / "source" / "eit_emg_annemijn",
    )
    _annemijn_dir = next(
        (_path for _path in _annemijn_dir_candidates if _path.exists()),
        _annemijn_dir_candidates[0],
    )
    _biopac_file = _annemijn_dir / "Paw_EMG_ajM3Resp_test.txt"
    _eit_file = _annemijn_dir / "ajM3resp_03_001_01.bin"
    _annemijn_sample_frequency = 2000.0

    @lru_cache(maxsize=1)
    def _load_annemijn_biopac(path):
        _data = pd.read_csv(
            path,
            sep="\t",
            skiprows=11,
            names=["paw", "emg_di", "aux_ignore"],
            usecols=[0, 1, 2],
            engine="c",
        )
        return (
            _data["paw"].to_numpy(dtype=float),
            _data["emg_di"].to_numpy(dtype=float),
        )

    @lru_cache(maxsize=1)
    def _load_annemijn_eit(path):
        from m3resp.adapters import EITProcessingAdapter

        _sequence = EITProcessingAdapter().load(str(path), vendor="draeger")
        _raw = _sequence.eit_data["raw"]
        _eit_time = np.asarray(_raw.time, dtype=float)
        _eit_time = _eit_time - _eit_time[0]
        _pixel_impedance = np.asarray(_raw.pixel_impedance, dtype=float)
        _global_impedance = np.nansum(_pixel_impedance, axis=(1, 2))
        return _eit_time, _global_impedance

    eit_paw_paw_time = eit_paw_paw_values = None
    eit_paw_eit_time = eit_paw_eit_values = None
    eit_paw_paw_unavailable_reason = None
    eit_paw_eit_unavailable_reason = None

    if source_mode.value != "uploaded_recording":
        eit_paw_paw_unavailable_reason = (
            "Only available for **Upload a recording** with an Annemijn "
            "Biopac slice file."
        )
    elif not _eit_sync_available:
        eit_paw_paw_unavailable_reason = (
            "Requires the optional `eitprocessing` dependency, which is not "
            "installed in this environment."
        )
    elif not _biopac_file.exists() or not _eit_file.exists():
        eit_paw_paw_unavailable_reason = (
            f"Annemijn dataset files not found under {_biopac_file.parent}."
        )
    elif abs(sample_frequency - _annemijn_sample_frequency) > 1e-6:
        eit_paw_paw_unavailable_reason = (
            "Assumes the Annemijn 2000 Hz Biopac clock; the loaded recording "
            f"is {sample_frequency:g} Hz."
        )
    else:
        _biopac_paw, _biopac_emg = _load_annemijn_biopac(str(_biopac_file))
        _n = contaminated_emg.size
        _located = _n <= _biopac_emg.size and np.allclose(
            _biopac_emg[-_n:], contaminated_emg, rtol=0, atol=1e-6
        )
        if not _located:
            eit_paw_paw_unavailable_reason = (
                "This recording isn't byte-for-byte the tail of the Annemijn "
                "Biopac sEMG channel, so it can't be located on the shared "
                "clock. Upload the exact `_last_30secs_emg_slice.txt` or "
                "`_last_2min_emg.txt` file."
            )
        else:
            _absolute_start = (_biopac_emg.size - _n) / _annemijn_sample_frequency
            _window_start, _window_end = displayed_time.value
            _abs_start = _absolute_start + _window_start
            _abs_end = _absolute_start + _window_end

            _paw_lo = int(round(_abs_start * _annemijn_sample_frequency))
            _paw_hi = int(round(_abs_end * _annemijn_sample_frequency))
            eit_paw_paw_time = (
                np.arange(_paw_lo, _paw_hi) / _annemijn_sample_frequency
                - _absolute_start
            )
            eit_paw_paw_values = _biopac_paw[_paw_lo:_paw_hi]

            from m3resp.synchronization import estimate_offset_from_interference

            _eit_time, _eit_values = _load_annemijn_eit(str(_eit_file))
            _eit_duration = float(_eit_time[-1])
            _sync = estimate_offset_from_interference(
                _biopac_emg, _annemijn_sample_frequency, _eit_duration
            )
            if not _sync.detected or _sync.offset_seconds is None:
                eit_paw_eit_unavailable_reason = (
                    "Could not detect the EIT-on/EIT-off interference edge "
                    "needed to synchronize the EIT recording to this Biopac "
                    "clock."
                )
            else:
                _eit_active_start = _sync.offset_seconds
                _eit_active_end = _sync.offset_seconds + _eit_duration
                _overlap_start = max(_eit_active_start, _abs_start)
                _overlap_end = min(_eit_active_end, _abs_end)
                if _overlap_end <= _overlap_start:
                    eit_paw_eit_unavailable_reason = (
                        f"EIT was only active {_eit_active_start:.1f}–"
                        f"{_eit_active_end:.1f}s into this recording; the "
                        f"displayed window ({_abs_start:.1f}–{_abs_end:.1f}s) "
                        "doesn't overlap it (per the dataset notes, EIT was "
                        "off for the final ~2 minutes of this Biopac "
                        "recording)."
                    )
                else:
                    _eit_mask = (_eit_time >= _abs_start - _sync.offset_seconds) & (
                        _eit_time <= _abs_end - _sync.offset_seconds
                    )
                    eit_paw_eit_time = (
                        _eit_time[_eit_mask] + _sync.offset_seconds - _absolute_start
                    )
                    eit_paw_eit_values = _eit_values[_eit_mask]

    return (
        eit_paw_eit_time,
        eit_paw_eit_unavailable_reason,
        eit_paw_eit_values,
        eit_paw_paw_time,
        eit_paw_paw_unavailable_reason,
        eit_paw_paw_values,
    )


@app.cell
def _(
    contaminated_emg,
    denormalized_qrs_segments,
    displayed_time,
    ees_result,
    eit_paw_eit_time,
    eit_paw_eit_unavailable_reason,
    eit_paw_eit_values,
    eit_paw_paw_time,
    eit_paw_paw_unavailable_reason,
    eit_paw_paw_values,
    np,
    original_qrs_segments,
    plt,
    promoted_ecg,
    qrs_segment_ends,
    qrs_segment_starts,
    rectified_ecg,
    qrs_window_seconds,
    sample_frequency,
    smoothed_detection,
    template_offsets,
    time,
):
    window_start, window_end = displayed_time.value
    visible = (time >= window_start) & (time <= window_end)
    template_time = template_offsets / sample_frequency
    selected_template_window_seconds = float(qrs_window_seconds.value)
    selected_template_half_window_seconds = selected_template_window_seconds / 2
    blue = "#3B7CCC"
    orange = "#F28E2B"
    gold = "#E5A823"
    charcoal = "#3F3F3F"
    red = "#E64B35"

    # Guards against NaN/Inf axis limits (matplotlib raises "Axis limits
    # cannot be NaN or Inf" otherwise) and against empty arrays. A filtered
    # signal can legitimately contain non-finite samples at extreme
    # low-cutoff/high-order combinations, and EIT/Paw windows can be short or
    # empty near a recording boundary.
    def _safe_abs_max(*arrays, fallback=1.0):
        _values = []
        for _array in arrays:
            _finite = np.asarray(_array)
            _finite = _finite[np.isfinite(_finite)]
            if _finite.size:
                _values.append(float(np.max(np.abs(_finite))))
        return max(_values) if _values else fallback

    def _safe_min_max(*arrays, fallback=(0.0, 1.0)):
        _mins, _maxs = [], []
        for _array in arrays:
            _finite = np.asarray(_array)
            _finite = _finite[np.isfinite(_finite)]
            if _finite.size:
                _mins.append(float(np.min(_finite)))
                _maxs.append(float(np.max(_finite)))
        return (min(_mins), max(_maxs)) if _mins else fallback

    # Derive limits from the complete recording, not the selected slider range.
    # This keeps the visual scale stable while comparing different time windows.
    # Deliberately excludes ees_result.cleaned: the output bandpass (either
    # stage) only changes cleaned's amplitude, and folding it in here would
    # rescale (and visually distort) every other panel, including the
    # steps 8-9 template-construction subplots whose underlying data the
    # output bandpass never touches (detection and the template are always
    # built from the raw signal).
    paper_amplitude_limit = float(
        1.05 * _safe_abs_max(contaminated_emg, promoted_ecg, ees_result.estimated_ecg)
    )
    paper_cleaned_amplitude_limit = float(
        max(
            paper_amplitude_limit,
            1.05 * _safe_abs_max(ees_result.cleaned, fallback=paper_amplitude_limit),
        )
    )
    paper_detection_limit = float(
        1.05
        * _safe_abs_max(smoothed_detection, ees_result.dynamic_threshold, fallback=1.0)
    )
    paper_normalized_min, paper_normalized_max = _safe_min_max(
        ees_result.normalized_segments, ees_result.normalized_template
    )
    paper_normalized_padding = max(
        0.05 * (paper_normalized_max - paper_normalized_min),
        np.finfo(float).eps,
    )

    paper_figure = plt.figure(figsize=(18, 29), constrained_layout=True)
    outer = paper_figure.add_gridspec(
        9,
        12,
        height_ratios=(1.0, 1.0, 1.55, 1.1, 0.95, 0.9, 0.9, 0.85, 0.85),
    )

    ax_input = paper_figure.add_subplot(outer[0, 0:6])
    ax_filter = paper_figure.add_subplot(outer[0, 6:12], sharex=ax_input)
    ax_rectified = paper_figure.add_subplot(outer[1, 0:6], sharex=ax_input)
    ax_threshold = paper_figure.add_subplot(outer[1, 6:12], sharex=ax_input)

    crossing_grid = outer[2, 0:6].subgridspec(
        2, 2, height_ratios=(2.0, 1.1), width_ratios=(1.15, 0.85)
    )
    ax_crossings = paper_figure.add_subplot(crossing_grid[0, :], sharex=ax_input)
    ax_correction = paper_figure.add_subplot(crossing_grid[1, 0])
    ax_correction_text = paper_figure.add_subplot(crossing_grid[1, 1])
    ax_qrs = paper_figure.add_subplot(outer[2, 6:12], sharex=ax_input)

    # Keep the processing sequence in reading order: select the window (8),
    # create the template (9), insert it (10), then show the subtraction (11).
    ax_window = paper_figure.add_subplot(outer[3, 0:12])
    template_grid = outer[4, 0:12].subgridspec(1, 4, wspace=0.16)
    ax_original = paper_figure.add_subplot(template_grid[0, 0])
    ax_normalized = paper_figure.add_subplot(template_grid[0, 1])
    ax_average = paper_figure.add_subplot(template_grid[0, 2])
    ax_denormalized = paper_figure.add_subplot(template_grid[0, 3])

    ax_inserted = paper_figure.add_subplot(outer[5, 0:12], sharex=ax_input)
    ax_cleaned = paper_figure.add_subplot(outer[6, 0:12], sharex=ax_input)
    ax_paw = paper_figure.add_subplot(outer[7, 0:12], sharex=ax_input)
    ax_eit = paper_figure.add_subplot(outer[8, 0:12], sharex=ax_input)

    def _paper_axis(axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
        axis.margins(x=0)

    def _time_title(axis, title):
        axis.set_title(title, loc="left", fontsize=10, pad=7)
        axis.set_ylabel("Amplitude", fontsize=8)
        axis.set_xlim(window_start, window_end)
        _paper_axis(axis)

    # Input and steps 1–4.
    ax_input.plot(time[visible], contaminated_emg[visible], color=blue, lw=0.6)
    _time_title(ax_input, "Input: EMGdi-DS")

    ax_filter.plot(time[visible], promoted_ecg[visible], color=blue, lw=0.6)
    _time_title(
        ax_filter,
        "1. Bandpass filter (4–50 Hz) to promote relative ECG amplitude",
    )

    ax_rectified.plot(time[visible], rectified_ecg[visible], color=blue, lw=0.65)
    _time_title(ax_rectified, "2. Rectification")

    ax_threshold.plot(time[visible], smoothed_detection[visible], color=blue, lw=0.75)
    ax_threshold.plot(
        time[visible],
        ees_result.dynamic_threshold[visible],
        color="black",
        lw=1.2,
    )
    _time_title(
        ax_threshold,
        "3. Moving average filter to create EMGdi-DS-A\n4. Apply threshold level",
    )

    # Step 5: threshold crossings.
    ax_crossings.plot(time[visible], smoothed_detection[visible], color=blue, lw=0.7)
    ax_crossings.plot(
        time[visible],
        ees_result.dynamic_threshold[visible],
        color="black",
        lw=1.0,
    )
    _start_visible = qrs_segment_starts[
        (time[qrs_segment_starts] >= window_start)
        & (time[qrs_segment_starts] <= window_end)
    ]
    _end_visible = qrs_segment_ends[
        (time[qrs_segment_ends] >= window_start)
        & (time[qrs_segment_ends] <= window_end)
    ]
    ax_crossings.scatter(
        time[_start_visible],
        smoothed_detection[_start_visible],
        marker="x",
        color=red,
        s=28,
        label="start QRS segment",
        zorder=4,
    )
    ax_crossings.scatter(
        time[_end_visible],
        smoothed_detection[_end_visible],
        marker="x",
        color=orange,
        s=28,
        label="end QRS segment",
        zorder=4,
    )
    _time_title(ax_crossings, "5. Detect crossings with threshold level")
    ax_crossings.legend(loc="upper left", fontsize=7, ncols=2, frameon=False)

    # Step 6 inset: center on the first rejected candidate, or a middle beat.
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
    ax_correction.plot(
        time[_correction_slice],
        smoothed_detection[_correction_slice],
        color=blue,
        lw=0.75,
    )
    ax_correction.plot(
        time[_correction_slice],
        ees_result.dynamic_threshold[_correction_slice],
        color="black",
        lw=0.9,
    )
    for _indices, _marker, _color, _label in (
        (ees_result.rejected_peak_indices, "x", red, "deleted"),
        (ees_result.restored_peak_indices, "D", blue, "restored"),
    ):
        _nearby = _indices[
            (_indices >= _correction_start) & (_indices < _correction_end)
        ]
        if len(_nearby):
            ax_correction.scatter(
                time[_nearby],
                smoothed_detection[_nearby],
                marker=_marker,
                color=_color,
                s=34,
                label=_label,
                zorder=5,
            )
    ax_correction.set_title(
        "6. Periodicity correction (detail)", loc="left", fontsize=9
    )
    ax_correction.set_xlabel("Time (s)", fontsize=8)
    ax_correction.set_ylabel("Normalized", fontsize=8)
    ax_correction.legend(loc="upper right", fontsize=7, frameon=False)
    _paper_axis(ax_correction)
    ax_correction_text.axis("off")
    ax_correction_text.text(
        0.03,
        0.68,
        "6. Remove (and restore) wrongfully\n"
        "detected (or deleted) QRS segments\n\n"
        f"Rejected: {len(ees_result.rejected_peak_indices)}\n"
        f"Restored: {len(ees_result.restored_peak_indices)}",
        fontsize=10,
        va="top",
    )
    # Step 7: Q, R, and S peaks on the unfiltered input.
    ax_qrs.plot(time[visible], contaminated_emg[visible], color=blue, lw=0.6)
    _qrs = ees_result.qrs_indices
    for _column, _color, _label, _size in (
        (0, charcoal, "Q", 28),
        (1, gold, "R", 32),
        (2, red, "S", 28),
    ):
        _indices = _qrs[:, _column]
        _inside = (time[_indices] >= window_start) & (time[_indices] <= window_end)
        ax_qrs.scatter(
            time[_indices[_inside]],
            contaminated_emg[_indices[_inside]],
            color=_color,
            s=_size,
            label=_label,
            zorder=4,
        )
    _time_title(ax_qrs, "7. Detect QRS peaks in EMGdi-DS")
    ax_qrs.legend(loc="upper right", fontsize=8, ncols=3, frameon=False)

    # Step 8: show three ECG complexes and bracket the 0.3 s template window.
    _middle_beat = len(_qrs) // 2
    _shown_beats = _qrs[max(0, _middle_beat - 1) : _middle_beat + 2]
    _window_left = max(0, int(_shown_beats[0, 1]) - int(0.35 * sample_frequency))
    _window_right = min(
        len(time), int(_shown_beats[-1, 1]) + int(0.35 * sample_frequency)
    )
    ax_window.plot(
        time[_window_left:_window_right],
        contaminated_emg[_window_left:_window_right],
        color=blue,
        lw=0.65,
    )
    for _column, _color, _label in (
        (0, charcoal, "Q"),
        (1, gold, "R"),
        (2, red, "S"),
    ):
        ax_window.scatter(
            time[_shown_beats[:, _column]],
            contaminated_emg[_shown_beats[:, _column]],
            color=_color,
            s=30,
            label=_label,
            zorder=4,
        )
    _selected_r_time = time[_shown_beats[-1, 1]]
    _bracket_y = 0.82 * paper_amplitude_limit
    ax_window.annotate(
        "",
        xy=(
            _selected_r_time - selected_template_half_window_seconds,
            _bracket_y,
        ),
        xytext=(
            _selected_r_time + selected_template_half_window_seconds,
            _bracket_y,
        ),
        arrowprops={"arrowstyle": "|-|", "lw": 1.2, "color": charcoal},
    )
    ax_window.text(
        _selected_r_time,
        _bracket_y,
        " 0.3 s ",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax_window.set_title(
        "8. Select window length around R wave\n"
        f"({selected_template_window_seconds:.2f} s selected) to capture the full artifact",
        loc="left",
        fontsize=9,
    )
    ax_window.set_xlabel("Time (s)", fontsize=8)
    ax_window.set_ylabel("Amplitude", fontsize=8)
    ax_window.legend(loc="upper left", fontsize=7, ncols=3, frameon=False)
    _paper_axis(ax_window)

    # Step 9: all four template-construction panels.
    _segment_colors = plt.cm.tab20(np.linspace(0, 1, len(original_qrs_segments)))
    for _segment, _color in zip(original_qrs_segments, _segment_colors):
        ax_original.plot(template_time, _segment, color=_color, lw=0.55, alpha=0.85)
    for _segment, _color in zip(ees_result.normalized_segments, _segment_colors):
        ax_normalized.plot(template_time, _segment, color=_color, lw=0.55, alpha=0.85)
    ax_average.plot(template_time, ees_result.normalized_template, color=orange, lw=1.2)
    for _segment, _color in zip(denormalized_qrs_segments, _segment_colors):
        ax_denormalized.plot(template_time, _segment, color=_color, lw=0.55, alpha=0.9)
    for _axis, _subtitle in (
        (ax_original, "a. original QRS segments"),
        (ax_normalized, "b. normalized QRS segments"),
        (ax_average, "c. average QRS segment"),
        (ax_denormalized, "d. denormalized QRS segments"),
    ):
        _axis.set_xlabel(_subtitle, fontsize=8, fontstyle="italic")
        _axis.set_xticks([])
        _axis.set_yticks([])
        _paper_axis(_axis)
    ax_original.set_title("9. Create QRS template", loc="left", fontsize=9)
    for _source_axis in (ax_original, ax_normalized, ax_average):
        _source_axis.annotate(
            "",
            xy=(1.12, 0.52),
            xytext=(1.01, 0.52),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 1.0},
            annotation_clip=False,
        )

    # Step 10: insert each denormalized template at its original R location.
    ax_inserted.plot(
        time[visible], ees_result.estimated_ecg[visible], color=orange, lw=0.75
    )
    _time_title(
        ax_inserted,
        "10. Insert denormalized QRS segments at original location",
    )
    _zoom_r = int(_qrs[len(_qrs) // 2, 1])
    _zoom_radius = int(round(0.18 * sample_frequency))
    _zoom_start = max(0, _zoom_r - _zoom_radius)
    _zoom_end = min(len(time), _zoom_r + _zoom_radius)
    ax_insert_zoom = ax_inserted.inset_axes([0.36, 0.08, 0.38, 0.5])
    ax_insert_zoom.plot(
        time[_zoom_start:_zoom_end],
        contaminated_emg[_zoom_start:_zoom_end],
        color=blue,
        lw=0.7,
        label="EMGdi-DS",
    )
    ax_insert_zoom.plot(
        time[_zoom_start:_zoom_end],
        ees_result.estimated_ecg[_zoom_start:_zoom_end],
        color=orange,
        lw=0.9,
        label="template",
    )
    ax_insert_zoom.set_xticks([])
    ax_insert_zoom.set_yticks([])
    ax_insert_zoom.set_ylim(-paper_amplitude_limit, paper_amplitude_limit)
    ax_insert_zoom.legend(loc="lower right", fontsize=7, frameon=False)
    ax_inserted.indicate_inset_zoom(ax_insert_zoom, edgecolor=charcoal, alpha=0.8)

    # Step 11: final cleaned EMGdi signal.
    ax_cleaned.plot(time[visible], ees_result.cleaned[visible], color=blue, lw=0.6)
    _time_title(
        ax_cleaned,
        "11. Subtract estimated ECG template from EMGdi-DS",
    )

    # Reference panels: EIT global impedance and airway pressure (Paw), for
    # the same real-world time window as everything above. Both come from a
    # different acquisition device than the EMG/EES processing (Annemijn
    # dataset only) and are sliced by eit_paw_context; a panel shows an
    # explanatory message instead of a curve when that context isn't
    # available (wrong source, missing files, or -- for EIT specifically --
    # no synchronized frames overlap this window).
    def _context_panel(
        axis,
        plot_time,
        plot_values,
        unavailable_reason,
        *,
        title,
        ylabel,
        color,
        reference,
    ):
        _time_title(axis, title)
        _has_data = (
            plot_time is not None
            and plot_values is not None
            and plot_time.size > 0
            and plot_values.size > 0
            and np.any(np.isfinite(plot_values))
        )
        if _has_data:
            _n = min(plot_time.size, plot_values.size)
            _plot_time, _plot_values = plot_time[:_n], plot_values[:_n]
            axis.plot(_plot_time, _plot_values, color=color, lw=0.7)
            axis.set_ylabel(ylabel, fontsize=8)
            _lo, _hi = _safe_min_max(_plot_values)
            _pad = 0.05 * max(_hi - _lo, np.finfo(float).eps)
            axis.set_ylim(_lo - _pad, _hi + _pad)
            if np.isfinite(reference):
                axis.axhline(reference, color="#777777", lw=0.35, alpha=0.5)
        else:
            axis.set_ylabel("")
            axis.set_yticks([])
            axis.text(
                0.5,
                0.5,
                unavailable_reason or "Not available for this recording.",
                ha="center",
                va="center",
                fontsize=9,
                color="#777777",
                wrap=True,
                transform=axis.transAxes,
            )

    _eit_reference = (
        float(np.nanmean(eit_paw_eit_values))
        if eit_paw_eit_values is not None
        and eit_paw_eit_values.size
        and np.any(np.isfinite(eit_paw_eit_values))
        else 0.0
    )
    _context_panel(
        ax_eit,
        eit_paw_eit_time,
        eit_paw_eit_values,
        eit_paw_eit_unavailable_reason or eit_paw_paw_unavailable_reason,
        title="EIT global impedance (synchronized to Biopac clock)",
        ylabel="Impedance (a.u.)",
        color="#3F7F5F",
        reference=_eit_reference,
    )
    _context_panel(
        ax_paw,
        eit_paw_paw_time,
        eit_paw_paw_values,
        eit_paw_paw_unavailable_reason,
        title="Airway pressure (Paw)",
        ylabel="Paw (cmH2O)",
        color="#8B5FA8",
        reference=0.0,
    )

    ax_inserted.set_xlabel("Time (s)", fontsize=8)
    ax_eit.set_xlabel("Time (s)", fontsize=8)

    # Keep comparable panels on one recording-wide amplitude scale.  The
    # rectified and detection signals have their own fixed non-negative scales.
    # ax_cleaned gets its own limit: the output bandpass (either stage) only
    # changes ees_result.cleaned, and should not rescale panels whose data it
    # cannot affect (see paper_cleaned_amplitude_limit above). EIT and Paw are
    # on unrelated physical scales and are limited inside _context_panel
    # instead.
    for _axis in (
        ax_input,
        ax_filter,
        ax_qrs,
        ax_window,
        ax_original,
        ax_denormalized,
        ax_inserted,
    ):
        _axis.set_ylim(-paper_amplitude_limit, paper_amplitude_limit)
    ax_cleaned.set_ylim(-paper_cleaned_amplitude_limit, paper_cleaned_amplitude_limit)
    ax_rectified.set_ylim(0, paper_amplitude_limit)
    for _axis in (ax_threshold, ax_crossings, ax_correction):
        _axis.set_ylim(0, paper_detection_limit)
    for _axis in (ax_normalized, ax_average):
        _axis.set_ylim(
            paper_normalized_min - paper_normalized_padding,
            paper_normalized_max + paper_normalized_padding,
        )

    for _axis in (
        ax_input,
        ax_filter,
        ax_rectified,
        ax_threshold,
        ax_crossings,
        ax_qrs,
        ax_inserted,
        ax_cleaned,
    ):
        _axis.axhline(0, color="#777777", lw=0.35, alpha=0.5)

    paper_figure.suptitle(
        "Estimated ECG Subtraction method — complete processing sequence",
        fontsize=15,
    )
    paper_figure
    return


@app.cell
def _(
    clean_emg,
    contaminated_emg,
    displayed_time,
    ees_result,
    mo,
    np,
    plt,
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
        comparison_figure, comparison_axes = plt.subplots(
            2, 1, figsize=(18, 7), sharex=True, constrained_layout=True
        )
        comparison_axes[0].plot(
            time[comparison_visible],
            reference_emg[comparison_visible],
            color="#222222",
            lw=0.8,
            label="Reference EMG without ECG contamination",
        )
        comparison_axes[0].plot(
            time[comparison_visible],
            ees_result.cleaned[comparison_visible],
            color="#3B7CCC",
            lw=0.75,
            label="EES-subtracted EMG",
        )
        comparison_axes[0].set_title(
            "Validation: EES-subtracted EMG versus known ECG-free reference", loc="left"
        )
        comparison_axes[0].set_ylabel("Amplitude")
        comparison_axes[0].set_ylim(
            -validation_amplitude_limit, validation_amplitude_limit
        )
        comparison_axes[0].legend(loc="upper right", frameon=False)
        comparison_axes[1].plot(
            time[comparison_visible],
            reference_error[comparison_visible],
            color="#D1495B",
            lw=0.75,
            label="EES-subtracted EMG − reference EMG",
        )
        comparison_axes[1].axhline(0, color="#555555", lw=0.6)
        for reference_beat_time in true_beat_times:
            if comparison_start <= reference_beat_time <= comparison_end:
                comparison_axes[1].axvspan(
                    reference_beat_time - 0.15,
                    reference_beat_time + 0.15,
                    color="#F59E0B",
                    alpha=0.12,
                )
        comparison_axes[1].set_title(
            "Residual error after subtraction (orange bands: known ECG windows)",
            loc="left",
        )
        comparison_axes[1].set_xlabel("Time (s)")
        comparison_axes[1].set_ylabel("Error")
        comparison_axes[1].set_ylim(
            -validation_amplitude_limit, validation_amplitude_limit
        )
        comparison_axes[1].legend(loc="upper right", frameon=False)
        comparison_axes[1].text(
            0.01,
            0.96,
            f"RMSE before: {rmse_before:.4f}\nRMSE after: {rmse_after:.4f}\nRMSE reduction: {error_reduction_percent:.1f}%\nMaximum |error|: {np.max(np.abs(reference_error)):.4f}",
            transform=comparison_axes[1].transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        for comparison_axis in comparison_axes:
            comparison_axis.spines["top"].set_visible(False)
            comparison_axis.spines["right"].set_visible(False)
            comparison_axis.margins(x=0)
            comparison_axis.set_xlim(comparison_start, comparison_end)
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
    .venv/bin/marimo edit tools/visualization_tools/estimated_ecg_subtraction.py
    ```

    Select **Upload a recording** and provide its sampling frequency to process
    a `.csv`, `.txt`, or `.npy` one-dimensional signal. The configurable
    synthetic source provides known clean EMG and ECG components for testing.
    For a clinical recording, retain ECG frequency content before EES and
    review the QRS detections and estimated ECG before accepting the cleaned
    signal.

    The optional output bandpass (below the main EES parameters) is a second,
    independent filter from the detection band. QRS detection and the
    template always run on the raw signal for both stages, so panels 1-9
    (including `ees_result.estimated_ecg` and the steps 8/9
    template-construction subpanels) never change when this filter is
    toggled. Only `ees_result.cleaned` (panel 11 and the validation plot)
    differs by stage: "before_subtraction" filters the raw signal and then
    subtracts the unfiltered estimated ECG; "after_subtraction" subtracts
    first and filters the result. The two give different `cleaned` signals
    because filtering and subtracting the QRS template don't commute.
    """)
    return


if __name__ == "__main__":
    app.run()
