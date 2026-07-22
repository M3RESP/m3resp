import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full", app_title="M3Resp Annemijn multimodal viewer")


@app.cell
def _imports():
    from functools import lru_cache
    from importlib.util import find_spec
    from pathlib import Path
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.signal import welch

    def _ensure_importable(module_name: str, local_path: Path) -> None:
        # Prefer a properly installed package (its build step generates files
        # like `__version__.py` that a raw sibling git checkout may lack) and
        # only fall back to the local checkout -- inserted at the *end* of
        # sys.path so it can't shadow a working install -- when the module
        # isn't installed at all.
        if find_spec(module_name) is not None:
            return
        if local_path.exists() and str(local_path) not in sys.path:
            sys.path.append(str(local_path))

    ROOT = Path(__file__).resolve().parents[2]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    _ensure_importable("eitprocessing", ROOT.parent / "eitprocessing")
    _ensure_importable("resurfemg", ROOT.parent / "ReSurfEMG")

    from m3resp.adapters import EITProcessingAdapter
    from m3resp.processing.ecg import estimated_ecg_subtraction
    from m3resp.processing.filters import bandpass_filter, harmonic_notch_filter
    from m3resp.processing.peaks import detect_emg_breath_peaks
    from m3resp.processing.windows import moving_average, rolling_arv, rolling_rms
    from m3resp.synchronization import (
        estimate_offset_from_interference,
        refine_offset_by_crosscorrelation,
    )
    from resurfemg.preprocessing.ecg_removal import (
        detect_ecg_peaks,
        gating,
        wavelet_denoising,
    )
    from resurfemg.preprocessing.filtering import emg_bandpass_butter

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
        bandpass_filter,
        detect_ecg_peaks,
        detect_emg_breath_peaks,
        emg_bandpass_butter,
        estimate_offset_from_interference,
        estimated_ecg_subtraction,
        gating,
        go,
        harmonic_notch_filter,
        lru_cache,
        make_subplots,
        mo,
        moving_average,
        np,
        pd,
        plt,
        refine_offset_by_crosscorrelation,
        rolling_arv,
        rolling_rms,
        wavelet_denoising,
        welch,
    )


@app.cell
def _loaders(BIOPAC_SAMPLE_FREQUENCY, EITProcessingAdapter, lru_cache, np, pd):
    @lru_cache(maxsize=2)
    def load_biopac(path):
        # Biopac channels: 1=Paw (mouthpiece), 2=diaphragm sEMG, 3=empty/ignore.
        data = pd.read_csv(
            path,
            sep="\t",
            skiprows=11,
            names=["paw", "emg_di", "aux_ignore"],
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
    # First guess: EIT stops ~120 s before Biopac ends (the EIT-off tail).
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
def _controls(biopac_duration, default_biopac_offset, mo):
    sync_mode = mo.ui.dropdown(
        options=[
            "Manual offset",
            "Interference anchor",
            "Interference + cross-corr",
        ],
        value="Interference + cross-corr",
        label="Sync method",
    )
    time_range = mo.ui.range_slider(
        start=0.0,
        stop=float(biopac_duration),
        step=1.0,
        value=[
            float(default_biopac_offset),
            float(min(default_biopac_offset + 180.0, biopac_duration)),
        ],
        label="Biopac time window (s)",
        full_width=True,
    )
    # Paw is the fixed reference (its window is exactly the time_range
    # selection); the EIT and EMG traces slide against it. These nudges are only
    # meaningful (and only shown) under "Manual offset" sync, to visually align
    # each modality's breaths with the Paw breaths.
    eit_manual_offset = mo.ui.slider(
        start=-100.0,
        stop=100.0,
        step=0.5,
        value=0.0,
        label="EIT manual offset (s)",
        full_width=True,
    )
    emg_manual_offset = mo.ui.slider(
        start=-100.0,
        stop=100.0,
        step=0.5,
        value=0.0,
        label="EMG manual offset (s)",
        full_width=True,
    )
    offset_fine = mo.ui.number(
        start=-30.0, stop=30.0, step=0.01, value=0.0, label="Fine offset (s)"
    )
    stretch = mo.ui.number(
        start=0.990,
        stop=1.010,
        step=0.0001,
        value=1.0,
        label="Drift stretch (Biopac s / EIT s)",
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
    notch_base_frequency = mo.ui.number(
        start=1.0, stop=200.0, step=1.0, value=50.0, label="Notch base freq (Hz)"
    )
    notch_quality_factor = mo.ui.number(
        start=1.0, stop=100.0, step=1.0, value=30.0, label="Notch quality factor"
    )
    # A harmonic sitting at or just past `low_pass_hz` (e.g. the EIT frame-rate
    # comb's 10th harmonic at 500 Hz) is only partially attenuated by the
    # low-pass filter's finite roll-off. If the notch's own reach stops at
    # that same edge (the old default), that harmonic is never fully
    # notched either. Two independent knobs to compare live:
    #  - `canonical_low_pass_hz`: where the band-pass cuts off.
    #  - `canonical_notch_max_hz`: how far the harmonic notch reaches.
    # Set them equal to reproduce the old buggy behaviour; set the notch
    # reach well past the low-pass cutoff (default here) for the fix.
    canonical_low_pass_hz = mo.ui.number(
        start=100.0, stop=1000.0, step=10.0, value=500.0, label="Low-pass cutoff (Hz)"
    )
    canonical_notch_max_hz = mo.ui.number(
        start=50.0,
        stop=1000.0,
        step=10.0,
        value=1000.0,
        label="Notch reach / max freq (Hz)",
    )
    breath_min_width_s = mo.ui.number(
        start=0.1, stop=5.0, step=0.1, value=0.8, label="Breath min width (s)"
    )
    breath_prominence_factor = mo.ui.number(
        start=0.01,
        stop=2.0,
        step=0.01,
        value=0.45,
        label="Breath prominence factor",
    )
    # Canonical step-by-step pipeline following Jonkman et al. 2024 best practices.
    canonical_ecg_method = mo.ui.dropdown(
        options=[
            "Wavelet denoising",
            "Gating",
            "High-pass 200 Hz",
            "EES (Estimated ECG Subtraction)",
        ],
        value="Wavelet denoising",
        label="Canonical ECG removal",
    )
    canonical_hpf_hz = mo.ui.number(
        start=0.5, stop=40.0, step=0.5, value=20.0, label="Low-freq HPF (Hz)"
    )
    # ECG (QRS) removal knobs. A missed R-peak gets no wavelet suppression /
    # no gate at all for that beat -- on this dataset the detector's default
    # `peak_fraction=0.4` misses roughly 1 in 6-7 beats (checked by sweeping
    # peak_fraction over a known EIT-off window: 0.4 -> 46 peaks / 50 s,
    # 0.2 -> 49 peaks with no spurious doubles, 0.1 -> 59 peaks already
    # over-detecting). Lowered the default to 0.2 accordingly.
    canonical_ecg_peak_fraction = mo.ui.number(
        start=0.02, stop=1.0, step=0.02, value=0.2, label="ECG peak sensitivity"
    )
    canonical_wavelet_threshold = mo.ui.number(
        start=1.0, stop=10.0, step=0.25, value=4.5, label="Wavelet threshold (σ)"
    )
    canonical_wavelet_level = mo.ui.number(
        start=1, stop=8, step=1, value=4, label="Wavelet level (n)"
    )
    canonical_gate_width_ms = mo.ui.number(
        start=20.0, stop=300.0, step=5.0, value=100.0, label="Gate width (ms)"
    )
    canonical_envelope_type = mo.ui.dropdown(
        options=["RMS", "ARV", "MAV (median)"],
        value="RMS",
        label="Envelope",
    )
    canonical_envelope_ms = mo.ui.number(
        start=50.0, stop=2000.0, step=10.0, value=1000.0, label="Envelope window (ms)"
    )
    canonical_baseline_correction = mo.ui.checkbox(
        value=True, label="Baseline offset correction"
    )

    # EES (Estimated ECG Subtraction, Jonkman et al. 2021) parameters. This is
    # a distinct QRS-template-and-subtract method: it detects an ECG-promoting
    # band, builds an averaged QRS template per beat, and subtracts it -- quite
    # different from the wavelet-shrinkage / gating / high-pass approaches
    # above. It's computed unconditionally in the canonical pipeline (not only
    # when selected as the active method below), so the detailed EES
    # diagnostic section at the end of the notebook always has a result to
    # show, sharing this single set of controls instead of a second,
    # easily-desynced copy.
    ees_detection_band_low = mo.ui.slider(
        1.0, 45.0, 1.0, 4.0, label="EES low cutoff (Hz)", show_value=True
    )
    ees_detection_band_high = mo.ui.slider(
        50.0, 200.0, 1.0, 50.0, label="EES high cutoff (Hz)", show_value=True
    )
    ees_filter_order = mo.ui.slider(
        1, 8, 1, 4, label="EES filter order", show_value=True
    )
    ees_detection_smoothing_s = mo.ui.slider(
        0.001,
        0.100,
        0.001,
        0.06,
        label="EES rectified-signal smoothing (s)",
        show_value=True,
    )
    ees_threshold_interval_s = mo.ui.slider(
        0.05, 2.0, 0.05, 1.5, label="EES threshold interval (s)", show_value=True
    )
    ees_threshold_smoothing_s = mo.ui.slider(
        0.001,
        0.100,
        0.001,
        0.0125,
        label="EES threshold smoothing (s)",
        show_value=True,
    )
    ees_qrs_window_s = mo.ui.slider(
        0.05,
        0.60,
        0.01,
        0.30,
        label="EES template window around R (s)",
        show_value=True,
    )
    ees_inter_qrs_tolerance = mo.ui.slider(
        0.0, 0.95, 0.01, 0.66, label="EES inter-QRS tolerance", show_value=True
    )
    ees_min_template_beats = mo.ui.slider(
        1, 12, 1, 3, label="EES minimum template beats", show_value=True
    )
    ees_use_min_qrs_interval = mo.ui.checkbox(
        value=True, label="Enforce minimum QRS interval"
    )
    ees_min_qrs_interval_s = mo.ui.slider(
        0.05,
        1.0,
        0.05,
        0.25,
        label="EES minimum QRS interval (s)",
        show_value=True,
    )
    ees_use_max_qrs_interval = mo.ui.checkbox(
        value=True, label="Enforce maximum QRS interval"
    )
    ees_max_qrs_interval_s = mo.ui.slider(
        1.1,
        4.0,
        0.1,
        2.0,
        label="EES maximum QRS interval (s)",
        show_value=True,
    )
    ees_use_output_bandpass = mo.ui.checkbox(
        value=False, label="Apply EES output bandpass"
    )
    ees_output_bandpass_low = mo.ui.slider(
        1.0, 45.0, 1.0, 4.0, label="EES output bandpass low (Hz)", show_value=True
    )
    ees_output_bandpass_high = mo.ui.slider(
        50.0,
        200.0,
        1.0,
        50.0,
        label="EES output bandpass high (Hz)",
        show_value=True,
    )
    ees_output_bandpass_order = mo.ui.slider(
        1, 8, 1, 4, label="EES output bandpass order", show_value=True
    )
    ees_output_bandpass_stage = mo.ui.radio(
        options={
            "Before subtraction": "before_subtraction",
            "After subtraction": "after_subtraction",
        },
        value="After subtraction",
        label="EES output bandpass stage",
    )
    return (
        breath_min_width_s,
        breath_prominence_factor,
        canonical_baseline_correction,
        canonical_ecg_method,
        canonical_ecg_peak_fraction,
        canonical_envelope_ms,
        canonical_envelope_type,
        canonical_gate_width_ms,
        canonical_hpf_hz,
        canonical_low_pass_hz,
        canonical_notch_max_hz,
        canonical_wavelet_level,
        canonical_wavelet_threshold,
        downsample,
        ees_detection_band_high,
        ees_detection_band_low,
        ees_detection_smoothing_s,
        ees_filter_order,
        ees_inter_qrs_tolerance,
        ees_max_qrs_interval_s,
        ees_min_qrs_interval_s,
        ees_min_template_beats,
        ees_output_bandpass_high,
        ees_output_bandpass_low,
        ees_output_bandpass_order,
        ees_output_bandpass_stage,
        ees_qrs_window_s,
        ees_threshold_interval_s,
        ees_threshold_smoothing_s,
        ees_use_max_qrs_interval,
        ees_use_min_qrs_interval,
        ees_use_output_bandpass,
        eit_manual_offset,
        emg_manual_offset,
        normalize,
        notch_base_frequency,
        notch_quality_factor,
        offset_fine,
        stretch,
        sync_mode,
        time_range,
    )


@app.cell
def _sync_controls_layout(
    downsample,
    eit_manual_offset,
    emg_manual_offset,
    mo,
    normalize,
    offset_fine,
    stretch,
    sync_mode,
    time_range,
):
    # A UI element's .value cannot be read in the cell that creates it, so the
    # sync-mode-dependent visibility of the manual offset sliders lives here,
    # in a separate cell from `_controls`.
    _manual_offset_controls = (
        [mo.hstack([eit_manual_offset, emg_manual_offset], justify="start", gap=2)]
        if sync_mode.value == "Manual offset"
        else []
    )
    mo.vstack(
        [
            mo.md("# Annemijn EIT + EMG + Paw"),
            sync_mode,
            time_range,
            *_manual_offset_controls,
            mo.hstack(
                [offset_fine, stretch, downsample, normalize],
                justify="start",
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _interference(
    BIOPAC_SAMPLE_FREQUENCY, biopac, eit_duration, estimate_offset_from_interference
):
    # Anchor 1: the EIT device injects interference into the sEMG while it is
    # recording. The last ~2 min is EIT-off, so the sEMG high-frequency power
    # steps down exactly when the 03_ EIT recording stops. That edge fixes the
    # EIT end in Biopac time -> offset = edge - eit_duration. See
    # m3resp.synchronization.offset_estimation for the implementation.
    _emg = biopac["emg_di"].to_numpy(dtype=float)
    _time = biopac["time_seconds"].to_numpy(dtype=float)
    _result = estimate_offset_from_interference(
        _emg, BIOPAC_SAMPLE_FREQUENCY, eit_duration, emg_time=_time
    )

    interference_edge = _result.edge_time_seconds
    interference_offset = _result.offset_seconds
    interference_power = {
        "time": _result.power_time,
        "values": _result.power_values,
        "threshold": _result.threshold,
    }
    return interference_edge, interference_offset, interference_power


@app.cell
def _base_offset(default_biopac_offset, interference_offset, offset_fine, sync_mode):
    if sync_mode.value == "Manual offset":
        _base = float(default_biopac_offset)
        base_source = "manual (default heuristic)"
    elif interference_offset is not None:
        _base = float(interference_offset)
        base_source = "interference edge"
    else:
        _base = float(default_biopac_offset)
        base_source = "manual (interference edge not found)"
    base_offset = _base + float(offset_fine.value)
    return base_offset, base_source


@app.cell
def _xcorr(
    base_offset,
    biopac,
    eit,
    np,
    refine_offset_by_crosscorrelation,
    stretch,
    sync_mode,
    time_range,
):
    # Anchor 2: EIT global impedance and Paw both track the breath cycle.
    # Cross-correlate them over the visible window to refine within a breath.
    # See m3resp.synchronization.offset_estimation for the implementation.
    xcorr_lag = 0.0
    xcorr_curve = None
    if sync_mode.value == "Interference + cross-corr":
        # time_range is now Biopac time; the target signal (EIT) is on its own
        # t~0 clock, so translate the window via the current base offset.
        _t0, _t1 = [float(v) for v in time_range.value]
        _s = float(stretch.value)
        _et = np.asarray(eit["time_seconds"], dtype=float)
        _ev = np.asarray(eit["global_impedance"], dtype=float)
        _eit_t0 = (_t0 - base_offset) / _s
        _eit_t1 = (_t1 - base_offset) / _s

        _bm = (biopac["time_seconds"] >= _t0 - 6.0) & (
            biopac["time_seconds"] <= _t1 + 6.0
        )
        _pt = biopac.loc[_bm, "time_seconds"].to_numpy(dtype=float)
        _pv = biopac.loc[_bm, "paw"].to_numpy(dtype=float)

        if _pt.size > 20:
            _result = refine_offset_by_crosscorrelation(
                _et,
                _ev,
                _pt,
                _pv,
                base_offset,
                stretch=_s,
                window=(_eit_t0, _eit_t1),
            )
            xcorr_lag = _result.lag_seconds
            if _result.lags_seconds.size:
                xcorr_curve = {
                    "lags": _result.lags_seconds,
                    "corr": _result.correlation,
                }
    return xcorr_curve, xcorr_lag


@app.cell
def _effective(base_offset, xcorr_lag):
    effective_offset = float(base_offset) + float(xcorr_lag)
    return (effective_offset,)


@app.cell
def _sync_status(base_offset, base_source, effective_offset, mo, sync_mode, xcorr_lag):
    mo.md(
        f"""
        **Sync method:** {sync_mode.value}
        **Base offset:** {base_offset:.3f} s ({base_source})
        **Cross-corr refine:** {xcorr_lag:+.3f} s
        **Effective offset (EIT t=0 → Biopac):** {effective_offset:.3f} s
        """
    )
    return


@app.cell
def _windowed_data(
    biopac,
    downsample,
    effective_offset,
    eit,
    eit_centered,
    eit_manual_offset,
    emg_manual_offset,
    normalize,
    np,
    stretch,
    sync_mode,
    time_range,
):
    # Paw is the fixed reference: its window is exactly the [t0, t1] selection
    # and the shared x-axis is locked to it. EIT and EMG are cropped to the same
    # window *width* but slid by their manual offsets, so the sliders scroll each
    # signal through the fixed Paw window until its breaths line up. A sample is
    # drawn at x = (its own time) + shift, so to keep the [t0, t1] window full we
    # crop the source window [t0 - shift, t1 - shift] and re-reference it back
    # onto the Paw axis (positive offset => trace moves right / later).
    t0, t1 = [float(value) for value in time_range.value]
    s = float(stretch.value)
    step = int(downsample.value)

    _manual = sync_mode.value == "Manual offset"
    _eit_shift = float(eit_manual_offset.value) if _manual else 0.0
    _emg_shift = float(emg_manual_offset.value) if _manual else 0.0

    # -- Paw: the fixed reference, window exactly [t0, t1], no offset. ---------
    paw_mask = (biopac["time_seconds"] >= t0) & (biopac["time_seconds"] <= t1)
    paw_window = biopac.loc[paw_mask]
    paw_plot = {
        "time": paw_window["time_seconds"].to_numpy()[::step],
        "values": paw_window["paw"].to_numpy(dtype=float)[::step],
    }

    # -- EMG: same-width window, slid by its manual offset. -------------------
    emg_mask = (biopac["time_seconds"] >= t0 - _emg_shift) & (
        biopac["time_seconds"] <= t1 - _emg_shift
    )
    emg_window = biopac.loc[emg_mask]
    emg_time_full = emg_window["time_seconds"].to_numpy() + _emg_shift
    # Raw EMG only -- no EIT notch or any other filtering here. Filtering
    # (EIT notch, ECG removal, etc.) lives entirely in the canonical pipeline
    # below; this main view shows the untouched signal.
    emg_raw_full = emg_window["emg_di"].to_numpy(dtype=float)
    emg_raw = emg_raw_full[::step]
    if normalize.value:
        emg_raw = emg_raw - np.nanmedian(emg_raw)
    emg_plot = {
        "time": emg_time_full[::step],
        "raw_values": emg_raw,
        # Full-resolution (undownsampled), for the spectrum plot and as the
        # canonical pipeline's input -- envelope windows need full resolution.
        "raw_full": emg_raw_full,
        "full_time": emg_time_full,
    }

    # -- EIT: separate clock. A frame at EIT-time tau maps to Biopac time
    # effective_offset + tau*s, then slid by its manual offset. ---------------
    eit_time = np.asarray(eit["time_seconds"], dtype=float)
    eit_values = (
        eit_centered
        if normalize.value
        else np.asarray(eit["global_impedance"], dtype=float)
    )
    eit_mapped = effective_offset + eit_time * s
    eit_mask = (eit_mapped >= t0 - _eit_shift) & (eit_mapped <= t1 - _eit_shift)
    eit_plot = {
        "time": eit_mapped[eit_mask] + _eit_shift,
        "values": eit_values[eit_mask],
    }

    window_range = (t0, t1)
    return emg_plot, eit_plot, paw_plot, window_range


@app.cell
def _stacked_plot(
    emg_plot,
    eit_plot,
    go,
    make_subplots,
    normalize,
    paw_plot,
    robust_range,
    window_range,
):
    # Paw (the fixed reference) on top, EIT and EMG below it. All three share the
    # x-axis, which is locked to the selected Paw window so the sliders crop each
    # signal to exactly the same window.
    _fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=[
            "Paw / airway pressure (reference)",
            "EIT global impedance",
            "EMGdi",
        ],
    )
    _fig.add_trace(
        go.Scattergl(
            x=paw_plot["time"],
            y=paw_plot["values"],
            mode="lines",
            name="Paw",
            line=dict(color="#D1495B", width=1.1),
            hovertemplate="t=%{x:.2f}s<br>Paw=%{y:.3f} cmH2O<extra></extra>",
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scattergl(
            x=eit_plot["time"],
            y=eit_plot["values"],
            mode="lines",
            name="EIT",
            line=dict(color="#185FA5", width=1.4),
            hovertemplate="t=%{x:.2f}s<br>EIT=%{y:.4g}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    _fig.add_trace(
        go.Scattergl(
            x=emg_plot["time"],
            y=emg_plot["raw_values"],
            mode="lines",
            name="EMGdi (raw)",
            line=dict(color="#2A9D8F", width=0.8),
            hovertemplate="t=%{x:.2f}s<br>EMGdi=%{y:.3f} mV<extra></extra>",
        ),
        row=3,
        col=1,
    )

    _fig.update_yaxes(title_text="cmH2O", row=1, col=1)
    _fig.update_yaxes(
        title_text="a.u. centered" if normalize.value else "a.u.", row=2, col=1
    )
    _fig.update_yaxes(
        title_text="mV centered" if normalize.value else "mV", row=3, col=1
    )
    _fig.update_xaxes(title_text="Biopac time (s)", row=3, col=1)
    # Lock every row's x-axis to the fixed Paw window.
    _fig.update_xaxes(range=list(window_range))

    for row, values in (
        (1, paw_plot["values"]),
        (2, eit_plot["values"]),
        (3, emg_plot["raw_values"]),
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
def _pipeline_controls_layout(
    breath_min_width_s,
    breath_prominence_factor,
    canonical_baseline_correction,
    canonical_ecg_method,
    canonical_ecg_peak_fraction,
    canonical_envelope_ms,
    canonical_envelope_type,
    canonical_gate_width_ms,
    canonical_hpf_hz,
    canonical_low_pass_hz,
    canonical_notch_max_hz,
    canonical_wavelet_level,
    canonical_wavelet_threshold,
    ees_detection_band_high,
    ees_detection_band_low,
    ees_detection_smoothing_s,
    ees_filter_order,
    ees_inter_qrs_tolerance,
    ees_max_qrs_interval_s,
    ees_min_qrs_interval_s,
    ees_min_template_beats,
    ees_output_bandpass_high,
    ees_output_bandpass_low,
    ees_output_bandpass_order,
    ees_output_bandpass_stage,
    ees_qrs_window_s,
    ees_threshold_interval_s,
    ees_threshold_smoothing_s,
    ees_use_max_qrs_interval,
    ees_use_min_qrs_interval,
    ees_use_output_bandpass,
    mo,
    notch_base_frequency,
    notch_quality_factor,
):
    # ECG-removal knobs only apply to the method that's currently selected:
    # peak detection feeds both Gating and Wavelet denoising (not the plain
    # High-pass 200 Hz fallback); wavelet threshold/level only affect Wavelet
    # denoising; gate width only affects Gating.
    _ecg_method = canonical_ecg_method.value

    # A small "i" hover button after each pipeline knob, explaining what it
    # does and its typical/recommended range -- browser-native tooltip via
    # the HTML `title` attribute, so it needs no extra JS.
    def _info(text):
        _title = text.replace('"', "'")
        return mo.md(
            f'<span title="{_title}" '
            'style="cursor:help; opacity:0.6; font-size:0.85em;'
            ' padding-left:2px;">&#9432;</span>'
        )

    def _lab(control, text):
        # `justify="start"` is required here: mo.hstack defaults to
        # "space-between", which stretches to fill the outer hstack's column
        # width and shoves the icon away from its control.
        return mo.hstack(
            [control, _info(text)], justify="start", align="center", gap=0.25
        )

    _ecg_controls = [
        _lab(
            canonical_ecg_method,
            "Method for removing ECG (cardiac) crosstalk from the sEMG. "
            "Wavelet denoising (paper default) shrinks wavelet coefficients "
            "around detected R-peaks; Gating blanks/interpolates a fixed "
            "window at each R-peak; High-pass 200 Hz is a rudimentary "
            "fallback that also removes most EMG content below 200 Hz.",
        )
    ]
    if _ecg_method in ("Gating", "Wavelet denoising"):
        _ecg_controls.append(
            _lab(
                canonical_ecg_peak_fraction,
                "R-peak detection sensitivity (fraction of peak "
                "normalized cross-correlation used as the threshold). Lower "
                "= more sensitive (finds more beats, risk of false "
                "positives); higher = misses beats, leaving unfiltered QRS "
                "spikes. Typical range: 0.1-0.3. The library default (0.4) "
                "under-detects on this dataset -- verified by sweeping "
                "peak_fraction over an EIT-off window and counting beats.",
            )
        )
    if _ecg_method == "Wavelet denoising":
        _ecg_controls += [
            _lab(
                canonical_wavelet_threshold,
                "Shrinkage threshold in noise-sigma units (σ): wavelet "
                "coefficients below this multiple of the estimated noise "
                "level are zeroed. Lower = more aggressive ECG removal but "
                "risks removing real EMG amplitude; higher leaves more "
                "residual QRS spikes. Typical range: 3.5-5 (Jonkman et al. "
                "2024 default: 4.5).",
            ),
            _lab(
                canonical_wavelet_level,
                "Depth of the à-trous wavelet decomposition. Higher levels "
                "capture lower-frequency structure (more thorough removal, "
                "slower, can over-smooth genuine EMG). Typical range: 3-5 "
                "(paper default: 4).",
            ),
        ]
    if _ecg_method == "Gating":
        _ecg_controls.append(
            _lab(
                canonical_gate_width_ms,
                "Width of the window blanked/interpolated around each "
                "detected R-peak. Should roughly match the QRS complex "
                "duration -- too narrow leaves QRS edges unremoved, too "
                "wide deletes real EMG around each heartbeat. Typical "
                "range: 80-120 ms.",
            )
        )

    _ees_controls = [
        _lab(
            ees_detection_band_low,
            "Low cutoff of the band-pass used to promote ECG relative to EMG "
            "before QRS detection (EES's own detection stage, separate from "
            "the shared notch/HPF/low-pass above). Paper default: 4 Hz.",
        ),
        _lab(
            ees_detection_band_high,
            "High cutoff of the ECG-promoting detection band. Paper default: 50 Hz.",
        ),
        _lab(ees_filter_order, "Order of the detection band-pass filter."),
        _lab(
            ees_detection_smoothing_s,
            "Moving-average window used to smooth the rectified "
            "ECG-promoted signal before threshold detection.",
        ),
        _lab(
            ees_threshold_interval_s,
            "Window length for the dynamic (locally adaptive) threshold.",
        ),
        _lab(
            ees_threshold_smoothing_s,
            "Smoothing applied to the dynamic threshold itself.",
        ),
        _lab(
            ees_qrs_window_s,
            "Width of the window around each detected R-peak used to build "
            "the QRS template. Paper default: 0.3 s.",
        ),
        _lab(
            ees_inter_qrs_tolerance,
            "Periodicity-correction tolerance used to reject/restore "
            "candidate QRS detections based on inter-beat interval "
            "consistency.",
        ),
        _lab(
            ees_min_template_beats,
            "Minimum number of complete beats required to build the "
            "averaged QRS template.",
        ),
        _lab(ees_use_min_qrs_interval, "Enforce a minimum physiological RR interval."),
        _lab(ees_min_qrs_interval_s, "Minimum RR interval, if enforced."),
        _lab(ees_use_max_qrs_interval, "Enforce a maximum physiological RR interval."),
        _lab(ees_max_qrs_interval_s, "Maximum RR interval, if enforced."),
        _lab(
            ees_use_output_bandpass,
            "Apply an extra band-pass filter, independent of the detection "
            "band, to the final EES output only. QRS detection and the "
            "template always run on the (already notch/HPF/low-pass "
            "filtered) input signal.",
        ),
        _lab(ees_output_bandpass_low, "Output bandpass low cutoff, if enabled."),
        _lab(ees_output_bandpass_high, "Output bandpass high cutoff, if enabled."),
        _lab(ees_output_bandpass_order, "Output bandpass filter order, if enabled."),
        ees_output_bandpass_stage,
    ]

    mo.vstack(
        [
            mo.md("### Canonical step-by-step pipeline (Jonkman et al. 2024)"),
            mo.md("**1-2. Line-noise & band-limiting**"),
            mo.hstack(
                [
                    _lab(
                        notch_base_frequency,
                        "Fundamental frequency of the periodic interference "
                        "to notch out (mains hum, or a co-recorded EIT "
                        "device's frame rate). Typical: 50 Hz (EU mains / "
                        "Draeger EIT frame rate) or 60 Hz (US mains).",
                    ),
                    _lab(
                        notch_quality_factor,
                        "Notch sharpness (f0 / bandwidth) at each harmonic. "
                        "Higher = narrower notch, less collateral signal "
                        "loss, but must land exactly on the true "
                        "frequency. Typical range: 20-40.",
                    ),
                    _lab(
                        canonical_notch_max_hz,
                        "Highest frequency the harmonic notch reaches. Must "
                        "be >= the low-pass cutoff -- a harmonic sitting "
                        "right at that edge is only partially attenuated by "
                        "either filter otherwise. Typical: Nyquist (fs/2). "
                        "Set equal to the low-pass cutoff to reproduce that "
                        "under-notching bug.",
                    ),
                    _lab(
                        canonical_low_pass_hz,
                        "Upper cutoff of the EMG band-pass. Diaphragm sEMG "
                        "content is mostly below ~250 Hz, but literature "
                        "commonly keeps headroom up to 400-500 Hz.",
                    ),
                    _lab(
                        canonical_hpf_hz,
                        "Lower cutoff removing baseline wander and residual "
                        "ECG P/T waves. Jonkman et al. 2024 recommend "
                        "0.5-20 Hz; higher strips more cardiac content but "
                        "risks removing genuine low-frequency diaphragm "
                        "EMG.",
                    ),
                ],
                justify="start",
                gap=2,
            ),
            mo.md("**3. ECG (QRS) removal**"),
            mo.hstack(_ecg_controls, justify="start", gap=2),
            mo.md(
                "**3b. EES (Estimated ECG Subtraction) parameters** — used "
                "when 'EES' is selected above, and always used by the "
                "detailed EES diagnostic section at the end of this notebook."
            ),
            mo.hstack(_ees_controls, wrap=True, justify="start", gap=2),
            mo.md("**4-6. Envelope & breath detection**"),
            mo.hstack(
                [
                    _lab(
                        canonical_envelope_type,
                        "How the burst amplitude envelope is computed. RMS "
                        "(paper default, required for variance-based "
                        "baseline correction), ARV (mean absolute value), "
                        "or MAV (rolling median of the absolute value, more "
                        "robust to spikes).",
                    ),
                    _lab(
                        canonical_envelope_ms,
                        "Length of the centered rolling window used to "
                        "compute the envelope. Shorter = more temporal "
                        "detail but noisier; longer = smoother but blurs "
                        "breath onset/offset timing. Typical range: "
                        "100-300 ms (paper default: 250 ms).",
                    ),
                    _lab(
                        canonical_baseline_correction,
                        "Subtracts the estimated noise floor (10th "
                        "percentile) from the envelope. For RMS, the noise "
                        "*variance* is subtracted, not the standard "
                        "deviation, per the paper's recommendation. "
                        "Recommended: on.",
                    ),
                    _lab(
                        breath_min_width_s,
                        "Minimum width of an EMG burst to count as a "
                        "breath; also sets the window used for per-breath "
                        "amplitude. Typical range: 0.5-1.5 s (matches "
                        "expected inspiratory duration).",
                    ),
                    _lab(
                        breath_prominence_factor,
                        "Minimum peak prominence (relative to signal scale) "
                        "for a burst to be counted as a breath. Lower = "
                        "more sensitive (more breaths, more false "
                        "positives from residual noise); higher may miss "
                        "weak breaths. Typical range: 0.05-0.2.",
                    ),
                ],
                justify="start",
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _canonical_pipeline(
    BIOPAC_SAMPLE_FREQUENCY,
    breath_min_width_s,
    breath_prominence_factor,
    canonical_baseline_correction,
    canonical_ecg_method,
    canonical_ecg_peak_fraction,
    canonical_envelope_ms,
    canonical_envelope_type,
    canonical_gate_width_ms,
    canonical_hpf_hz,
    canonical_low_pass_hz,
    canonical_notch_max_hz,
    canonical_wavelet_level,
    canonical_wavelet_threshold,
    detect_ecg_peaks,
    detect_emg_breath_peaks,
    downsample,
    ees_detection_band_high,
    ees_detection_band_low,
    ees_detection_smoothing_s,
    ees_filter_order,
    ees_inter_qrs_tolerance,
    ees_max_qrs_interval_s,
    ees_min_qrs_interval_s,
    ees_min_template_beats,
    ees_output_bandpass_high,
    ees_output_bandpass_low,
    ees_output_bandpass_order,
    ees_output_bandpass_stage,
    ees_qrs_window_s,
    ees_threshold_interval_s,
    ees_threshold_smoothing_s,
    ees_use_max_qrs_interval,
    ees_use_min_qrs_interval,
    ees_use_output_bandpass,
    emg_bandpass_butter,
    emg_plot,
    estimated_ecg_subtraction,
    gating,
    harmonic_notch_filter,
    normalize,
    notch_base_frequency,
    notch_quality_factor,
    np,
    pd,
    rolling_arv,
    rolling_rms,
    wavelet_denoising,
):
    # Canonical respiratory-sEMG preprocessing pipeline, step by step, following
    # Jonkman et al. 2024 (Critical Care 28:2), "Preprocessing" + Table 2.
    # Runs from the RAW sEMG so every stage is explicit.
    _fs = BIOPAC_SAMPLE_FREQUENCY
    _raw = emg_plot["raw_full"]
    _time = emg_plot["full_time"]
    _low_pass = min(float(canonical_low_pass_hz.value), _fs / 2 * 0.95)
    _notch_max = min(float(canonical_notch_max_hz.value), _fs / 2 * 0.95)

    # -- Stage 1: low-frequency artifact + power-line removal ------------------
    # Paper: HPF 0.5-20 Hz (baseline wander, P/T waves) + power-line (50/60 Hz)
    # suppression. The 50 Hz harmonic notch here also removes the EIT comb, and
    # the 20 Hz high-pass is the paper's recommended way to strip the P/T waves
    # (rather than a wider gate).
    #
    # `_notch_max` and `_low_pass` are independent controls: a harmonic sitting
    # at or just past the low-pass cutoff (e.g. the EIT frame-rate comb's 10th
    # harmonic at 500 Hz) is only partially attenuated by the low-pass
    # filter's finite roll-off, so it must still be fully inside the notch's
    # stopband rather than at its edge. Set `_notch_max == _low_pass` to
    # reproduce that under-notching; the default here extends the notch past
    # the low-pass cutoff to fix it.
    _notched = harmonic_notch_filter(
        _raw,
        base_frequency=float(notch_base_frequency.value),
        sample_frequency=_fs,
        max_frequency=_notch_max,
        quality_factor=float(notch_quality_factor.value),
    )
    _hpf = emg_bandpass_butter(
        emg_raw=_notched,
        high_pass=float(canonical_hpf_hz.value),
        low_pass=_low_pass,
        fs_emg=_fs,
    )

    # -- EES (Estimated ECG Subtraction, Jonkman et al. 2021) -----------------
    # Computed unconditionally (not just when selected as the active method
    # below) on the same stage-1 (notch + band-pass) signal every other
    # method uses, so the detailed EES diagnostic section at the end of the
    # notebook always has a result without duplicating its parameter
    # controls.
    _ees_min_qrs_interval = (
        float(ees_min_qrs_interval_s.value) if ees_use_min_qrs_interval.value else None
    )
    _ees_max_qrs_interval = (
        float(ees_max_qrs_interval_s.value) if ees_use_max_qrs_interval.value else None
    )
    _ees_output_bandpass_hz = (
        (float(ees_output_bandpass_low.value), float(ees_output_bandpass_high.value))
        if ees_use_output_bandpass.value
        else None
    )
    ees_result = estimated_ecg_subtraction(
        _hpf,
        sample_frequency=_fs,
        detection_band_hz=(
            float(ees_detection_band_low.value),
            float(ees_detection_band_high.value),
        ),
        filter_order=int(ees_filter_order.value),
        detection_smoothing_seconds=float(ees_detection_smoothing_s.value),
        threshold_interval_seconds=float(ees_threshold_interval_s.value),
        threshold_smoothing_seconds=float(ees_threshold_smoothing_s.value),
        qrs_window_seconds=float(ees_qrs_window_s.value),
        inter_qrs_tolerance=float(ees_inter_qrs_tolerance.value),
        minimum_template_beats=int(ees_min_template_beats.value),
        minimum_qrs_interval_seconds=_ees_min_qrs_interval,
        maximum_qrs_interval_seconds=_ees_max_qrs_interval,
        output_bandpass_hz=_ees_output_bandpass_hz,
        output_bandpass_stage=ees_output_bandpass_stage.value,
        output_bandpass_order=int(ees_output_bandpass_order.value),
    )

    # -- Stage 2: ECG (cardiac crosstalk) removal -----------------------------
    # A missed R-peak gets no wavelet suppression / no gate at all for that
    # beat, which shows up as leftover QRS spikes riding through stage 2/3 even
    # in EIT-off windows. `_peak_fraction` trades detection sensitivity
    # against false positives -- too high misses beats, too low starts
    # double-detecting and gating/denoising real EMG content around them.
    _peak_fraction = float(canonical_ecg_peak_fraction.value)
    _method = canonical_ecg_method.value
    if _method == "Gating":
        _peaks = detect_ecg_peaks(
            ecg_raw=_hpf, fs=int(_fs), peak_fraction=_peak_fraction
        )
        _gate_w = max(2, int(float(canonical_gate_width_ms.value) / 1000.0 * _fs))
        _ecg_removed = gating(_hpf, _peaks, gate_width=_gate_w, method=1)
        _stage2_label = (
            f"ECG removed — gating ({float(canonical_gate_width_ms.value):.0f} ms"
            " window, interpolated)"
        )
    elif _method == "High-pass 200 Hz":
        _ecg_removed = emg_bandpass_butter(
            emg_raw=_notched, high_pass=200.0, low_pass=_low_pass, fs_emg=_fs
        )
        _stage2_label = "ECG removed — high-pass 200 Hz (rudimentary)"
    elif _method == "EES (Estimated ECG Subtraction)":
        _ecg_removed = ees_result.cleaned
        _stage2_label = (
            f"ECG removed — EES ({len(ees_result.qrs_indices)} QRS templates, "
            f"{len(ees_result.rejected_peak_indices)} rejected, "
            f"{len(ees_result.restored_peak_indices)} restored)"
        )
    else:  # Wavelet denoising (paper's go-to method)
        _peaks = detect_ecg_peaks(
            ecg_raw=_hpf, fs=int(_fs), peak_fraction=_peak_fraction
        )
        _wavelet_threshold = float(canonical_wavelet_threshold.value)
        _wavelet_level = int(canonical_wavelet_level.value)
        _ecg_removed, _, _, _ = wavelet_denoising(
            _hpf,
            _peaks,
            fs=int(_fs),
            n=_wavelet_level,
            fixed_threshold=_wavelet_threshold,
        )
        _stage2_label = (
            f"ECG removed — wavelet denoising (db2, {_wavelet_threshold:g}σ,"
            f" level {_wavelet_level}, {len(_peaks)} R-peaks found)"
        )

    # -- Stage 3: envelope ----------------------------------------------------
    # Paper general recommendation: centered window, length 250 ms.
    _env_win = max(1, int(float(canonical_envelope_ms.value) / 1000.0 * _fs))
    _env_type = canonical_envelope_type.value
    if _env_type == "ARV":
        _envelope = rolling_arv(_ecg_removed, window_length=_env_win)
    elif _env_type.startswith("MAV"):
        _envelope = (
            pd.Series(np.abs(_ecg_removed))
            .rolling(_env_win, min_periods=1, center=True)
            .median()
            .to_numpy()
        )
    else:  # RMS
        _envelope = rolling_rms(_ecg_removed, window_length=_env_win)

    # -- Stage 4: baseline offset correction ----------------------------------
    # Paper: subtract baseline noise; for RMS remove the noise *variance*
    # (not the standard deviation).
    if canonical_baseline_correction.value:
        if _env_type == "RMS":
            _baseline_var = float(np.nanpercentile(_envelope**2, 10))
            _envelope = np.sqrt(np.clip(_envelope**2 - _baseline_var, 0.0, None))
        else:
            _baseline = float(np.nanpercentile(_envelope, 10))
            _envelope = np.clip(_envelope - _baseline, 0.0, None)

    # -- Stage 5: postprocessing (breath detection + robust amplitude) --------
    # Paper: breath-wise amplitude via 95th-5th percentile is more robust than
    # max-min; EMG-time product = area under the envelope per breath.
    # np.trapz was removed in numpy 2.4 in favour of np.trapezoid.
    _trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    _min_width = max(1, int(float(breath_min_width_s.value) * _fs))
    _peak_idx = detect_emg_breath_peaks(
        _envelope,
        min_peak_width_samples=_min_width,
        prominence_factor=float(breath_prominence_factor.value),
    )
    _amps = []
    _time_products = []
    for _p in _peak_idx:
        _lo = max(0, _p - _min_width)
        _hi = min(len(_envelope), _p + _min_width)
        _seg = _envelope[_lo:_hi]
        _amps.append(float(np.nanpercentile(_seg, 95) - np.nanpercentile(_seg, 5)))
        _time_products.append(float(_trapezoid(_seg, dx=1.0 / _fs)))

    _window_seconds = _time[-1] - _time[0] if _time.size else 0.0
    _rate = (
        60.0 * len(_peak_idx) / _window_seconds if _window_seconds > 0 else float("nan")
    )

    _step = int(downsample.value)

    def _prep(signal):
        _values = signal[::_step]
        if normalize.value:
            _values = _values - float(np.nanmedian(signal))
        return _values

    canonical_plot = {
        "ecg_method": _method,
        "stage2_label": _stage2_label,
        "envelope_type": _env_type,
        "envelope_ms": float(canonical_envelope_ms.value),
        "hpf_hz": float(canonical_hpf_hz.value),
        "low_pass_hz": _low_pass,
        "notch_max_hz": _notch_max,
        "baseline_corrected": bool(canonical_baseline_correction.value),
        "time": _time[::_step],
        "raw": _prep(_raw),
        "stage1": _prep(_hpf),
        "stage1_full": _hpf,
        "stage2": _prep(_ecg_removed),
        "envelope_time": _time[::_step],
        "envelope": _envelope[::_step],
        "breath_times": _time[_peak_idx],
        "breath_values": _envelope[_peak_idx],
        "breath_amplitudes": np.asarray(_amps),
        "breath_time_products": np.asarray(_time_products),
        "breath_count": int(_peak_idx.size),
        "breath_rate": _rate,
    }
    return canonical_plot, ees_result


@app.cell
def _canonical_pipeline_status(canonical_plot, mo, np):
    _amps = canonical_plot["breath_amplitudes"]
    _tps = canonical_plot["breath_time_products"]
    _median_amp = f"{np.median(_amps):.4f}" if _amps.size else "n/a"
    _median_tp = f"{np.median(_tps):.4g}" if _tps.size else "n/a"
    _rate = canonical_plot["breath_rate"]
    _rate_text = f"{_rate:.0f}" if np.isfinite(_rate) else "n/a"
    mo.md(
        f"""
        1. **Acquisition** — Biopac sEMGdi at 2000 Hz (paper advises ≥500,
           ideally 1000 Hz; respiratory content 25-250 Hz).
        2. **Low-frequency artifact removal** — 50 Hz harmonic notch, reaching
           up to {canonical_plot["notch_max_hz"]:.0f} Hz (power-line / EIT
           interference) + {canonical_plot["hpf_hz"]:.0f} Hz high-pass
           (baseline wander, P/T waves); low-pass cutoff at
           {canonical_plot["low_pass_hz"]:.0f} Hz. If the notch's reach stops
           at or before the low-pass cutoff, a harmonic sitting on that edge
           (e.g. the EIT frame-rate comb's 10th harmonic) only gets partial
           attenuation from either filter — set "Notch reach" ≥ "Low-pass
           cutoff" to avoid that.
        3. **ECG removal** — {canonical_plot["stage2_label"]}.
        4. **Envelope** — {canonical_plot["envelope_type"]},
           {canonical_plot["envelope_ms"]:.0f} ms centered window.
        5. **Baseline offset correction** —
           {"on (RMS: noise variance removed)" if canonical_plot["baseline_corrected"] else "off"}.
        6. **Postprocessing** — {canonical_plot["breath_count"]} breaths
           (~{_rate_text}/min); median breath amplitude (95th-5th pct)
           = {_median_amp}; median EMG-time product = {_median_tp}.
        """
    )
    return


@app.cell
def _canonical_pipeline_plot(
    canonical_plot, go, make_subplots, normalize, paw_plot, robust_range
):
    _unit = "mV centered" if normalize.value else "mV"
    _fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "1-2. Raw sEMGdi",
            f"2. Low-freq artifact removal (notch + {canonical_plot['hpf_hz']:.0f} Hz HPF)",
            f"3. {canonical_plot['stage2_label']}",
            f"4-6. Envelope ({canonical_plot['envelope_type']}) + detected breaths",
            "Raw Paw / airway pressure",
        ],
    )
    _fig.add_trace(
        go.Scattergl(
            x=canonical_plot["time"],
            y=canonical_plot["raw"],
            mode="lines",
            line=dict(color="#B0B0B0", width=0.6),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scattergl(
            x=canonical_plot["time"],
            y=canonical_plot["stage1"],
            mode="lines",
            line=dict(color="#7A5195", width=0.6),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    _fig.add_trace(
        go.Scattergl(
            x=canonical_plot["time"],
            y=canonical_plot["stage2"],
            mode="lines",
            line=dict(color="#2A9D8F", width=0.6),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    _fig.add_trace(
        go.Scattergl(
            x=canonical_plot["envelope_time"],
            y=canonical_plot["envelope"],
            mode="lines",
            name="envelope",
            line=dict(color="#E07A00", width=1.6),
        ),
        row=4,
        col=1,
    )
    if canonical_plot["breath_count"]:
        _fig.add_trace(
            go.Scattergl(
                x=canonical_plot["breath_times"],
                y=canonical_plot["breath_values"],
                mode="markers",
                name="detected breaths",
                marker=dict(color="#2A9D8F", size=7, symbol="triangle-down"),
            ),
            row=4,
            col=1,
        )
    _fig.add_trace(
        go.Scattergl(
            x=paw_plot["time"],
            y=paw_plot["values"],
            mode="lines",
            name="raw Paw",
            line=dict(color="#D1495B", width=1.1),
            hovertemplate="t=%{x:.2f}s<br>Paw=%{y:.3f} cmH2O<extra></extra>",
        ),
        row=5,
        col=1,
    )

    for _r, _key in ((1, "raw"), (2, "stage1"), (3, "stage2")):
        _fig.update_yaxes(title_text=_unit, row=_r, col=1)
        _yrange = robust_range(canonical_plot[_key])
        if _yrange is not None:
            _fig.update_yaxes(range=_yrange, row=_r, col=1)
    _fig.update_yaxes(title_text="envelope", row=4, col=1)
    _paw_range = robust_range(paw_plot["values"])
    _fig.update_yaxes(title_text="cmH2O", row=5, col=1)
    if _paw_range is not None:
        _fig.update_yaxes(range=_paw_range, row=5, col=1)
    _fig.update_xaxes(title_text="Biopac time (s)", row=5, col=1)

    _fig.update_layout(
        template="plotly_white",
        height=900,
        showlegend=True,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=64, r=24, b=44, l=72),
    )
    _fig
    return


# @app.cell
# def _emg_spectrum_plot(
#     BIOPAC_SAMPLE_FREQUENCY,
#     emg_plot,
#     go,
#     np,
#     welch,
# ):
#     # Power spectrum of the raw EMG in the visible window. The EIT interference
#     # (removed only inside the canonical pipeline below) shows up here as a
#     # comb at the frame rate and its harmonics.
#     _raw = emg_plot["raw_full"]
#     _n_fft = min(4096, max(256, 1 << int(np.log2(max(_raw.size, 1)))))
#     _f_raw, _p_raw = welch(_raw, fs=BIOPAC_SAMPLE_FREQUENCY, nperseg=_n_fft)

#     _fig = go.Figure()
#     _fig.add_trace(
#         go.Scatter(
#             x=_f_raw,
#             y=_p_raw,
#             mode="lines",
#             name="raw",
#             line=dict(color="#185FA5", width=1),
#         )
#     )
#     _fig.update_layout(
#         title="EMGdi power spectrum — raw (visible window)",
#         template="plotly_white",
#         height=280,
#         margin=dict(t=40, r=20, b=44, l=64),
#         xaxis_title="frequency (Hz)",
#         yaxis_title="PSD",
#         yaxis_type="log",
#         xaxis_range=[0, min(500, BIOPAC_SAMPLE_FREQUENCY / 2)],
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#     )
#     _fig
#     return


# @app.cell
# def _interference_plot(go, interference_edge, interference_power):
#     _fig = go.Figure()
#     _fig.add_trace(
#         go.Scatter(
#             x=interference_power["time"],
#             y=interference_power["values"],
#             mode="lines",
#             name="sEMG HF power",
#             line=dict(color="#E07A00", width=1),
#         )
#     )
#     if interference_power["threshold"] is not None:
#         _fig.add_hline(
#             y=interference_power["threshold"],
#             line=dict(color="#888888", dash="dot"),
#             annotation_text="threshold",
#         )
#     if interference_edge is not None:
#         _fig.add_vline(
#             x=interference_edge,
#             line=dict(color="#D1495B", dash="dash"),
#             annotation_text="EIT off",
#         )
#     _fig.update_layout(
#         title="sEMG interference power — EIT-off edge detection",
#         template="plotly_white",
#         height=260,
#         margin=dict(t=40, r=20, b=36, l=64),
#         xaxis_title="Biopac time (s)",
#         yaxis_title="HF power (a.u.)",
#     )
#     _fig
#     return


@app.cell
def _xcorr_plot(go, xcorr_curve):
    if xcorr_curve is None:
        _out = None
    else:
        _out = go.Figure(
            go.Scatter(
                x=xcorr_curve["lags"],
                y=xcorr_curve["corr"],
                mode="lines",
                line=dict(color="#185FA5"),
            )
        )
        _out.update_layout(
            title="EIT ↔ Paw cross-correlation (peak = best lag)",
            template="plotly_white",
            height=240,
            margin=dict(t=40, r=20, b=36, l=64),
            xaxis_title="lag (s)",
            yaxis_title="norm. corr",
        )
    _out
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


@app.cell
def _ees_section_header(mo):
    mo.md(r"""
    ---
    # Estimated ECG Subtraction (EES) — diagnostic detail

    [Jonkman et al. (2021)](https://doi.org/10.1152/japplphysiol.00298.2021)'s
    EES method builds an averaged QRS template per detected heartbeat and
    subtracts it, rather than shrinking wavelet coefficients or blanking a
    fixed gate around each beat. It's available as **"EES (Estimated ECG
    Subtraction)"** in the canonical pipeline's ECG removal dropdown above
    (feeding the same envelope/breath-detection stages as the other
    methods); this section always shows its full step-by-step diagnostics —
    reproducing Figure 1 of the paper — using the shared EES parameters set
    above, regardless of which method is currently selected there.

    EES runs on the same stage-1 signal (notch + band-pass filtered) as
    every other method, over the currently selected Biopac time window.
    """)
    return


@app.cell
def _ees_display_controls(mo, window_range):
    # Defaults to (and stays bounded by) the main "Biopac time window"
    # selector above -- moving that slider resets this one to match, since a
    # window outside it has no EES data anyway. Still independently
    # adjustable to zoom into a narrower slice of that window for the dense
    # step-by-step figure below.
    _t0, _t1 = [float(v) for v in window_range]
    ees_display_time = mo.ui.range_slider(
        start=_t0,
        stop=_t1,
        step=0.1,
        value=[_t0, _t1],
        label="EES display window (s, Biopac time) — synced to the main time window",
        full_width=True,
    )
    ees_display_time
    return (ees_display_time,)


@app.cell
def _ees_metrics(ees_result, mo):
    mo.md(
        f"""
        **Detected template beats:** {len(ees_result.qrs_indices)} &nbsp; · &nbsp;
        **Candidate peaks:** {len(ees_result.candidate_peak_indices)} &nbsp; · &nbsp;
        **Rejected candidates:** {len(ees_result.rejected_peak_indices)} &nbsp; · &nbsp;
        **Restored candidates:** {len(ees_result.restored_peak_indices)}
        """
    )
    return


@app.cell
def _ees_intermediate(
    bandpass_filter,
    canonical_plot,
    ees_detection_band_high,
    ees_detection_band_low,
    ees_detection_smoothing_s,
    ees_filter_order,
    ees_result,
    emg_plot,
    moving_average,
    np,
    BIOPAC_SAMPLE_FREQUENCY,
):
    """Reproduce EES's internal steps 1-3 and 5/8/9 for display; the public
    `estimated_ecg_subtraction` API only returns the final result, not these
    intermediate arrays."""

    _input_signal = canonical_plot["stage1_full"]
    _fs = BIOPAC_SAMPLE_FREQUENCY

    ees_promoted = bandpass_filter(
        _input_signal,
        cutoff_frequency=(
            float(ees_detection_band_low.value),
            float(ees_detection_band_high.value),
        ),
        sample_frequency=_fs,
        order=int(ees_filter_order.value),
    )
    ees_rectified = np.abs(ees_promoted)
    _detection_window_samples = max(
        1, int(round(float(ees_detection_smoothing_s.value) * _fs))
    )
    ees_smoothed_detection = moving_average(
        ees_rectified, window_size=_detection_window_samples
    ) / np.ptp(_input_signal)

    _above_threshold = ees_smoothed_detection > ees_result.dynamic_threshold
    _padded = np.pad(_above_threshold.astype(np.int8), (1, 1))
    _changes = np.diff(_padded)
    ees_segment_starts = np.flatnonzero(_changes == 1)
    ees_segment_ends = np.flatnonzero(_changes == -1) - 1

    _template_offsets = ees_result.template_sample_offsets
    _template_length = len(_template_offsets)
    _half_template = _template_length // 2
    ees_original_segments = np.stack(
        [
            _input_signal[
                int(_r) - _half_template : int(_r) - _half_template + _template_length
            ]
            for _r in ees_result.qrs_indices[:, 1]
        ]
    )
    _denormalized_segments = []
    for _q, _r, _s in ees_result.qrs_indices:
        _q_value = _input_signal[_q]
        _r_value = _input_signal[_r]
        _s_value = _input_signal[_s]
        _denormalized = np.empty_like(ees_result.normalized_template)
        _denormalized[: _half_template + 1] = (
            ees_result.normalized_template[: _half_template + 1] * (_r_value - _q_value)
            + _q_value
        )
        _denormalized[_half_template:] = (
            ees_result.normalized_template[_half_template:] * (_r_value - _s_value)
            + _s_value
        )
        _denormalized_segments.append(_denormalized)
    ees_denormalized_segments = np.asarray(_denormalized_segments)

    return (
        ees_denormalized_segments,
        ees_original_segments,
        ees_promoted,
        ees_rectified,
        ees_segment_ends,
        ees_segment_starts,
        ees_smoothed_detection,
    )


@app.cell
def _ees_figure(
    BIOPAC_SAMPLE_FREQUENCY,
    canonical_plot,
    ees_denormalized_segments,
    ees_display_time,
    ees_original_segments,
    ees_promoted,
    ees_qrs_window_s,
    ees_rectified,
    ees_result,
    ees_segment_ends,
    ees_segment_starts,
    ees_smoothed_detection,
    eit_plot,
    emg_plot,
    np,
    paw_plot,
    plt,
):
    _input_signal = canonical_plot["stage1_full"]
    _time = emg_plot["full_time"]
    _fs = BIOPAC_SAMPLE_FREQUENCY
    _window_start, _window_end = [float(v) for v in ees_display_time.value]
    _visible = (_time >= _window_start) & (_time <= _window_end)
    _template_time = ees_result.template_sample_offsets / _fs
    _template_window_seconds = float(ees_qrs_window_s.value)
    _template_half_window_seconds = _template_window_seconds / 2
    _blue = "#3B7CCC"
    _orange = "#F28E2B"
    _gold = "#E5A823"
    _charcoal = "#3F3F3F"
    _red = "#E64B35"

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

    _amplitude_limit = float(
        1.05 * _safe_abs_max(_input_signal, ees_promoted, ees_result.estimated_ecg)
    )
    _cleaned_amplitude_limit = float(
        max(
            _amplitude_limit,
            1.05 * _safe_abs_max(ees_result.cleaned, fallback=_amplitude_limit),
        )
    )
    _detection_limit = float(
        1.05
        * _safe_abs_max(
            ees_smoothed_detection, ees_result.dynamic_threshold, fallback=1.0
        )
    )
    _normalized_min, _normalized_max = _safe_min_max(
        ees_result.normalized_segments, ees_result.normalized_template
    )
    _normalized_padding = max(
        0.05 * (_normalized_max - _normalized_min), np.finfo(float).eps
    )

    _figure = plt.figure(figsize=(18, 27), constrained_layout=True)
    _outer = _figure.add_gridspec(
        8, 12, height_ratios=(1.0, 1.0, 1.55, 1.1, 0.95, 0.9, 0.9, 0.85)
    )

    _ax_input = _figure.add_subplot(_outer[0, 0:6])
    _ax_filter = _figure.add_subplot(_outer[0, 6:12], sharex=_ax_input)
    _ax_rectified = _figure.add_subplot(_outer[1, 0:6], sharex=_ax_input)
    _ax_threshold = _figure.add_subplot(_outer[1, 6:12], sharex=_ax_input)

    _crossing_grid = _outer[2, 0:6].subgridspec(
        2, 2, height_ratios=(2.0, 1.1), width_ratios=(1.15, 0.85)
    )
    _ax_crossings = _figure.add_subplot(_crossing_grid[0, :], sharex=_ax_input)
    _ax_correction = _figure.add_subplot(_crossing_grid[1, 0])
    _ax_correction_text = _figure.add_subplot(_crossing_grid[1, 1])
    _ax_qrs = _figure.add_subplot(_outer[2, 6:12], sharex=_ax_input)

    _ax_window = _figure.add_subplot(_outer[3, 0:12])
    _template_grid = _outer[4, 0:12].subgridspec(1, 4, wspace=0.16)
    _ax_original = _figure.add_subplot(_template_grid[0, 0])
    _ax_normalized = _figure.add_subplot(_template_grid[0, 1])
    _ax_average = _figure.add_subplot(_template_grid[0, 2])
    _ax_denormalized = _figure.add_subplot(_template_grid[0, 3])

    _ax_inserted = _figure.add_subplot(_outer[5, 0:12], sharex=_ax_input)
    _ax_cleaned = _figure.add_subplot(_outer[6, 0:12], sharex=_ax_input)
    _ax_paw = _figure.add_subplot(_outer[7, 0:6], sharex=_ax_input)
    _ax_eit = _figure.add_subplot(_outer[7, 6:12], sharex=_ax_input)

    def _paper_axis(axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
        axis.margins(x=0)

    def _time_title(axis, title):
        axis.set_title(title, loc="left", fontsize=10, pad=7)
        axis.set_ylabel("Amplitude", fontsize=8)
        axis.set_xlim(_window_start, _window_end)
        _paper_axis(axis)

    _ax_input.plot(_time[_visible], _input_signal[_visible], color=_blue, lw=0.6)
    _time_title(_ax_input, "Input: EMGdi (post notch + band-pass)")

    _ax_filter.plot(_time[_visible], ees_promoted[_visible], color=_blue, lw=0.6)
    _time_title(_ax_filter, "1. Bandpass filter to promote relative ECG amplitude")

    _ax_rectified.plot(_time[_visible], ees_rectified[_visible], color=_blue, lw=0.65)
    _time_title(_ax_rectified, "2. Rectification")

    _ax_threshold.plot(
        _time[_visible], ees_smoothed_detection[_visible], color=_blue, lw=0.75
    )
    _ax_threshold.plot(
        _time[_visible], ees_result.dynamic_threshold[_visible], color="black", lw=1.2
    )
    _time_title(
        _ax_threshold,
        "3. Moving-average detection signal\n4. Apply dynamic threshold",
    )

    _ax_crossings.plot(
        _time[_visible], ees_smoothed_detection[_visible], color=_blue, lw=0.7
    )
    _ax_crossings.plot(
        _time[_visible], ees_result.dynamic_threshold[_visible], color="black", lw=1.0
    )
    _start_visible = ees_segment_starts[
        (_time[ees_segment_starts] >= _window_start)
        & (_time[ees_segment_starts] <= _window_end)
    ]
    _end_visible = ees_segment_ends[
        (_time[ees_segment_ends] >= _window_start)
        & (_time[ees_segment_ends] <= _window_end)
    ]
    _ax_crossings.scatter(
        _time[_start_visible],
        ees_smoothed_detection[_start_visible],
        marker="x",
        color=_red,
        s=28,
        label="start QRS segment",
        zorder=4,
    )
    _ax_crossings.scatter(
        _time[_end_visible],
        ees_smoothed_detection[_end_visible],
        marker="x",
        color=_orange,
        s=28,
        label="end QRS segment",
        zorder=4,
    )
    _time_title(_ax_crossings, "5. Detect crossings with threshold level")
    _ax_crossings.legend(loc="upper left", fontsize=7, ncols=2, frameon=False)

    if len(ees_result.rejected_peak_indices):
        _correction_center = int(ees_result.rejected_peak_indices[0])
    elif len(ees_result.candidate_peak_indices):
        _correction_center = int(
            ees_result.candidate_peak_indices[
                len(ees_result.candidate_peak_indices) // 2
            ]
        )
    else:
        _correction_center = int(_time.size // 2)
    _correction_radius = int(round(0.45 * _fs))
    _correction_start = max(0, _correction_center - _correction_radius)
    _correction_end = min(len(_time), _correction_center + _correction_radius)
    _correction_slice = slice(_correction_start, _correction_end)
    _ax_correction.plot(
        _time[_correction_slice],
        ees_smoothed_detection[_correction_slice],
        color=_blue,
        lw=0.75,
    )
    _ax_correction.plot(
        _time[_correction_slice],
        ees_result.dynamic_threshold[_correction_slice],
        color="black",
        lw=0.9,
    )
    for _indices, _marker, _color, _label in (
        (ees_result.rejected_peak_indices, "x", _red, "deleted"),
        (ees_result.restored_peak_indices, "D", _blue, "restored"),
    ):
        _nearby = _indices[
            (_indices >= _correction_start) & (_indices < _correction_end)
        ]
        if len(_nearby):
            _ax_correction.scatter(
                _time[_nearby],
                ees_smoothed_detection[_nearby],
                marker=_marker,
                color=_color,
                s=34,
                label=_label,
                zorder=5,
            )
    _ax_correction.set_title(
        "6. Periodicity correction (detail)", loc="left", fontsize=9
    )
    _ax_correction.set_xlabel("Time (s)", fontsize=8)
    _ax_correction.set_ylabel("Normalized", fontsize=8)
    _ax_correction.legend(loc="upper right", fontsize=7, frameon=False)
    _paper_axis(_ax_correction)
    _ax_correction_text.axis("off")
    _ax_correction_text.text(
        0.03,
        0.68,
        "6. Remove (and restore) wrongfully\n"
        "detected (or deleted) QRS segments\n\n"
        f"Rejected: {len(ees_result.rejected_peak_indices)}\n"
        f"Restored: {len(ees_result.restored_peak_indices)}",
        fontsize=10,
        va="top",
    )

    _ax_qrs.plot(_time[_visible], _input_signal[_visible], color=_blue, lw=0.6)
    _qrs = ees_result.qrs_indices
    for _column, _color, _label, _size in (
        (0, _charcoal, "Q", 28),
        (1, _gold, "R", 32),
        (2, _red, "S", 28),
    ):
        _indices = _qrs[:, _column]
        _inside = (_time[_indices] >= _window_start) & (_time[_indices] <= _window_end)
        _ax_qrs.scatter(
            _time[_indices[_inside]],
            _input_signal[_indices[_inside]],
            color=_color,
            s=_size,
            label=_label,
            zorder=4,
        )
    _time_title(_ax_qrs, "7. Detect QRS peaks")
    _ax_qrs.legend(loc="upper right", fontsize=8, ncols=3, frameon=False)

    _middle_beat = len(_qrs) // 2
    _shown_beats = _qrs[max(0, _middle_beat - 1) : _middle_beat + 2]
    if len(_shown_beats):
        _window_left = max(0, int(_shown_beats[0, 1]) - int(0.35 * _fs))
        _window_right = min(len(_time), int(_shown_beats[-1, 1]) + int(0.35 * _fs))
        _ax_window.plot(
            _time[_window_left:_window_right],
            _input_signal[_window_left:_window_right],
            color=_blue,
            lw=0.65,
        )
        for _column, _color, _label in (
            (0, _charcoal, "Q"),
            (1, _gold, "R"),
            (2, _red, "S"),
        ):
            _ax_window.scatter(
                _time[_shown_beats[:, _column]],
                _input_signal[_shown_beats[:, _column]],
                color=_color,
                s=30,
                label=_label,
                zorder=4,
            )
        _selected_r_time = _time[_shown_beats[-1, 1]]
        _bracket_y = 0.82 * _amplitude_limit
        _ax_window.annotate(
            "",
            xy=(_selected_r_time - _template_half_window_seconds, _bracket_y),
            xytext=(_selected_r_time + _template_half_window_seconds, _bracket_y),
            arrowprops={"arrowstyle": "|-|", "lw": 1.2, "color": _charcoal},
        )
        _ax_window.text(
            _selected_r_time,
            _bracket_y,
            f" {_template_window_seconds:.2f} s ",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _ax_window.set_title(
        f"8. Select window length around R wave ({_template_window_seconds:.2f} s)",
        loc="left",
        fontsize=9,
    )
    _ax_window.set_xlabel("Time (s)", fontsize=8)
    _ax_window.set_ylabel("Amplitude", fontsize=8)
    _ax_window.legend(loc="upper left", fontsize=7, ncols=3, frameon=False)
    _paper_axis(_ax_window)

    _segment_colors = plt.cm.tab20(
        np.linspace(0, 1, max(len(ees_original_segments), 1))
    )
    for _segment, _color in zip(ees_original_segments, _segment_colors):
        _ax_original.plot(_template_time, _segment, color=_color, lw=0.55, alpha=0.85)
    for _segment, _color in zip(ees_result.normalized_segments, _segment_colors):
        _ax_normalized.plot(_template_time, _segment, color=_color, lw=0.55, alpha=0.85)
    _ax_average.plot(
        _template_time, ees_result.normalized_template, color=_orange, lw=1.2
    )
    for _segment, _color in zip(ees_denormalized_segments, _segment_colors):
        _ax_denormalized.plot(
            _template_time, _segment, color=_color, lw=0.55, alpha=0.9
        )
    for _axis, _subtitle in (
        (_ax_original, "a. original QRS segments"),
        (_ax_normalized, "b. normalized QRS segments"),
        (_ax_average, "c. average QRS segment"),
        (_ax_denormalized, "d. denormalized QRS segments"),
    ):
        _axis.set_xlabel(_subtitle, fontsize=8, fontstyle="italic")
        _axis.set_xticks([])
        _axis.set_yticks([])
        _paper_axis(_axis)
    _ax_original.set_title("9. Create QRS template", loc="left", fontsize=9)

    _ax_inserted.plot(
        _time[_visible], ees_result.estimated_ecg[_visible], color=_orange, lw=0.75
    )
    _time_title(
        _ax_inserted, "10. Insert denormalized QRS segments at original location"
    )

    _ax_cleaned.plot(_time[_visible], ees_result.cleaned[_visible], color=_blue, lw=0.6)
    _time_title(_ax_cleaned, "11. Subtract estimated ECG template from EMGdi")

    def _context_panel(axis, plot_time, plot_values, *, title, ylabel, color):
        _time_title(axis, title)
        _has_data = (
            plot_time is not None
            and plot_values is not None
            and np.asarray(plot_time).size > 0
            and np.asarray(plot_values).size > 0
            and np.any(np.isfinite(plot_values))
        )
        if _has_data:
            _n = min(len(plot_time), len(plot_values))
            axis.plot(plot_time[:_n], plot_values[:_n], color=color, lw=0.7)
            axis.set_ylabel(ylabel, fontsize=8)
            _lo, _hi = _safe_min_max(plot_values)
            _pad = 0.05 * max(_hi - _lo, np.finfo(float).eps)
            axis.set_ylim(_lo - _pad, _hi + _pad)
        else:
            axis.set_ylabel("")
            axis.set_yticks([])
            axis.text(
                0.5,
                0.5,
                "No data in this window.",
                ha="center",
                va="center",
                fontsize=9,
                color="#777777",
                transform=axis.transAxes,
            )

    _context_panel(
        _ax_paw,
        paw_plot["time"],
        paw_plot["values"],
        title="Airway pressure (Paw)",
        ylabel="Paw (cmH2O)",
        color="#8B5FA8",
    )
    _context_panel(
        _ax_eit,
        eit_plot["time"],
        eit_plot["values"],
        title="EIT global impedance",
        ylabel="Impedance (a.u.)",
        color="#3F7F5F",
    )

    _ax_inserted.set_xlabel("Time (s)", fontsize=8)
    _ax_paw.set_xlabel("Time (s)", fontsize=8)
    _ax_eit.set_xlabel("Time (s)", fontsize=8)

    for _axis in (
        _ax_input,
        _ax_filter,
        _ax_qrs,
        _ax_window,
        _ax_original,
        _ax_denormalized,
        _ax_inserted,
    ):
        _axis.set_ylim(-_amplitude_limit, _amplitude_limit)
    _ax_cleaned.set_ylim(-_cleaned_amplitude_limit, _cleaned_amplitude_limit)
    _ax_rectified.set_ylim(0, _amplitude_limit)
    for _axis in (_ax_threshold, _ax_crossings, _ax_correction):
        _axis.set_ylim(0, _detection_limit)
    for _axis in (_ax_normalized, _ax_average):
        _axis.set_ylim(
            _normalized_min - _normalized_padding, _normalized_max + _normalized_padding
        )

    for _axis in (
        _ax_input,
        _ax_filter,
        _ax_rectified,
        _ax_threshold,
        _ax_crossings,
        _ax_qrs,
        _ax_inserted,
        _ax_cleaned,
    ):
        _axis.axhline(0, color="#777777", lw=0.35, alpha=0.5)

    _figure.suptitle(
        "Estimated ECG Subtraction (EES) — complete processing sequence", fontsize=15
    )
    _figure
    return


if __name__ == "__main__":
    app.run()
