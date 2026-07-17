"""Pure signal/waveform synthesis for synthetic EIT and Medibus data."""

from __future__ import annotations

import numpy as np

from m3resp.synthetic.config import (
    EITWaveformConfig,
    EventSeriesConfig,
    LungTemplateConfig,
    MedibusConfig,
    SyntheticGeneratorConfig,
)
from m3resp.synthetic.drift import (
    _eit_timing_drift_config,
    _shift_eit_signal_components,
    generate_drift,
)

DEFAULT_ZERO = 0.0
DEFAULT_ONE = 1.0
DEFAULT_TWO = 2.0
DEFAULT_SECONDS_PER_MINUTE = 60.0
DEFAULT_FLOAT32_DTYPE = np.float32
DEFAULT_INT32_DTYPE = np.int32
DEFAULT_MIN_SAMPLES_FOR_EXTREMA = 3
DEFAULT_LOCAL_MAX_FLAG = 1
DEFAULT_LOCAL_MIN_FLAG = -1
DEFAULT_NO_EXTREMA_FLAG = 0
DEFAULT_EMPTY_STRING = ""
DEFAULT_COMPONENT_BASELINE_INDEX = 0
DEFAULT_COMPONENT_BREATHING_INDEX = 1
DEFAULT_COMPONENT_DRIFT_INDEX = 2
DEFAULT_COMPONENT_CARDIAC_INDEX = 3
DEFAULT_COMPONENT_NOISE_INDEX = 4
DEFAULT_EIT_COMPONENT_LABELS = ("baseline", "breathing", "drift", "cardiac", "noise")


def generate_realistic_eit_signal(
    config: SyntheticGeneratorConfig,
    duration: float,
    fs: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Generate a realistic synthetic breathing-related impedance waveform."""

    rng = np.random.default_rng(seed)
    time_seconds = np.arange(0, duration, 1 / fs)
    drift = generate_drift(time_seconds, config.eit.drift)
    breathing = np.zeros_like(time_seconds)
    current_time = DEFAULT_ZERO

    while current_time < duration:
        this_bpm = max(
            config.eit_waveform.min_respiratory_rate_bpm,
            config.respiratory.respiratory_rate_bpm
            + rng.normal(
                DEFAULT_ZERO,
                config.respiratory.respiratory_rate_variation_bpm,
            ),
        )
        cycle_duration = DEFAULT_SECONDS_PER_MINUTE / this_bpm
        this_amp = max(
            config.eit_waveform.min_tidal_amplitude_au,
            config.eit.tidal_amplitude_au
            * (
                DEFAULT_ONE
                + rng.normal(
                    DEFAULT_ZERO,
                    config.eit_waveform.tidal_amplitude_variation_fraction,
                )
            ),
        )
        inhale_fraction = np.clip(
            config.respiratory.ie_ratio / (DEFAULT_ONE + config.respiratory.ie_ratio)
            + rng.normal(DEFAULT_ZERO, config.eit_waveform.inhale_fraction_variation),
            config.eit_waveform.min_inhale_fraction,
            config.eit_waveform.max_inhale_fraction,
        )
        inhale_duration = cycle_duration * inhale_fraction
        exhale_duration = cycle_duration - inhale_duration

        idx = (time_seconds >= current_time) & (
            time_seconds < current_time + cycle_duration
        )
        tau = time_seconds[idx] - current_time
        wave = np.zeros_like(tau)
        insp_idx = tau < inhale_duration
        exp_idx = ~insp_idx

        if np.any(insp_idx):
            x = tau[insp_idx] / inhale_duration
            wave[insp_idx] = config.eit_waveform.breath_wave_scale * (
                DEFAULT_ONE - np.cos(np.pi * x)
            )

        if np.any(exp_idx):
            x = (tau[exp_idx] - inhale_duration) / exhale_duration
            wave[exp_idx] = config.eit_waveform.breath_wave_scale * (
                DEFAULT_ONE + np.cos(np.pi * x)
            )

        breathing[idx] = this_amp * wave
        current_time += cycle_duration

    heart_rate_hz = config.eit.heart_rate_bpm / DEFAULT_SECONDS_PER_MINUTE
    cardiac = config.eit.cardiac_amplitude_au * (
        config.eit_waveform.cardiac_primary_weight
        * np.sin(2 * np.pi * heart_rate_hz * time_seconds)
        + config.eit_waveform.cardiac_harmonic_weight
        * np.sin(
            DEFAULT_TWO
            * np.pi
            * config.eit_waveform.cardiac_harmonic_multiplier
            * heart_rate_hz
            * time_seconds
            + config.eit_waveform.cardiac_harmonic_phase_radians
        )
    )
    noise = rng.normal(DEFAULT_ZERO, config.eit.noise_std_au, size=time_seconds.shape)
    baseline = np.full_like(time_seconds, config.eit.base_impedance_au)
    signal = baseline + breathing + drift + cardiac + noise
    timing_drift = _eit_timing_drift_config(config)
    if timing_drift is not None:
        time_seconds, signal, components = _shift_eit_signal_components(
            time_seconds,
            {
                DEFAULT_EIT_COMPONENT_LABELS[
                    DEFAULT_COMPONENT_BASELINE_INDEX
                ]: baseline,
                DEFAULT_EIT_COMPONENT_LABELS[
                    DEFAULT_COMPONENT_BREATHING_INDEX
                ]: breathing,
                DEFAULT_EIT_COMPONENT_LABELS[DEFAULT_COMPONENT_DRIFT_INDEX]: drift,
                DEFAULT_EIT_COMPONENT_LABELS[DEFAULT_COMPONENT_CARDIAC_INDEX]: cardiac,
                DEFAULT_EIT_COMPONENT_LABELS[DEFAULT_COMPONENT_NOISE_INDEX]: noise,
            },
            timing_drift,
        )
        return time_seconds, signal, components
    return (
        time_seconds,
        signal,
        {
            DEFAULT_EIT_COMPONENT_LABELS[DEFAULT_COMPONENT_BASELINE_INDEX]: baseline,
            DEFAULT_EIT_COMPONENT_LABELS[DEFAULT_COMPONENT_BREATHING_INDEX]: breathing,
            DEFAULT_EIT_COMPONENT_LABELS[DEFAULT_COMPONENT_DRIFT_INDEX]: drift,
            DEFAULT_EIT_COMPONENT_LABELS[DEFAULT_COMPONENT_CARDIAC_INDEX]: cardiac,
            DEFAULT_EIT_COMPONENT_LABELS[DEFAULT_COMPONENT_NOISE_INDEX]: noise,
        },
    )


def make_lung_template(config: LungTemplateConfig) -> np.ndarray:
    """Create a simple 32x32 synthetic lung-shaped template."""

    nx = config.nx
    ny = config.ny
    y, x = np.mgrid[int(DEFAULT_ZERO) : ny, int(DEFAULT_ZERO) : nx]
    x_half_range = (nx - DEFAULT_ONE) / DEFAULT_TWO
    y_half_range = (ny - DEFAULT_ONE) / DEFAULT_TWO
    xn = (x - x_half_range) / x_half_range
    yn = (y - y_half_range) / y_half_range
    left = np.exp(
        -(
            ((xn - config.left_center_x) / config.lung_width) ** DEFAULT_TWO
            + ((yn + config.center_y) / config.lung_height) ** DEFAULT_TWO
        )
    )
    right = np.exp(
        -(
            ((xn - config.right_center_x) / config.lung_width) ** DEFAULT_TWO
            + ((yn + config.center_y) / config.lung_height) ** DEFAULT_TWO
        )
    )
    center_penalty = config.center_penalty_weight * np.exp(
        -(
            (xn / config.center_penalty_width) ** DEFAULT_TWO
            + ((yn + config.center_y) / config.center_penalty_height) ** DEFAULT_TWO
        )
    )
    template = np.clip(left + right - center_penalty, config.min_template_value, None)
    max_val = template.max()
    if max_val > DEFAULT_ZERO:
        template = template / max_val
    return template.astype(DEFAULT_FLOAT32_DTYPE)


def signal_to_pixel_impedance(
    signal: np.ndarray,
    seed: int,
    waveform_config: EITWaveformConfig,
    template_config: LungTemplateConfig,
) -> np.ndarray:
    """Convert a 1D breathing waveform into 32x32 impedance frames."""

    rng = np.random.default_rng(seed)
    template = make_lung_template(template_config)
    dynamic = signal - np.min(signal)
    pixel_impedance = np.empty(
        (len(signal), template_config.ny, template_config.nx),
        dtype=DEFAULT_FLOAT32_DTYPE,
    )

    for index, amp in enumerate(dynamic):
        frame = amp * template
        frame += rng.normal(
            DEFAULT_ZERO,
            waveform_config.pixel_noise_fraction * max(float(amp), np.finfo(float).eps),
            size=frame.shape,
        )
        pixel_impedance[index] = frame.astype(DEFAULT_FLOAT32_DTYPE)

    return pixel_impedance


def detect_min_max_flags(signal: np.ndarray) -> np.ndarray:
    """Detect local minima and maxima for Draeger min_max_flag."""

    flags = np.zeros(len(signal), dtype=DEFAULT_INT32_DTYPE)
    if len(signal) < DEFAULT_MIN_SAMPLES_FOR_EXTREMA:
        return flags

    d_signal = np.diff(signal)
    for index in range(int(DEFAULT_ONE), len(signal) - int(DEFAULT_ONE)):
        prev_slope = d_signal[index - int(DEFAULT_ONE)]
        next_slope = d_signal[index]
        if prev_slope > DEFAULT_ZERO and next_slope <= DEFAULT_ZERO:
            flags[index] = DEFAULT_LOCAL_MAX_FLAG
        elif prev_slope < DEFAULT_ZERO and next_slope >= DEFAULT_ZERO:
            flags[index] = DEFAULT_LOCAL_MIN_FLAG
    return flags


def generate_medibus_data(
    time_seconds: np.ndarray,
    signal: np.ndarray,
    n_medibus_fields: int,
    respiratory_rate_bpm: float,
    config: MedibusConfig,
) -> np.ndarray:
    """Create synthetic Medibus channels with pressure, flow, and volume."""

    medibus = np.zeros(
        (n_medibus_fields, len(time_seconds)),
        dtype=DEFAULT_FLOAT32_DTYPE,
    )
    signal_range = np.ptp(signal)
    if signal_range == DEFAULT_ZERO:
        scaled = np.zeros_like(signal)
    else:
        scaled = (signal - signal.min()) / signal_range
    medibus[config.pressure_channel] = (
        config.pressure_baseline_cm_h2o + config.pressure_amplitude_cm_h2o * scaled
    ).astype(DEFAULT_FLOAT32_DTYPE)
    medibus[config.flow_channel] = np.gradient(signal, time_seconds).astype(
        DEFAULT_FLOAT32_DTYPE
    )

    signal_std = np.std(signal)
    if signal_std == DEFAULT_ZERO:
        medibus[config.volume_channel] = np.full_like(
            signal,
            config.volume_baseline_ml,
            dtype=DEFAULT_FLOAT32_DTYPE,
        )
    else:
        medibus[config.volume_channel] = (
            config.volume_baseline_ml
            + config.volume_amplitude_ml * (signal - signal.mean()) / signal_std
        ).astype(DEFAULT_FLOAT32_DTYPE)

    if n_medibus_fields > config.respiratory_rate_channel:
        medibus[config.respiratory_rate_channel] = respiratory_rate_bpm
    if n_medibus_fields > config.fio2_channel:
        medibus[config.fio2_channel] = config.fio2_percent
    return medibus


def generate_event_series(
    time_seconds: np.ndarray,
    config: EventSeriesConfig,
) -> tuple[np.ndarray, list[str]]:
    """Create monotonically nondecreasing event markers and sparse text."""

    event_markers = np.zeros(len(time_seconds), dtype=DEFAULT_INT32_DTYPE)
    event_texts = [DEFAULT_EMPTY_STRING] * len(time_seconds)
    midpoint_index = int(len(time_seconds) * config.midpoint_fraction)
    midpoint_index = min(max(midpoint_index, int(DEFAULT_ZERO)), len(time_seconds) - 1)
    events_to_insert = [
        (config.start_time_seconds, config.start_marker, config.start_text),
        (
            time_seconds[midpoint_index],
            config.midpoint_marker,
            config.midpoint_text,
        ),
    ]
    current_marker = DEFAULT_NO_EXTREMA_FLAG
    event_index = int(DEFAULT_ZERO)
    for index, time_value in enumerate(time_seconds):
        while (
            event_index < len(events_to_insert)
            and time_value >= events_to_insert[event_index][0]
        ):
            current_marker = events_to_insert[event_index][1]
            event_texts[index] = events_to_insert[event_index][2]
            event_index += int(DEFAULT_ONE)
        event_markers[index] = current_marker
    return event_markers, event_texts
