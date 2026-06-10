"""Unified synthetic EIT, EMG, and ventilator data generator.

This module is intentionally examples-only. It provides a single configuration
surface for generating reproducible synthetic respiratory datasets without
adding public API to :mod:`m3resp`.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import struct
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any

import numpy as np


FORMAT_SPECS = {
    "original": {
        "frame_size": 4358,
        "n_medibus_fields": 52,
    },
    "pressure_pod": {
        "frame_size": 4382,
        "n_medibus_fields": 58,
    },
}
DEFAULT_GENERATOR_MODULE = (
    "notebooks.examples.synthetic_data_generators.unified_generator"
)
DEFAULT_CONFIG_FILENAME = "synthetic_generator_config.yaml"
DEFAULT_OUTPUT_DIR = os.path.join("data", "source")
DEFAULT_BASENAME = "synthetic"
DEFAULT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
DEFAULT_TIMESTAMP_SUFFIX_WIDTH = 2
DEFAULT_SECONDS_PER_DAY = 24.0 * 60.0 * 60.0
DEFAULT_DRAEGER_EVENT_TEXT_LENGTH = 30
DEFAULT_DRAEGER_PIXEL_SHAPE = (32, 32)
DEFAULT_NATIVE_EXTENSION = ".Poly5"
DEFAULT_CSV_TIME_LABEL = "time_seconds"
DEFAULT_EIT_LABELS = ("pixel_impedance",)
DEFAULT_EIT_UNITS = ("a.u.",)
DEFAULT_EMG_UNIT = "uV"
DEFAULT_VENTILATOR_LABELS = ("pressure", "flow", "volume")
DEFAULT_VENTILATOR_UNITS = ("cmH2O", "L/s", "L")
DEFAULT_EIT_COMPONENT_LABELS = ("baseline", "breathing", "drift", "cardiac", "noise")
DEFAULT_RETRY_ECG_ACCELERATION = 1.0
DEFAULT_RANDOM_HEART_RATE_MIN_BPM = 60
DEFAULT_RANDOM_HEART_RATE_MAX_BPM = 100
DEFAULT_LUNG_TEMPLATE_SIZE = 32
DEFAULT_FLOAT32_DTYPE = np.float32
DEFAULT_FLOAT64_DTYPE = np.float64
DEFAULT_INT32_DTYPE = np.int32
DEFAULT_LITTLE_ENDIAN_FLOAT32 = "<f4"
DEFAULT_LITTLE_ENDIAN_FLOAT64_PACK = "<d"
DEFAULT_LITTLE_ENDIAN_FLOAT32_PACK = "<f"
DEFAULT_LITTLE_ENDIAN_INT32_PACK = "<i"
DEFAULT_POLY5_MAGIC = b"POLY SAMPLE FILEversion 2.03\r\n\x1a"
DEFAULT_POLY5_VERSION = 203
DEFAULT_POLY5_HEADER_FORMAT = "=31sH81phhBHi4xHHHHHHHiHHH64x"
DEFAULT_POLY5_CHANNEL_FORMAT = "=41p4x11pffffH62x"
DEFAULT_POLY5_HEADER_BYTES = 217
DEFAULT_POLY5_CHANNEL_BYTES = 136
DEFAULT_POLY5_SIGNAL_BLOCK_HEADER_BYTES = 86
DEFAULT_POLY5_SAMPLES_PER_BLOCK = 256
DEFAULT_METADATA_INDENT = 2
DEFAULT_SECONDS_PER_MINUTE = 60.0
DEFAULT_ZERO = 0.0
DEFAULT_ONE = 1.0
DEFAULT_TWO = 2.0
DEFAULT_EMPTY_STRING = ""
DEFAULT_COMPONENT_BASELINE_INDEX = 0
DEFAULT_COMPONENT_BREATHING_INDEX = 1
DEFAULT_COMPONENT_DRIFT_INDEX = 2
DEFAULT_COMPONENT_CARDIAC_INDEX = 3
DEFAULT_COMPONENT_NOISE_INDEX = 4
DEFAULT_MIN_SAMPLES_FOR_EXTREMA = 3
DEFAULT_LOCAL_MAX_FLAG = 1
DEFAULT_LOCAL_MIN_FLAG = -1
DEFAULT_NO_EXTREMA_FLAG = 0
DEFAULT_GLOBAL_IMPEDANCE_AXES = (1, 2)
DEFAULT_GLOBAL_IMPEDANCE_LABEL = "global_impedance"
DEFAULT_ROW_VECTOR_SHAPE = (1, -1)
DEFAULT_ONE_DIMENSION = 1
DEFAULT_TWO_DIMENSIONS = 2
DEFAULT_FIRST_AXIS = 0
DEFAULT_SECOND_AXIS = 1


@dataclass
class DriftConfig:
    """Optional low-frequency baseline drift configuration."""

    enabled: bool = False
    amplitude: float = 0.0
    kind: str = "sinusoidal"
    time_shift_seconds: float = 0.0
    time_shift_fill_mode: str = "edge"
    frequency_hz: float = 0.008
    secondary_frequency_hz: float = 0.003
    phase_radians: float = 1.2
    slope_per_second: float = 0.0
    primary_weight: float = 0.6
    secondary_weight: float = 0.4


@dataclass
class EITWaveformConfig:
    """Detailed EIT waveform-shape parameters."""

    min_respiratory_rate_bpm: float = 6.0
    tidal_amplitude_variation_fraction: float = 0.12
    min_tidal_amplitude_au: float = 0.3
    inhale_fraction_variation: float = 0.04
    min_inhale_fraction: float = 0.25
    max_inhale_fraction: float = 0.55
    breath_wave_scale: float = 0.5
    cardiac_primary_weight: float = 0.75
    cardiac_harmonic_weight: float = 0.25
    cardiac_harmonic_multiplier: float = 2.0
    cardiac_harmonic_phase_radians: float = 0.5
    pixel_noise_fraction: float = 0.003


@dataclass
class LungTemplateConfig:
    """Shape parameters for the synthetic EIT lung template."""

    nx: int = DEFAULT_LUNG_TEMPLATE_SIZE
    ny: int = DEFAULT_LUNG_TEMPLATE_SIZE
    left_center_x: float = -0.42
    right_center_x: float = 0.42
    center_y: float = 0.02
    lung_width: float = 0.28
    lung_height: float = 0.42
    center_penalty_weight: float = 0.35
    center_penalty_width: float = 0.16
    center_penalty_height: float = 0.7
    min_template_value: float = 0.0


@dataclass
class MedibusConfig:
    """Synthetic Medibus channel layout and plausible values."""

    pressure_channel: int = 0
    flow_channel: int = 1
    volume_channel: int = 2
    respiratory_rate_channel: int = 36
    fio2_channel: int = 44
    pressure_baseline_cm_h2o: float = 8.0
    pressure_amplitude_cm_h2o: float = 6.0
    volume_baseline_ml: float = 400.0
    volume_amplitude_ml: float = 150.0
    fio2_percent: float = 21.0


@dataclass
class EventSeriesConfig:
    """Synthetic event marker configuration."""

    start_time_seconds: float = 0.0
    start_marker: int = 1
    start_text: str = "start"
    midpoint_fraction: float = 0.5
    midpoint_marker: int = 2
    midpoint_text: str = "midpoint"


@dataclass
class ExportConfig:
    """Output naming and serialization settings."""

    timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT
    timestamp_suffix_width: int = DEFAULT_TIMESTAMP_SUFFIX_WIDTH
    csv_time_label: str = DEFAULT_CSV_TIME_LABEL
    native_extension: str = DEFAULT_NATIVE_EXTENSION
    draeger_event_text_length: int = DEFAULT_DRAEGER_EVENT_TEXT_LENGTH
    seconds_per_day: float = DEFAULT_SECONDS_PER_DAY
    metadata_indent: int = DEFAULT_METADATA_INDENT


@dataclass
class RespiratoryPatternConfig:
    """Shared respiratory timing and occlusion settings."""

    respiratory_rate_bpm: float = 14.0
    respiratory_rate_variation_bpm: float = 2.0
    ie_ratio: float = 0.5
    occlusion_times_seconds: tuple[float, ...] = ()


@dataclass
class EITGeneratorConfig:
    """Synthetic EIT signal and Draeger export settings."""

    sample_frequency_hz: float = 20.0
    format_name: str = "original"
    base_impedance_au: float = 10.0
    tidal_amplitude_au: float = 0.9
    cardiac_amplitude_au: float = 0.04
    heart_rate_bpm: float = 72.0
    noise_std_au: float = 0.02


@dataclass
class EMGGeneratorConfig:
    """Synthetic EMG settings passed to ReSurfEMG."""

    sample_frequency_hz: float = 2048.0
    channel_amplitudes_uv: tuple[float, ...] = (0.2, 5.0)
    drift_amplitude_uv: float = 100.0
    noise_amplitude_uv: float = 2.0
    heart_rate_bpm: float = 72.0
    ecg_acceleration: float = 1.6
    tau_mus_up_seconds: float = 0.3
    tau_mus_down_seconds: float = 0.3


@dataclass
class VentilatorGeneratorConfig:
    """Synthetic ventilator settings passed to ReSurfEMG."""

    sample_frequency_hz: float = 100.0
    driving_pressure_cm_h2o: float = 8.0
    muscle_pressure_amplitude_cm_h2o: float = 5.0


@dataclass
class SyntheticGeneratorConfig:
    """Top-level configuration for a synthetic multimodal dataset."""

    duration_seconds: float = 60.0
    seed: int = 42
    output_dir: str = DEFAULT_OUTPUT_DIR
    basename: str = DEFAULT_BASENAME
    timestamp_output_dir: bool = True
    generate_eit: bool = True
    generate_emg: bool = True
    generate_ventilator: bool = True
    write_native_outputs: bool = False
    respiratory: RespiratoryPatternConfig = field(
        default_factory=RespiratoryPatternConfig
    )
    drift: DriftConfig = field(default_factory=DriftConfig)
    eit: EITGeneratorConfig = field(default_factory=EITGeneratorConfig)
    emg: EMGGeneratorConfig = field(default_factory=EMGGeneratorConfig)
    ventilator: VentilatorGeneratorConfig = field(
        default_factory=VentilatorGeneratorConfig
    )
    eit_waveform: EITWaveformConfig = field(default_factory=EITWaveformConfig)
    lung_template: LungTemplateConfig = field(default_factory=LungTemplateConfig)
    medibus: MedibusConfig = field(default_factory=MedibusConfig)
    events: EventSeriesConfig = field(default_factory=EventSeriesConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


@dataclass
class SyntheticRecord:
    """Generated signal plus metadata and export paths."""

    time: np.ndarray
    array: np.ndarray
    sample_frequency: float
    labels: list[str]
    units: list[str]
    paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticDataset:
    """Generated multimodal dataset."""

    eit: SyntheticRecord | None = None
    emg: SyntheticRecord | None = None
    ventilator: SyntheticRecord | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticDraegerData:
    """Arrays needed for a parser-compatible Draeger binary file."""

    format_name: str
    frame_size: int
    sample_frequency: float
    time: np.ndarray
    pixel_impedance: np.ndarray
    medibus_data: np.ndarray
    min_max_flags: np.ndarray
    event_markers: np.ndarray
    event_texts: list[str]
    timing_errors: np.ndarray
    unused_float32: np.ndarray
    components: dict[str, np.ndarray] = field(default_factory=dict)


def load_synthetic_generator_config(path: str) -> SyntheticGeneratorConfig:
    """Load a synthetic generator configuration from a YAML file."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML configuration requires `PyYAML`. Install the project "
            "dependencies or construct SyntheticGeneratorConfig directly."
        ) from exc

    config_path = os.path.abspath(os.path.normpath(path))
    with open(config_path, encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}

    if not isinstance(data, dict):
        raise ValueError("Synthetic generator YAML must contain a mapping.")

    config = synthetic_generator_config_from_dict(data)
    if not os.path.isabs(config.output_dir):
        config.output_dir = os.path.normpath(
            os.path.join(os.path.dirname(config_path), config.output_dir)
        )
    return config


def synthetic_generator_config_from_dict(
    data: dict[str, Any],
) -> SyntheticGeneratorConfig:
    """Create a typed generator config from a dictionary."""

    unknown = set(data) - _field_names(SyntheticGeneratorConfig)
    if unknown:
        raise ValueError(f"Unknown synthetic generator config keys: {sorted(unknown)}")

    values = dict(data)
    values["respiratory"] = _nested_dataclass(
        RespiratoryPatternConfig,
        values.get("respiratory", {}),
    )
    values["drift"] = _nested_dataclass(DriftConfig, values.get("drift", {}))
    values["eit"] = _nested_dataclass(EITGeneratorConfig, values.get("eit", {}))
    values["emg"] = _nested_dataclass(EMGGeneratorConfig, values.get("emg", {}))
    values["ventilator"] = _nested_dataclass(
        VentilatorGeneratorConfig,
        values.get("ventilator", {}),
    )
    values["eit_waveform"] = _nested_dataclass(
        EITWaveformConfig,
        values.get("eit_waveform", {}),
    )
    values["lung_template"] = _nested_dataclass(
        LungTemplateConfig,
        values.get("lung_template", {}),
    )
    values["medibus"] = _nested_dataclass(MedibusConfig, values.get("medibus", {}))
    values["events"] = _nested_dataclass(EventSeriesConfig, values.get("events", {}))
    values["export"] = _nested_dataclass(ExportConfig, values.get("export", {}))
    return SyntheticGeneratorConfig(**values)


def generate_synthetic_dataset(config: SyntheticGeneratorConfig) -> SyntheticDataset:
    """Generate synthetic EIT, EMG, and ventilator data from one configuration."""

    output_root = os.path.abspath(os.path.normpath(config.output_dir))
    output_dir = _resolve_run_output_dir(
        output_root=output_root,
        timestamp_output_dir=config.timestamp_output_dir,
        export_config=config.export,
    )
    os.makedirs(output_dir, exist_ok=True)

    provenance = {
        "generator": DEFAULT_GENERATOR_MODULE,
        "config": _config_to_dict(config),
        "output_root": output_root,
        "output_dir": output_dir,
    }
    dataset = SyntheticDataset(provenance=provenance)

    if config.generate_eit:
        dataset.eit = generate_eit_record(config, output_dir)

    if config.generate_emg:
        dataset.emg = generate_emg_record(config, output_dir)

    if config.generate_ventilator:
        dataset.ventilator = generate_ventilator_record(config, output_dir)

    metadata_path = os.path.join(output_dir, f"{config.basename}_metadata.json")
    _write_json(metadata_path, _dataset_metadata(dataset), config.export)
    provenance["metadata_path"] = metadata_path
    return dataset


def generate_eit_record(
    config: SyntheticGeneratorConfig,
    output_dir: str,
) -> SyntheticRecord:
    """Generate EIT waveform, 32x32 frames, and portable plus Draeger outputs."""

    draeger = generate_synthetic_draeger_data(
        config=config,
        duration=config.duration_seconds,
        fs=config.eit.sample_frequency_hz,
        seed=config.seed,
    )
    global_impedance = draeger.pixel_impedance.sum(
        axis=DEFAULT_GLOBAL_IMPEDANCE_AXES
    ).astype(DEFAULT_FLOAT32_DTYPE)

    npy_path = os.path.join(output_dir, f"{config.basename}_eit_pixels.npy")
    csv_path = os.path.join(output_dir, f"{config.basename}_eit_global.csv")
    bin_path = os.path.join(output_dir, f"{config.basename}_eit_draeger.bin")
    components_path = os.path.join(
        output_dir,
        f"{config.basename}_eit_components.npz",
    )

    np.save(npy_path, draeger.pixel_impedance)
    np.savez(components_path, **draeger.components)
    _write_timeseries_csv(
        csv_path,
        draeger.time,
        global_impedance.reshape(DEFAULT_ROW_VECTOR_SHAPE),
        [DEFAULT_GLOBAL_IMPEDANCE_LABEL],
        config.export,
    )
    save_draeger_bin(bin_path, draeger, config.export)

    return SyntheticRecord(
        time=draeger.time,
        array=draeger.pixel_impedance,
        sample_frequency=draeger.sample_frequency,
        labels=list(DEFAULT_EIT_LABELS),
        units=list(DEFAULT_EIT_UNITS),
        paths={
            "npy": npy_path,
            "csv": csv_path,
            "components_npz": components_path,
            "native": bin_path,
        },
        metadata={
            "modality": "eit",
            "vendor": "draeger",
            "format_name": draeger.format_name,
            "frame_size_bytes": draeger.frame_size,
            "time_units": "seconds",
            "signal_units": "relative impedance (a.u.)",
            "component_labels": sorted(draeger.components),
        },
    )


def generate_emg_record(
    config: SyntheticGeneratorConfig,
    output_dir: str,
) -> SyntheticRecord:
    """Generate EMG data with ReSurfEMG and write portable exports."""

    synth = _load_resurfemg_synthetic()
    rng = np.random.default_rng(config.seed)
    fs_emg = int(round(config.emg.sample_frequency_hz))
    expected_samples = int(round(config.duration_seconds * fs_emg))
    channels = []
    for amplitude_uv in config.emg.channel_amplitudes_uv:
        heart_rate_bpm = int(round(config.emg.heart_rate_bpm))
        if config.emg.heart_rate_bpm <= 0:
            heart_rate_bpm = int(
                rng.integers(
                    DEFAULT_RANDOM_HEART_RATE_MIN_BPM,
                    DEFAULT_RANDOM_HEART_RATE_MAX_BPM + 1,
                )
            )
        signal = _simulate_raw_emg_length_safe(
            synth=synth,
            expected_samples=expected_samples,
            t_p_occs=np.asarray(
                config.respiratory.occlusion_times_seconds,
                dtype=float,
            ),
            t_end=config.duration_seconds,
            fs_emg=fs_emg,
            rr=config.respiratory.respiratory_rate_bpm,
            ie_ratio=config.respiratory.ie_ratio,
            tau_mus_up=config.emg.tau_mus_up_seconds,
            tau_mus_down=config.emg.tau_mus_down_seconds,
            emg_amp=amplitude_uv,
            drift_amp=config.emg.drift_amplitude_uv,
            noise_amp=config.emg.noise_amplitude_uv,
            heart_rate=heart_rate_bpm,
            ecg_acceleration=config.emg.ecg_acceleration,
        )
        channels.append(signal)

    array = np.asarray(channels, dtype=np.float32)
    time = np.arange(array.shape[1], dtype=float) / fs_emg
    labels = [f"emg_{index}" for index in range(array.shape[0])]
    paths = _write_record_exports(
        output_dir=output_dir,
        basename=f"{config.basename}_emg",
        time=time,
        array=array,
        labels=labels,
        export_config=config.export,
    )

    if config.write_native_outputs:
        native_path = _write_native_resurfemg_output(
            synth=synth,
            output_dir=output_dir,
            basename=f"{config.basename}_emg",
            kind="emg",
            time=time,
            array=array,
            sample_frequency=fs_emg,
            labels=labels,
            units=[DEFAULT_EMG_UNIT] * len(labels),
            export_config=config.export,
        )
        if native_path is not None:
            paths["native"] = native_path

    return SyntheticRecord(
        time=time,
        array=array,
        sample_frequency=fs_emg,
        labels=labels,
        units=[DEFAULT_EMG_UNIT] * len(labels),
        paths=paths,
        metadata={
            "modality": "emg",
            "time_units": "seconds",
            "signal_units": DEFAULT_EMG_UNIT,
        },
    )


def generate_ventilator_record(
    config: SyntheticGeneratorConfig,
    output_dir: str,
) -> SyntheticRecord:
    """Generate ventilator data with ReSurfEMG and write portable exports."""

    synth = _load_resurfemg_synthetic()
    fs_vent = int(round(config.ventilator.sample_frequency_hz))
    y_vent, p_mus = synth.simulate_ventilator_data(
        t_end=config.duration_seconds,
        fs_vent=fs_vent,
        p_mus_amp=config.ventilator.muscle_pressure_amplitude_cm_h2o,
        rr=config.respiratory.respiratory_rate_bpm,
        dp=config.ventilator.driving_pressure_cm_h2o,
        t_p_occs=np.asarray(config.respiratory.occlusion_times_seconds, dtype=float),
    )

    array = np.asarray(y_vent, dtype=np.float32)
    p_mus = np.asarray(p_mus, dtype=np.float32)
    time = np.arange(array.shape[1], dtype=float) / fs_vent
    labels = list(DEFAULT_VENTILATOR_LABELS)
    paths = _write_record_exports(
        output_dir=output_dir,
        basename=f"{config.basename}_ventilator",
        time=time,
        array=array,
        labels=labels,
        export_config=config.export,
    )
    p_mus_path = os.path.join(output_dir, f"{config.basename}_ventilator_p_mus.npy")
    np.save(p_mus_path, p_mus)
    paths["p_mus_npy"] = p_mus_path

    if config.write_native_outputs:
        native_path = _write_native_resurfemg_output(
            synth=synth,
            output_dir=output_dir,
            basename=f"{config.basename}_ventilator",
            kind="ventilator",
            time=time,
            array=array,
            sample_frequency=fs_vent,
            labels=labels,
            units=list(DEFAULT_VENTILATOR_UNITS),
            export_config=config.export,
        )
        if native_path is not None:
            paths["native"] = native_path

    return SyntheticRecord(
        time=time,
        array=array,
        sample_frequency=fs_vent,
        labels=labels,
        units=list(DEFAULT_VENTILATOR_UNITS),
        paths=paths,
        metadata={
            "modality": "ventilator",
            "time_units": "seconds",
            "p_mus_npy": p_mus_path,
        },
    )


def generate_synthetic_draeger_data(
    config: SyntheticGeneratorConfig,
    duration: float,
    fs: float,
    seed: int,
) -> SyntheticDraegerData:
    """Generate all arrays needed for a parser-compatible Draeger file."""

    if config.eit.format_name not in FORMAT_SPECS:
        raise ValueError(f"Unsupported format_name: {config.eit.format_name}")

    spec = FORMAT_SPECS[config.eit.format_name]
    time_seconds, signal, components = generate_realistic_eit_signal(
        config=config,
        duration=duration,
        fs=fs,
        seed=seed,
    )
    pixel_impedance = signal_to_pixel_impedance(
        signal,
        seed=seed + 1,
        waveform_config=config.eit_waveform,
        template_config=config.lung_template,
    )
    medibus_data = generate_medibus_data(
        time_seconds,
        signal,
        n_medibus_fields=spec["n_medibus_fields"],
        respiratory_rate_bpm=config.respiratory.respiratory_rate_bpm,
        config=config.medibus,
    )
    min_max_flags = detect_min_max_flags(signal)
    event_markers, event_texts = generate_event_series(time_seconds, config.events)

    data = SyntheticDraegerData(
        format_name=config.eit.format_name,
        frame_size=spec["frame_size"],
        sample_frequency=fs,
        time=time_seconds.astype(DEFAULT_FLOAT64_DTYPE),
        pixel_impedance=pixel_impedance.astype(DEFAULT_FLOAT32_DTYPE),
        medibus_data=medibus_data.astype(DEFAULT_FLOAT32_DTYPE),
        min_max_flags=min_max_flags,
        event_markers=event_markers,
        event_texts=event_texts,
        timing_errors=np.zeros(len(time_seconds), dtype=DEFAULT_INT32_DTYPE),
        unused_float32=np.zeros(len(time_seconds), dtype=DEFAULT_FLOAT32_DTYPE),
        components=components,
    )
    return data


def generate_realistic_eit_signal(
    config: SyntheticGeneratorConfig,
    duration: float,
    fs: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Generate a realistic synthetic breathing-related impedance waveform."""

    rng = np.random.default_rng(seed)
    time_seconds = np.arange(0, duration, 1 / fs)
    drift = generate_drift(time_seconds, config.drift)
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
    if is_time_shift_drift(config.drift):
        shifted_signal = shift_signal_in_time(signal, time_seconds, config.drift)
        drift = shifted_signal - signal
        signal = shifted_signal
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


def generate_drift(time_seconds: np.ndarray, config: DriftConfig) -> np.ndarray:
    """Generate a named drift component for any time series."""

    if is_time_shift_drift(config):
        return np.zeros_like(time_seconds, dtype=float)

    if not config.enabled or config.amplitude == DEFAULT_ZERO:
        return np.zeros_like(time_seconds, dtype=float)

    if config.kind == "sinusoidal":
        return config.amplitude * config.primary_weight * np.sin(
            2 * np.pi * config.frequency_hz * time_seconds
        ) + config.amplitude * config.secondary_weight * np.sin(
            DEFAULT_TWO * np.pi * config.secondary_frequency_hz * time_seconds
            + config.phase_radians
        )
    if config.kind == "linear":
        centered_time = time_seconds - float(time_seconds[0])
        slope = config.slope_per_second or config.amplitude / max(
            float(time_seconds[-1] - time_seconds[0]),
            DEFAULT_ONE,
        )
        return slope * centered_time
    if config.kind == "constant":
        return np.full_like(time_seconds, config.amplitude, dtype=float)

    raise ValueError(
        "drift.kind must be one of: sinusoidal, linear, constant, time_shift"
    )


def is_time_shift_drift(config: DriftConfig) -> bool:
    """Return whether drift should be modeled as a signal time shift."""

    return config.enabled and config.kind == "time_shift"


def shift_signal_in_time(
    signal: np.ndarray,
    time_seconds: np.ndarray,
    config: DriftConfig,
) -> np.ndarray:
    """Shift a signal in time using interpolation on the original time grid."""

    if signal.shape != time_seconds.shape:
        raise ValueError("signal and time_seconds must have the same shape")
    if len(time_seconds) == int(DEFAULT_ZERO):
        return np.asarray(signal, dtype=float)

    shift_seconds = float(config.time_shift_seconds)
    if shift_seconds == DEFAULT_ZERO:
        return np.asarray(signal, dtype=float).copy()

    sample_times = time_seconds - shift_seconds
    fill_mode = config.time_shift_fill_mode
    if fill_mode == "edge":
        left = float(signal[0])
        right = float(signal[-1])
    elif fill_mode == "zero":
        left = DEFAULT_ZERO
        right = DEFAULT_ZERO
    else:
        raise ValueError("drift.time_shift_fill_mode must be one of: edge, zero")

    return np.interp(sample_times, time_seconds, signal, left=left, right=right)


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


def pack_draeger_frame(
    time_seconds: float,
    pixel_frame: np.ndarray,
    min_max_flag: int,
    event_marker: int,
    event_text: str,
    timing_error: int,
    medibus_values: np.ndarray,
    export_config: ExportConfig,
    unused_float32: float = DEFAULT_ZERO,
) -> bytes:
    """Pack one Draeger frame exactly as expected by the Draeger loader."""

    pixel_frame = np.asarray(pixel_frame, dtype=DEFAULT_FLOAT32_DTYPE)
    if pixel_frame.shape != DEFAULT_DRAEGER_PIXEL_SHAPE:
        raise ValueError(f"pixel_frame must have shape {DEFAULT_DRAEGER_PIXEL_SHAPE}")

    time_fraction_of_day = float(time_seconds) / export_config.seconds_per_day
    return b"".join(
        [
            struct.pack(DEFAULT_LITTLE_ENDIAN_FLOAT64_PACK, time_fraction_of_day),
            struct.pack(DEFAULT_LITTLE_ENDIAN_FLOAT32_PACK, float(unused_float32)),
            np.asarray(pixel_frame, dtype=DEFAULT_LITTLE_ENDIAN_FLOAT32)
            .reshape(-1, order="C")
            .tobytes(),
            struct.pack(DEFAULT_LITTLE_ENDIAN_INT32_PACK, int(min_max_flag)),
            struct.pack(DEFAULT_LITTLE_ENDIAN_INT32_PACK, int(event_marker)),
            _encode_event_text(
                event_text,
                length=export_config.draeger_event_text_length,
            ),
            struct.pack(DEFAULT_LITTLE_ENDIAN_INT32_PACK, int(timing_error)),
            np.asarray(medibus_values, dtype=DEFAULT_LITTLE_ENDIAN_FLOAT32)
            .reshape(-1)
            .tobytes(),
        ]
    )


def save_draeger_bin(
    path: str,
    data: SyntheticDraegerData,
    export_config: ExportConfig,
) -> str:
    """Save a complete Draeger-style binary file."""

    n_frames = len(data.time)
    if data.pixel_impedance.shape != (n_frames, *DEFAULT_DRAEGER_PIXEL_SHAPE):
        raise ValueError("pixel_impedance shape mismatch")
    if data.medibus_data.shape[1] != n_frames:
        raise ValueError("medibus_data shape mismatch")

    with open(path, "wb") as file_obj:
        for index in range(n_frames):
            frame_bytes = pack_draeger_frame(
                time_seconds=float(data.time[index]),
                pixel_frame=data.pixel_impedance[index],
                min_max_flag=int(data.min_max_flags[index]),
                event_marker=int(data.event_markers[index]),
                event_text=data.event_texts[index],
                timing_error=int(data.timing_errors[index]),
                medibus_values=data.medibus_data[:, index],
                export_config=export_config,
                unused_float32=float(data.unused_float32[index]),
            )
            if len(frame_bytes) != data.frame_size:
                raise ValueError(
                    f"Packed frame has {len(frame_bytes)} bytes, "
                    f"expected {data.frame_size}"
                )
            file_obj.write(frame_bytes)
    return path


def _write_record_exports(
    output_dir: str,
    basename: str,
    time: np.ndarray,
    array: np.ndarray,
    labels: list[str],
    export_config: ExportConfig,
) -> dict[str, str]:
    npy_path = os.path.join(output_dir, f"{basename}.npy")
    csv_path = os.path.join(output_dir, f"{basename}.csv")
    np.save(npy_path, array)
    _write_timeseries_csv(csv_path, time, array, labels, export_config)
    return {"npy": npy_path, "csv": csv_path}


def _write_timeseries_csv(
    path: str,
    time: np.ndarray,
    array: np.ndarray,
    labels: list[str],
    export_config: ExportConfig,
) -> None:
    values = np.asarray(array)
    if values.ndim == DEFAULT_ONE_DIMENSION:
        values = values.reshape(DEFAULT_ROW_VECTOR_SHAPE)
    if values.shape[DEFAULT_FIRST_AXIS] != len(labels):
        raise ValueError("labels length must match the first data dimension")
    if values.shape[DEFAULT_SECOND_AXIS] != len(time):
        raise ValueError("time length must match the second data dimension")

    with open(path, "w", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([export_config.csv_time_label, *labels])
        for index, time_value in enumerate(time):
            row_values = [float(value) for value in values[:, index]]
            writer.writerow([float(time_value), *row_values])


def _write_native_resurfemg_output(
    synth: Any,
    output_dir: str,
    basename: str,
    kind: str,
    time: np.ndarray,
    array: np.ndarray,
    sample_frequency: float,
    labels: list[str],
    units: list[str],
    export_config: ExportConfig,
) -> str | None:
    writer = getattr(synth, "write_synthetic_recording", None)

    native_path = os.path.join(
        output_dir,
        f"{basename}{export_config.native_extension}",
    )
    if writer is None:
        _write_poly5_file(
            path=native_path,
            array=array,
            sample_frequency=sample_frequency,
            labels=labels,
            units=units,
        )
        return native_path

    result = writer(
        path=native_path,
        kind=kind,
        time=time,
        array=array,
        sample_frequency=sample_frequency,
        labels=labels,
        units=units,
    )
    if result is not None:
        native_path = str(result)
    return native_path


def _write_poly5_file(
    path: str,
    array: np.ndarray,
    sample_frequency: float,
    labels: list[str],
    units: list[str],
) -> None:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == DEFAULT_ONE_DIMENSION:
        values = values.reshape(DEFAULT_ROW_VECTOR_SHAPE)
    if values.ndim != DEFAULT_TWO_DIMENSIONS:
        raise ValueError("Poly5 export requires a 2D channel-by-sample array")
    if values.shape[DEFAULT_FIRST_AXIS] != len(labels):
        raise ValueError("labels length must match the first data dimension")
    if values.shape[DEFAULT_FIRST_AXIS] != len(units):
        raise ValueError("units length must match the first data dimension")
    if values.shape[DEFAULT_SECOND_AXIS] <= 0:
        raise ValueError("Poly5 export requires at least one sample")

    sample_rate = _coerce_poly5_sample_rate(sample_frequency)
    num_channels = int(values.shape[DEFAULT_FIRST_AXIS])
    num_samples = int(values.shape[DEFAULT_SECOND_AXIS])
    samples_per_block = DEFAULT_POLY5_SAMPLES_PER_BLOCK
    num_data_blocks = int(np.ceil(num_samples / samples_per_block))
    now = datetime.now()

    header = struct.pack(
        DEFAULT_POLY5_HEADER_FORMAT,
        DEFAULT_POLY5_MAGIC,
        DEFAULT_POLY5_VERSION,
        b"Synthetic M3Resp recording",
        sample_rate,
        sample_rate,
        0,
        num_channels * 2,
        num_samples,
        now.year,
        now.month,
        now.day,
        now.weekday(),
        now.hour,
        now.minute,
        now.second,
        num_data_blocks,
        samples_per_block,
        0,
        0,
    )

    with open(path, "wb") as file_obj:
        file_obj.write(header)
        for label, unit in zip(labels, units):
            file_obj.write(_pack_poly5_channel_description(label, unit))
            file_obj.write(bytes(DEFAULT_POLY5_CHANNEL_BYTES))

        for block_index in range(num_data_blocks):
            start = block_index * samples_per_block
            stop = min(start + samples_per_block, num_samples)
            block = values[:, start:stop].T.astype(DEFAULT_LITTLE_ENDIAN_FLOAT32)
            file_obj.write(bytes(DEFAULT_POLY5_SIGNAL_BLOCK_HEADER_BYTES))
            file_obj.write(block.ravel().tobytes())


def _coerce_poly5_sample_rate(sample_frequency: float) -> int:
    sample_rate = int(round(float(sample_frequency)))
    if sample_rate <= 0:
        raise ValueError("Poly5 sample_frequency must be positive")
    if not np.isclose(float(sample_frequency), float(sample_rate)):
        raise ValueError("Poly5 sample_frequency must be an integer number of Hz")
    return sample_rate


def _pack_poly5_channel_description(label: str, unit: str) -> bytes:
    label_bytes = str(label).encode("ascii")
    unit_bytes = str(unit).encode("utf-8")
    if len(label_bytes) > 36:
        raise ValueError("Poly5 channel labels must be at most 36 ASCII bytes")
    if len(unit_bytes) > 10:
        raise ValueError("Poly5 channel units must be at most 10 UTF-8 bytes")
    return struct.pack(
        DEFAULT_POLY5_CHANNEL_FORMAT,
        b"     " + label_bytes,
        unit_bytes,
        0.0,
        0.0,
        1.0,
        0.0,
        0,
    )


def _load_resurfemg_synthetic() -> Any:
    try:
        return importlib.import_module("resurfemg.pipelines.synthetic_data")
    except ImportError as exc:
        raise RuntimeError(
            "EMG and ventilator synthetic generation requires the optional "
            "dependency `resurfemg`. Install m3resp with the EMG/all extra or "
            "disable generate_emg and generate_ventilator."
        ) from exc


def _resolve_run_output_dir(
    output_root: str,
    timestamp_output_dir: bool,
    export_config: ExportConfig,
) -> str:
    if not timestamp_output_dir:
        return output_root

    timestamp = datetime.now().strftime(export_config.timestamp_format)
    candidate = os.path.join(output_root, timestamp)
    suffix = int(DEFAULT_ONE)
    while os.path.exists(candidate):
        formatted_suffix = f"{suffix:0{export_config.timestamp_suffix_width}d}"
        candidate = os.path.join(output_root, f"{timestamp}_{formatted_suffix}")
        suffix += int(DEFAULT_ONE)
    return candidate


def _simulate_raw_emg_length_safe(
    synth: Any,
    expected_samples: int,
    **kwargs: Any,
) -> np.ndarray:
    """Call ReSurfEMG and guard against its ECG/EMG length mismatch."""

    try:
        signal = synth.simulate_raw_emg(**kwargs)
    except ValueError as exc:
        if not _is_resurfemg_length_mismatch(exc):
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs["ecg_acceleration"] = DEFAULT_RETRY_ECG_ACCELERATION
        signal = synth.simulate_raw_emg(**retry_kwargs)

    return _normalize_signal_length(signal, expected_samples)


def _is_resurfemg_length_mismatch(exc: ValueError) -> bool:
    message = str(exc)
    return "operands could not be broadcast together" in message


def _normalize_signal_length(signal: Any, expected_samples: int) -> np.ndarray:
    values = np.asarray(signal, dtype=DEFAULT_FLOAT32_DTYPE).reshape(-1)
    if len(values) == expected_samples:
        return values
    if len(values) > expected_samples:
        return values[:expected_samples]
    if len(values) == int(DEFAULT_ZERO):
        return np.zeros(expected_samples, dtype=DEFAULT_FLOAT32_DTYPE)
    return np.pad(
        values,
        (int(DEFAULT_ZERO), expected_samples - len(values)),
        mode="edge",
    )


def _encode_event_text(
    text: str,
    length: int = DEFAULT_DRAEGER_EVENT_TEXT_LENGTH,
) -> bytes:
    raw = text.encode("ascii", errors="replace")[:length]
    return raw.ljust(length, b"\x00")


def _nested_dataclass(cls: type, data: Any) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} configuration must be a mapping.")

    unknown = set(data) - _field_names(cls)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")

    values = dict(data)
    if cls is RespiratoryPatternConfig and "occlusion_times_seconds" in values:
        values["occlusion_times_seconds"] = tuple(
            float(value) for value in values["occlusion_times_seconds"]
        )
    if cls is EMGGeneratorConfig and "channel_amplitudes_uv" in values:
        values["channel_amplitudes_uv"] = tuple(
            float(value) for value in values["channel_amplitudes_uv"]
        )
    return cls(**values)


def _field_names(cls: type) -> set[str]:
    return {field.name for field in fields(cls)}


def _config_to_dict(config: SyntheticGeneratorConfig) -> dict[str, Any]:
    return asdict(config)


def _dataset_metadata(dataset: SyntheticDataset) -> dict[str, Any]:
    return {
        "provenance": dataset.provenance,
        "records": {
            "eit": _record_metadata(dataset.eit),
            "emg": _record_metadata(dataset.emg),
            "ventilator": _record_metadata(dataset.ventilator),
        },
    }


def _record_metadata(record: SyntheticRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "sample_frequency": record.sample_frequency,
        "labels": record.labels,
        "units": record.units,
        "paths": record.paths,
        "metadata": record.metadata,
        "n_samples": int(len(record.time)),
        "array_shape": list(record.array.shape),
    }


def _write_json(
    path: str,
    data: dict[str, Any],
    export_config: ExportConfig | None = None,
) -> None:
    indent = (
        DEFAULT_METADATA_INDENT
        if export_config is None
        else export_config.metadata_indent
    )
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=indent, sort_keys=True)


def main(config_path: str | None = None) -> None:
    """Generate a synthetic dataset from YAML configuration."""

    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            DEFAULT_CONFIG_FILENAME,
        )

    config = load_synthetic_generator_config(config_path)
    dataset = generate_synthetic_dataset(config)
    print(
        json.dumps(
            {
                "config_path": os.path.abspath(os.path.normpath(config_path)),
                **_dataset_metadata(dataset),
            },
            indent=config.export.metadata_indent,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > DEFAULT_ONE_DIMENSION else None)
