"""Typed configuration surface for synthetic respiratory data generation."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

DEFAULT_OUTPUT_DIR = os.path.join("data", "source")
DEFAULT_BASENAME = "synthetic"
DEFAULT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
DEFAULT_TIMESTAMP_SUFFIX_WIDTH = 2
DEFAULT_SECONDS_PER_DAY = 24.0 * 60.0 * 60.0
DEFAULT_DRAEGER_EVENT_TEXT_LENGTH = 30
DEFAULT_CSV_TIME_LABEL = "time_seconds"
DEFAULT_METADATA_INDENT = 2
DEFAULT_LUNG_TEMPLATE_SIZE = 32
DEFAULT_NATIVE_EXTENSION = ".Poly5"


@dataclass
class DriftConfig:
    """Optional low-frequency baseline drift configuration."""

    enabled: bool = False
    amplitude: float = 0.0
    kind: str = "sinusoidal"
    frequency_hz: float = 0.008
    secondary_frequency_hz: float = 0.003
    phase_radians: float = 1.2
    slope_per_second: float = 0.0
    primary_weight: float = 0.6
    secondary_weight: float = 0.4


@dataclass
class TimingDriftConfig:
    """Optional modality-specific temporal drift configuration."""

    enabled: bool = False
    time_shift_seconds: float = 0.0


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
    drift: DriftConfig = field(default_factory=DriftConfig)
    timing_drift: TimingDriftConfig | None = None


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
    timing_drift: TimingDriftConfig | None = None


@dataclass
class VentilatorGeneratorConfig:
    """Synthetic ventilator settings passed to ReSurfEMG."""

    sample_frequency_hz: float = 100.0
    driving_pressure_cm_h2o: float = 8.0
    muscle_pressure_amplitude_cm_h2o: float = 5.0
    timing_drift: TimingDriftConfig | None = None


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
    if "drift" in values:
        values["drift"] = _nested_dataclass(DriftConfig, values["drift"])
    if "timing_drift" in values:
        values["timing_drift"] = _nested_dataclass(
            TimingDriftConfig,
            values["timing_drift"],
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
