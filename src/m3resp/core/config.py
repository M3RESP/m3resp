"""Configuration models for M3Resp workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class M3Config:
    """Small Stage 1 configuration object."""

    time_unit: str = "seconds"
    default_alignment_method: str = "manual_offset"


@dataclass(frozen=True)
class ModuleFlags:
    """On/off switches for configured workflows."""

    eit: bool = True
    emg: bool = True
    vent: bool = True


@dataclass(frozen=True)
class EITWorkflowConfig:
    """EIT-specific configured workflow settings."""

    file: Path | None = None
    vendor: str = "draeger"
    processing: EITProcessingConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.processing is None:
            object.__setattr__(self, "processing", EITProcessingConfig())


@dataclass(frozen=True)
class EMGWorkflowConfig:
    """EMG-specific configured workflow settings."""

    file: Path | None = None
    processing: EMGProcessingConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.processing is None:
            object.__setattr__(self, "processing", EMGProcessingConfig())


@dataclass(frozen=True)
class VentWorkflowConfig:
    """Ventilator-specific configured workflow settings."""

    file: Path | None = None


@dataclass(frozen=True)
class AlignmentConfig:
    """Inter-modality alignment settings."""

    method: str = "manual_offset"
    manual_offset_seconds: float = 0.0
    offset_seconds: dict[str, float] | None = None
    reference_modality: str | None = None


@dataclass(frozen=True)
class OutputConfig:
    """Configured workflow output directories."""

    combined: Path = Path(os.path.join("output", "multimodal-summary"))
    eit_only: Path = Path(os.path.join("output", "eit-summary"))
    emg_only: Path = Path(os.path.join("output", "emg-summary"))


@dataclass(frozen=True)
class SwitchConfig:
    """Generic enabled switch."""

    enabled: bool = True


@dataclass(frozen=True)
class EITFilterConfig:
    """EIT filter settings."""

    enabled: bool = True
    mode: str = "mdn"
    lowpass_hz: float = 1.0
    highpass_hz: float = 0.05
    order: int = 4


@dataclass(frozen=True)
class EITOutputsConfig:
    """EIT output switches."""

    rates: bool = True
    breath_intervals: bool = True
    continuous_tiv: bool = True
    eeli: bool = True
    pixel_tiv: bool = True
    filtered_data: bool = True
    global_impedance: bool = True


@dataclass(frozen=True)
class EITProcessingConfig:
    """EIT processing switches and parameters."""

    preprocess: SwitchConfig = None  # type: ignore[assignment]
    filter: EITFilterConfig = None  # type: ignore[assignment]
    breath_detection: SwitchConfig = None  # type: ignore[assignment]
    outputs: EITOutputsConfig = None  # type: ignore[assignment]
    subject_type: str = "adult"
    welch_window_seconds: float = 30.0
    breath_min_duration_seconds: float = 2 / 3

    def __post_init__(self) -> None:
        if self.preprocess is None:
            object.__setattr__(self, "preprocess", SwitchConfig())
        if self.filter is None:
            object.__setattr__(self, "filter", EITFilterConfig())
        if self.breath_detection is None:
            object.__setattr__(self, "breath_detection", SwitchConfig())
        if self.outputs is None:
            object.__setattr__(self, "outputs", EITOutputsConfig())


@dataclass(frozen=True)
class EMGPreprocessConfig:
    """EMG preprocessing settings."""

    enabled: bool = True
    channel: int = 0
    high_pass_hz: float = 80
    low_pass_hz: float | None = None
    envelope_window_seconds: float = 0.5


@dataclass(frozen=True)
class EMGDetectionConfig:
    """EMG breath detection settings."""

    enabled: bool = True
    min_breath_width_seconds: float = 1.0
    half_window_seconds: float = 0.5


@dataclass(frozen=True)
class EMGPostprocessingGroupsConfig:
    """Switches for ReSurfEMG postprocessing function groups."""

    baseline: dict[str, bool] = None  # type: ignore[assignment]
    event_detection: dict[str, bool] = None  # type: ignore[assignment]
    features: dict[str, bool] = None  # type: ignore[assignment]
    quality_assessment: dict[str, bool] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        from m3resp.adapters.resurfemg_adapter import POSTPROCESSING_FUNCTIONS

        for category, functions in POSTPROCESSING_FUNCTIONS.items():
            value = getattr(self, category)
            if value is None:
                value = {}
            object.__setattr__(
                self,
                category,
                {
                    function_name: value.get(function_name, True)
                    for function_name in functions
                },
            )


@dataclass(frozen=True)
class EMGPostprocessingConfig:
    """EMG postprocessing switches and parameters."""

    enabled: bool = True
    functions: EMGPostprocessingGroupsConfig = None  # type: ignore[assignment]
    ventilator_pressure_channel: int = 0
    ventilator_flow_channel: int = 1
    ventilator_volume_channel: int = 2
    ventilator_fs: float | None = None
    ventilator_breath_width_seconds: float = 0.5
    peep: float | None = None
    baseline_window_seconds: float = 30.0
    baseline_step_seconds: float = 1.0
    baseline_percentile: float = 33.0
    slope_window_seconds: float = 0.5
    aub_window_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.functions is None:
            object.__setattr__(self, "functions", EMGPostprocessingGroupsConfig())


@dataclass(frozen=True)
class EMGProcessingConfig:
    """EMG processing switches and parameters."""

    preprocess: EMGPreprocessConfig = None  # type: ignore[assignment]
    breath_detection: EMGDetectionConfig = None  # type: ignore[assignment]
    postprocessing: EMGPostprocessingConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.preprocess is None:
            object.__setattr__(self, "preprocess", EMGPreprocessConfig())
        if self.breath_detection is None:
            object.__setattr__(self, "breath_detection", EMGDetectionConfig())
        if self.postprocessing is None:
            object.__setattr__(self, "postprocessing", EMGPostprocessingConfig())


@dataclass(frozen=True)
class ResultsConfig:
    """Configured workflow output artifact switches."""

    summary_json: bool = True
    parameters_csv: bool = True
    event_csvs: bool = True
    postprocessing: bool = True
    figures: bool = True


@dataclass(frozen=True)
class WorkflowConfig:
    """Typed, resolved YAML workflow configuration."""

    modules: ModuleFlags
    eit: EITWorkflowConfig
    emg: EMGWorkflowConfig
    vent: VentWorkflowConfig
    alignment: AlignmentConfig
    output: OutputConfig
    results: ResultsConfig


def load_workflow_config(
    config_path: str | Path,
    *,
    root: str | Path | None = None,
) -> WorkflowConfig:
    """Load a YAML workflow config and resolve paths to absolute paths."""

    path = Path(config_path).expanduser().resolve()
    base = Path(root).expanduser().resolve() if root is not None else _infer_root(path)

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    modules = ModuleFlags(**raw.get("modules", {}))
    eit_raw: dict[str, Any] = raw.get("eit", {})
    emg_raw: dict[str, Any] = raw.get("emg", {})
    vent_raw: dict[str, Any] = raw.get("vent", {})
    output_raw: dict[str, Any] = raw.get("output", {})
    results_raw: dict[str, Any] = raw.get("results", {})

    return WorkflowConfig(
        modules=modules,
        eit=EITWorkflowConfig(
            file=_resolve_optional_path(base, eit_raw.get("file")),
            vendor=eit_raw.get("vendor", "draeger"),
            processing=_load_eit_processing(eit_raw.get("processing", {})),
        ),
        emg=EMGWorkflowConfig(
            file=_resolve_optional_path(base, emg_raw.get("file")),
            processing=_load_emg_processing(emg_raw.get("processing", {})),
        ),
        vent=VentWorkflowConfig(
            file=_resolve_optional_path(base, vent_raw.get("file"))
        ),
        alignment=AlignmentConfig(**raw.get("alignment", {})),
        output=OutputConfig(
            **{
                key: _resolve_optional_path(base, value)
                for key, value in output_raw.items()
                if value is not None
            }
        ),
        results=ResultsConfig(**results_raw),
    )


def _infer_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (
            Path(os.path.join(candidate, "pyproject.toml")).exists()
            and Path(os.path.join(candidate, "src", "m3resp")).exists()
        ):
            return candidate
    return config_path.parent


def _resolve_optional_path(base: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(os.path.join(base, path)).resolve()


def _load_eit_processing(raw: dict[str, Any]) -> EITProcessingConfig:
    return EITProcessingConfig(
        preprocess=SwitchConfig(**raw.get("preprocess", {})),
        filter=EITFilterConfig(**raw.get("filter", {})),
        breath_detection=SwitchConfig(**raw.get("breath_detection", {})),
        outputs=EITOutputsConfig(**raw.get("outputs", {})),
        subject_type=raw.get("subject_type", "adult"),
        welch_window_seconds=raw.get("welch_window_seconds", 30.0),
        breath_min_duration_seconds=raw.get("breath_min_duration_seconds", 2 / 3),
    )


def _load_emg_processing(raw: dict[str, Any]) -> EMGProcessingConfig:
    post_raw = raw.get("postprocessing", {})
    functions_raw = post_raw.get("functions", {})
    post_kwargs = {key: value for key, value in post_raw.items() if key != "functions"}
    return EMGProcessingConfig(
        preprocess=EMGPreprocessConfig(**raw.get("preprocess", {})),
        breath_detection=EMGDetectionConfig(**raw.get("breath_detection", {})),
        postprocessing=EMGPostprocessingConfig(
            functions=EMGPostprocessingGroupsConfig(**functions_raw),
            **post_kwargs,
        ),
    )
