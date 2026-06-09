"""Config-to-adapter kwargs for YAML-configured workflows."""

from __future__ import annotations

from typing import Any

from m3resp.core.config import WorkflowConfig


def eit_preprocess_kwargs(cfg: WorkflowConfig) -> dict[str, Any]:
    """Return EIT preprocessing kwargs from workflow config."""

    processing = cfg.eit.processing
    filter_config = processing.filter
    outputs = processing.outputs
    if not outputs.breath_intervals and (
        outputs.continuous_tiv or outputs.eeli or outputs.pixel_tiv
    ):
        raise ValueError(
            "EIT TIV, EELI, and pixel TIV require "
            "eit.processing.outputs.breath_intervals: true."
        )
    return {
        "subject_type": processing.subject_type,
        "welch_window_seconds": processing.welch_window_seconds,
        "filter_enabled": filter_config.enabled,
        "filter_mode": filter_config.mode,
        "lowpass_hz": filter_config.lowpass_hz,
        "highpass_hz": filter_config.highpass_hz,
        "filter_order": filter_config.order,
        "breath_min_duration_seconds": processing.breath_min_duration_seconds,
        "compute_rates": outputs.rates,
        "compute_breath_intervals": outputs.breath_intervals,
        "compute_continuous_tiv": outputs.continuous_tiv,
        "compute_eeli": outputs.eeli,
        "compute_pixel_tiv": outputs.pixel_tiv,
        "include_filtered_data": outputs.filtered_data,
        "include_global_impedance": outputs.global_impedance,
    }


def emg_preprocess_kwargs(cfg: WorkflowConfig) -> dict[str, Any]:
    """Return EMG preprocessing kwargs from workflow config."""

    preprocess = cfg.emg.processing.preprocess
    return {
        "channel": preprocess.channel,
        "high_pass_hz": preprocess.high_pass_hz,
        "low_pass_hz": preprocess.low_pass_hz,
        "envelope_window_seconds": preprocess.envelope_window_seconds,
    }


def emg_detection_kwargs(cfg: WorkflowConfig) -> dict[str, Any]:
    """Return EMG breath detection kwargs from workflow config."""

    detection = cfg.emg.processing.breath_detection
    return {
        "min_breath_width_seconds": detection.min_breath_width_seconds,
        "half_window_seconds": detection.half_window_seconds,
    }


def emg_postprocessing_kwargs(
    cfg: WorkflowConfig,
    *,
    ventilator: Any | None,
) -> dict[str, Any]:
    """Return EMG postprocessing kwargs from workflow config."""

    postprocessing = cfg.emg.processing.postprocessing
    functions = postprocessing.functions
    _validate_emg_postprocessing_functions(
        {
            "baseline": functions.baseline,
            "event_detection": functions.event_detection,
            "features": functions.features,
            "quality_assessment": functions.quality_assessment,
        }
    )
    return {
        "ventilator": ventilator,
        "ventilator_pressure_channel": postprocessing.ventilator_pressure_channel,
        "ventilator_flow_channel": postprocessing.ventilator_flow_channel,
        "ventilator_volume_channel": postprocessing.ventilator_volume_channel,
        "ventilator_fs": postprocessing.ventilator_fs,
        "ventilator_breath_width_seconds": (
            postprocessing.ventilator_breath_width_seconds
        ),
        "peep": postprocessing.peep,
        "baseline_window_seconds": postprocessing.baseline_window_seconds,
        "baseline_step_seconds": postprocessing.baseline_step_seconds,
        "baseline_percentile": postprocessing.baseline_percentile,
        "slope_window_seconds": postprocessing.slope_window_seconds,
        "aub_window_seconds": postprocessing.aub_window_seconds,
        "selected_functions": {
            "baseline": functions.baseline,
            "event_detection": functions.event_detection,
            "features": functions.features,
            "quality_assessment": functions.quality_assessment,
        },
    }


def _validate_emg_postprocessing_functions(
    selected_functions: dict[str, dict[str, bool]],
) -> None:
    baseline = selected_functions["baseline"]
    event_detection = selected_functions["event_detection"]
    features = selected_functions["features"]
    quality = selected_functions["quality_assessment"]

    baseline_selected = any(baseline.values())
    needs_baseline = any(
        [
            event_detection["onoffpeak_baseline_crossing"],
            features["amplitude"],
            features["time_product"],
            features["area_under_baseline"],
            quality["snr_pseudo"],
            quality["percentage_under_baseline"],
        ]
    )
    if needs_baseline and not baseline_selected:
        raise ValueError(
            "Selected EMG postprocessing functions require at least one baseline "
            "function. Enable emg.processing.postprocessing.functions.baseline."
        )

    needs_onoff_crossing = any(
        [
            features["time_to_peak"],
            features["pseudo_slope"],
            features["time_product"],
            features["area_under_baseline"],
            quality["percentage_under_baseline"],
            quality["evaluate_bell_curve_error"],
        ]
    )
    if needs_onoff_crossing and not event_detection["onoffpeak_baseline_crossing"]:
        raise ValueError(
            "Selected EMG feature/quality functions require "
            "event_detection.onoffpeak_baseline_crossing."
        )

    if quality["detect_local_high_aub"] and not features["area_under_baseline"]:
        raise ValueError(
            "quality_assessment.detect_local_high_aub requires "
            "features.area_under_baseline."
        )
    if (
        quality["detect_extreme_time_products"] or quality["evaluate_bell_curve_error"]
    ) and not features["time_product"]:
        raise ValueError(
            "Selected EMG quality functions require features.time_product."
        )
