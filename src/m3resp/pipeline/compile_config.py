"""Compile a ``WorkflowConfig`` into declarative pipeline specs.

This is the bridge that keeps the legacy switch-based ``config.yaml`` working:
the boolean switches select which steps are emitted into a generated spec, which
is then executed by the engine. The EIT processing pipeline is fully granular
(one step per ``eitprocessing`` operation); EMG processing wraps the session
stage methods, which already dispatch ReSurfEMG at per-function granularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from m3resp.core.config import WorkflowConfig
from m3resp.core.session import M3Session
from m3resp.workflows.configured.kwargs import (
    eit_preprocess_kwargs,
    emg_detection_kwargs,
    emg_postprocessing_kwargs,
    emg_preprocess_kwargs,
)


@dataclass
class EitProcessingPlan:
    """A compiled EIT processing spec plus the metadata needed to summarize it."""

    steps: list[dict[str, Any]] = field(default_factory=list)
    filter_mode: str = "none"
    include_filtered_data: bool = True
    include_global_impedance: bool = True


def build_eit_processing_plan(cfg: WorkflowConfig) -> EitProcessingPlan:
    """Compile the granular EIT processing spec from config switches.

    Raw EIT signals are expected to be seeded into the pipeline context (the
    session is already loaded), so this spec contains no load step.
    """

    kwargs = eit_preprocess_kwargs(cfg)  # also validates TIV-needs-breaths
    filter_mode = (
        "none" if not kwargs["filter_enabled"] else kwargs["filter_mode"].lower()
    )
    compute_rates = kwargs["compute_rates"]
    rates_required = compute_rates or filter_mode == "mdn"
    include_global_impedance = kwargs["include_global_impedance"]
    compute_breaths = kwargs["compute_breath_intervals"]
    needs_global = (
        include_global_impedance
        or compute_breaths
        or any(
            (
                kwargs["compute_continuous_tiv"],
                kwargs["compute_eeli"],
                kwargs["compute_pixel_tiv"],
            )
        )
    )

    plan = EitProcessingPlan(
        filter_mode=filter_mode,
        include_filtered_data=kwargs["include_filtered_data"],
        include_global_impedance=include_global_impedance,
    )

    if rates_required:
        plan.steps.append(
            {
                "uses": "eit.detect_rates",
                "in": {"signal": "raw_eit"},
                "with": {
                    "subject_type": kwargs["subject_type"],
                    "welch_window_seconds": kwargs["welch_window_seconds"],
                    "capture": True,
                },
            }
        )

    eit_key = "raw_eit"
    if filter_mode == "mdn":
        plan.steps.append(
            {
                "uses": "eit.mdn_filter",
                "in": {"signal": "raw_eit"},
                "with": {"label": "mdn_filtered"},
            }
        )
        eit_key = "filtered_eit"
    elif filter_mode in {"lowpass", "bandpass"}:
        plan.steps.append(
            {
                "uses": "eit.butterworth_filter",
                "in": {"signal": "raw_eit"},
                "with": {
                    "mode": filter_mode,
                    "lowpass_hz": kwargs["lowpass_hz"],
                    "highpass_hz": kwargs["highpass_hz"],
                    "order": kwargs["filter_order"],
                    "label": f"{filter_mode}_filtered",
                },
            }
        )
        eit_key = "filtered_eit"

    if needs_global:
        plan.steps.append({"uses": "eit.global_impedance", "in": {"signal": eit_key}})

    if compute_breaths:
        plan.steps.append(
            {
                "uses": "eit.detect_breaths",
                "in": {"signal": "global_impedance"},
                "with": {"min_duration_s": kwargs["breath_min_duration_seconds"]},
            }
        )
        if cfg.eit.processing.breath_detection.enabled:
            plan.steps.append({"uses": "eit.normalize_breaths"})
    if kwargs["compute_continuous_tiv"]:
        plan.steps.append(
            {"uses": "eit.continuous_tiv", "in": {"signal": "global_impedance"}}
        )
    if kwargs["compute_eeli"]:
        plan.steps.append({"uses": "eit.eeli", "in": {"signal": "global_impedance"}})
    if kwargs["compute_pixel_tiv"]:
        plan.steps.append(
            {
                "uses": "eit.pixel_tiv",
                "in": {"signal": "global_impedance", "filtered_eit": eit_key},
            }
        )

    return plan


def build_multimodal_spec(cfg: WorkflowConfig) -> dict[str, Any]:
    """Build a declarative full multimodal pipeline spec from a workflow config.

    The generated spec covers every stage that ``run_configured_workflow`` runs:
    loading, optional raw synchronization, EIT processing (granular), EMG
    processing, and event alignment. Run it with :func:`run_pipeline` and then
    call :func:`~m3resp.workflows.configured.steps.assemble_eit_processed` to
    populate ``session.processed["eit"]`` for summary functions.

    The ventilator array is **not** embedded in the spec (it can't be
    serialized to YAML). ``emg.postprocess`` reads it from
    ``session.raw["vent"]`` automatically if not provided in ``with:``.
    """

    steps: list[dict[str, Any]] = []

    if cfg.modules.eit and cfg.eit.file is not None:
        steps.append(
            {
                "uses": "eit.load",
                "with": {"file": str(cfg.eit.file), "vendor": cfg.eit.vendor},
            }
        )

    if cfg.modules.emg and cfg.emg.file is not None:
        vent_file = (
            str(cfg.vent.file)
            if cfg.modules.vent and cfg.vent.file is not None
            else None
        )
        emg_load: dict[str, Any] = {
            "uses": "emg.load",
            "with": {"file": str(cfg.emg.file)},
        }
        if vent_file is not None:
            emg_load["with"]["vent_file"] = vent_file
        steps.append(emg_load)

    if cfg.modules.eit and cfg.modules.emg:
        offset = (
            cfg.alignment.offset_seconds
            if cfg.alignment.offset_seconds is not None
            else cfg.alignment.manual_offset_seconds
        )
        steps.append(
            {
                "uses": "session.sync_raw",
                "with": {
                    "method": cfg.alignment.method,
                    "offset_seconds": offset,
                    "reference_modality": cfg.alignment.reference_modality,
                },
            }
        )

    if cfg.modules.eit and cfg.eit.processing.preprocess.enabled:
        plan = build_eit_processing_plan(cfg)
        steps.extend(plan.steps)

    if cfg.modules.emg:
        steps.extend(_build_emg_steps_no_vent(cfg))

    return {"name": "multimodal", "steps": steps}


def _build_emg_steps_no_vent(cfg: WorkflowConfig) -> list[dict[str, Any]]:
    """Build EMG processing steps without embedding ventilator data.

    The ``emg.postprocess`` step reads the ventilator from the session at
    execution time, so the spec remains YAML-serializable.
    """

    processing = cfg.emg.processing
    steps: list[dict[str, Any]] = []

    if processing.preprocess.enabled:
        steps.append({"uses": "emg.preprocess", "with": emg_preprocess_kwargs(cfg)})

    if processing.breath_detection.enabled:
        if not processing.preprocess.enabled:
            raise ValueError(
                "Configured EMG breath detection requires "
                "emg.processing.preprocess.enabled: true."
            )
        steps.append({"uses": "emg.detect_breaths", "with": emg_detection_kwargs(cfg)})

    if processing.postprocessing.enabled:
        if not processing.preprocess.enabled:
            raise ValueError(
                "Configured EMG postprocessing requires "
                "emg.processing.preprocess.enabled: true."
            )
        post_kwargs = emg_postprocessing_kwargs(cfg, ventilator=None)
        post_kwargs.pop("ventilator", None)
        steps.append({"uses": "emg.postprocess", "with": post_kwargs})

    return steps


def build_emg_processing_steps(
    cfg: WorkflowConfig, session: M3Session
) -> list[dict[str, Any]]:
    """Compile the EMG processing spec from config switches."""

    processing = cfg.emg.processing
    steps: list[dict[str, Any]] = []

    if processing.preprocess.enabled:
        steps.append({"uses": "emg.preprocess", "with": emg_preprocess_kwargs(cfg)})

    if processing.breath_detection.enabled:
        if not processing.preprocess.enabled:
            raise ValueError(
                "Configured EMG breath detection requires "
                "emg.processing.preprocess.enabled: true."
            )
        steps.append({"uses": "emg.detect_breaths", "with": emg_detection_kwargs(cfg)})

    if processing.postprocessing.enabled:
        if not processing.preprocess.enabled:
            raise ValueError(
                "Configured EMG postprocessing requires "
                "emg.processing.preprocess.enabled: true."
            )
        ventilator = session.raw.get("vent")
        steps.append(
            {
                "uses": "emg.postprocess",
                "with": emg_postprocessing_kwargs(cfg, ventilator=ventilator),
            }
        )

    return steps
