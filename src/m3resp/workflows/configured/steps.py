"""Modality steps for YAML-configured workflows.

EIT and EMG processing are compiled from the config switches into declarative
pipeline specs and executed by :func:`m3resp.pipeline.run_pipeline`. Loading and
cross-modality synchronization/alignment remain session-level concerns handled by
the runner.
"""

from __future__ import annotations

from typing import Any

from m3resp.core.config import WorkflowConfig
from m3resp.core.session import M3Session
from m3resp.pipeline import run_pipeline
from m3resp.pipeline.compile_config import (
    EitProcessingPlan,
    build_eit_processing_plan,
    build_emg_processing_steps,
)


def run_configured_eit_steps(session: M3Session, cfg: WorkflowConfig) -> None:
    """Run enabled EIT loading and processing steps."""

    load_configured_eit(session, cfg)
    process_configured_eit(session, cfg)


def load_configured_eit(session: M3Session, cfg: WorkflowConfig) -> None:
    """Load configured EIT input."""

    if cfg.eit.file is None:
        raise ValueError("Configured EIT workflow requires eit.file.")

    session.load_eit(cfg.eit.file, vendor=cfg.eit.vendor)


def process_configured_eit(session: M3Session, cfg: WorkflowConfig) -> None:
    """Run the granular EIT processing pipeline selected by config."""

    if not cfg.eit.processing.preprocess.enabled:
        if cfg.eit.processing.breath_detection.enabled:
            raise ValueError(
                "Configured EIT breath detection requires "
                "eit.processing.preprocess.enabled: true."
            )
        return

    plan = build_eit_processing_plan(cfg)
    recording = session.eit
    assert recording is not None  # loaded by load_configured_eit
    seed = {
        "raw_eit": recording.raw,
        "raw_global_impedance": recording.global_impedance,
        "eit_sequence": recording.data,
    }

    context_values: dict[str, Any] = dict(seed)
    if plan.steps:
        result = run_pipeline(
            {"name": "configured-eit", "steps": plan.steps},
            session=session,
            extra_context=seed,
        )
        context_values = result.context.values

    session.processed["eit"] = assemble_eit_processed(plan, seed, context_values)


def assemble_eit_processed(
    plan: EitProcessingPlan,
    seed: dict[str, Any],
    values: dict[str, Any],
) -> dict[str, Any]:
    """Reassemble the legacy ``processed['eit']`` shape from pipeline context."""

    filtered_eit = values.get("filtered_eit")
    return {
        "sequence": seed["eit_sequence"],
        "raw_eit": seed["raw_eit"],
        "raw_global_impedance": seed.get("raw_global_impedance"),
        "filter_mode": plan.filter_mode,
        "filter_captures": values.get("filter_captures", {}),
        "rate_detector": values.get("rate_detector"),
        "rate_captures": values.get("rate_captures", {}),
        "respiratory_rate_hz": values.get("respiratory_rate_hz"),
        "heart_rate_hz": values.get("heart_rate_hz"),
        "filtered_eit": filtered_eit if plan.include_filtered_data else None,
        "filtered_global_impedance": values.get(
            "global_impedance", seed.get("raw_global_impedance")
        ),
        "breath_intervals": values.get("breath_intervals"),
        "continuous_tiv": values.get("continuous_tiv"),
        "eeli": values.get("eeli"),
        "pixel_tiv": values.get("pixel_tiv"),
    }


def run_configured_emg_steps(session: M3Session, cfg: WorkflowConfig) -> None:
    """Run enabled EMG loading and processing steps."""

    load_configured_emg(session, cfg)
    process_configured_emg(session, cfg)


def load_configured_emg(session: M3Session, cfg: WorkflowConfig) -> None:
    """Load configured EMG and optional ventilator inputs."""

    if cfg.emg.file is None:
        raise ValueError("Configured EMG workflow requires emg.file.")

    session.load_emg(cfg.emg.file, verbose=False)
    if cfg.modules.vent and cfg.vent.file is not None:
        session.raw["vent"] = session.emg_adapter.load(
            str(cfg.vent.file), verbose=False
        )


def process_configured_emg(session: M3Session, cfg: WorkflowConfig) -> None:
    """Run the EMG processing pipeline selected by config."""

    steps = build_emg_processing_steps(cfg, session)
    if steps:
        run_pipeline(
            {"name": "configured-emg", "steps": steps},
            session=session,
        )
