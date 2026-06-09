"""Modality steps for YAML-configured workflows."""

from __future__ import annotations

from m3resp.core.config import WorkflowConfig
from m3resp.core.session import M3Session
from m3resp.workflows.configured.kwargs import (
    eit_preprocess_kwargs,
    emg_detection_kwargs,
    emg_postprocessing_kwargs,
    emg_preprocess_kwargs,
)


def run_configured_eit_steps(session: M3Session, cfg: WorkflowConfig) -> None:
    """Run enabled EIT loading and processing steps."""

    if cfg.eit.file is None:
        raise ValueError("Configured EIT workflow requires eit.file.")

    session.load_eit(cfg.eit.file, vendor=cfg.eit.vendor)
    if cfg.eit.processing.preprocess.enabled:
        session.preprocess_eit(**eit_preprocess_kwargs(cfg))
    if cfg.eit.processing.breath_detection.enabled:
        if not cfg.eit.processing.preprocess.enabled:
            raise ValueError(
                "Configured EIT breath detection requires "
                "eit.processing.preprocess.enabled: true."
            )
        session.detect_eit_breaths()


def run_configured_emg_steps(session: M3Session, cfg: WorkflowConfig) -> None:
    """Run enabled EMG loading and processing steps."""

    if cfg.emg.file is None:
        raise ValueError("Configured EMG workflow requires emg.file.")

    session.load_emg(cfg.emg.file, verbose=False)
    if cfg.emg.processing.preprocess.enabled:
        session.preprocess_emg(**emg_preprocess_kwargs(cfg))
    if cfg.emg.processing.breath_detection.enabled:
        if not cfg.emg.processing.preprocess.enabled:
            raise ValueError(
                "Configured EMG breath detection requires "
                "emg.processing.preprocess.enabled: true."
            )
        session.detect_emg_breaths(**emg_detection_kwargs(cfg))

    ventilator = None
    if cfg.modules.vent and cfg.vent.file is not None:
        ventilator = session.emg_adapter.load(str(cfg.vent.file), verbose=False)
    if cfg.emg.processing.postprocessing.enabled:
        if not cfg.emg.processing.preprocess.enabled:
            raise ValueError(
                "Configured EMG postprocessing requires "
                "emg.processing.preprocess.enabled: true."
            )
        session.postprocess_emg(**emg_postprocessing_kwargs(cfg, ventilator=ventilator))
