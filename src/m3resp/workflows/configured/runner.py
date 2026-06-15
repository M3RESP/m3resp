"""Runner for YAML-configured workflows."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from m3resp.core.config import WorkflowConfig, load_workflow_config
from m3resp.core.session import M3Session
from m3resp.workflows.configured.artifacts import (
    export_configured_session,
    save_workflow_figures,
)
from m3resp.workflows.configured.steps import (
    load_configured_eit,
    load_configured_emg,
    process_configured_eit,
    process_configured_emg,
)
from m3resp.workflows.configured.summaries import (
    summarize_eit,
    summarize_emg,
    summarize_multimodal,
)
from m3resp.workflows.configured.types import (
    ConfiguredWorkflowKind,
    WorkflowResult,
)


def run_configured_workflow(
    config: str | Path | WorkflowConfig,
    *,
    root: str | Path | None = None,
    export: bool = True,
    save_figures: bool = True,
    eit_adapter: Any = None,
    emg_adapter: Any = None,
) -> WorkflowResult:
    """Run the YAML-configured workflow selected by enabled module flags."""

    cfg = coerce_workflow_config(config, root=root)
    selected = select_configured_workflow(cfg)
    session = M3Session(eit_adapter=eit_adapter, emg_adapter=emg_adapter)

    if cfg.modules.eit:
        load_configured_eit(session, cfg)
    if cfg.modules.emg:
        load_configured_emg(session, cfg)

    if cfg.modules.eit and cfg.modules.emg:
        session.synchronize_raw_modalities(
            method=cfg.alignment.method,
            offset_seconds=(
                cfg.alignment.offset_seconds
                if cfg.alignment.offset_seconds is not None
                else cfg.alignment.manual_offset_seconds
            ),
            reference_modality=cfg.alignment.reference_modality,
        )

    if cfg.modules.eit:
        process_configured_eit(session, cfg)
    if cfg.modules.emg:
        process_configured_emg(session, cfg)

    output_dir = _timestamped_output_dir(_configured_output_dir(cfg, selected))
    if export:
        output_dir = export_configured_session(session, output_dir, cfg)
    figures = (
        save_workflow_figures(
            session,
            output_dir,
            include_eit=cfg.modules.eit,
            include_emg=cfg.modules.emg,
        )
        if save_figures and cfg.results.figures
        else {}
    )
    return WorkflowResult(
        session=session,
        summary=_configured_summary(session, selected),
        output_dir=Path(output_dir),
        figures=figures,
    )


def coerce_workflow_config(
    config: str | Path | WorkflowConfig,
    *,
    root: str | Path | None = None,
) -> WorkflowConfig:
    """Return a resolved workflow config from a path or existing config object."""

    if isinstance(config, WorkflowConfig):
        return config
    return load_workflow_config(config, root=root)


def select_configured_workflow(config: WorkflowConfig) -> ConfiguredWorkflowKind:
    """Select the configured workflow from module flags."""

    if config.modules.eit and config.modules.emg:
        return "multimodal"
    if config.modules.eit:
        return "eit"
    if config.modules.emg:
        return "emg"
    raise ValueError(
        "Workflow config must enable at least one primary module: modules.eit or modules.emg."
    )


def _configured_output_dir(
    cfg: WorkflowConfig, selected: ConfiguredWorkflowKind
) -> Path:
    if selected == "multimodal":
        return cfg.output.combined
    if selected == "eit":
        return cfg.output.eit_only
    return cfg.output.emg_only


def _timestamped_output_dir(output_dir: Path) -> Path:
    run_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(os.path.join(output_dir, run_folder))


def _configured_summary(
    session: M3Session, selected: ConfiguredWorkflowKind
) -> dict[str, Any]:
    if selected == "multimodal":
        return summarize_multimodal(
            session,
            include_eit="eit" in session.processed,
            include_emg="emg" in session.processed,
        )
    if selected == "eit":
        return summarize_eit(session) if "eit" in session.processed else {}
    return summarize_emg(session) if "emg" in session.processed else {}
