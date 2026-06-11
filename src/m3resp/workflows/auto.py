"""Automatic YAML-configured workflow selection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from m3resp.core.config import WorkflowConfig
from m3resp.workflows.configured import (
    WorkflowResult,
    coerce_workflow_config,
    run_configured_workflow,
    select_configured_workflow,
)

WorkflowKind = Literal["eit", "emg", "multimodal"]
CONFIG_PATH = Path(os.path.join("config.yaml"))


def select_workflow(
    config: str | Path | WorkflowConfig, *, root: str | Path | None = None
) -> WorkflowKind:
    """Select the configured workflow from module flags."""

    return select_configured_workflow(coerce_workflow_config(config, root=root))


def run_workflow(
    config: str | Path | WorkflowConfig,
    *,
    root: str | Path | None = None,
    export: bool = True,
    save_figures: bool = True,
    eit_adapter: Any = None,
    emg_adapter: Any = None,
) -> WorkflowResult:
    """Run the workflow selected by the YAML module flags."""

    return run_configured_workflow(
        config,
        root=root,
        export=export,
        save_figures=save_figures,
        eit_adapter=eit_adapter,
        emg_adapter=emg_adapter,
    )


def run(config: str | Path | WorkflowConfig | None = None) -> WorkflowResult:
    """Run the workflow selected by the YAML module flags."""

    return run_workflow(config=CONFIG_PATH if config is None else config)
