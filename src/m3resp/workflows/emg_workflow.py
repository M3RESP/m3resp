"""Minimal EMG workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from m3resp import M3Session
from m3resp.core.config import WorkflowConfig
from m3resp.workflows.configured import (
    WorkflowResult,
    run_configured_workflow,
)


def run_emg_workflow(
    path: str | Path | None = None,
    *,
    config: str | Path | WorkflowConfig | None = None,
    root: str | Path | None = None,
    export: bool = True,
    save_figures: bool = True,
    emg_adapter: Any = None,
    **kwargs: Any,
) -> M3Session | WorkflowResult:
    """Run the EMG workflow.

    Positional ``path`` calls preserve the original lightweight API and return
    an ``M3Session``. Passing ``config=`` runs the YAML-configured workflow and
    returns a ``WorkflowResult``.
    """

    if config is not None:
        return run_configured_workflow(
            config,
            root=root,
            export=export,
            save_figures=save_figures,
            emg_adapter=emg_adapter,
        )

    if path is None:
        raise TypeError("run_emg_workflow() requires path or config")

    session = M3Session(emg_adapter=emg_adapter)
    session.load_emg(path, **kwargs)
    return session
