"""Minimal EIT workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from m3resp import M3Session
from m3resp.core.config import WorkflowConfig
from m3resp.workflows.configured import (
    WorkflowResult,
    run_configured_workflow,
)


def run_eit_workflow(
    path: str | Path | None = None,
    vendor: str | None = None,
    *,
    config: str | Path | WorkflowConfig | None = None,
    root: str | Path | None = None,
    export: bool = True,
    save_figures: bool = True,
    eit_adapter: Any = None,
    **kwargs: Any,
) -> M3Session | WorkflowResult:
    """Run the EIT workflow.

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
            eit_adapter=eit_adapter,
        )

    if path is None:
        raise TypeError("run_eit_workflow() requires path or config")

    session = M3Session(eit_adapter=eit_adapter)
    session.load_eit(path, vendor=vendor, **kwargs)
    return session
