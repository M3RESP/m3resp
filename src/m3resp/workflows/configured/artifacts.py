"""Artifact helpers for YAML-configured workflows."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from m3resp.core.config import WorkflowConfig
from m3resp.core.session import M3Session
from m3resp.export.session_export import export_session_summary


def save_workflow_figures(
    session: M3Session,
    output_dir: str | Path,
    *,
    include_eit: bool = False,
    include_emg: bool = False,
    max_seconds: float | None = 120.0,
) -> dict[str, Path]:
    """Save available workflow figures and return created paths."""

    from m3resp.visualization.session import (
        plot_eit_processing_summary,
        plot_session_overview,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Any] = {}

    if include_emg:
        _add_figure(
            figures,
            "overview.png",
            lambda: plot_session_overview(session, max_seconds=max_seconds),
        )

    if include_eit:
        _add_figure(
            figures, "eit-processing.png", lambda: plot_eit_processing_summary(session)
        )
        processed = session.processed.get("eit")
        if isinstance(processed, Mapping):
            _add_figure(
                figures,
                "eit-rate-detection.png",
                lambda: processed["rate_detector"].plotting.plot(
                    **processed["rate_captures"]
                ),
            )

    saved: dict[str, Path] = {}
    for filename, figure in figures.items():
        path = Path(os.path.join(output_path, filename))
        figure.savefig(path, dpi=150)
        saved[filename] = path
    return saved


def export_configured_session(
    session: M3Session,
    output_dir: str | Path,
    cfg: WorkflowConfig,
) -> Path:
    """Export configured workflow artifacts selected by config."""

    return export_session_summary(
        session,
        output_dir,
        summary_json=cfg.results.summary_json,
        event_csvs=cfg.results.event_csvs,
        parameters_csv=cfg.results.parameters_csv,
        postprocessing=cfg.results.postprocessing,
    )


def _add_figure(
    figures: dict[str, Any],
    filename: str,
    factory: Any,
) -> None:
    try:
        figures[filename] = factory()
    except (AttributeError, ImportError, KeyError, TypeError, ValueError):
        return
