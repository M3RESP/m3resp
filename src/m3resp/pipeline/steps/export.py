"""Registered export pipeline steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from m3resp.core.session import M3Session
from m3resp.export.session_export import export_session_summary
from m3resp.pipeline.registry import register_step
from m3resp.workflows.toolbox import write_json


@register_step(
    "export.scalar_file",
    reads={"value": "value"},
    writes=("result_path",),
    summary="Write a single scalar value to a text file.",
)
def scalar_file(value: float, *, path: str, precision: int = 8) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{float(value):.{precision}f}", encoding="utf-8")
    return {"result_path": str(target)}


@register_step(
    "export.json_file",
    reads={"payload": "summary"},
    writes=("json_path",),
    summary="Write a mapping payload to a JSON file.",
)
def json_file(payload: dict[str, Any], *, path: str) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, payload)
    return {"json_path": str(target)}


@register_step(
    "export.session_summary",
    reads={"session": "session"},
    writes=("output_dir",),
    summary="Export the session summary (JSON, event CSVs, parameters) to disk.",
)
def session_summary(
    session: M3Session,
    *,
    output_dir: str,
    summary_json: bool = True,
    event_csvs: bool = True,
    parameters_csv: bool = True,
    postprocessing: bool = True,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    export_session_summary(
        session,
        target,
        summary_json=summary_json,
        event_csvs=event_csvs,
        parameters_csv=parameters_csv,
        postprocessing=postprocessing,
    )
    return {"output_dir": str(target)}


__all__ = ["scalar_file", "json_file", "session_summary"]
