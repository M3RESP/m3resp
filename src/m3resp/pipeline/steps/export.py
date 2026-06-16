"""Registered export pipeline steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from m3resp.core.session import M3Session
from m3resp.export.session_export import export_session_summary
from m3resp.pipeline.registry import register_step
from m3resp.pipeline.utils import write_json


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


@register_step(
    "export.rotarc_result",
    reads={
        "value": "cv",
        "_spec_outputs": "_spec_outputs",
        "_spec_experiment": "_spec_experiment",
        "session": "session",
    },
    writes=("result_path",),
    summary="Write ROTARC breath-duration CV to a named result file and rotarc_summary.json.",
)
def rotarc_result(
    value: float,
    _spec_outputs: Any,
    _spec_experiment: Any,
    session: M3Session,
    *,
    precision: int = 8,
) -> dict[str, Any]:
    """Derives the output path from the spec's ``experiment:`` and ``outputs:`` sections.

    Output path: ``<outputs.dir>/subject_results/<run_identifier>/<subject>-<mode>-<tp>-<selection>.txt``
    """

    from m3resp.pipeline.utils import subject_result_filename

    exp = _spec_experiment
    out = _spec_outputs

    if out is None or out.dir is None:
        raise ValueError(
            "export.rotarc_result requires 'outputs.dir' to be set in the pipeline spec."
        )
    for field_name, field_val in [
        ("experiment.subject_id", exp.subject_id),
        ("experiment.run_identifier", exp.run_identifier),
    ]:
        if not field_val:
            raise ValueError(
                f"export.rotarc_result requires '{field_name}' in the pipeline spec."
            )

    output_dir = Path(out.dir) / "subject_results" / str(exp.run_identifier)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_filename = subject_result_filename(
        str(exp.subject_id),
        str(exp.mode),
        exp.timepoint,
        str(exp.selection),
    )
    result_path = output_dir / result_filename
    result_path.write_text(f"{float(value):.{precision}f}", encoding="utf-8")

    rotarc_summary = {
        "subject_id": exp.subject_id,
        "mode": exp.mode,
        "timepoint": exp.timepoint,
        "selection": exp.selection,
        "run_identifier": exp.run_identifier,
        "result_path": str(result_path),
        "breath_duration_cv": float(value),
    }
    write_json(output_dir / "rotarc_summary.json", rotarc_summary)

    export_session_summary(
        session,
        output_dir,
        summary_json=out.summary_json,
        event_csvs=out.event_csvs,
        parameters_csv=out.parameters_csv,
        postprocessing=out.postprocessing,
    )

    return {"result_path": str(result_path)}


__all__ = ["scalar_file", "json_file", "session_summary", "rotarc_result"]
