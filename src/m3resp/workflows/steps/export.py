"""Registered export pipeline steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from m3resp.core.session import M3Session
from m3resp.export.session_export import export_session_summary
from m3resp.workflows.context import RESOLVED_OUTPUT_DIR_KEY
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step
from m3resp.workflows.utils import write_json


@register_step(
    "export.scalar_file",
    reads={"value": "value"},
    writes=("result_path",),
    summary="Write a single scalar value to a text file.",
    description="Write a single scalar value to a plain text file, formatted to a fixed decimal precision.",
    category="export",
    input_artifacts=(
        StepArtifact(
            name="value",
            artifact_type="scalar_metric",
            description="Scalar value to write.",
        ),
    ),
    parameters=(
        StepParameter(
            name="path",
            value_type="path",
            required=True,
            path_kind="file",
            description="Output file path.",
        ),
        StepParameter(
            name="precision",
            value_type="integer",
            default=8,
            minimum=0,
            description="Number of decimal places written.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="result_path",
            artifact_type="file_path",
            description="Path to the written text file.",
        ),
    ),
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
    description="Write a JSON-safe mapping payload to disk as a formatted JSON file.",
    category="export",
    input_artifacts=(
        StepArtifact(
            name="payload",
            artifact_type="mapping",
            description="JSON-safe mapping to write.",
        ),
    ),
    parameters=(
        StepParameter(
            name="path",
            value_type="path",
            required=True,
            path_kind="file",
            description="Output file path.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="json_path",
            artifact_type="file_path",
            description="Path to the written JSON file.",
        ),
    ),
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
    description=(
        "Export the full session summary to a directory: JSON summary, event "
        "CSVs, parameters CSV, postprocessing artifacts, and/or the "
        "structured (array + scalar) export, each independently toggleable."
    ),
    category="export",
    input_artifacts=(
        StepArtifact(
            name="session",
            artifact_type="m3session",
            default_context_key="session",
            description="Session whose accumulated results are exported.",
            public=False,
        ),
    ),
    parameters=(
        StepParameter(
            name="output_dir",
            value_type="path",
            required=True,
            path_kind="directory",
            description="Directory the summary is written into (created if missing).",
        ),
        StepParameter(
            name="summary_json",
            value_type="boolean",
            default=True,
            description="Write a JSON session summary.",
        ),
        StepParameter(
            name="event_csvs",
            value_type="boolean",
            default=True,
            description="Write one CSV per event collection.",
        ),
        StepParameter(
            name="parameters_csv",
            value_type="boolean",
            default=True,
            description="Write a CSV of scalar derived parameters.",
        ),
        StepParameter(
            name="postprocessing",
            value_type="boolean",
            default=True,
            description="Write postprocessing (figure/report) artifacts.",
        ),
        StepParameter(
            name="structured_export",
            value_type="boolean",
            default=True,
            description="Write the structured scalar-CSV plus array-archive export.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="output_dir",
            artifact_type="directory_path",
            description="Directory the summary was written into.",
        ),
    ),
)
def session_summary(
    session: M3Session,
    *,
    output_dir: str,
    summary_json: bool = True,
    event_csvs: bool = True,
    parameters_csv: bool = True,
    postprocessing: bool = True,
    structured_export: bool = True,
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
        structured_export=structured_export,
    )
    return {"output_dir": str(target)}


@register_step(
    "export.rotarc_result",
    reads={
        "value": "cv",
        "_spec_outputs": "_spec_outputs",
        "_spec_experiment": "_spec_experiment",
        "_resolved_output_dir": RESOLVED_OUTPUT_DIR_KEY,
        "session": "session",
    },
    writes=("result_path",),
    summary="Write ROTARC breath-duration CV to a named result file and rotarc_summary.json.",
    description=(
        "ROTARC-specific export: derives the output path from the spec's "
        "'experiment:'/'outputs:' sections, writes the scalar result plus "
        "'rotarc_summary.json', and runs the same session export as "
        "'export.session_summary' alongside it."
    ),
    category="export",
    input_artifacts=(
        StepArtifact(
            name="value",
            artifact_type="scalar_metric",
            description="Breath-duration CV to write.",
        ),
        StepArtifact(
            name="session",
            artifact_type="m3session",
            default_context_key="session",
            description="Session whose accumulated results are exported alongside the result file.",
            public=False,
        ),
        StepArtifact(
            name="_spec_outputs",
            artifact_type="spec_outputs_config",
            default_context_key="_spec_outputs",
            description="The spec's own 'outputs:' section, auto-injected by run_spec (internal engine plumbing, not user-bindable).",
            public=False,
            compatibility_only=True,
        ),
        StepArtifact(
            name="_spec_experiment",
            artifact_type="spec_experiment_config",
            default_context_key="_spec_experiment",
            description="The spec's own 'experiment:' section, auto-injected by run_spec (internal engine plumbing, not user-bindable).",
            public=False,
            compatibility_only=True,
        ),
        StepArtifact(
            name="_resolved_output_dir",
            artifact_type="directory_path",
            default_context_key=RESOLVED_OUTPUT_DIR_KEY,
            description="The run's one resolved output directory, auto-injected by run_spec (see RESOLVED_OUTPUT_DIR_KEY).",
            public=False,
        ),
    ),
    parameters=(
        StepParameter(
            name="precision",
            value_type="integer",
            default=8,
            minimum=0,
            description="Number of decimal places written to the result file.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="result_path",
            artifact_type="file_path",
            description="Path to the written per-subject result file.",
        ),
    ),
)
def rotarc_result(
    value: float,
    _spec_outputs: Any,
    _spec_experiment: Any,
    _resolved_output_dir: Path | None,
    session: M3Session,
    *,
    precision: int = 8,
) -> dict[str, Any]:
    """Derives the output path from the spec's ``experiment:`` and ``outputs:`` sections.

    Output path: ``<outputs.dir>/[<timestamp>/]subject_results/<run_identifier>/<subject>-<mode>-<tp>-<selection>.txt``.
    The optional timestamp segment is included when ``outputs.timestamped`` is
    set - see ``_resolved_output_dir`` in ``run_spec`` for how it's computed
    once per run and shared across every export step.
    """

    from m3resp.workflows.utils import subject_result_filename

    exp = _spec_experiment
    out = _spec_outputs

    if _resolved_output_dir is None:
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

    output_dir = (
        Path(_resolved_output_dir) / "subject_results" / str(exp.run_identifier)
    )
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
        structured_export=out.structured_export,
    )

    return {"result_path": str(result_path)}


__all__ = ["scalar_file", "json_file", "session_summary", "rotarc_result"]
