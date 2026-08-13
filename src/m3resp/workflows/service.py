"""A framework-neutral pipeline service (Phase 7.3/7.4 of the
pipeline-structure plan).

``PipelineService`` is the intended integration surface for the Stage 3 GUI
and any other application embedding m3resp: every method takes a spec
(path/dict/mapping) and returns only JSON-safe dictionaries - never a
session, adapter instance, upstream package object, NumPy array, or
arbitrary Python callable. It calls the existing synchronous engine
directly; GUI threading/process management is the caller's responsibility,
not this service's.

``event_sink``/``cancellation_token``, when supplied by the caller, are the
one exception to "JSON-safe only": they are the caller's own objects, used
exactly as ``run_pipeline`` already uses them (Phase 4.4/4.5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from m3resp.workflows.compiler import compile_pipeline as _compile_pipeline
from m3resp.workflows.compiler import validate_pipeline as _validate_pipeline
from m3resp.workflows.engine import PipelineResult
from m3resp.workflows.engine import run_pipeline as _run_pipeline
from m3resp.workflows.lifecycle import (
    CancellationToken,
    EventSink,
    summarize_output_value,
)
from m3resp.workflows.registry import describe_step, describe_steps
from m3resp.workflows.spec import PipelineSpec, load_spec


class PipelineService:
    """Framework-neutral facade over discovery, validation, compilation, and
    execution - the integration surface for a GUI or other application."""

    def list_capabilities(self, *, prefix: str | None = None) -> list[dict[str, Any]]:
        """Every registered step's discovery description (Phase 1.5),
        optionally filtered to one operation prefix (e.g. ``"eit."``)."""

        return [description.as_dict() for description in describe_steps(prefix=prefix)]

    def describe_capability(self, operation_id: str) -> dict[str, Any]:
        """One registered step's discovery description, including its
        capability state (``"available"``/``"missing_optional_dependency"``/
        ``"deprecated"``) without importing its optional packages."""

        return describe_step(operation_id).as_dict()

    def validate_pipeline(
        self,
        spec: str | Path | dict[str, Any] | PipelineSpec,
        *,
        readiness: bool = False,
    ) -> dict[str, Any]:
        """Structural (and, if ``readiness=True``, capability/file-existence)
        validation report (Phase 3.5) - never raises for an invalid spec,
        report is returned either way via ``ValidationReport.as_dict()``."""

        parsed = load_spec(spec)
        return _validate_pipeline(parsed, readiness=readiness).as_dict()

    def compile_pipeline(
        self, spec: str | Path | dict[str, Any] | PipelineSpec
    ) -> dict[str, Any]:
        """The fully-resolved, read-only execution plan (Phase 3.1), as a
        JSON-safe dict. Raises the same way ``compile_pipeline`` does for an
        invalid spec - call ``validate_pipeline`` first to check without
        raising."""

        parsed = load_spec(spec)
        return _compile_pipeline(parsed).as_dict()

    def run_pipeline(
        self,
        spec: str | Path | dict[str, Any] | PipelineSpec,
        *,
        event_sink: EventSink | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Run the pipeline and return a JSON-safe run summary (Phase 4.7's
        ``PipelineResult``, minus the live session/context/raw outputs -
        see :func:`summarize_pipeline_result`). Raises
        ``PipelineExecutionError`` on a step failure, exactly like
        ``run_pipeline`` itself (Phase 4.2) - this service does not swallow
        it into a return value, since a Python caller can already catch it."""

        result = _run_pipeline(
            spec, event_sink=event_sink, cancellation_token=cancellation_token
        )
        return summarize_pipeline_result(result)


def summarize_pipeline_result(result: PipelineResult) -> dict[str, Any]:
    """JSON-safe summary of a ``PipelineResult`` (Phase 7.4): run metadata,
    step records, diagnostics/warnings, and a *summary* of each produced
    output (via ``summarize_output_value``, the same type/shape-only
    summary step records already use) - never the raw session, context, or
    output objects themselves."""

    return {
        "name": result.name,
        "run_id": result.run_id,
        "status": result.status,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "duration_seconds": result.duration_seconds,
        "processing_run_id": result.processing_run_id,
        "resolved_output_dir": str(result.resolved_output_dir)
        if result.resolved_output_dir is not None
        else None,
        "manifest_path": str(result.manifest_path)
        if result.manifest_path is not None
        else None,
        "step_records": [record.as_dict() for record in result.step_records],
        "diagnostics": [d.as_dict() for d in result.diagnostics],
        "warnings": [w.as_dict() for w in result.warnings],
        "outputs": {
            name: summarize_output_value(value)
            for name, value in result.outputs.items()
        },
    }
