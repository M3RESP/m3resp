"""Workflow discovery, validation and execution for application callers.

Methods return dictionaries describing steps, validation, compiled settings
or execution results. Execution is synchronous; callers manage background
work and supply progress receivers or cancellation flags when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from m3resp.workflows.compiler import compile_workflow as _compile_workflow
from m3resp.workflows.compiler import validate_workflow as _validate_workflow
from m3resp.workflows.engine import WorkflowResult
from m3resp.workflows.engine import run_workflow as _run_workflow
from m3resp.workflows.lifecycle import (
    CancellationToken,
    EventSink,
    summarize_output_value,
)
from m3resp.workflows.registry import describe_step, describe_steps
from m3resp.workflows.spec import WorkflowSpec, load_spec


class WorkflowService:
    """Discover workflow steps, inspect specs and run workflows for an application."""

    def list_capabilities(self, *, prefix: str | None = None) -> list[dict[str, Any]]:
        """Return registered step descriptions, optionally filtered by name prefix.

        Each dictionary includes parameters, input/output descriptions and current
        availability. Names are sorted; prefix can select e.g. ``"eit."`` steps.
        """

        return [description.as_dict() for description in describe_steps(prefix=prefix)]

    def describe_capability(self, operation_id: str) -> dict[str, Any]:
        """Return one step's settings, input/output descriptions and availability.

        Raises UnknownStepError if operation_id is unregistered. Former step names
        resolve through registered aliases.
        """

        return describe_step(operation_id).as_dict()

    def validate_workflow(
        self,
        spec: str | Path | dict[str, Any] | WorkflowSpec,
        *,
        readiness: bool = False,
    ) -> dict[str, Any]:
        """Load a workflow and return its structural and optional readiness report.

        Args:
            spec: A WorkflowSpec, dictionary, or YAML/JSON file path.
            readiness: Also check package availability and input-file existence.

        Returns:
            dict[str, Any]: is_valid plus structural and readiness diagnostic lists.
                Structural problems in a parsed spec are returned as diagnostics.
                is_valid reflects structural errors only.

        Raises:
            WorkflowSpecError: If the spec cannot be parsed or fails format validation.
            OSError: If a spec file cannot be read.
        """

        parsed = load_spec(spec)
        return _validate_workflow(parsed, readiness=readiness).as_dict()

    def compile_workflow(
        self, spec: str | Path | dict[str, Any] | WorkflowSpec
    ) -> dict[str, Any]:
        """Load and compile a workflow into an ordered description of its steps.

        Args:
            spec: A WorkflowSpec, dictionary, or YAML/JSON file path.

        Returns:
            dict[str, Any]: Workflow name, schema version and compiled steps with
                resolved input/output names, defaults, parameter values and paths.

        Raises:
            WorkflowSpecError: If parsing or structural validation fails.
            UnknownStepError: If the first structural error names an unknown step.
            OSError: If a spec file cannot be read.
        """

        parsed = load_spec(spec)
        return _compile_workflow(parsed).as_dict()

    def run_workflow(
        self,
        spec: str | Path | dict[str, Any] | WorkflowSpec,
        *,
        event_sink: EventSink | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Run a workflow synchronously and return an execution summary.

        Args:
            spec: A WorkflowSpec, dictionary, or YAML/JSON file path.
            event_sink: Optional callable receiving progress-event dictionaries.
            cancellation_token: Optional flag checked before and after each step.

        Returns:
            dict[str, Any]: Run identifiers, status, timings, step records, diagnostics,
                warnings and output summaries. Completed work is preserved when
                cancelled. Output data are represented by scalar values or brief
                type/shape/length descriptions, as in `summarize_workflow_result`.

        Raises:
            WorkflowSpecError: If the spec, bindings or returned outputs are invalid.
            UnknownStepError: If the first structural error names an unknown step.
            WorkflowExecutionError: If a step function raises, with its cause and
                gathered records attached.
            OSError: If a spec file cannot be read.
        """

        result = _run_workflow(
            spec, event_sink=event_sink, cancellation_token=cancellation_token
        )
        return summarize_workflow_result(result)


def summarize_workflow_result(result: WorkflowResult) -> dict[str, Any]:
    """Return run metadata, execution records and compact output summaries.

    Scalar output values are retained. Arrays are described by shape and type,
    collections by length or keys, and other objects by type. Recorded step
    parameters are included as stored in the execution records.
    """

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
