"""Shared execution-result type and step-registration bookkeeping for the
engine package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from m3resp.core.session import M3Session
from m3resp.workflows.context import (
    WorkflowContext,
)
from m3resp.workflows.diagnostics import Diagnostic
from m3resp.workflows.lifecycle import (
    CapturedWarning,
    ExecutionContext,
    StepExecutionRecord,
    WorkflowStatus,
)

if TYPE_CHECKING:
    # Deferred at runtime: compiler.py imports collect_diagnostics from this
    # package, so importing it back at module scope here would be circular.
    from m3resp.workflows.compiler import CompiledWorkflow
_STEPS_REGISTERED = False


def _ensure_steps_registered() -> None:
    """Import the built-in step package so every step is registered.

    Done lazily to avoid a circular import between the step modules and the
    workflows package.
    """

    global _STEPS_REGISTERED
    if not _STEPS_REGISTERED:
        from m3resp.workflows import steps as _steps  # noqa: F401

        _STEPS_REGISTERED = True


@dataclass
class WorkflowResult:
    """Results, timing and execution records for one workflow run.

    Attributes:
        name: Workflow name.
        context: Shared values and the session used by the steps.
        outputs: Produced context values and externally supplied values,
            excluding the session and keys named in spec.inputs.
        processing_run_id: Stored ProcessingRun identifier when a data-model
            recorder is attached, otherwise None.
        run_id: Identifier used in execution records and progress events.
        status: Run state; completed executions are succeeded or cancelled.
        started_at: Run start time as an ISO 8601 UTC string, or None.
        finished_at: Run finish time as an ISO 8601 UTC string, or None.
        duration_seconds: Elapsed execution time in seconds, or None.
        compiled_workflow: Ordered description of the steps and resolved settings.
        step_records: Records of steps that executed, in execution order.
        diagnostics: Structural diagnostics collected before execution.
        warnings: Python warnings captured during executed steps, in order.
        execution_context: Recorded software versions and seed information.
        resolved_output_dir: Shared output directory, when supplied in context.
        manifest_path: Run-manifest path when written by run_spec.
    """

    name: str
    context: WorkflowContext
    outputs: dict[str, Any] = field(default_factory=dict)
    #: The `ProcessingRun` id `record_workflow_result` created for this run,
    #: when a `DataModelRecorder` is attached to the session. `None`
    #: otherwise (including for a session without a recorder).
    processing_run_id: str | None = None
    run_id: str | None = None
    status: WorkflowStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    compiled_workflow: CompiledWorkflow | None = None
    step_records: tuple[StepExecutionRecord, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    warnings: tuple[CapturedWarning, ...] = ()
    execution_context: ExecutionContext | None = None
    resolved_output_dir: Path | None = None
    manifest_path: Path | None = None

    @property
    def session(self) -> M3Session:
        """Return the session used by this workflow's context."""

        return self.context.session

    def value(self, key: str) -> Any:
        """Return a context value by key.

        Raises WorkflowSpecError if the key is unavailable.
        """

        return self.context.get(key)
