"""Shared execution-result type and step-registration bookkeeping for the
engine package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from m3resp.core.session import M3Session
from m3resp.workflows.context import (
    PipelineContext,
)
from m3resp.workflows.diagnostics import Diagnostic
from m3resp.workflows.lifecycle import (
    CapturedWarning,
    ExecutionContext,
    PipelineStatus,
    StepExecutionRecord,
)

if TYPE_CHECKING:
    # Deferred at runtime: compiler.py imports collect_diagnostics from this
    # package, so importing it back at module scope here would be circular.
    from m3resp.workflows.compiler import CompiledPipeline
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
class PipelineResult:
    """Outcome of running a pipeline."""

    name: str
    context: PipelineContext
    outputs: dict[str, Any] = field(default_factory=dict)
    #: The `ProcessingRun` id `record_pipeline_result` created for this run,
    #: when a `DataModelRecorder` is attached to the session. `None`
    #: otherwise (including for a session without a recorder).
    processing_run_id: str | None = None
    #: All additive; existing fields above are unchanged.
    run_id: str | None = None
    status: PipelineStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    compiled_pipeline: "CompiledPipeline | None" = None
    step_records: tuple[StepExecutionRecord, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    warnings: tuple[CapturedWarning, ...] = ()
    execution_context: ExecutionContext | None = None
    resolved_output_dir: Path | None = None
    manifest_path: Path | None = None

    @property
    def session(self) -> M3Session:
        return self.context.session

    def value(self, key: str) -> Any:
        """Return a produced artifact by context key."""

        return self.context.get(key)
