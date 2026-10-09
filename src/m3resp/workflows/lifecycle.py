"""Execution lifecycle types for declarative workflows: workflow/step status, per-step execution records,
a structured execution error, deliberate warning capture, framework-neutral
progress events, cooperative cancellation, and deterministic execution
context metadata.
"""

from __future__ import annotations

import platform
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import Any, Literal

import numpy as np

#: Shared by both the overall run and each step (Phase 4.1).
WorkflowStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]
StepStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]

ProgressEventType = Literal[
    "workflow_started",
    "step_started",
    "step_warning",
    "step_completed",
    "step_failed",
    "workflow_completed",
    "workflow_failed",
    "workflow_cancelled",
]

#: Receives one JSON-safe event mapping per call (Phase 4.4). Never raises
#: framework/GUI objects or scientific arrays at the caller.
EventSink = Callable[[dict[str, Any]], None]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_run_id() -> str:
    return uuid.uuid4().hex


def optional_package_version(name: str) -> str | None:
    """Installed version of ``name``, or ``None`` if not installed - used for
    provenance without ever importing the package itself."""

    try:
        return _package_version(name)
    except PackageNotFoundError:
        return None


def summarize_output_value(value: Any) -> Any:
    """A small, JSON-safe summary of a produced value for lifecycle logs.

    Never the raw data (Phase 4.1: "do not serialize raw signal arrays into
    lifecycle logs") - only type/shape/length information.
    """

    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return {"type": type(value).__name__, "shape": list(shape)}
        except TypeError:
            pass
    if isinstance(value, list | tuple):
        return {"type": type(value).__name__, "length": len(value)}
    if isinstance(value, dict):
        return {"type": "mapping", "keys": sorted(str(key) for key in value)}
    return {"type": type(value).__name__}


@dataclass(frozen=True)
class CapturedWarning:
    """One Python warning captured during a step's execution (Phase 4.3)."""

    message: str
    category: str

    def as_dict(self) -> dict[str, Any]:
        return {"message": self.message, "category": self.category}


@dataclass
class StepExecutionRecord:
    """Provenance/lifecycle record for one executed step (Phase 4.1)."""

    step_id: str
    position: int
    operation_id: str
    status: StepStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    #: Resolved static parameters (already non-sensitive: static config
    #: values, not signal data).
    parameters: dict[str, Any] = field(default_factory=dict)
    input_context_keys: dict[str, str] = field(default_factory=dict)
    output_context_keys: dict[str, str] = field(default_factory=dict)
    output_summaries: dict[str, Any] = field(default_factory=dict)
    warnings: list[CapturedWarning] = field(default_factory=list)
    error: dict[str, Any] | None = None
    optional_package_versions: dict[str, str | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "position": self.position,
            "operation_id": self.operation_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "parameters": dict(self.parameters),
            "input_context_keys": dict(self.input_context_keys),
            "output_context_keys": dict(self.output_context_keys),
            "output_summaries": dict(self.output_summaries),
            "warnings": [w.as_dict() for w in self.warnings],
            "error": self.error,
            "optional_package_versions": dict(self.optional_package_versions),
        }


class WorkflowExecutionError(RuntimeError):
    """A step failure with its original cause and gathered execution records.

    Attributes:
        step_id: ID of the failing step.
        position: Zero-based position in the workflow.
        operation_id: Registered operation name.
        run_id: Workflow execution identifier, when supplied.
        started_at: Workflow start time as an ISO 8601 UTC string, when supplied.
        step_records: Gathered execution records, including the failed step.

    The message includes the step and cause message. The original exception
    is retained as __cause__.
    """

    def __init__(
        self,
        *,
        step_id: str,
        position: int,
        operation_id: str,
        message: str,
        cause: BaseException,
        run_id: str | None = None,
        started_at: str | None = None,
        step_records: tuple[StepExecutionRecord, ...] = (),
    ) -> None:
        """Attach the failing step, original exception and gathered run records."""

        super().__init__(
            f"Step #{position} '{operation_id}' (id={step_id}) failed: {message}"
        )
        self.step_id = step_id
        self.position = position
        self.operation_id = operation_id
        self.__cause__ = cause
        self.run_id = run_id
        self.started_at = started_at
        self.step_records = step_records


class CancellationToken:
    """A small cooperative cancellation flag (Phase 4.5).

    Checked before and after each step - never interrupts a step already
    running. Cancellation is not rollback: completed work is preserved.
    """

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


@dataclass(frozen=True)
class ExecutionContext:
    """Deterministic, JSON-safe execution-environment metadata (Phase 4.6).

    Never sets a global random seed in a third-party library - ``seed`` is
    recorded for provenance; a step that uses randomness must accept and
    record its own seed explicitly.
    """

    run_id: str
    run_timestamp: str
    m3resp_version: str | None
    python_version: str
    platform: str
    seed: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "m3resp_version": self.m3resp_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "seed": self.seed,
        }


def build_execution_context(
    *, run_id: str, run_timestamp: str, seed: int | None
) -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        run_timestamp=run_timestamp,
        m3resp_version=optional_package_version("m3resp"),
        python_version=sys.version,
        platform=platform.platform(),
        seed=seed,
    )


def make_event(event_type: ProgressEventType, **fields: Any) -> dict[str, Any]:
    """Build one JSON-safe progress event (Phase 4.4)."""

    return {"event": event_type, "timestamp": utc_now_iso(), **fields}
