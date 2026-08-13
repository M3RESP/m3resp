"""Structured, JSON-safe validation diagnostics (Phase 3.6 of the
pipeline-structure plan).

A :class:`Diagnostic` describes one independent problem found while
compiling/validating a spec, without raising - so a caller (CLI, GUI, or
:func:`m3resp.workflows.compiler.validate_pipeline`) can report every
problem in one pass instead of stopping at the first. ``validate_spec``
remains a raising compatibility wrapper around the same collection logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class Diagnostic:
    """One structural or readiness finding."""

    severity: Severity
    code: str
    message: str
    step_id: str | None = None
    step_position: int | None = None
    operation_id: str | None = None
    document_path: str | None = None
    suggestion: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "step_id": self.step_id,
            "step_position": self.step_position,
            "operation_id": self.operation_id,
            "document_path": self.document_path,
            "suggestion": self.suggestion,
        }
