"""``QualityFlag``: a pass/fail quality check result (plan_stage2.md Sec 12).

Mirrors the persisted ``QualityAnnotation`` entity
(``m3resp.datamodel.entities``) but is the lightweight, in-memory object a
quality check actually produces during a pipeline run; conversion to
``QualityAnnotation`` happens at the ``DataModelRecorder`` boundary
(``plan/stage2_consolidation.md``), not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

#: Controlled vocabulary for ``QualityFlag.severity`` (doc Sec 12).
Severity = Literal["info", "warning", "error", "critical"]
_VALID_SEVERITIES = frozenset(get_args(Severity))


@dataclass
class QualityFlag:
    """The outcome of one quality check against a signal, breath, or run."""

    name: str
    passed: bool
    severity: Severity
    modality: str | None = None
    signal_name: str | None = None
    breath_id: str | None = None
    message: str | None = None
    value: float | None = None
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Unknown QualityFlag.severity {self.severity!r}; expected one "
                f"of {sorted(_VALID_SEVERITIES)}"
            )
