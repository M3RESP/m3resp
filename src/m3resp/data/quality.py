"""``QualityFlag``: a pass/fail quality check result.

Mirrors the persisted ``QualityAnnotation`` entity
(``m3resp.datamodel.entities``) but is the lightweight, in-memory object a
quality check actually produces during a workflow run; conversion to
``QualityAnnotation`` happens in ``DataModelRecorder``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from m3resp.data.categories import normalize_category
from m3resp.data.parameters import check_breath_scope

#: Controlled vocabulary for ``QualityFlag.severity`` (doc Sec 12).
Severity = Literal["info", "warning", "error", "critical"]
_VALID_SEVERITIES = frozenset(get_args(Severity))


@dataclass
class QualityFlag:
    """The outcome of one quality check against a signal, breath, or run.

    A flag covers the whole signal by default. To represent a signal that's
    only bad for part of its duration (e.g. good except for a noisy period
    in the middle), emit one `QualityFlag` per affected window with
    `start_time`/`end_time` set, rather than a single flag for the whole
    signal - `signal_name` ties them all back to the same signal.

    A flag about one breath sets `breath_id`. A flag about several breaths
    judged together (for example ten breaths marked as a stable stretch, or
    a set of breaths with gaps between them) sets `breath_ids`. Set one or
    the other; setting both, or an empty `breath_ids` list, raises
    `ValueError`. The per-breath steps use the breath's zero-based position
    in its breath list as the id (``"0"``, ``"1"``, ...), the same as for
    `ParameterResult.breath_id`.
    """

    name: str
    passed: bool
    severity: Severity
    modality: str | None = None
    category: str | None = None
    signal_name: str | None = None
    breath_id: str | None = None
    breath_ids: list[str] | None = None
    start_time: float | None = None
    end_time: float | None = None
    message: str | None = None
    value: float | None = None
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_breath_scope(self.breath_id, self.breath_ids, "QualityFlag")
        self.category = normalize_category(self.category) or self.category
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Unknown QualityFlag.severity {self.severity!r}; expected one "
                f"of {sorted(_VALID_SEVERITIES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation, one key per field."""

        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "modality": self.modality,
            "category": self.category,
            "signal_name": self.signal_name,
            "breath_id": self.breath_id,
            "breath_ids": None if self.breath_ids is None else list(self.breath_ids),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "metadata": dict(self.metadata),
        }
