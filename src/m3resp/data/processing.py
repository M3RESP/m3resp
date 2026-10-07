"""``ProcessingStep``/``ProcessingHistory``: what produced a result.

This is the runtime counterpart of ``ProvenanceRecord``
(``m3resp.core.provenance``): where ``ProvenanceRecord`` is Stage 1's minimal
"action + modality + parameters" log entry, ``ProcessingStep`` also names the
context keys a step read and wrote, so the persisted
``ProcessingRun.input_file_ids`` can be filled in precisely. Both stay in use
until every pipeline step emits ``ProcessingStep``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ProcessingStep:
    """One step of a processing pipeline."""

    name: str
    input_keys: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    software: str = "m3resp"
    version: str | None = None
    #: Installed version of each optional upstream package (e.g.
    #: ``resurfemg``, ``eitprocessing``) this step's operation declared a
    #: dependency on, or ``None`` if that package wasn't importable when the
    #: step ran. Without this, exact reconstruction of a step's numeric
    #: output isn't possible even with deterministic code, since an upstream
    #: library upgrade between runs can change results under the same
    #: operation name and parameters.
    optional_package_versions: dict[str, str | None] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    #: The step's outcome: "succeeded", "failed" or "cancelled".
    status: str = "succeeded"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingHistory:
    """An ordered log of :class:`ProcessingStep` entries for one session or run."""

    steps: list[ProcessingStep] = field(default_factory=list)

    def record(self, name: str, **kwargs: Any) -> ProcessingStep:
        """Create, append, and return a new :class:`ProcessingStep`."""

        step = ProcessingStep(name=name, **kwargs)
        self.steps.append(step)
        return step

    def to_list(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self.steps]

    def __iter__(self) -> Iterator[ProcessingStep]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)
