"""``ProcessingStep``/``ProcessingHistory``: what produced a result (plan_stage2.md Sec 13).

This is the runtime counterpart of ``ProvenanceRecord``
(``m3resp.core.provenance``): where ``ProvenanceRecord`` is Stage 1's minimal
"action + modality + parameters" log entry, ``ProcessingStep`` additionally
names the context keys a step read and wrote, which is what lets the
persisted ``ProcessingRun.input_file_ids`` (see
``plan/stage2_consolidation.md``) be filled in precisely instead of guessed.
``ProvenanceRecord`` is not replaced by this - both stay in use until pipeline
steps are migrated to emit ``ProcessingStep`` (Milestone 2.3+).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

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
