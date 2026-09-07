"""Helpers for recording how session data was produced."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ProvenanceRecord:
    """A single processing step recorded in a session."""

    action: str
    modality: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def record(
    action: str, modality: str | None = None, **parameters: Any
) -> ProvenanceRecord:
    """Create a provenance record with UTC timestamp."""

    return ProvenanceRecord(action=action, modality=modality, parameters=parameters)
