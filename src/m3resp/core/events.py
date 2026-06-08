"""Common event models used across modalities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """A timestamped event from one modality."""

    name: str
    modality: str
    time: float
    sample_index: int | None = None
    label: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BreathEvent:
    """A respiratory event represented on a common time axis."""

    modality: str
    start_time: float
    end_time: float
    peak_time: float | None = None
    source: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
