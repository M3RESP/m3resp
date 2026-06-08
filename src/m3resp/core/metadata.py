"""Metadata containers for sessions and modalities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionMetadata:
    """Lightweight metadata attached to an M3Resp session."""

    subject_id: str | None = None
    recording_id: str | None = None
    notes: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
