"""EIT modality containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EITRecording:
    """Loaded EIT recording with source metadata."""

    data: Any
    path: Path
    vendor: str | None = None
