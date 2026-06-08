"""EMG modality containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EMGRecording:
    """Loaded EMG recording with source metadata."""

    data: Any
    path: Path
