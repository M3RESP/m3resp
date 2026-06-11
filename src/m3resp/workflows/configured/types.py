"""Types for YAML-configured M3Resp workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from m3resp.core.session import M3Session

ConfiguredWorkflowKind = Literal["eit", "emg", "multimodal"]


@dataclass(frozen=True)
class WorkflowResult:
    """Result returned by YAML-configured workflows."""

    session: M3Session
    summary: dict[str, Any]
    output_dir: Path
    figures: dict[str, Path]
