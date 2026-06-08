"""Minimal EMG workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from m3resp import M3Session


def run_emg_workflow(path: str | Path, **kwargs: Any) -> M3Session:
    """Load EMG data and return the populated session."""

    session = M3Session()
    session.load_emg(path, **kwargs)
    return session
