"""Minimal EIT workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from m3resp import M3Session


def run_eit_workflow(
    path: str | Path, vendor: str | None = None, **kwargs: Any
) -> M3Session:
    """Load EIT data and return the populated session."""

    session = M3Session()
    session.load_eit(path, vendor=vendor, **kwargs)
    return session
