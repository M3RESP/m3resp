"""Minimal combined EIT and EMG workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from m3resp import M3Session


def run_multimodal_workflow(
    eit_path: str | Path,
    emg_path: str | Path,
    eit_vendor: str | None = None,
    **kwargs: Any,
) -> M3Session:
    """Load EIT and EMG data into one session."""

    session = M3Session()
    session.load_eit(eit_path, vendor=eit_vendor, **kwargs.get("eit", {}))
    session.load_emg(emg_path, **kwargs.get("emg", {}))
    return session
