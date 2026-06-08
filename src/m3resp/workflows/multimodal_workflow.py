"""Minimal combined EIT and EMG workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from m3resp import M3Session


def run_multimodal_workflow(
    eit_path: str | Path,
    emg_path: str | Path,
    eit_vendor: str | None = None,
    process_eit: bool = True,
    detect_eit_breaths: bool = True,
    process_emg: bool = True,
    detect_emg_breaths: bool = True,
    postprocess_emg: bool = True,
    eit_adapter: Any = None,
    emg_adapter: Any = None,
    **kwargs: Any,
) -> M3Session:
    """Load EIT and EMG data, then run default Stage 1 pipelines."""

    if detect_eit_breaths and not process_eit:
        raise ValueError("detect_eit_breaths=True requires process_eit=True")
    if detect_emg_breaths and not process_emg:
        raise ValueError("detect_emg_breaths=True requires process_emg=True")
    if postprocess_emg and not process_emg:
        raise ValueError("postprocess_emg=True requires process_emg=True")

    session = M3Session(eit_adapter=eit_adapter, emg_adapter=emg_adapter)
    session.load_eit(eit_path, vendor=eit_vendor, **kwargs.get("eit", {}))
    session.load_emg(emg_path, **kwargs.get("emg", {}))

    if process_eit:
        session.preprocess_eit(**kwargs.get("eit_preprocess", {}))
    if detect_eit_breaths:
        session.detect_eit_breaths(**kwargs.get("eit_detection", {}))
    if process_emg:
        session.preprocess_emg(**kwargs.get("emg_preprocess", {}))
    if detect_emg_breaths:
        session.detect_emg_breaths(**kwargs.get("emg_detection", {}))
    if postprocess_emg:
        session.postprocess_emg(**kwargs.get("emg_postprocessing", {}))

    return session
