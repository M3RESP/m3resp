"""Registered EMG pipeline steps.

These wrap the ``M3Session`` EMG stage methods so EMG processing runs through the
same engine as everything else. The methods already dispatch ReSurfEMG functions
at per-function granularity (see ``ReSurfEMGAdapter`` ``POSTPROCESSING_FUNCTIONS``),
so wrapping them keeps that granularity while unifying orchestration.
"""

from __future__ import annotations

from typing import Any  # noqa: F401 (used in step signatures below)

from m3resp.core.session import M3Session
from m3resp.pipeline.registry import register_step


@register_step(
    "emg.load",
    reads={"session": "session"},
    writes=(),
    summary="Load EMG (and optionally ventilator) recordings into the session.",
)
def load(
    session: M3Session,
    *,
    file: str,
    vent_file: str | None = None,
) -> dict[str, Any]:
    session.load_emg(file, verbose=False)
    if vent_file is not None:
        session.raw["vent"] = session.emg_adapter.load(str(vent_file), verbose=False)
    return {}


@register_step(
    "emg.preprocess",
    reads={"session": "session"},
    writes=("processed_emg",),
    summary="Filter EMG and compute its envelope.",
)
def preprocess(session: M3Session, **kwargs: Any) -> dict[str, Any]:
    return {"processed_emg": session.preprocess_emg(**kwargs)}


@register_step(
    "emg.detect_breaths",
    reads={"session": "session"},
    writes=(),
    summary="Detect EMG breaths from the envelope.",
)
def detect_breaths(session: M3Session, **kwargs: Any) -> dict[str, Any]:
    session.detect_emg_breaths(**kwargs)
    return {}


@register_step(
    "emg.postprocess",
    reads={"session": "session"},
    writes=(),
    summary="Run selected ReSurfEMG postprocessing functions.",
)
def postprocess(session: M3Session, **kwargs: Any) -> dict[str, Any]:
    # Fall back to the loaded ventilator data from the session when the caller
    # didn't embed it in the spec (e.g. a pure YAML spec can't hold numpy arrays).
    if "ventilator" not in kwargs:
        kwargs["ventilator"] = session.raw.get("vent")
    session.postprocess_emg(**kwargs)
    return {}
