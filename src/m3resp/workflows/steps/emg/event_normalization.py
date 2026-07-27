"""Registered ventilator-breath-event normalization pipeline step."""

from __future__ import annotations

from typing import Any

from m3resp.core.session import M3Session
from m3resp.synchronization.ventilator import (
    iter_ventilator_detections,
    normalize_ventilator_breath,
)
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import (
    _SESSION_ARTIFACT,
)


@register_step(
    "ventilator.normalize_breaths",
    aliases=("emg.normalize_ventilator_breaths",),
    reads={
        "ventilator_breath_indices": "ventilator_breath_indices",
        "ventilator_signals": "ventilator_signals",
        "session": "session",
    },
    writes=(),
    summary="Normalize detected ventilator breath indices into session events.",
    description="Convert detected ventilator breath peak indices into native BreathEvents and store them on the session as 'ventilator_breaths'.",
    category="detection",
    modality="ventilator",
    input_artifacts=(
        StepArtifact(
            name="ventilator_breath_indices",
            artifact_type="index_array",
            description="Ventilator breath peak indices from 'ventilator.detect_breaths'.",
        ),
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle supplying 'fs'.",
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="breath_width_seconds",
            value_type="number",
            default=0.5,
            unit="s",
            minimum=0,
            description="Assumed breath width used to derive start/end times around each peak.",
        ),
    ),
)
def normalize_ventilator_breaths(
    ventilator_breath_indices: Any,
    ventilator_signals: Any,
    session: M3Session,
    *,
    breath_width_seconds: float = 0.5,
) -> dict[str, Any]:
    fs = float(ventilator_signals["fs"])
    events = [
        normalize_ventilator_breath(
            detection, fs=fs, width_seconds=breath_width_seconds
        )
        for detection in iter_ventilator_detections(ventilator_breath_indices)
    ]
    session.add_events("ventilator_breaths", events)
    return {}
