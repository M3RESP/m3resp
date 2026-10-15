"""Registered ventilator-breath-event normalization workflow step."""

from __future__ import annotations

from typing import Any

from m3resp.adapters.ventilator_adapter import (
    iter_ventilator_detections,
    normalize_ventilator_breath,
)
from m3resp.core.session import M3Session
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
    session_writes=("session.events.ventilator_breaths",),
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
def normalize_breaths(
    ventilator_breath_indices: Any,
    ventilator_signals: Any,
    session: M3Session,
    *,
    breath_width_seconds: float = 0.5,
) -> dict[str, Any]:
    """Store ventilator breath detections as session events.

    Args:
        ventilator_breath_indices (Any): Sample indices, or detections containing
            existing breath times.
        ventilator_signals (Any): Channel bundle supplying fs in Hz and a channel
            array whose length defines the recording duration.
        session (M3Session): Session receiving ventilator_breaths events.
        breath_width_seconds (float): Total width of the window centered on a
            sample-index detection. Defaults to 0.5 s. Windows are clipped to the
            recording's bounds when its duration is available. Existing event
            boundaries are retained.

    Returns:
        dict[str, Any]: Empty mapping. Events are stored by `session.add_events`,
            which also records provenance. Event times are in seconds from the
            recording's first sample, with modality "ventilator".
    """

    fs = float(ventilator_signals["fs"])
    # All channels of one recording have the same length, so any of them
    # gives the recording length; airway pressure may not have been asked for.
    channels = ventilator_signals.get("channels") or {
        name: ventilator_signals.get(name)
        for name in ("airway_pressure", "flow", "volume")
    }
    lengths = [len(values) for values in channels.values() if values is not None]
    duration_seconds = lengths[0] / fs if lengths and fs else None
    events = [
        normalize_ventilator_breath(
            detection,
            fs=fs,
            width_seconds=breath_width_seconds,
            duration_seconds=duration_seconds,
        )
        for detection in iter_ventilator_detections(ventilator_breath_indices)
    ]
    session.add_events("ventilator_breaths", events)
    return {}
