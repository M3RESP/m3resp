"""Registered ventilator cutting step: keep a time window of the standalone
ventilator recordings."""

from __future__ import annotations

from typing import Any

from m3resp.core.session import M3Session
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import _SESSION_ARTIFACT


@register_step(
    "ventilator.slice_recording",
    reads={"session": "session"},
    writes=("ventilator_slice",),
    summary="Keep only a time window of the standalone ventilator recordings.",
    description="Keep only the part of the standalone ventilator recordings (loaded with source 'ventilator') between two times: the one named, or all of them. Each cut recording's start time moves later so it stays lined up. Ventilator data from the EIT or EMG file is cut with its host by eit.slice_recording or emg.slice_recording instead. Run before the ventilator preprocessing steps.",
    category="preprocessing",
    modality="ventilator",
    session_reads=("session.raw.ventilator",),
    session_writes=("session.raw.ventilator", "session.start_times"),
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="start_seconds",
            value_type="number",
            required=True,
            unit="s",
            minimum=0,
            description="Start of the part to keep, in seconds from the start of the recording as it is now (after any earlier slicing).",
        ),
        StepParameter(
            name="end_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="End of the part to keep, in seconds. Leave empty to keep everything up to the end.",
        ),
        StepParameter(
            name="name",
            value_type="string",
            required=False,
            default=None,
            description="Cut only the ventilator recording loaded under this name. Leave empty to cut every standalone ventilator recording.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ventilator_slice",
            artifact_type="ventilator_slice_summary",
            description="The kept window and how many samples were removed at each end, per recording.",
        ),
    ),
)
def slice_recording(
    session: M3Session,
    *,
    start_seconds: float,
    end_seconds: float | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    return {
        "ventilator_slice": session.slice_ventilator(
            start_seconds, end_seconds, name=name
        )
    }
