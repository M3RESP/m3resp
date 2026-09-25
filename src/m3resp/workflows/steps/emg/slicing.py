"""Registered EMG cutting step: keep a time window of the loaded recording."""

from __future__ import annotations

from typing import Any

from m3resp.core.session import M3Session
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import _SESSION_ARTIFACT


@register_step(
    "emg.slice_recording",
    reads={"session": "session"},
    # Named `emg.slice` until the slicing steps were renamed:
    # `*.slice_recording` cuts the loaded recording, `*.slice_signal` one signal.
    aliases=("emg.slice",),
    writes=("emg_slice",),
    summary="Keep only a time window of the loaded EMG recording.",
    description="Keep only the part of the loaded EMG recording between two times, for example to leave out a stretch where another device disturbed the EMG. Ventilator channels from the same file are cut the same way. Run before emg.preprocess.",
    category="preprocessing",
    modality="emg",
    session_reads=("session.raw.emg",),
    session_writes=("session.raw.emg", "session.raw.ventilator", "session.start_times"),
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
    ),
    output_artifacts=(
        StepArtifact(
            name="emg_slice",
            artifact_type="emg_slice_summary",
            description="The kept window and how many samples were removed at each end.",
        ),
    ),
)
def slice_recording(
    session: M3Session,
    *,
    start_seconds: float,
    end_seconds: float | None = None,
) -> dict[str, Any]:
    return {"emg_slice": session.slice_emg(start_seconds, end_seconds)}
