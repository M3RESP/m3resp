"""Registered EIT cutting steps: keep a time window of the loaded recording
(`eit.slice_recording`) or of one signal (`eit.slice_signal`)."""

from __future__ import annotations

from typing import Any, Literal

from m3resp.core.session import M3Session
from m3resp.processing.slicing import slice_signal_by_mode
from m3resp.workflows.registry import (
    ANY_ARTIFACT_TYPE,
    StepArtifact,
    StepParameter,
    register_step,
)

from ._shared import _EITPROCESSING, _SESSION_ARTIFACT


@register_step(
    "eit.slice_signal",
    reads={"signal": "raw_eit"},
    # Named `eit.slice` until the slicing steps were renamed:
    # `*.slice_recording` cuts the loaded recording, `*.slice_signal` one signal.
    aliases=("eit.slice",),
    writes=("result",),
    summary="Cut one EIT signal to a window, by sample index or time.",
    description="Cut one EIT signal passed along in the workflow to a window, e.g. to select a detection window. Works on m3resp Signals and on eitprocessing data (raw EIT, global impedance, a Sequence). The loaded recording in the session is not changed; use eit.slice_recording for that. In time mode, times are the signal's own time values: for data read from an EIT file that is the time of day stored in the file.",
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="signal",
            # Genuine passthrough - this step accepts *any* upstream EIT
            # signal (raw, filtered, global impedance, ...), not one fixed
            # type, so it uses the "any" sentinel rather than a specific
            # artifact type (Phase 10 artifact-type compatibility check).
            artifact_type=ANY_ARTIFACT_TYPE,
            default_context_key="raw_eit",
            description="EIT signal to cut: an m3resp Signal or eitprocessing data.",
            compatibility_only=True,
        ),
    ),
    parameters=(
        StepParameter(
            name="start",
            value_type="number",
            required=True,
            description="Start of the window: a sample index (mode='index') or a time value on the signal's own time axis, in seconds (mode='time').",
        ),
        StepParameter(
            name="end",
            value_type="number",
            required=True,
            description="End of the window, not included: a sample index (mode='index') or a time value on the signal's own time axis, in seconds (mode='time').",
        ),
        StepParameter(
            name="mode",
            value_type="choice",
            default="index",
            choices=("index", "time"),
            description="Whether 'start'/'end' are sample indices or seconds.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="result",
            artifact_type=ANY_ARTIFACT_TYPE,
            description="The cut signal, of the same type as the input.",
            compatibility_only=True,
        ),
    ),
)
def slice_signal(
    signal: Any,
    *,
    start: float,
    end: float,
    mode: Literal["index", "time"] = "index",
) -> dict[str, Any]:
    return {
        "result": slice_signal_by_mode(signal, start=start, end=end, slicing_mode=mode)
    }


@register_step(
    "eit.slice_recording",
    reads={"session": "session"},
    writes=("eit_slice",),
    summary="Keep only a time window of the loaded EIT recording.",
    description="Keep only the part of the loaded EIT recording between two times, for example to analyse one ventilator setting or leave out a disturbed stretch. The cutting is done by eitprocessing, so pixel data, global impedance, pressure/flow channels and markers are cut to the same frames. A ventilator recording from the same EIT file is cut the same way. Run before eit.preprocess.",
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    session_reads=("session.raw.eit",),
    session_writes=("session.raw.eit", "session.raw.ventilator", "session.start_times"),
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="start_seconds",
            value_type="number",
            required=True,
            unit="s",
            minimum=0,
            description="Start of the part to keep, in seconds from the first frame of the recording as it is now (not the time of day stored in the file).",
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
            name="eit_slice",
            artifact_type="eit_slice_summary",
            description="The kept window and how many frames were removed at each end.",
        ),
    ),
)
def slice_recording(
    session: M3Session,
    *,
    start_seconds: float,
    end_seconds: float | None = None,
) -> dict[str, Any]:
    return {"eit_slice": session.slice_eit(start_seconds, end_seconds)}
