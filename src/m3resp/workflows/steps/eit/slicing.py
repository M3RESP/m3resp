"""Registered EIT cutting steps: keep a time window of the loaded recording
(`eit.slice_recording`) or of one signal (`eit.slice_signal`)."""

from __future__ import annotations

from typing import Any, Literal

from m3resp.core.session import M3Session
from m3resp.processing.slicing import slice_by_time, slice_signal_by_mode
from m3resp.workflows.registry import (
    ANY_ARTIFACT_TYPE,
    StepArtifact,
    StepParameter,
    register_step,
)

from ._shared import _EITPROCESSING, _SESSION_ARTIFACT
from .loading import _LOAD_OUTPUTS, _LOAD_WRITES, _loaded_recording_outputs


@register_step(
    "eit.slice_signal",
    reads={"signal": "raw_eit"},
    aliases=("eit.slice",),
    writes=("result",),
    summary="Cut one EIT signal to a window, by sample index or time.",
    description="Copy a window of one EIT Signal, by sample index or by time in seconds on its own clock. For Draeger data the clock may be the time of day. The boundary flags in time mode follow eitprocessing's sample-selection rules. Use eit.slice_recording to cut the loaded recording.",
    category="preprocessing",
    modality="eit",
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type=ANY_ARTIFACT_TYPE,
            default_context_key="raw_eit",
            description="EIT signal to cut: an m3resp Signal or eitprocessing data.",
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
        StepParameter(
            name="start_inclusive",
            value_type="boolean",
            default=False,
            description="In time mode, keep the sample before a start that falls between samples. An exact start is kept with either setting.",
        ),
        StepParameter(
            name="end_inclusive",
            value_type="boolean",
            default=False,
            description="In time mode, keep the first sample at or after the end time.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="result",
            artifact_type=ANY_ARTIFACT_TYPE,
            description="The cut signal, of the same type as the input.",
        ),
    ),
)
def slice_signal(
    signal: Any,
    *,
    start: float,
    end: float,
    mode: Literal["index", "time"] = "index",
    start_inclusive: bool = False,
    end_inclusive: bool = False,
) -> dict[str, Any]:
    if mode == "time":
        return {
            "result": slice_by_time(
                signal,
                start=start,
                end=end,
                start_inclusive=start_inclusive,
                end_inclusive=end_inclusive,
            )
        }
    if start_inclusive or end_inclusive:
        raise ValueError("The inclusive boundary flags apply to mode='time'.")
    return {
        "result": slice_signal_by_mode(signal, start=start, end=end, slicing_mode=mode)
    }


@register_step(
    "eit.slice_recording",
    reads={"session": "session"},
    writes=("eit_slice", *_LOAD_WRITES),
    replaces=_LOAD_WRITES,
    summary="Keep only a time window of the loaded EIT recording.",
    description="Keep a time window of the native EIT images, waveforms, device marks and breaths, and rebuild the eitprocessing Sequence. A ventilator recording from the same EIT file is cut at the same frames. The outputs have the same names as eit.load (raw_eit, eit_sequence, ...), and the cut raw Signals replace the full ones in session.signals. Run before preprocessing.",
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    session_reads=("session.raw.eit",),
    session_writes=(
        "session.raw.eit",
        "session.raw.ventilator",
        "session.start_times",
        "session.signals",
    ),
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
        *_LOAD_OUTPUTS,
    ),
)
def slice_recording(
    session: M3Session,
    *,
    start_seconds: float,
    end_seconds: float | None = None,
) -> dict[str, Any]:
    """Cut the loaded EIT recording to a window.

    Returns:
        dict[str, Any]: ``eit_slice`` (the kept window, see
            `M3Session.slice_eit`) and the cut recording under the names
            `eit.load` uses: ``raw_eit``, ``raw_global_impedance``,
            ``eit_sequence``, ``raw_global_impedance_signal`` and
            ``raw_pixel_impedance_signal``.
    """

    summary = session.slice_eit(start_seconds, end_seconds)
    return {
        "eit_slice": summary,
        **_loaded_recording_outputs(session, replace_raw_signals=True),
    }
