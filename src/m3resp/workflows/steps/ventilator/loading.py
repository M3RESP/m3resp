"""Registered ventilator loading/channel-splitting pipeline steps."""

from __future__ import annotations

from typing import Any

from m3resp.adapters.resurfemg_adapter import ventilator_signals
from m3resp.core.session import M3Session
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import _RESURFEMG, _SESSION_ARTIFACT


@register_step(
    "ventilator.load",
    aliases=("emg.load_ventilator",),
    reads={"session": "session"},
    writes=("ventilator_raw",),
    summary="Load a ventilator recording into the session.",
    description="Load a ventilator recording file through ReSurfEMGAdapter.",
    category="loading",
    modality="ventilator",
    optional_packages=_RESURFEMG,
    session_writes=("session.raw.ventilator",),
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="file",
            value_type="path",
            required=True,
            path_kind="file",
            description="Ventilator recording file to load.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ventilator_raw",
            artifact_type="ventilator_recording",
            description="Raw upstream ventilator recording dict.",
            compatibility_only=True,
        ),
    ),
)
def load(session: M3Session, *, file: str) -> dict[str, Any]:
    # Delegates to the session method - the same shape as `emg.load` calling
    # `session.load_emg` - so provenance and `session.raw` bookkeeping happen
    # in one place. The step still emits the raw payload dict, which is what
    # `ventilator.channels` downstream expects.
    session.load_ventilator(file, verbose=False)
    recording = session.ventilator
    assert recording is not None
    return {"ventilator_raw": recording.data}


@register_step(
    "ventilator.channels",
    aliases=("emg.ventilator_channels",),
    reads={"ventilator_raw": "ventilator_raw"},
    writes=("ventilator_signals",),
    summary="Split a raw ventilator recording into pressure/flow/volume channels.",
    description="Split a raw ventilator recording into named pressure/flow/volume channel arrays plus sample frequency.",
    category="preprocessing",
    modality="ventilator",
    input_artifacts=(
        StepArtifact(
            name="ventilator_raw",
            artifact_type="ventilator_recording",
            description="Raw ventilator recording from 'ventilator.load'.",
            compatibility_only=True,
        ),
    ),
    parameters=(
        StepParameter(
            name="pressure_channel",
            value_type="integer",
            default=0,
            minimum=0,
            description="Channel index of airway pressure.",
        ),
        StepParameter(
            name="flow_channel",
            value_type="integer",
            default=1,
            minimum=0,
            description="Channel index of flow.",
        ),
        StepParameter(
            name="volume_channel",
            value_type="integer",
            default=2,
            minimum=0,
            description="Channel index of volume.",
        ),
        StepParameter(
            name="fs",
            value_type="number",
            required=False,
            default=None,
            unit="Hz",
            description="Sample frequency override; defaults to the recording's own metadata.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Mapping of 'pressure'/'flow'/'volume' arrays plus 'fs'.",
        ),
    ),
)
def channels(
    ventilator_raw: Any,
    *,
    pressure_channel: int = 0,
    flow_channel: int = 1,
    volume_channel: int = 2,
    fs: float | None = None,
) -> dict[str, Any]:
    signals = ventilator_signals(
        ventilator_raw,
        pressure_channel=pressure_channel,
        flow_channel=flow_channel,
        volume_channel=volume_channel,
        fs=fs,
    )
    return {"ventilator_signals": signals}
