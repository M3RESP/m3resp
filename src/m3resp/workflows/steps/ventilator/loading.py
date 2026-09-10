"""Registered ventilator loading/channel-splitting pipeline steps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from m3resp.adapters.resurfemg_adapter import ventilator_signals
from m3resp.adapters.ventilator_adapter import DEFAULT_CHANNELS
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
            name="file_path",
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
def load(session: M3Session, *, file_path: str) -> dict[str, Any]:
    # Delegates to the session method - the same shape as `emg.load` calling
    # `session.load_emg` - so provenance and `session.raw` bookkeeping happen
    # in one place. The step still emits the raw payload dict, which is what
    # `ventilator.channels` downstream expects.
    session.load_ventilator(file_path, verbose=False)
    recording = session.ventilator
    assert recording is not None
    return {"ventilator_raw": recording.data}


@register_step(
    "ventilator.channels",
    aliases=("emg.ventilator_channels",),
    reads={"ventilator_raw": "ventilator_raw"},
    writes=("ventilator_signals",),
    summary="Split a raw ventilator recording into its named channels.",
    description="Split a raw ventilator recording into named channel arrays plus sample frequency, resolving channels by label where the recording provides them.",
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
            name="channels",
            value_type="list",
            default=DEFAULT_CHANNELS,
            description="Quantities to extract, found by channel label. Beyond the three standard ones a pressure pod can also supply 'esophageal_pressure', 'transpulmonary_pressure' or 'gastric_pressure'.",
        ),
        StepParameter(
            name="pressure_channel",
            value_type="integer",
            required=False,
            default=None,
            minimum=0,
            description="Explicit column index for airway pressure, overriding the label match. The way to read an unlabelled recording whose columns are not in the default order.",
            advanced=True,
        ),
        StepParameter(
            name="flow_channel",
            value_type="integer",
            required=False,
            default=None,
            minimum=0,
            description="Explicit column index for flow, overriding the label match.",
            advanced=True,
        ),
        StepParameter(
            name="volume_channel",
            value_type="integer",
            required=False,
            default=None,
            minimum=0,
            description="Explicit column index for volume, overriding the label match.",
            advanced=True,
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
    channels: Sequence[str] = DEFAULT_CHANNELS,
    pressure_channel: int | None = None,
    flow_channel: int | None = None,
    volume_channel: int | None = None,
    fs: float | None = None,
) -> dict[str, Any]:
    """Split a ventilator recording into channels found by name.

    `channels` selects which quantities to extract - the three standard ones
    by default, but a recording from a pressure pod can also be asked for
    ``esophageal_pressure``, ``transpulmonary_pressure`` or
    ``gastric_pressure``. The per-channel index parameters override the name
    match with an explicit column, and remain the way to read an unlabelled
    recording whose columns are not in the default order.
    """

    signals = ventilator_signals(
        ventilator_raw,
        channels=tuple(channels),
        pressure_channel=pressure_channel,
        flow_channel=flow_channel,
        volume_channel=volume_channel,
        fs=fs,
    )
    return {"ventilator_signals": signals}
