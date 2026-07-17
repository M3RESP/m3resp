"""Registered EIT loading/slicing pipeline steps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from m3resp.adapters.eitprocessing_adapter import (
    continuous_data_to_signal,
)
from m3resp.core.session import M3Session
from m3resp.workflows.registry import (
    ANY_ARTIFACT_TYPE,
    StepArtifact,
    StepParameter,
    register_step,
)
from m3resp.workflows.utils import slice_signal_by_mode

from ._shared import (
    _EITPROCESSING,
    _SESSION_ARTIFACT,
    _record_step,
    _upstream_metadata,
)


@register_step(
    "eit.load",
    reads={"session": "session"},
    writes=(
        "raw_eit",
        "raw_global_impedance",
        "eit_sequence",
        "raw_global_impedance_signal",
    ),
    summary="Load an EIT recording into the session.",
    description="Load a vendor EIT recording file through EITProcessingAdapter and expose the raw pixel/global-impedance data.",
    category="loading",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="file",
            value_type="path",
            required=True,
            path_kind="file",
            description="EIT recording file to load.",
        ),
        StepParameter(
            name="vendor",
            value_type="choice",
            required=False,
            default=None,
            choices=("draeger", "sentec", "timpel"),
            description="Recording vendor. Required unless a custom loader was injected into the adapter.",
        ),
        StepParameter(
            name="loader_options",
            value_type="mapping",
            required=False,
            default=None,
            description="Extra keyword arguments forwarded to M3Session.load_eit().",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="raw_eit",
            artifact_type="eit_pixel_signal",
            description="Raw upstream EITData object (pixel impedance).",
            compatibility_only=True,
        ),
        StepArtifact(
            name="raw_global_impedance",
            artifact_type="eit_global_impedance",
            required=False,
            description="Raw upstream summed/global impedance signal object, when present.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="eit_sequence",
            artifact_type="eit_sequence",
            description="Upstream EIT Sequence container used by downstream adapter calls.",
            public=False,
            compatibility_only=True,
        ),
        StepArtifact(
            name="raw_global_impedance_signal",
            artifact_type="signal",
            required=False,
            description="Native Signal wrapping the raw global impedance, when present.",
        ),
    ),
)
def load(
    session: M3Session,
    *,
    file: str,
    vendor: str | None = None,
    loader_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if loader_options is not None and not isinstance(loader_options, Mapping):
        raise TypeError(
            "eit.load 'loader_options' must be a mapping of keyword arguments "
            f"for M3Session.load_eit(), got {type(loader_options).__name__}."
        )

    session.load_eit(file, vendor=vendor, **dict(loader_options or {}))
    recording = session.eit
    assert recording is not None

    raw_global_impedance_signal = None
    if recording.global_impedance is not None:
        raw_global_impedance_signal = continuous_data_to_signal(
            recording.global_impedance,
            modality="eit",
            channel="global_impedance",
            processing_state="raw",
            source="eitprocessing",
        )
        session.signals.add(raw_global_impedance_signal)

    _record_step(
        session,
        "eit.load",
        metadata=_upstream_metadata(
            source_function="eitprocessing.datahandling.loading.load_eit_data",
            operation="eit.load",
            parameters={"vendor": vendor, "loader_options": dict(loader_options or {})},
        ),
    )
    return {
        "raw_eit": recording.raw,
        "raw_global_impedance": recording.global_impedance,
        "eit_sequence": recording.data,
        "raw_global_impedance_signal": raw_global_impedance_signal,
    }


@register_step(
    "eit.slice",
    reads={"signal": "raw_eit"},
    writes=("result",),
    summary="Slice an EIT signal by sample index or time.",
    description="Slice any upstream EIT signal by sample index or time window, e.g. to select a detection window.",
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
            description="Upstream EIT signal to slice (any signal type).",
            compatibility_only=True,
        ),
    ),
    parameters=(
        StepParameter(
            name="start",
            value_type="number",
            required=True,
            description="Slice start: a sample index (mode='index') or seconds (mode='time').",
        ),
        StepParameter(
            name="end",
            value_type="number",
            required=True,
            description="Slice end: a sample index (mode='index') or seconds (mode='time').",
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
            description="Sliced signal, of the same type as the input.",
            compatibility_only=True,
        ),
    ),
)
def slice_signal(
    signal: Any, *, start: int | float, end: int | float, mode: str = "index"
) -> dict[str, Any]:
    return {
        "result": slice_signal_by_mode(signal, start=start, end=end, slicing_mode=mode)
    }
