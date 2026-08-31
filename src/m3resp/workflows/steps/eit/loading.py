"""Registered EIT loading pipeline steps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from m3resp.adapters.eitprocessing_adapter import (
    continuous_data_to_signal,
)
from m3resp.core.session import M3Session
from m3resp.workflows.registry import (
    StepArtifact,
    StepParameter,
    register_step,
)

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
    session_writes=("session.raw.eit", "session.signals"),
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="file_path",
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
            name="sample_frequency",
            value_type="number",
            required=False,
            default=None,
            unit="Hz",
            minimum=0,
            description=(
                "Sampling rate of the recording. Left unset, Draeger files are "
                "read from the file itself, Timpel assumes 50 Hz and Sentec "
                "50.2 Hz. Setting it on a Draeger file warns if it disagrees "
                "with what the file says."
            ),
        ),
        StepParameter(
            name="first_frame",
            value_type="integer",
            required=False,
            default=0,
            minimum=0,
            description="Index of the first frame to read. Defaults to the start of the recording.",
        ),
        StepParameter(
            name="max_frames",
            value_type="integer",
            required=False,
            default=None,
            minimum=1,
            description=(
                "Stop after this many frames. Left unset, the whole recording "
                "is read; a recording shorter than this is read in full."
            ),
        ),
        StepParameter(
            name="loader_options",
            value_type="mapping",
            required=False,
            default=None,
            description=(
                "Remaining keyword arguments for the vendor loader (its "
                "label/name/description metadata). The reading parameters "
                "above are declared in their own right and should be set "
                "there, not here."
            ),
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
    file_path: str,
    vendor: str | None = None,
    sample_frequency: float | None = None,
    first_frame: int = 0,
    max_frames: int | None = None,
    loader_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if loader_options is not None and not isinstance(loader_options, Mapping):
        raise TypeError(
            "eit.load 'loader_options' must be a mapping of keyword arguments "
            f"for M3Session.load_eit(), got {type(loader_options).__name__}."
        )

    extra = dict(loader_options or {})
    # The three reading parameters are declared in their own right, so a GUI
    # can offer them. Accepting them a second time through the bag would let
    # a spec set one value in each and silently pick a winner.
    duplicated = {"sample_frequency", "first_frame", "max_frames"} & extra.keys()
    if duplicated:
        raise ValueError(
            f"eit.load: {', '.join(sorted(duplicated))} "
            f"{'is' if len(duplicated) == 1 else 'are'} declared parameter(s) "
            "of this step and must be set directly, not inside "
            "'loader_options'."
        )

    read_options: dict[str, Any] = {"first_frame": first_frame}
    if sample_frequency is not None:
        read_options["sample_frequency"] = sample_frequency
    if max_frames is not None:
        read_options["max_frames"] = max_frames

    session.load_eit(file_path, vendor=vendor, **read_options, **extra)
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
            parameters={
                "vendor": vendor,
                "sample_frequency": sample_frequency,
                "first_frame": first_frame,
                "max_frames": max_frames,
                "loader_options": extra,
            },
        ),
    )
    return {
        "raw_eit": recording.raw,
        "raw_global_impedance": recording.global_impedance,
        "eit_sequence": recording.data,
        "raw_global_impedance_signal": raw_global_impedance_signal,
    }
