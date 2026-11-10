"""Registered EIT loading workflow steps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from m3resp.core.session import M3Session
from m3resp.workflows.registry import (
    StepArtifact,
    StepParameter,
    register_step,
)

from ._shared import (
    _SESSION_ARTIFACT,
    _step_provenance,
)

_LOAD_WRITES = (
    "raw_eit",
    "raw_global_impedance",
    "raw_global_impedance_signal",
    "raw_pixel_impedance_signal",
)

_LOAD_SESSION_WRITES = (
    "session.raw.eit",
    "session.signals",
    # A new recording is not synchronized: its start time is cleared.
    "session.start_times",
    "session.sync_methods",
)

_LOAD_OUTPUTS = (
    StepArtifact(
        name="raw_eit",
        artifact_type="eit_pixel_signal",
        description="Raw EIT images (pixel impedance) as a Signal with values of shape (frames, rows, columns).",
    ),
    StepArtifact(
        name="raw_global_impedance",
        artifact_type="eit_impedance_waveform",
        description=(
            "Raw global impedance as a Signal: the waveform of the file "
            "(Draeger, Timpel), or the sum of all pixels of each raw image "
            "(Sentec)."
        ),
    ),
    StepArtifact(
        name="raw_global_impedance_signal",
        artifact_type="eit_impedance_waveform",
        description="The same Signal as 'raw_global_impedance'.",
    ),
    StepArtifact(
        name="raw_pixel_impedance_signal",
        artifact_type="eit_pixel_signal",
        description="The same Signal as 'raw_eit'.",
    ),
)


@register_step(
    "eit.load",
    reads={"session": "session"},
    writes=_LOAD_WRITES,
    summary="Load an EIT recording into the session.",
    description="Load a Draeger, Sentec or Timpel EIT recording file and give the raw images (pixel impedance) and the raw global impedance as Signals.",
    category="loading",
    modality="eit",
    session_writes=_LOAD_SESSION_WRITES,
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="file_path",
            value_type="path",
            required=True,
            path_kind="file",
            allow_list=True,
            description=(
                "EIT recording file to load, or a list of files recorded one "
                "after the other, which are joined into one recording in the "
                "given order. A pause between two files stays in the time "
                "axis and gives a warning."
            ),
        ),
        StepParameter(
            name="vendor",
            value_type="choice",
            required=True,
            choices=("draeger", "sentec", "timpel"),
            description="Device maker of the recording. For other files, use 'eit.load_custom'.",
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
    ),
    output_artifacts=_LOAD_OUTPUTS,
    record=True,
)
def load(
    session: M3Session,
    *,
    file_path: str | list[str],
    vendor: str,
    sample_frequency: float | None = None,
    first_frame: int = 0,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Load an EIT recording and add its raw signals to the session.

    Loading and signal conversion are recorded in the session's history.

    Args:
        session: Session that loads and stores the recording and signals.
        file_path: EIT recording to read, or a list of files joined in the
            given order.
        vendor: Device maker: ``"draeger"``, ``"sentec"`` or ``"timpel"``.
        sample_frequency: Sampling rate in Hz; ``None`` uses the reader's value.
        first_frame: Index of the first frame to read, starting at zero.
        max_frames: Maximum number of frames to read; ``None`` reads to the end.

    Returns:
        dict[str, Any]: ``raw_eit`` (also ``raw_pixel_impedance_signal``),
            the raw images as a Signal with time, row and column axes, and
            ``raw_global_impedance`` (also ``raw_global_impedance_signal``),
            the raw global impedance as a Signal. The Signals keep the times,
            sampling rate and impedance unit of the recording.

    Raises:
        ValueError: If ``first_frame`` or ``max_frames`` is set with several
            files, or if the files are out of order, overlap, or cannot be
            joined.
    """

    read_options: dict[str, Any] = {"first_frame": first_frame}
    if sample_frequency is not None:
        read_options["sample_frequency"] = sample_frequency
    if max_frames is not None:
        read_options["max_frames"] = max_frames

    session.load_eit(file_path, vendor=vendor, loader="vendor", **read_options)
    return {
        **_loaded_recording_outputs(session),
        "_provenance": _step_provenance(
            source_function="m3resp.eit.io.read_eit_file",
            source_package="m3resp",
            implementation="m3resp.eit.io",
            operation="eit.load",
            parameters={
                "vendor": vendor,
                "sample_frequency": sample_frequency,
                "first_frame": first_frame,
                "max_frames": max_frames,
            },
        ),
    }


@register_step(
    "eit.load_custom",
    reads={"session": "session"},
    writes=_LOAD_WRITES,
    summary="Load an EIT recording with your own loader.",
    description=(
        "Load an EIT recording with the custom loader of the session "
        "(M3Session(eit_loader=...)), for files that the vendor readers "
        "cannot read. The loader returns an EITFileData; the outputs are the "
        "same as those of eit.load."
    ),
    category="loading",
    modality="eit",
    session_writes=_LOAD_SESSION_WRITES,
    alternatives=("eit.load",),
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="file_path",
            value_type="path",
            required=True,
            allow_list=True,
            description="File or folder the loader reads, or a list of them (passed to the loader as a list).",
        ),
        StepParameter(
            name="loader_options",
            value_type="mapping",
            required=False,
            default=None,
            description="Keyword arguments passed on to the loader.",
        ),
    ),
    output_artifacts=_LOAD_OUTPUTS,
    record=True,
)
def load_custom(
    session: M3Session,
    *,
    file_path: str | list[str],
    loader_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load an EIT recording with the session's custom loader.

    The loader is set in Python, before the workflow runs:
    ``M3Session(eit_loader=my_loader)``, or
    ``run_workflow(spec, eit_loader=my_loader)``. It is called as
    ``my_loader(file_path, vendor=None, **loader_options)`` and returns an
    ``m3resp.eit.io.EITFileData``.

    Args:
        session: Session that loads and stores the recording and signals.
        file_path: File or folder the loader reads, or a list of them.
        loader_options: Keyword arguments for the loader.

    Returns:
        dict[str, Any]: The same outputs as ``eit.load``.

    Raises:
        TypeError: If ``loader_options`` is other than a mapping, or if the
            loader returns something other than an ``EITFileData``.
        UnsupportedWorkflowError: If the session has no custom EIT loader.
    """

    if loader_options is not None and not isinstance(loader_options, Mapping):
        raise TypeError(
            "eit.load_custom 'loader_options' must be a mapping of keyword "
            f"arguments for the loader, got {type(loader_options).__name__}."
        )
    extra = dict(loader_options or {})
    session.load_eit(file_path, loader="custom", **extra)
    loader = session.eit_loader
    return {
        **_loaded_recording_outputs(session),
        "_provenance": {
            "source_package": getattr(loader, "__module__", None),
            "source_function": getattr(loader, "__qualname__", repr(loader)),
            "implementation": "custom_loader",
            "parameters": {"loader_options": extra},
            "operation": "eit.load_custom",
        },
    }


#: Channels of the raw EIT signals that the loading steps add to
#: `session.signals`.
_RAW_EIT_CHANNELS = ("global_impedance", "pixel_impedance")


def _loaded_recording_outputs(
    session: M3Session, *, replace_raw_signals: bool = False
) -> dict[str, Any]:
    """The outputs shared by both loading steps, from the session's new recording.

    Adds the raw global impedance and the raw pixel impedance to
    `session.signals`. With `replace_raw_signals`, the
    raw EIT signals already in `session.signals` are removed first; the slicing
    step uses this so only the cut signals remain.
    """

    recording = session.eit
    assert recording is not None

    if replace_raw_signals:
        session.signals.items[:] = [
            existing
            for existing in session.signals.items
            if not (
                existing.modality == "eit"
                and existing.processing_state == "raw"
                and existing.channel in _RAW_EIT_CHANNELS
            )
        ]

    # Copies with their own metadata, so a step that adds metadata to an
    # output leaves the recording as it is.
    global_impedance = recording.raw_global_impedance
    raw_global_impedance_signal = replace(
        global_impedance, metadata=dict(global_impedance.metadata)
    )
    pixels = recording.pixel_impedance
    raw_pixel_impedance_signal = replace(pixels, metadata=dict(pixels.metadata))
    session.signals.add(raw_global_impedance_signal)
    # The pixel impedance has the same category as the global impedance (both
    # are impedances) and is told apart by its channel.
    session.signals.add(raw_pixel_impedance_signal)

    return {
        "raw_eit": raw_pixel_impedance_signal,
        "raw_global_impedance": raw_global_impedance_signal,
        "raw_global_impedance_signal": raw_global_impedance_signal,
        "raw_pixel_impedance_signal": raw_pixel_impedance_signal,
    }
