"""Registered synchronization pipeline steps.

``sync.raw_modalities`` sets each loaded recording's start time on a shared
clock from a manual offset, and ``sync.skip`` uses the recordings as they are,
for recordings that really did start together. ``sync.raw_modalities`` used
to be named ``session.sync_raw``; the old name still works in pipeline
specs.

``sync.estimate_offset`` returns a manually supplied constant time offset and
writes it into the pipeline context. Downstream, ``sync.apply_estimated_offset``
consumes that value and sets the modalities' start times, keeping estimation and
application as separate, declarative steps.

There is no robust, general-purpose automatic sync estimator in this package:
find the offset interactively with the marimo multimodal viewer
(``tools/visualization_tools/2_annemijn_multimodal_vis.py`` and its
protocol-specific estimators in
``tools/visualization_tools/utils/offset_estimation.py``), then hardcode the
result as ``manual_offset_seconds`` here.

Run ``sync.estimate_offset`` *after* the ``*.load`` steps but *before*
``sync.raw_modalities``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from m3resp.core.session import M3Session
from m3resp.modalities.names import VENTILATOR, normalize_modality
from m3resp.synchronization.offset_estimation import estimate_sync_offset
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step


@register_step(
    "sync.raw_modalities",
    aliases=("session.sync_raw",),
    reads={"session": "session"},
    writes=("sync_summary",),
    summary="Set each recording's start time on a shared clock, from a manual offset.",
    description=(
        "Direct cross-modality session operation: stores a start time per "
        "loaded recording (a fixed manual offset per modality) so all "
        "recordings share one clock. No samples are removed; the start times "
        "are applied when breaths are compared across modalities."
    ),
    category="synchronization",
    session_reads=("session.raw",),
    session_writes=(
        "session.start_times",
        "session.sync_methods",
        "session.parameters.raw_alignment",
    ),
    input_artifacts=(
        StepArtifact(
            name="session",
            artifact_type="m3session",
            default_context_key="session",
            description="Backing M3Session whose recordings get a start time.",
            public=False,
        ),
    ),
    parameters=(
        StepParameter(
            name="method",
            value_type="choice",
            default="manual_offset",
            choices=("manual_offset",),
            description=(
                "Synchronization method. Only 'manual_offset' is currently "
                "supported; other values raise ValueError."
            ),
        ),
        StepParameter(
            name="offset_seconds",
            value_type="number",
            default=0.0,
            unit="s",
            description=(
                "Offset applied to each modality, relative to "
                "'reference_modality'. Also accepts a mapping of "
                "{modality: offset_seconds} for per-modality offsets."
            ),
        ),
        StepParameter(
            name="reference_modality",
            value_type="string",
            required=False,
            default=None,
            description=(
                "Modality whose offset is held at zero; others shift relative "
                "to it. Defaults to the adapter's own resolution rule when unset."
            ),
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="sync_summary",
            artifact_type="sync_summary",
            description="Per-modality applied offset and before/after trace summary.",
        ),
    ),
)
def raw_modalities(
    session: M3Session,
    *,
    method: str = "manual_offset",
    offset_seconds: float | Mapping[str, float] = 0.0,
    reference_modality: str | None = None,
) -> dict[str, Any]:
    summary = session.synchronize_raw_modalities(
        method=method,
        offset_seconds=offset_seconds,
        reference_modality=reference_modality,
    )
    return {"sync_summary": summary}


@register_step(
    "sync.skip",
    reads={"session": "session"},
    writes=("skipped_recordings",),
    summary="Use the recordings as they are, without synchronizing them.",
    description=(
        "For recordings that really did start at the same moment, e.g. when "
        "one trigger started every device. Records 'none' as the "
        "synchronization method of each loaded recording not yet "
        "synchronized, so steps that compare recordings do not warn. The "
        "choice is kept in the provenance log and the exported summary."
    ),
    category="synchronization",
    session_reads=("session.raw",),
    session_writes=("session.sync_methods",),
    # Nothing to tune: it only records a choice.
    parameters_reviewed=True,
    input_artifacts=(
        StepArtifact(
            name="session",
            artifact_type="m3session",
            default_context_key="session",
            description="Backing M3Session whose recordings are marked as used as they are.",
            public=False,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="skipped_recordings",
            artifact_type="recording_list",
            description="The recordings marked 'none', e.g. ['eit', 'emg'].",
        ),
    ),
)
def skip(session: M3Session) -> dict[str, Any]:
    return {"skipped_recordings": session.skip_synchronization()}


@register_step(
    "sync.estimate_offset",
    reads={"session": "session"},
    writes=("estimated_offset_seconds", "offset_estimation"),
    summary="Return a manually supplied EIT-to-Biopac time offset.",
    description=(
        "Return a manually supplied constant time offset. There is no "
        "robust, general-purpose automatic sync method in this package; "
        "find the offset interactively (see "
        "docs/developer/offset-estimation.md) and hardcode "
        "it as manual_offset_seconds. Run after the '*.load' steps but "
        "before 'sync.raw_modalities'/'sync.apply_estimated_offset'."
    ),
    category="synchronization",
    modality=None,
    session_writes=("session.parameters.offset_estimation",),
    input_artifacts=(
        StepArtifact(
            name="session",
            artifact_type="m3session",
            default_context_key="session",
            description="Backing M3Session used to record the estimate for provenance.",
            public=False,
        ),
    ),
    parameters=(
        StepParameter(
            name="method",
            value_type="choice",
            default="manual",
            choices=("manual",),
            description=(
                "'manual': return manual_offset_seconds unchanged. The only "
                "supported method - there is no robust general-purpose "
                "automatic sync."
            ),
        ),
        StepParameter(
            name="manual_offset_seconds",
            value_type="number",
            default=0.0,
            unit="s",
            description="The offset to use, supplied manually.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="estimated_offset_seconds",
            artifact_type="scalar_metric",
            unit="s",
            description="The manually supplied offset, ready to bind onto 'sync.raw_modalities's offset_seconds.",
        ),
        StepArtifact(
            name="offset_estimation",
            artifact_type="diagnostic_summary",
            description="JSON-friendly summary, for provenance/QA.",
        ),
    ),
)
def estimate_offset(
    session: M3Session,
    *,
    method: str = "manual",
    manual_offset_seconds: float = 0.0,
) -> dict[str, Any]:
    """Return the manually supplied sync offset and record it for provenance.

    Writes two context artifacts: ``estimated_offset_seconds`` (a float, ready to
    bind onto ``sync.raw_modalities``'s ``offset_seconds``) and ``offset_estimation``
    (a JSON-friendly summary, for provenance/QA).
    """

    result = estimate_sync_offset(
        method=method, manual_offset_seconds=manual_offset_seconds
    )

    summary: dict[str, Any] = {
        "method": result.method,
        "offset_seconds": result.offset_seconds,
        "source": result.source,
    }

    session.parameters["offset_estimation"] = summary
    return {
        "estimated_offset_seconds": result.offset_seconds,
        "offset_estimation": summary,
    }


@register_step(
    "sync.apply_estimated_offset",
    reads={
        "session": "session",
        "offset_seconds": "estimated_offset_seconds",
    },
    writes=("sync_summary",),
    summary="Apply an estimated EIT-to-source offset to the raw modalities.",
    description=(
        "Put source-clock recordings on the target modality's clock, using "
        "the offset 'sync.estimate_offset' reported. The target modality "
        "starts at zero; each source modality's start time is the negative "
        "estimate. No samples are removed."
    ),
    category="synchronization",
    session_reads=("session.raw",),
    session_writes=(
        "session.start_times",
        "session.sync_methods",
        "session.parameters.raw_alignment",
    ),
    input_artifacts=(
        StepArtifact(
            name="session",
            artifact_type="m3session",
            default_context_key="session",
            description="Backing M3Session whose recordings get a start time.",
            public=False,
        ),
        StepArtifact(
            name="offset_seconds",
            artifact_type="scalar_metric",
            default_context_key="estimated_offset_seconds",
            unit="s",
            description="Estimated target-clock-t0-on-source-clock offset, as written by 'sync.estimate_offset'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="target_modality",
            value_type="string",
            default="eit",
            description="Modality whose start time is zero; every source modality is placed relative to it.",
        ),
        StepParameter(
            name="source_modalities",
            value_type="list",
            default=("emg", VENTILATOR),
            description="Modalities whose start time is the negative estimated offset. Must not include 'target_modality'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="sync_summary",
            artifact_type="sync_summary",
            description="Per-modality applied offset and before/after trace summary.",
        ),
    ),
)
def apply_estimated_offset(
    session: M3Session,
    offset_seconds: float,
    *,
    target_modality: str = "eit",
    source_modalities: tuple[str, ...] | list[str] = ("emg", VENTILATOR),
) -> dict[str, Any]:
    """Place source-clock recordings on the target recording's clock.

    ``sync.estimate_offset`` reports where target ``t=0`` falls on the source
    clock. Each source recording therefore starts at the negative estimate on
    the target clock, with the target modality starting at zero.
    """

    target = normalize_modality(target_modality)
    sources = tuple(normalize_modality(modality) for modality in source_modalities)
    if not sources:
        raise ValueError("source_modalities must contain at least one modality")
    if target in sources:
        raise ValueError("target_modality cannot also be a source modality")

    estimate = float(offset_seconds)
    configured_offsets = {target: 0.0}
    configured_offsets.update({modality: -estimate for modality in sources})
    summary = session.synchronize_raw_modalities(
        method="manual_offset",
        offset_seconds=configured_offsets,
        reference_modality=target,
    )
    session.parameters["raw_alignment"]["estimated_offset_seconds"] = estimate
    session.parameters["raw_alignment"]["estimate_direction"] = (
        "target_t0_on_source_clock"
    )
    return {"sync_summary": summary}
