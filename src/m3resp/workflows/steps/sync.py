"""Registered synchronization pipeline steps.

``sync.estimate_offset`` returns a manually supplied constant time offset and
writes it into the pipeline context. Downstream, ``sync.apply_estimated_offset``
consumes that value and crops the modalities, keeping estimation and
application as separate, declarative steps.

There is no robust, general-purpose automatic sync estimator in this package:
find the offset interactively with the marimo multimodal viewer
(``tools/visualization_tools/2_annemijn_multimodal_vis.py`` and its
protocol-specific estimators in
``tools/visualization_tools/utils/offset_estimation.py``), then hardcode the
result as ``manual_offset_seconds`` here.

Run this *after* the ``*.load`` steps but *before* ``session.sync_raw``.
"""

from __future__ import annotations

from typing import Any

from m3resp.core.session import M3Session
from m3resp.synchronization.cropping import VENTILATOR, normalize_modality
from m3resp.synchronization.offset_estimation import estimate_sync_offset
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step


@register_step(
    "sync.estimate_offset",
    reads={"session": "session"},
    writes=("estimated_offset_seconds", "offset_estimation"),
    summary="Return a manually supplied EIT-to-Biopac time offset.",
    description=(
        "Return a manually supplied constant time offset. There is no "
        "robust, general-purpose automatic sync method in this package; "
        "find the offset interactively (see "
        "tools/visualization_tools/utils/offset_estimation.md) and hardcode "
        "it as manual_offset_seconds. Run after the '*.load' steps but "
        "before 'session.sync_raw'/'sync.apply_estimated_offset'."
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
            description="The manually supplied offset, ready to bind onto 'session.sync_raw's offset_seconds.",
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
    bind onto ``session.sync_raw``'s ``offset_seconds``) and ``offset_estimation``
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
        "Crop source-clock raw modalities so they start with the target "
        "modality, using the offset 'sync.estimate_offset' reported. The "
        "target modality is held at zero; each source modality is cropped by "
        "the negative estimate."
    ),
    category="synchronization",
    session_reads=("session.raw",),
    session_writes=("session.raw", "session.parameters.raw_alignment"),
    input_artifacts=(
        StepArtifact(
            name="session",
            artifact_type="m3session",
            default_context_key="session",
            description="Backing M3Session whose raw modality signals are cropped in place.",
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
            description="Modality held at zero offset; every source modality is cropped relative to it.",
        ),
        StepParameter(
            name="source_modalities",
            value_type="list",
            default=("emg", VENTILATOR),
            description="Modalities cropped by the negative estimated offset. Must not include 'target_modality'.",
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
    """Crop source-clock recordings so they start with the target recording.

    ``sync.estimate_offset`` reports where target ``t=0`` falls on the source
    clock. The corresponding Stage-1 crop is therefore the negative estimate
    on each source modality, with the target modality held at zero.
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
