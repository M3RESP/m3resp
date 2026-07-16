"""Registered session-level pipeline steps.

These wrap ``M3Session`` cross-modality operations so multimodal workflows can be
expressed as a declarative pipeline spec.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from m3resp.core.session import M3Session
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step


@register_step(
    "session.sync_raw",
    reads={"session": "session"},
    writes=("sync_summary",),
    summary="Crop raw modality signals by a manual time offset before processing.",
    description=(
        "Direct cross-modality session operation: crops each loaded raw "
        "modality signal so they share a common start time, using a fixed "
        "manual offset per modality. Run before per-modality preprocessing."
    ),
    category="synchronization",
    input_artifacts=(
        StepArtifact(
            name="session",
            artifact_type="m3session",
            default_context_key="session",
            description="Backing M3Session whose raw modality signals are cropped in place.",
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
def sync_raw(
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
