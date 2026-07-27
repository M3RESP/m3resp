"""Registered ventilator per-breath feature pipeline steps."""

from __future__ import annotations

from typing import Any

from m3resp.processing.metrics import respiratory_rate_from_indices
from m3resp.workflows.registry import StepArtifact, register_step


@register_step(
    "ventilator.respiratory_rate",
    aliases=("emg.ventilator_respiratory_rate",),
    reads={
        "ventilator_breath_indices": "ventilator_breath_indices",
        "ventilator_signals": "ventilator_signals",
    },
    writes=("ventilator_respiratory_rate",),
    summary="Compute respiratory rate from detected ventilator breaths.",
    description="Compute respiratory rate from the detected ventilator breath peak indices.",
    category="parameters",
    modality="ventilator",
    input_artifacts=(
        StepArtifact(
            name="ventilator_breath_indices",
            artifact_type="index_array",
            description="Ventilator breath peak indices from 'ventilator.detect_breaths'.",
        ),
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle supplying 'fs'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ventilator_respiratory_rate",
            artifact_type="scalar_metric",
            unit="breaths/min",
            description="Ventilator-derived respiratory rate.",
        ),
    ),
)
def respiratory_rate(
    ventilator_breath_indices: Any, ventilator_signals: Any
) -> dict[str, Any]:
    fs = float(ventilator_signals["fs"])
    return {
        "ventilator_respiratory_rate": respiratory_rate_from_indices(
            ventilator_breath_indices, fs
        )
    }
