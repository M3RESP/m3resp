"""Registered ventilator event-quality pipeline steps."""

from __future__ import annotations

from typing import Any

from m3resp.core.session import M3Session
from m3resp.workflows.registry import StepArtifact, register_step

from ._shared import (
    _RESURFEMG,
    _SESSION_ARTIFACT,
    _per_breath_flags,
    _record_step,
    _upstream_metadata,
)


@register_step(
    "ventilator.detect_non_consecutive_manoeuvres",
    aliases=("emg.detect_non_consecutive_manoeuvres",),
    reads={
        "session": "session",
        "ventilator_breath_indices": "ventilator_breath_indices",
        "pocc_indices": "pocc_indices",
    },
    writes=(
        "detect_non_consecutive_manoeuvres",
        "detect_non_consecutive_manoeuvres_flags",
    ),
    summary="Flag non-consecutive occlusion manoeuvres against ventilator breaths.",
    description="Flag Pocc manoeuvres that are not consecutive ventilator breaths, since a valid occlusion trial requires uninterrupted breaths.",
    category="quality",
    modality="ventilator",
    optional_packages=_RESURFEMG,
    parameters_reviewed=True,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="ventilator_breath_indices",
            artifact_type="index_array",
            description="Ventilator breath peak indices.",
        ),
        StepArtifact(
            name="pocc_indices",
            artifact_type="index_array",
            description="Pocc manoeuvre peak indices.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="detect_non_consecutive_manoeuvres",
            artifact_type="boolean_array",
            description="Whether each manoeuvre is flagged as non-consecutive.",
        ),
        StepArtifact(
            name="detect_non_consecutive_manoeuvres_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per manoeuvre.",
        ),
    ),
)
def detect_non_consecutive_manoeuvres(
    session: M3Session, ventilator_breath_indices: Any, pocc_indices: Any
) -> dict[str, Any]:
    result = session.emg_adapter.detect_non_consecutive_manoeuvres(
        ventilator_breath_indices, pocc_indices
    )
    flags = _per_breath_flags(
        "detect_non_consecutive_manoeuvres",
        result,
        modality="ventilator",
        category="airway_pressure",
        peak_indices=pocc_indices,
    )
    for flag in flags:
        session.quality.add(flag)
    _record_step(
        session,
        "ventilator.detect_non_consecutive_manoeuvres",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.detect_non_consecutive_manoeuvres",
            operation="ventilator.detect_non_consecutive_manoeuvres",
            parameters={},
        ),
    )
    return {
        "detect_non_consecutive_manoeuvres": result,
        "detect_non_consecutive_manoeuvres_flags": flags,
    }
