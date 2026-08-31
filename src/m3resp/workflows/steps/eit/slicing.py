"""Registered EIT slicing pipeline steps."""

from __future__ import annotations

from typing import Any, Literal

from m3resp.workflows.registry import (
    ANY_ARTIFACT_TYPE,
    StepArtifact,
    StepParameter,
    register_step,
)
from m3resp.workflows.utils import slice_signal_by_mode

from ._shared import _EITPROCESSING


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
    signal: Any,
    *,
    start: float,
    end: float,
    mode: Literal["index", "time"] = "index",
) -> dict[str, Any]:
    return {
        "result": slice_signal_by_mode(signal, start=start, end=end, slicing_mode=mode)
    }
