"""Registered reducer/metric pipeline steps."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.workflows.registry import StepArtifact, register_step


@register_step(
    "metric.interval_cv",
    reads={"intervals": "breath_intervals"},
    writes=("cv", "mean", "std", "n"),
    summary="Coefficient of variation of interval durations.",
    description=(
        "Backend-neutral reducer: coefficient of variation (std/mean) of "
        "interval durations, plus the underlying mean, std, and count. Usable "
        "for EIT breath intervals, EMG breath intervals, or any other "
        "modality's interval collection."
    ),
    category="reducer",
    input_artifacts=(
        StepArtifact(
            name="intervals",
            artifact_type="interval_collection",
            description=(
                "Interval/breath collection exposing a `.intervals` list of "
                "(start, end) time pairs."
            ),
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="cv",
            artifact_type="scalar_metric",
            description="Coefficient of variation (std/mean) of interval durations.",
        ),
        StepArtifact(
            name="mean",
            artifact_type="scalar_metric",
            unit="s",
            description="Mean interval duration.",
        ),
        StepArtifact(
            name="std",
            artifact_type="scalar_metric",
            unit="s",
            description="Standard deviation of interval durations.",
        ),
        StepArtifact(
            name="n",
            artifact_type="count",
            description="Number of intervals.",
        ),
    ),
)
def interval_cv(intervals: Any) -> dict[str, Any]:
    """Compute the coefficient of variation of breath/interval durations.

    Mirrors the ROTARC breath-duration calculation: durations are
    ``end - start`` per interval, and CV is ``std / mean``.
    """

    durations = np.asarray(
        [interval[1] - interval[0] for interval in intervals.intervals],
        dtype=float,
    )
    mean = float(durations.mean())
    std = float(durations.std())
    return {
        "cv": float(std / mean),
        "mean": mean,
        "std": std,
        "n": len(intervals.intervals),
    }
