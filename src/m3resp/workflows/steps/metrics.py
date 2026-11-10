"""Registered reducer/metric workflow steps."""

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
    parameters_reviewed=True,
    input_artifacts=(
        StepArtifact(
            name="intervals",
            artifact_type="interval_data",
            description=(
                "Intervals or breaths with a `.intervals` list: m3resp "
                "Intervals or BreathEvents, or (start, end) time pairs."
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

    Args:
        intervals: An object with ``.intervals``: m3resp ``Interval``s or
            ``BreathEvent``s (an m3resp ``IntervalData``), objects with
            ``start_time`` and ``end_time``, or (start, end) pairs. Times
            in seconds.

    Returns:
        dict[str, Any]: ``cv`` (std / mean), ``mean`` and ``std`` of the
            durations in seconds, and ``n``, the number of intervals.
    """

    durations = np.asarray(
        [_duration(interval) for interval in intervals.intervals],
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


def _duration(interval: Any) -> float:
    """Return ``end - start`` of one interval, in seconds."""

    if hasattr(interval, "end_time"):
        return interval.end_time - interval.start_time
    return interval[1] - interval[0]
