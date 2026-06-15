"""Registered reducer/metric pipeline steps."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.pipeline.registry import register_step


@register_step(
    "metric.interval_cv",
    reads={"intervals": "breath_intervals"},
    writes=("cv", "mean", "std", "n"),
    summary="Coefficient of variation of interval durations.",
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
        "n": int(len(intervals.intervals)),
    }
