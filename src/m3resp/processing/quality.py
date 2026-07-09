"""Shared quality-flag mapping helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from m3resp.data import QualityFlag


def threshold_flag(
    name: str,
    value: float,
    *,
    threshold: float,
    comparison: str | Callable[[float, float], bool],
    modality: str | None = None,
    severity: str = "info",
) -> QualityFlag:
    """Create a quality flag by comparing a scalar value to a threshold."""

    scalar = float(value)
    limit = float(threshold)
    passed = _compare(scalar, limit, comparison)
    return QualityFlag(
        name=name,
        passed=passed,
        severity=severity,
        modality=modality,
        value=scalar,
        threshold=limit,
        metadata={"comparison": _comparison_name(comparison)},
    )


def fraction_flag(
    name: str,
    passed_fraction: float,
    *,
    minimum_fraction: float,
    modality: str | None = None,
    severity: str = "info",
) -> QualityFlag:
    """Create a quality flag for a fraction that must meet a minimum value."""

    return threshold_flag(
        name,
        passed_fraction,
        threshold=minimum_fraction,
        comparison=">=",
        modality=modality,
        severity=severity,
    )


def timing_window_flag(
    name: str,
    deltas: np.ndarray,
    *,
    min_delta: float | None = None,
    max_delta: float | None = None,
    modality: str | None = None,
    severity: str = "info",
) -> QualityFlag:
    """Create a quality flag for event timing deltas inside a time window."""

    values = np.asarray(deltas, dtype=float)
    valid = np.ones(values.shape, dtype=bool)
    if min_delta is not None:
        valid &= values >= min_delta
    if max_delta is not None:
        valid &= values <= max_delta
    valid &= np.isfinite(values)
    return QualityFlag(
        name=name,
        passed=bool(valid.all()),
        severity=severity,
        modality=modality,
        value=_scalar_or_none(values),
        threshold=max_delta if max_delta is not None else min_delta,
        metadata={
            "min_delta": min_delta,
            "max_delta": max_delta,
            "deltas": values.tolist(),
            "valid": valid.tolist(),
        },
    )


def quality_flag_from_result(
    name: str,
    result: Any,
    *,
    modality: str | None = None,
    severity: str = "info",
    source_method: str | None = None,
) -> QualityFlag:
    """Map a heterogeneous quality result into a native `QualityFlag`."""

    return QualityFlag(
        name=name,
        passed=_result_passed(result),
        severity=severity,
        modality=modality,
        value=_scalar_or_none(result),
        metadata=_result_metadata(result, source_method=source_method),
    )


def skipped_quality_flag(
    name: str,
    reason: str,
    *,
    modality: str | None = None,
    severity: str = "warning",
) -> QualityFlag:
    """Create a warning flag for a skipped quality calculation."""

    return QualityFlag(
        name=name,
        passed=False,
        severity=severity,
        modality=modality,
        message=str(reason),
        metadata={"skipped": True},
    )


def _compare(
    value: float,
    threshold: float,
    comparison: str | Callable[[float, float], bool],
) -> bool:
    if callable(comparison):
        return bool(comparison(value, threshold))
    operators = {
        ">": value > threshold,
        ">=": value >= threshold,
        "<": value < threshold,
        "<=": value <= threshold,
        "==": value == threshold,
        "!=": value != threshold,
    }
    if comparison not in operators:
        raise ValueError(
            "comparison must be one of '>', '>=', '<', '<=', '==', '!=' or a callable"
        )
    return bool(operators[comparison])


def _comparison_name(comparison: str | Callable[[float, float], bool]) -> str:
    if isinstance(comparison, str):
        return comparison
    return getattr(comparison, "__name__", "callable")


def _result_passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    array = np.asarray(value)
    if array.dtype == bool:
        return bool(array.all())
    return True


def _scalar_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    array = np.asarray(value)
    if array.ndim == 0 and array.dtype != bool:
        return float(array)
    return None


def _result_metadata(
    value: Any,
    *,
    source_method: str | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if source_method is not None:
        metadata["source_method"] = source_method
    array = np.asarray(value)
    if array.ndim > 0:
        metadata["shape"] = tuple(int(dim) for dim in array.shape)
        metadata["dtype"] = str(array.dtype)
    return metadata
