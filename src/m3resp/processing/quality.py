"""Shared quality-flag mapping helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from m3resp.data import QualityFlag
from m3resp.data.quality import Severity


def threshold_flag(
    name: str,
    value: float,
    *,
    threshold: float,
    comparison: str | Callable[[float, float], bool],
    modality: str | None = None,
    severity: Severity = "info",
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
    severity: Severity = "info",
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
    severity: Severity = "info",
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
    severity: Severity = "info",
    source_method: str | None = None,
) -> QualityFlag:
    """Map a heterogeneous quality result into a native `QualityFlag`.

    Only a plain `bool` or a boolean array carries a real pass/fail
    criterion from ReSurfEMG. Everything else - a scalar measurement, a
    numeric array, or a heterogeneous tuple like `snr_pseudo`'s
    `(mean, per-peak array)` - is a measurement, not a criterion, so this
    does not invent a pass/fail meaning the upstream function never defined.
    `passed=False` for a measurement-only result follows the same
    "not evaluated" convention `skipped_quality_flag` uses for a skipped
    result, not a failed check; `metadata["measurement_only"]` distinguishes
    the two. Operation-specific conversions (Phase 5) replace this generic
    mapping once each quality function's scientific meaning is encoded.
    """

    is_boolean, passed = _boolean_verdict(result)
    metadata = _result_metadata(result, source_method=source_method)
    if not is_boolean:
        metadata["measurement_only"] = True
    return QualityFlag(
        name=name,
        passed=passed,
        severity=severity,
        modality=modality,
        value=_scalar_or_none(result),
        metadata=metadata,
    )


def skipped_quality_flag(
    name: str,
    reason: str,
    *,
    modality: str | None = None,
    severity: Severity = "warning",
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


def _boolean_verdict(value: Any) -> tuple[bool, bool]:
    """Return `(is_boolean_criterion, passed)`.

    `is_boolean_criterion` is `False` for anything that is not a plain
    `bool` or a boolean array, including values `np.asarray` cannot convert
    to a single homogeneous array at all (e.g. `snr_pseudo`'s
    `(mean, per-peak array)` tuple, whose elements have different shapes).
    """

    if isinstance(value, bool):
        return True, value
    array = _safe_asarray(value)
    if array is not None and array.dtype == bool:
        return True, bool(array.all())
    return False, False


def _scalar_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    array = _safe_asarray(value)
    if array is not None and array.ndim == 0 and array.dtype != bool:
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
    array = _safe_asarray(value)
    if array is not None and array.ndim > 0:
        metadata["shape"] = tuple(int(dim) for dim in array.shape)
        metadata["dtype"] = str(array.dtype)
    return metadata


def _safe_asarray(value: Any) -> np.ndarray | None:
    """`np.asarray(value)`, or `None` if `value` has no single homogeneous
    array shape (e.g. a tuple of differently-shaped elements)."""

    try:
        return np.asarray(value)
    except ValueError:
        return None
