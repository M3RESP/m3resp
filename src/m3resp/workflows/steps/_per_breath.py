"""Helpers for steps that give one result per breath (EMG and ventilator).

Each per-breath result or quality flag records the sample of its breath's
turning point in ``metadata["extremum_sample_index"]``, so it can be matched
back to the breath it belongs to. These helpers write and read that key in one
place.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np

from m3resp.data import ParameterResult, QualityFlag
from m3resp.data.quality import Severity

#: Metadata key holding the sample of the breath's turning point.
EXTREMUM_SAMPLE_INDEX = "extremum_sample_index"
#: The name this key had before it was renamed (#93).
_OLD_EXTREMUM_SAMPLE_INDEX = "peak_sample_index"


def _require_equal_length(**named_arrays: Any) -> None:
    """Raise a clear error instead of silently cutting paired arrays to the
    shortest length when they disagree in length."""

    lengths = {name: len(array) for name, array in named_arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Arrays must have equal length; got {lengths}.")


def _breath_metadata(extremum_index: Any, *, fs: float | None = None) -> dict[str, Any]:
    """Metadata tying a per-breath result to its breath: the sample of the
    breath's turning point and, when the sampling rate is known, its time."""

    metadata: dict[str, Any] = {EXTREMUM_SAMPLE_INDEX: int(extremum_index)}
    if fs is not None:
        metadata["extremum_time"] = float(extremum_index) / fs
    return metadata


def extremum_sample_index(metadata: Mapping[str, Any]) -> int | None:
    """The turning-point sample a per-breath result or flag was made for, or
    None when it is not tied to one breath.

    A flag made before the rename still carries ``peak_sample_index``; that
    key is read too, with a warning naming the new key.
    """

    value = metadata.get(EXTREMUM_SAMPLE_INDEX)
    if value is None and metadata.get(_OLD_EXTREMUM_SAMPLE_INDEX) is not None:
        warnings.warn(
            f"Metadata key {_OLD_EXTREMUM_SAMPLE_INDEX!r} is now called "
            f"{EXTREMUM_SAMPLE_INDEX!r}; please rename it.",
            UserWarning,
            stacklevel=2,
        )
        value = metadata[_OLD_EXTREMUM_SAMPLE_INDEX]
    return None if value is None else int(value)


def _per_breath_results(
    name: str,
    values: Any,
    *,
    modality: str,
    category: str | None = None,
    peak_indices: Any,
    unit: str | None = None,
    method: str | None = None,
    fs: float | None = None,
    extra_metadata_per_item: list[dict[str, Any]] | None = None,
) -> list[ParameterResult]:
    """One `ParameterResult` per breath, each tied to its breath by the
    sample of the breath's turning point (``peak_indices``)."""

    _require_equal_length(values=values, peak_indices=peak_indices)
    results = []
    for position, (value, extremum_index) in enumerate(zip(values, peak_indices)):
        metadata = _breath_metadata(extremum_index, fs=fs)
        if extra_metadata_per_item is not None:
            metadata.update(extra_metadata_per_item[position])
        results.append(
            ParameterResult(
                name=name,
                value=value if np.ndim(value) > 0 else float(value),
                modality=modality,
                category=category,
                unit=unit,
                breath_id=str(position),
                method=method,
                metadata=metadata,
            )
        )
    return results


def _per_breath_flags(
    name: str,
    valid: Any,
    *,
    modality: str,
    category: str | None = None,
    peak_indices: Any,
    severity: Severity = "info",
    fs: float | None = None,
    threshold: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[QualityFlag]:
    """Create one quality flag for each breath extremum.

    Args:
        name (str): Quality check name.
        valid (Any): Pass/fail values in the same order as peak_indices.
        modality (str): Recording modality for each flag.
        category (str | None): Physical quantity assessed.
        peak_indices (Any): Zero-based samples of each breath's turning point.
        severity (Severity): Severity of a failed check. Defaults to "info".
        fs (float | None): Sampling rate in Hz. When supplied, metadata also
            records extremum_time in seconds from the recording's first sample.
        threshold (float | None): Threshold used by the quality check.
        extra_metadata (dict[str, Any] | None): Fields added to every flag.
            These values override matching extremum metadata keys.

    Returns:
        list[QualityFlag]: Flags with passed=bool(valid), breath_id equal to the
            string-valued position, and extremum_sample_index in metadata.

    Raises:
        ValueError: If valid and peak_indices differ in length.
    """

    _require_equal_length(valid=valid, peak_indices=peak_indices)
    flags = []
    for position, (is_valid, peak_index) in enumerate(zip(valid, peak_indices)):
        metadata = _breath_metadata(peak_index, fs=fs)
        if extra_metadata:
            metadata.update(extra_metadata)
        flags.append(
            QualityFlag(
                name=name,
                passed=bool(is_valid),
                severity=severity,
                modality=modality,
                category=category,
                breath_id=str(position),
                threshold=threshold,
                metadata=metadata,
            )
        )
    return flags
