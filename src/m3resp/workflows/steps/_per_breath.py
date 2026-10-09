"""Helpers for EMG and ventilator steps that give one result per breath.

These steps store each per-breath value as its own ``ParameterResult`` or
``QualityFlag``. Each one records the sample of its breath's turning point in
``metadata["extremum_sample_index"]``, so it can be matched back to the breath
it belongs to. These helpers write and read that key in one place.

EIT per-breath values (TIV, EELI, pixel TIV) are stored as ``IntervalData``,
which holds each value together with its ``BreathEvent``.
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
#: Older metadata key accepted when the current key has no value.
_OLD_EXTREMUM_SAMPLE_INDEX = "peak_sample_index"


def _require_equal_length(**named_arrays: Any) -> None:
    """Raise ValueError if the named arrays have different lengths."""

    lengths = {name: len(array) for name, array in named_arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Arrays must have equal length; got {lengths}.")


def _breath_metadata(extremum_index: Any, *, fs: float | None = None) -> dict[str, Any]:
    """Return turning-point metadata for one breath.

    Store extremum_index as an integer under ``extremum_sample_index``.
    When fs is supplied in Hz, also store ``extremum_time`` in seconds from
    recording start as extremum_index/fs.
    """

    metadata: dict[str, Any] = {EXTREMUM_SAMPLE_INDEX: int(extremum_index)}
    if fs is not None:
        metadata["extremum_time"] = float(extremum_index) / fs
    return metadata


def extremum_sample_index(metadata: Mapping[str, Any]) -> int | None:
    """Read a breath's turning-point sample position from result metadata.

    Args:
        metadata: Result or quality-flag metadata. ``extremum_sample_index``
            takes precedence when its value is non-None. Otherwise the older
            ``peak_sample_index`` is read with a UserWarning naming the
            current key.

    Returns:
        int | None: The value converted to an integer, or None when neither
            key has a value.

    Raises:
        TypeError: If the selected value has an incompatible type.
        ValueError: If the selected value cannot be converted to an integer.
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
    extremum_indices: Any,
    unit: str | None = None,
    method: str | None = None,
    fs: float | None = None,
    extra_metadata_per_item: list[dict[str, Any]] | None = None,
) -> list[ParameterResult]:
    """Build one `ParameterResult` per breath in input order.

    Args:
        name: Measurement name shared by all results.
        values: Scalar or array values, one entry per breath.
        modality: Device or technique that produced the measurement.
        category: Optional measurement category.
        extremum_indices: Turning-point sample positions, paired with values.
        unit: Optional measurement unit shared by all results.
        method: Optional method name shared by all results.
        fs: Optional signal sampling rate in Hz.
        extra_metadata_per_item: Optional metadata dictionaries in breath order,
            one per value. Their entries override generated metadata.

    Returns:
        list[ParameterResult]: Results with ``breath_id=str(position)`` using
            zero-based input positions. Metadata records ``extremum_sample_index``
            and, when fs is supplied, ``extremum_time`` in seconds from recording
            start. Scalars become floats; array values are retained.

    Raises:
        ValueError: If values and extremum_indices have different lengths.
        IndexError: If extra_metadata_per_item has fewer entries than values.
    """

    _require_equal_length(values=values, extremum_indices=extremum_indices)
    results = []
    for position, (value, extremum_index) in enumerate(zip(values, extremum_indices)):
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
    extremum_indices: Any,
    severity: Severity = "info",
    fs: float | None = None,
    threshold: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[QualityFlag]:
    """Create one quality flag for each breath extremum.

    Args:
        name (str): Quality check name.
        valid (Any): Pass/fail values in the same order as extremum_indices.
        modality (str): Recording modality for each flag.
        category (str | None): Physical quantity assessed.
        extremum_indices (Any): Zero-based samples of each breath's turning point.
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
        ValueError: If valid and extremum_indices differ in length.
    """

    _require_equal_length(valid=valid, extremum_indices=extremum_indices)
    flags = []
    for position, (is_valid, extremum_index) in enumerate(zip(valid, extremum_indices)):
        metadata = _breath_metadata(extremum_index, fs=fs)
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
