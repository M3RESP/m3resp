"""Shared helpers for the registered EMG workflow step modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.workflows.registry import StepArtifact

if TYPE_CHECKING:
    from m3resp.core.session import M3Session
    from m3resp.data.quality import Severity

#: Steps that call resurfemg directly or through ReSurfEMGAdapter declare this.
_RESURFEMG = ("resurfemg",)

_SESSION_ARTIFACT = StepArtifact(
    name="session",
    artifact_type="m3session",
    default_context_key="session",
    description="Backing M3Session the step reads from and/or records provenance onto.",
    public=False,
)


def _upstream_metadata(
    *,
    source_function: str,
    operation: str,
    parameters: dict[str, Any],
    source_package: str = "resurfemg",
    implementation: str = "upstream_adapter",
) -> dict[str, Any]:
    """The provenance details an EMG step returns under ``"_provenance"``.

    The same fields as `m3resp.workflows.steps.eit._upstream_metadata`: which
    package and function computed the value, the settings used, and the
    step name.
    """The provenance details an EMG step returns under ``"_provenance"``.

    The same fields as `m3resp.workflows.steps.eit._upstream_metadata`: which
    package and function computed the value, the settings used, and the
    step name.

    `source_package`/`implementation` default to the ReSurfEMG-adapter case;
    pass `source_package="m3resp"`, `implementation="m3resp.processing.<module>"`
    for a step whose value comes from a native primitive instead.
    """
    return {
        "source_package": source_package,
        "source_function": source_function,
        "implementation": implementation,
        "parameters": parameters,
        "operation": operation,
    }


def resolve_emg_source(processed_emg: Any, source: str | None, step_name: str) -> str:
    """The bundle key an EMG step should read, given an optional explicit one.

    Band-pass filtering and ECG removal are two separate steps and each keeps
    its own result: ``"filtered"`` is the band-passed signal, ``"ecg_cleaned"``
    the one with the heartbeat removed. With no explicit ``source``, a step
    works on the most-processed trace available - the cleaned one where an
    ECG-removal step has already run, the band-passed one otherwise. That is
    the ``clean`` -> ``filt`` order ReSurfEMG resolves signals in, and it lets
    two removal steps chain without either naming a key.
    """
    if not isinstance(processed_emg, Mapping):
        msg = f"{step_name} needs a processed EMG bundle."
        raise TypeError(msg)

    key = source
    if key is None:
        key = (
            "ecg_cleaned"
            if processed_emg.get("ecg_cleaned") is not None
            else "filtered"
        )
    if processed_emg.get(key) is None:
        msg = (
            f"{step_name} source {key!r} is not present in processed_emg; "
            f"available keys: {sorted(processed_emg.keys())}."
        )
        raise ValueError(msg)
    return key


def _store_processed_emg(session: M3Session, processed_emg: dict[str, Any]) -> None:
    """Make `processed_emg` the session's current processed EMG.

    Sets `session.processed["emg"]`, which breath detection reads, and the
    trace fields of `session.emg` (mirrors what `M3Session.preprocess_emg`
    does).
def _store_processed_emg(session: M3Session, processed_emg: dict[str, Any]) -> None:
    """Make `processed_emg` the session's current processed EMG.

    Sets `session.processed["emg"]`, which breath detection reads, and the
    trace fields of `session.emg` (mirrors what `M3Session.preprocess_emg`
    does).
    """

    session.processed["emg"] = processed_emg
    if session.emg is not None:
        session.emg.filtered = processed_emg.get("filtered")
        session.emg.ecg_cleaned = processed_emg.get("ecg_cleaned")
        session.emg.envelope = processed_emg.get("envelope")
        session.emg.channel = processed_emg.get("channel")
        session.emg.fs = processed_emg.get("fs")


def require_envelope(processed_emg: Any, step_name: str) -> np.ndarray:
    """The envelope in a processed EMG bundle, as a float array.

    Raises a ValueError naming `emg.envelope` when the bundle has none: the
    envelope is computed by its own step, after filtering and ECG removal.
    """

    envelope = (
        processed_emg.get("envelope") if isinstance(processed_emg, Mapping) else None
    )
    if envelope is None:
        raise ValueError(
            f"{step_name} needs an EMG envelope, and the processed EMG has "
            "none. Add an 'emg.envelope' step after filtering and ECG removal."
        )
    return np.asarray(envelope, dtype=float)
        session.emg.filtered = processed_emg.get("filtered")
        session.emg.ecg_cleaned = processed_emg.get("ecg_cleaned")
        session.emg.envelope = processed_emg.get("envelope")
        session.emg.channel = processed_emg.get("channel")
        session.emg.fs = processed_emg.get("fs")


def require_envelope(processed_emg: Any, step_name: str) -> np.ndarray:
    """The envelope in a processed EMG bundle, as a float array.

    Raises a ValueError naming `emg.envelope` when the bundle has none: the
    envelope is computed by its own step, after filtering and ECG removal.
    """

    envelope = (
        processed_emg.get("envelope") if isinstance(processed_emg, Mapping) else None
    )
    if envelope is None:
        raise ValueError(
            f"{step_name} needs an EMG envelope, and the processed EMG has "
            "none. Add an 'emg.envelope' step after filtering and ECG removal."
        )
    return np.asarray(envelope, dtype=float)


def _require_positive_seconds(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        msg = f"{name} must be a finite, positive number; got {value!r}."
        raise ValueError(msg)


def _require_percentile(name: str, value: float) -> None:
    if not (0.0 <= float(value) <= 100.0):
        msg = f"{name} must be between 0 and 100; got {value!r}."
        raise ValueError(msg)


def _processed_channel_label_and_unit(processed_emg: Any) -> tuple[str, str | None]:
    channel_index = processed_emg.get("channel")
    metadata = processed_emg.get("metadata") or {}
    labels = list(metadata.get("labels") or [])
    units = list(metadata.get("units") or [])
    label = (
        labels[channel_index]
        if isinstance(channel_index, int) and channel_index < len(labels)
        else str(channel_index)
    )
    unit = (
        units[channel_index]
        if isinstance(channel_index, int) and channel_index < len(units)
        else None
    )
    return label, unit


def _require_equal_length(**named_arrays: Any) -> None:
    """Raise a clear error instead of silently truncating with
    `min(len(...))` when paired arrays disagree in length (plan Phase 5.4:
    "Do not truncate arrays... without reporting unmatched events").
    """
    lengths = {name: len(array) for name, array in named_arrays.items()}
    if len(set(lengths.values())) > 1:
        msg = f"Arrays must have equal length; got {lengths}."
        raise ValueError(msg)


def _breath_metadata(peak_index: Any, *, fs: float | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"peak_sample_index": int(peak_index)}
    if fs is not None:
        metadata["peak_time"] = float(peak_index) / fs
    return metadata


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
    """One `QualityFlag` per breath - `breath_id=str(position)` until a
    stable event ID is available, with the source peak sample index
    recorded in metadata (plan Phase 5.4).
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
    _require_equal_length(values=values, peak_indices=peak_indices)
    results = []
    for position, (value, peak_index) in enumerate(zip(values, peak_indices)):
        metadata = _breath_metadata(peak_index, fs=fs)
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
