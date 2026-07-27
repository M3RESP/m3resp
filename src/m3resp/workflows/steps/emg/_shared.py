"""Shared helpers for the registered EMG pipeline step modules."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, QualityFlag
from m3resp.data.quality import Severity
from m3resp.workflows.registry import StepArtifact

#: Steps that call resurfemg directly or through ReSurfEMGAdapter declare this.
_RESURFEMG = ("resurfemg",)

_SESSION_ARTIFACT = StepArtifact(
    name="session",
    artifact_type="m3session",
    default_context_key="session",
    description="Backing M3Session the step reads from and/or records provenance onto.",
    public=False,
)


def _resurfemg_version() -> str | None:
    """Installed `resurfemg` version, read from package metadata without
    importing the package itself (so this stays optional-dependency-safe)."""

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("resurfemg")
    except PackageNotFoundError:
        return None


def _upstream_metadata(
    *,
    source_function: str,
    operation: str,
    parameters: dict[str, Any],
    source_package: str = "resurfemg",
    implementation: str = "upstream_adapter",
) -> dict[str, Any]:
    """Build the Stage 2 EMG provenance metadata schema shared by native
    `Signal`/`ParameterResult` outputs (see `plan/stage2/
    2_resurfemg_gap_migration_implementation_plan.md`, "Use one provenance
    schema"). Mirrors `m3resp.workflows.steps.eit._upstream_metadata`.

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


def _record_step(
    session: M3Session, step_name: str, *, metadata: dict[str, Any]
) -> None:
    """Record per-step EMG provenance through the existing
    `M3Session._record()` seam, reusing the step's declared reads/writes
    from the registry rather than a second EMG-only history mechanism."""

    from m3resp.workflows.registry import get_step

    definition = get_step(step_name)
    session._record(
        step_name,
        "emg",
        parameters={
            "step": step_name,
            "reads": sorted(definition.reads),
            "writes": list(definition.writes),
            "upstream_version": _resurfemg_version(),
            **metadata,
        },
    )


def _update_session_after_ecg_removal(
    session: M3Session,
    processed_emg_after_ecg: dict[str, Any],
) -> None:
    """Update `session.processed["emg"]` and the `EMGRecording` filtered/
    envelope fields so existing breath detection operates on the
    ECG-cleaned data (mirrors what `M3Session.preprocess_emg` does)."""

    session.processed["emg"] = processed_emg_after_ecg
    if session.emg is not None:
        session.emg.filtered = processed_emg_after_ecg.get("filtered")
        session.emg.envelope = processed_emg_after_ecg.get("envelope")


def _require_positive_seconds(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite, positive number; got {value!r}.")


def _require_percentile(name: str, value: float) -> None:
    if not (0.0 <= float(value) <= 100.0):
        raise ValueError(f"{name} must be between 0 and 100; got {value!r}.")


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
    "Do not truncate arrays... without reporting unmatched events")."""

    lengths = {name: len(array) for name, array in named_arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Arrays must have equal length; got {lengths}.")


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
    recorded in metadata (plan Phase 5.4)."""

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
