"""Shared helpers for the registered ventilator pipeline step modules.

Mirrors `m3resp.workflows.steps.eit._shared`/`m3resp.workflows.steps.emg._shared`:
each modality's step package keeps its own small copy of these helpers rather
than importing another package's private module, so `_record_step` can
hardcode the right modality string for its own steps without a cross-package
dependency. Before this package existed, ventilator steps used the EMG copy of
`_record_step`, which recorded their provenance under `modality="emg"`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, QualityFlag
from m3resp.data.quality import Severity
from m3resp.workflows.registry import StepArtifact

#: Ventilator loading/quality steps currently go through ReSurfEMGAdapter
#: (loading shares the sEMG's file; Pocc quality assessment wraps
#: resurfemg's quality_assessment module), so they declare this the same way
#: EMG steps do.
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
    """Build the shared provenance metadata schema (see
    `m3resp.workflows.steps.emg._shared._upstream_metadata`).

    `source_package`/`implementation` default to the ReSurfEMG-adapter case
    (`ventilator.pocc_quality`); pass `source_package="m3resp"`,
    `implementation="m3resp.processing.<module>"` for a step whose value comes
    from a native primitive instead (`ventilator.pocc_intervals`/
    `.pocc_time_product`).
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
    """Record per-step ventilator provenance through the existing
    `M3Session._record()` seam."""

    from m3resp.workflows.registry import get_step

    definition = get_step(step_name)
    session._record(
        step_name,
        "ventilator",
        parameters={
            "step": step_name,
            "reads": sorted(definition.reads),
            "writes": list(definition.writes),
            "upstream_version": _resurfemg_version(),
            **metadata,
        },
    )


def _require_equal_length(**named_arrays: Any) -> None:
    """Raise a clear error instead of silently truncating with
    `min(len(...))` when paired arrays disagree in length."""

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
    stable event ID is available, with the source peak sample index recorded
    in metadata."""

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
