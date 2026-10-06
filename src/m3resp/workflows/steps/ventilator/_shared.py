"""Helpers for recording ventilator calculations and per-breath quality flags."""

from __future__ import annotations

from typing import Any

from m3resp.core.session import M3Session
from m3resp.data import QualityFlag
from m3resp.data.quality import Severity
from m3resp.workflows.registry import StepArtifact
from m3resp.workflows.steps._per_breath import (
    _breath_metadata,
    _require_equal_length,
)

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
    """Return one QualityFlag per breath in input order.

    Pair valid values with peak_indices and use ``breath_id=str(position)``
    for the zero-based input position. Metadata stores the turning-point
    sample as ``extremum_sample_index`` and, when fs is supplied in Hz,
    ``extremum_time`` in seconds from recording start. Extra metadata
    overrides generated entries. Raises ValueError if the arrays differ
    in length.
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
