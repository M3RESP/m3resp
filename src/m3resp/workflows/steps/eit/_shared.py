"""Shared helpers for the registered EIT pipeline step modules."""

from __future__ import annotations

from typing import Any

from m3resp.core.session import M3Session
from m3resp.workflows.registry import StepArtifact

#: Every eit.* step ultimately calls eitprocessing, directly or through
#: EITProcessingAdapter, so all of them declare the same optional dependency.
_EITPROCESSING = ("eitprocessing",)


_SESSION_ARTIFACT = StepArtifact(
    name="session",
    artifact_type="m3session",
    default_context_key="session",
    description="Backing M3Session the step reads from and/or records provenance onto.",
    public=False,
)


def _upstream_metadata(
    *, source_function: str, operation: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    """Build the Stage 2 EIT provenance metadata schema shared by native
    `Signal`/`ParameterResult` outputs (see `plan/stage2/
    1_eit_gap_migration_implementation_plan.md`, "Use one provenance
    schema")."""

    return {
        "source_package": "eitprocessing",
        "source_function": source_function,
        "implementation": "upstream_adapter",
        "parameters": parameters,
        "operation": operation,
    }


def _eitprocessing_version() -> str | None:
    """Installed `eitprocessing` version, read from package metadata without
    importing the package itself (so this stays optional-dependency-safe)."""

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("eitprocessing")
    except PackageNotFoundError:
        return None


def _record_step(
    session: M3Session, step_name: str, *, metadata: dict[str, Any]
) -> None:
    """Record per-step EIT provenance through the existing
    `M3Session._record()` seam (see `plan/stage2/
    1_eit_gap_migration_implementation_plan.md` Phase 5.2), reusing the
    step's declared reads/writes from the registry rather than a second
    EIT-only history mechanism."""

    from m3resp.workflows.registry import get_step

    definition = get_step(step_name)
    session._record(
        step_name,
        "eit",
        parameters={
            "step": step_name,
            "reads": sorted(definition.reads),
            "writes": list(definition.writes),
            "upstream_version": _eitprocessing_version(),
            **metadata,
        },
    )
