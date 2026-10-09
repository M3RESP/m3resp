"""Shared helpers for the registered EIT workflow step modules."""

from __future__ import annotations

import weakref
from typing import Any

from m3resp.adapters.eitprocessing_adapter import breath_intervals_to_breath_events
from m3resp.core.session import M3Session
from m3resp.data import BreathEvent
from m3resp.data.events import reuse_matching_breaths
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


#: The breaths already found in each session, per (detector, signal) pair.
#: Kept per session and dropped with it.
_BREATHS_FOUND: weakref.WeakKeyDictionary[
    M3Session, list[tuple[Any, Any, list[BreathEvent]]]
] = weakref.WeakKeyDictionary()


def _breaths_used_by(
    breath_detector: Any, signal: Any, session: M3Session
) -> list[BreathEvent]:
    """Return the breaths used for TIV and EELI on this detector-signal pair.

    Detection runs once per detector and signal object in each session.
    This assumes repeated detection with unchanged settings and data gives
    the same breaths. Matching breaths in ``session.events['eit_breaths']``
    are reused on the first call; later calls use the remembered list.

    Args:
        breath_detector: eitprocessing detector with ``find_breaths``.
        signal: Waveform used by the per-breath calculations.
        session: Session that remembers detected breaths and supplies stored
            breaths for matching.

    Returns:
        list[BreathEvent]: Breaths in detector order, shared by per-breath
            results using this detector and signal.
    """

    found = _BREATHS_FOUND.setdefault(session, [])
    for detector, detected_on, breaths in found:
        if detector is breath_detector and detected_on is signal:
            return breaths

    breaths = _use_stored_eit_breaths(
        breath_intervals_to_breath_events(breath_detector.find_breaths(signal)),
        session,
    )
    found.append((breath_detector, signal, breaths))
    return breaths


def _use_stored_eit_breaths(
    breaths: list[BreathEvent], session: M3Session
) -> list[BreathEvent]:
    """Reuse stored EIT breaths with exactly matching modality and start-end times."""

    return reuse_matching_breaths(breaths, session.get_events("eit_breaths", None))


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
    """Record per-step EIT provenance through `M3Session._record()`, using
    the step's declared reads/writes from the registry."""

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
