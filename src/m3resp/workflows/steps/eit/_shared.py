"""Shared helpers for the registered EIT workflow step modules."""

from __future__ import annotations

import weakref
from typing import Any

from m3resp.adapters.eitprocessing_adapter import (
    as_continuous_data,
    as_signal,
    breath_intervals_to_breath_events,
)
from m3resp.core.session import M3Session
from m3resp.data import BreathEvent
from m3resp.data.events import reuse_matching_breaths
from m3resp.eit.processing import BreathDetectionSettings, find_breaths
from m3resp.workflows.registry import StepArtifact

#: Steps that build or read eitprocessing objects require this package.
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
        breath_detector: ``BreathDetectionSettings`` (the breaths are found
            with ``m3resp.eit.processing.find_breaths``), or an
            eitprocessing detector with ``find_breaths``.
        signal: Waveform used by the per-breath calculations: a ``Signal``
            or an eitprocessing ``ContinuousData``.
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

    if isinstance(breath_detector, BreathDetectionSettings):
        detected = find_breaths(as_signal(signal), breath_detector)
    else:
        detected = breath_intervals_to_breath_events(
            breath_detector.find_breaths(as_continuous_data(signal))
        )
    breaths = _use_stored_eit_breaths(detected, session)
    found.append((breath_detector, signal, breaths))
    return breaths


def _use_stored_eit_breaths(
    breaths: list[BreathEvent], session: M3Session
) -> list[BreathEvent]:
    """Reuse stored EIT breaths with exactly matching modality and start-end times."""

    return reuse_matching_breaths(breaths, session.get_events("eit_breaths", None))


def _upstream_metadata(
    *,
    source_function: str,
    operation: str,
    parameters: dict[str, Any],
    source_package: str = "eitprocessing",
    implementation: str = "upstream_adapter",
) -> dict[str, Any]:
    """The provenance details an EIT step returns under ``"_provenance"``.

    They say which package and function computed the value, the settings
    used, and the step name. `source_package`/`implementation` default to
    the eitprocessing-adapter case; pass `source_package="m3resp"` and
    `implementation="m3resp.<module>"` for a step whose value comes from
    m3resp code.
    """

    return {
        "source_package": source_package,
        "source_function": source_function,
        "implementation": implementation,
        "parameters": parameters,
        "operation": operation,
    }
