"""Shared helpers for the registered EIT workflow step modules."""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable
from typing import Any

from m3resp.adapters.eitprocessing_adapter import (
    as_continuous_data,
    as_signal,
    breath_intervals_to_breath_events,
    given_breath_events,
)
from m3resp.core.session import M3Session
from m3resp.data import BreathEvent, IntervalData, Signal
from m3resp.data.events import Interval, reuse_matching_breaths
from m3resp.eit.processing import BreathDetectionSettings, find_breaths
from m3resp.modalities.names import normalize_modality
from m3resp.synchronization.start_times import on_clock_of
from m3resp.synchronization.sync_methods import warn_if_not_synchronized
from m3resp.workflows.registry import BREATHS_ARTIFACT_TYPE, StepArtifact

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


#: Input of TIV and EELI: breaths of any modality, from another step.
_BREATHS_ARTIFACT = StepArtifact(
    name="breaths",
    artifact_type=BREATHS_ARTIFACT_TYPE,
    required=False,
    description=(
        "Breaths of any modality for the values; without them, the step finds "
        "the breaths with 'breath_detector'. A list of BreathEvents or an "
        "IntervalData of breaths, for example "
        "the breaths of 'eit.detect_breaths', 'emg.detect_breaths' or "
        "'ventilator.normalize_breaths'. Breaths of another modality are moved "
        "to the EIT clock with the start times of the session. The values "
        "are a TIV and an EELI only when the breath times mark the start and "
        "the end of inspiration and the end of expiration in the impedance."
    ),
)


def _on_eit_clock(
    breaths: list[BreathEvent], session: M3Session, *, action: str
) -> list[BreathEvent]:
    """Return copies of the breaths with their times on the EIT clock.

    Args:
        breaths: The breaths, each on the clock of its modality.
        session: Session with the start time of each recording.
        action: Step name, for the warning.

    Returns:
        list[BreathEvent]: One copy per breath, with start, extremum and end
            in seconds on the EIT clock (see ``on_clock_of``).

    Warns:
        UnsynchronizedDataWarning: When breaths of another modality are used
            and the recordings were never synchronized.
    """

    modalities = sorted({normalize_modality(breath.modality) for breath in breaths})
    if any(modality != "eit" for modality in modalities):
        warn_if_not_synchronized(session, ["eit", *modalities], action=action)
    return on_clock_of(session, breaths, "eit")


def _per_breath_values(
    calculate: Callable[[Signal, list[BreathEvent]], IntervalData],
    signal: Any,
    *,
    breaths: Any,
    breath_detector: Any,
    session: M3Session,
    action: str,
) -> IntervalData:
    """Run a TIV or EELI calculation on the given breaths or on the found ones.

    Args:
        calculate: ``tidal_impedance_variation`` or
            ``end_expiratory_lung_impedance`` with its settings, taking the
            waveform and the breaths.
        signal: The impedance waveform (``Signal`` or eitprocessing
            ``ContinuousData``).
        breaths: Breaths of the step input ``breaths``, or None.
        breath_detector: Breath detection settings, used when ``breaths`` is
            None.
        session: Session with the start times and the stored EIT breaths.
        action: Step name, for the warning.

    Returns:
        IntervalData: The result of ``calculate``. With given breaths, it
            holds the given breath objects, on the clock of their modality.

    Raises:
        ValueError: If both ``breaths`` and ``breath_detector`` are None.
    """

    waveform = as_signal(signal)
    if breaths is None:
        if breath_detector is None:
            raise ValueError(
                f"{action} needs the breaths: connect 'breaths', or "
                "'breath_detector' from 'eit.detect_breaths'."
            )
        return calculate(waveform, _breaths_used_by(breath_detector, signal, session))
    given = given_breath_events(breaths, function=action)
    result = calculate(waveform, _on_eit_clock(given, session, action=action))
    return dataclasses.replace(
        result, intervals=list[Interval](given), metadata=dict(result.metadata)
    )
