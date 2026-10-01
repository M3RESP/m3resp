"""Basic Stage 1 modality alignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, overload

from m3resp.core.events import BreathEvent, Event
from m3resp.modalities.names import VENTILATOR, normalize_modality

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


@overload
def align_events_manual_offset(
    events: Sequence[Event], offset_seconds: float
) -> list[Event]: ...


@overload
def align_events_manual_offset(
    events: Sequence[BreathEvent], offset_seconds: float
) -> list[BreathEvent]: ...


@overload
def align_events_manual_offset(
    events: Sequence[Event | BreathEvent], offset_seconds: float
) -> list[Event | BreathEvent]: ...


def align_events_manual_offset(
    events: Sequence[Event | BreathEvent], offset_seconds: float
) -> list[Any]:
    """Return copies of events shifted by a manual offset."""

    offset = float(offset_seconds)
    aligned: list[Any] = []
    for event in events:
        if isinstance(event, BreathEvent):
            aligned.append(
                replace(
                    event,
                    start_time=event.start_time + offset,
                    end_time=event.end_time + offset,
                    peak_time=(
                        None if event.peak_time is None else event.peak_time + offset
                    ),
                )
            )
        elif isinstance(event, Event):
            aligned.append(replace(event, time=event.time + offset))
        else:
            raise TypeError(
                "Manual offset alignment supports only Event and BreathEvent objects."
            )
    return aligned


@overload
def align_events_by_modality_offset(
    events: Sequence[Event],
    offsets_seconds: Mapping[str, float],
) -> list[Event]: ...


@overload
def align_events_by_modality_offset(
    events: Sequence[BreathEvent],
    offsets_seconds: Mapping[str, float],
) -> list[BreathEvent]: ...


@overload
def align_events_by_modality_offset(
    events: Sequence[Event | BreathEvent],
    offsets_seconds: Mapping[str, float],
) -> list[Event | BreathEvent]: ...


def align_events_by_modality_offset(
    events: Sequence[Event | BreathEvent],
    offsets_seconds: Mapping[str, float],
) -> list[Any]:
    """Return event copies shifted by the offset configured for each modality.

    Both the event's modality and the offset keys are canonicalized before
    matching, so an event still tagged with a legacy spelling (``"vent"``) is
    shifted by an offset given under the canonical name (``"ventilator"``), and
    vice versa. Without this, a mismatch would silently apply a zero offset
    rather than raise.
    """

    offsets_by_canonical_modality = {
        normalize_modality(modality): offset
        for modality, offset in offsets_seconds.items()
    }
    aligned: list[Any] = []
    for event in events:
        modality = normalize_modality(_event_modality(event))
        offset = float(offsets_by_canonical_modality.get(modality, 0.0))
        aligned.extend(align_events_manual_offset([event], offset))
    return aligned


def _event_modality(event: Any) -> str:
    if isinstance(event, (BreathEvent, Event)):
        return event.modality
    raise TypeError(
        "Manual offset alignment supports only Event and BreathEvent objects."
    )


def compute_offsets_from_timestamps(
    reference_modality: str, timestamps: Mapping[str, float]
) -> dict[str, float]:
    """Convert absolute per-device start timestamps into manual offsets
    (plan_stage2.md Sec 20, "timestamp alignment").

    Each modality's events are relative to that modality's own recording
    start. If a modality's device started recording ``timestamps[modality]``
    seconds after ``reference_modality``'s device did, its events must be
    shifted forward by that same amount to land on the reference timeline.
    Feed the result straight into `align_events_by_modality_offset`.
    """

    if reference_modality not in timestamps:
        raise KeyError(
            f"reference_modality {reference_modality!r} is not in timestamps"
        )
    reference_time = timestamps[reference_modality]
    return {modality: value - reference_time for modality, value in timestamps.items()}


def ventilator_start_key(name: str) -> str:
    """The `session.start_times` / `offset_seconds` key for one standalone
    ventilator recording, e.g. ``"ventilator:monitor"``."""

    return f"{VENTILATOR}:{name}"


def normalize_offset_key(key: str) -> str:
    """Canonicalize an `offset_seconds` / `start_times` key.

    A modality name is normalized as by `normalize_modality`. A key for one
    ventilator recording (``"vent:Monitor"``, ``"ventilator:Monitor"``) keeps
    the recording's name exactly as given, since names are case-sensitive.
    """

    head, separator, name = str(key).partition(":")
    if not separator:
        return normalize_modality(key)
    return f"{normalize_modality(head)}:{name}"


def resolve_alignment_offsets(
    offset_seconds: float | Mapping[str, float],
) -> dict[str, float]:
    if isinstance(offset_seconds, Mapping):
        offsets = {"eit": 0.0, "emg": 0.0, VENTILATOR: 0.0}
        for modality, offset in offset_seconds.items():
            offsets[normalize_offset_key(modality)] = float(offset)
        return offsets
    return {"eit": 0.0, "emg": float(offset_seconds), VENTILATOR: 0.0}


def offsets_relative_to_reference(
    offsets: Mapping[str, float],
    reference_modality: str,
) -> dict[str, float]:
    reference = normalize_offset_key(reference_modality)
    reference_offset = float(offsets.get(reference, 0.0))
    return {
        normalize_offset_key(modality): float(offset) - reference_offset
        for modality, offset in offsets.items()
    }


def raw_reference_modality(session: M3Session, reference_modality: str | None) -> str:
    """The recording `M3Session.synchronize_raw_modalities` keeps at start
    time 0; every other start time is relative to it.

    The one asked for when `reference_modality` is given (a modality, or
    ``"ventilator:<name>"``). Otherwise the ventilator if one is loaded, then
    EIT, then EMG; EIT when nothing is loaded.
    """

    if reference_modality is not None:
        return normalize_offset_key(reference_modality)
    if VENTILATOR in session.raw or "vent" in session.raw:
        return VENTILATOR
    if "eit" in session.raw:
        return "eit"
    if "emg" in session.raw:
        return "emg"
    return "eit"


def breath_reference_modality(
    session: M3Session, reference_modality: str | None
) -> tuple[str, str | None]:
    """The modality whose breaths `M3Session.synchronize_multimodal_breaths`
    does not move by an extra offset; every other offset is relative to it.

    The one asked for when `reference_modality` is given. Otherwise the
    ventilator if there are ventilator breaths, and EIT when there are none.

    Returns ``(reference, fallback)``. `fallback` is ``"eit"`` when EIT was
    picked only because there were no ventilator breaths, and None otherwise.
    """

    if reference_modality is not None:
        return normalize_modality(reference_modality), None
    if session.events.get("ventilator_breaths"):
        return VENTILATOR, None
    return "eit", "eit"
