"""Basic Stage 1 modality alignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, overload

from m3resp.core.events import BreathEvent, Event


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
    """Return event copies shifted by the offset configured for each modality."""

    aligned: list[Any] = []
    for event in events:
        modality = _event_modality(event)
        offset = float(offsets_seconds.get(modality, 0.0))
        aligned.extend(align_events_manual_offset([event], offset))
    return aligned


def _event_modality(event: Any) -> str:
    if isinstance(event, (BreathEvent, Event)):
        return event.modality
    raise TypeError(
        "Manual offset alignment supports only Event and BreathEvent objects."
    )
