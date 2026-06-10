"""Basic Stage 1 modality alignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, TypeVar

from m3resp.core.events import BreathEvent, Event

TEvent = TypeVar("TEvent", Event, BreathEvent)


def align_events_manual_offset(
    events: list[TEvent], offset_seconds: float
) -> list[TEvent]:
    """Return copies of events shifted by a manual offset."""

    offset = float(offset_seconds)
    aligned: list[TEvent] = []
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


def align_events_by_modality_offset(
    events: Sequence[TEvent],
    offsets_seconds: Mapping[str, float],
) -> list[TEvent]:
    """Return event copies shifted by the offset configured for each modality."""

    aligned: list[TEvent] = []
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
