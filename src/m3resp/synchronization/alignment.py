"""Basic Stage 1 modality alignment."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

from m3resp.core.events import BreathEvent, Event

TEvent = TypeVar("TEvent", Event, BreathEvent)


def align_events_manual_offset(
    events: list[TEvent], offset_seconds: float
) -> list[TEvent]:
    """Return copies of events shifted by a manual offset."""

    aligned: list[TEvent] = []
    for event in events:
        if isinstance(event, BreathEvent):
            aligned.append(
                replace(
                    event,
                    start_time=event.start_time + offset_seconds,
                    end_time=event.end_time + offset_seconds,
                    peak_time=(
                        None
                        if event.peak_time is None
                        else event.peak_time + offset_seconds
                    ),
                )
            )
        else:
            aligned.append(replace(event, time=event.time + offset_seconds))
    return aligned
