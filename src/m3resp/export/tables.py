"""Convert M3Resp objects into table rows."""

from __future__ import annotations

from typing import Any

from m3resp.core.events import BreathEvent, Event, event_to_dict


def events_to_rows(events: list[Event] | list[BreathEvent]) -> list[dict[str, Any]]:
    """Convert event dataclasses to serializable rows."""

    rows: list[dict[str, Any]] = []
    for event in events:
        rows.append(event_to_dict(event))
    return rows


def parameters_to_rows(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten top-level parameter groups into serializable rows."""

    rows: list[dict[str, Any]] = []
    for modality, values in parameters.items():
        if isinstance(values, dict):
            for name, value in values.items():
                rows.append({"modality": modality, "name": name, "value": value})
        else:
            rows.append({"modality": modality, "name": "value", "value": values})
    return rows
