"""Convert M3Resp objects into table rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from m3resp.core.events import BreathEvent, Event, event_to_dict

if TYPE_CHECKING:
    from m3resp.data.linked_breath import LinkedBreath

#: `LinkedBreath` modality slot -> attribute name, in row-column order.
_LINKED_BREATH_SLOTS = ("eit", "emg", "ventilator")


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


def linked_breaths_to_rows(linked_breaths: list[LinkedBreath]) -> list[dict[str, Any]]:
    """Flatten `LinkedBreath` objects into one row per link (Milestone 2.5/2.6).

    Each modality's breath fields are prefixed (``eit_start_time``,
    ``emg_peak_time``, ...) and left ``None`` when that modality has no
    breath in the link, so the CSV has a stable column set regardless of
    which modalities matched.
    """

    rows: list[dict[str, Any]] = []
    for linked in linked_breaths:
        row: dict[str, Any] = {
            "modalities": "+".join(linked.modalities),
            "confidence": linked.confidence,
            "time_tolerance": linked.time_tolerance,
        }
        for slot in _LINKED_BREATH_SLOTS:
            breath = linked.breaths.get(slot)
            row[f"{slot}_start_time"] = None if breath is None else breath.start_time
            row[f"{slot}_end_time"] = None if breath is None else breath.end_time
            row[f"{slot}_peak_time"] = None if breath is None else breath.peak_time
        rows.append(row)
    return rows
