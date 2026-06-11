"""Common event models and normalization helpers used across modalities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


@dataclass
class Event:
    """A timestamped event from one modality."""

    name: str
    modality: str
    time: float
    sample_index: int | None = None
    label: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BreathEvent:
    """A respiratory event represented on a common time axis."""

    modality: str
    start_time: float
    end_time: float
    peak_time: float | None = None
    source: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def coerce_event(
    value: Any,
    *,
    name: str | None = None,
    modality: str | None = None,
    label: str | None = None,
) -> Event:
    """Normalize a generic timestamped event into an `Event`."""

    if isinstance(value, Event):
        return value

    if isinstance(value, Mapping):
        return Event(
            name=_required_str(value.get("name", name), "name"),
            modality=_required_str(value.get("modality", modality), "modality"),
            time=float(value["time"]),
            sample_index=value.get("sample_index"),
            label=value.get("label", label),
            confidence=value.get("confidence"),
            metadata=dict(value.get("metadata", {})),
        )

    if hasattr(value, "time"):
        return Event(
            name=_required_str(getattr(value, "name", name), "name"),
            modality=_required_str(getattr(value, "modality", modality), "modality"),
            time=float(getattr(value, "time")),
            sample_index=getattr(value, "sample_index", None),
            label=getattr(value, "label", label),
            confidence=getattr(value, "confidence", None),
            metadata=dict(getattr(value, "metadata", {}) or {}),
        )

    event_name, event_modality, time, *rest = value
    sample_index = rest[0] if rest else None
    return Event(
        name=_required_str(name if name is not None else event_name, "name"),
        modality=_required_str(
            modality if modality is not None else event_modality,
            "modality",
        ),
        time=float(time),
        sample_index=sample_index,
        label=label,
    )


def coerce_breath_event(
    value: Any,
    *,
    modality: str | None = None,
    source: str | None = None,
) -> BreathEvent:
    """Normalize one breath-like input into a `BreathEvent`."""

    if isinstance(value, BreathEvent):
        return value

    if isinstance(value, Mapping):
        return BreathEvent(
            modality=_required_str(value.get("modality", modality), "modality"),
            start_time=float(value["start_time"]),
            end_time=float(value["end_time"]),
            peak_time=_optional_float(value.get("peak_time")),
            source=value.get("source", source),
            confidence=value.get("confidence"),
            metadata=dict(value.get("metadata", {})),
        )

    if hasattr(value, "start_time") and hasattr(value, "end_time"):
        peak_time = getattr(value, "peak_time", None)
        if peak_time is None:
            peak_time = getattr(value, "middle_time", None)
        return BreathEvent(
            modality=_required_str(
                getattr(value, "modality", modality),
                "modality",
            ),
            start_time=float(getattr(value, "start_time")),
            end_time=float(getattr(value, "end_time")),
            peak_time=_optional_float(peak_time),
            source=getattr(value, "source", source),
            confidence=getattr(value, "confidence", None),
            metadata=dict(getattr(value, "metadata", {}) or {}),
        )

    start_time, end_time, *rest = value
    peak_time = rest[0] if rest else None
    return BreathEvent(
        modality=_required_str(modality, "modality"),
        start_time=float(start_time),
        end_time=float(end_time),
        peak_time=_optional_float(peak_time),
        source=source,
    )


def coerce_breath_events(
    values: Iterable[Any],
    *,
    modality: str | None = None,
    source: str | None = None,
) -> list[BreathEvent]:
    """Normalize breath-like inputs into `BreathEvent` objects."""

    return [
        coerce_breath_event(value, modality=modality, source=source) for value in values
    ]


def event_to_dict(
    value: Event | BreathEvent | Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Convert an event-like object to a serializable dictionary."""

    if isinstance(value, Mapping):
        return dict(value)
    if _is_dataclass_instance(value):
        return asdict(cast("DataclassInstance", value))
    return dict(cast(Any, value))


def _is_dataclass_instance(value: Any) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _required_str(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    return str(value)
