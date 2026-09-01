"""Common event models and normalization helpers used across modalities."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


@dataclass
class Event:
    """A timestamped event from one modality.

    Attributes:
        name: What kind of event this is, e.g. ``'blood_gas_draw'`` or
            ``'intervention'``.
        modality: The device or technique the event came from, e.g.
            ``'ventilator'`` or ``'eit'``.
        time: Real-world time at which the event occurred.
        id: Per-process, in-memory identifier (not persisted or globally
            unique like Layer 2's ids) that lets other Layer 1 objects -
            notably ``ParameterResult.event_id`` - reference this specific
            event. Generated automatically, and excluded from equality
            (``compare=False``) so two structurally identical events - the
            common case in tests and deduplication - still compare equal.
        sample_index: Position of this event in the signal it was detected
            in. Only meaningful together with ``signal_name`` and
            ``sample_frequency``, which say which signal and time axis it is
            relative to - a modality can have multiple signals at different
            sampling rates, so ``sample_index`` alone is ambiguous. ``None``
            when the event wasn't derived from indexing into a signal.
        signal_name: Which signal ``sample_index`` refers to. ``None`` when
            the event wasn't derived from indexing into a signal.
        sample_frequency: Sampling rate of that signal, in Hz. ``None`` when
            the event wasn't derived from indexing into a signal.
        label: Optional name for this particular occurrence, as opposed to
            ``name``, which says what kind of event it is: an intervention
            event carries ``name='intervention'`` and, say,
            ``label='baydur_maneuver'``. Nothing in the library reads it
            today; it is carried through for downstream use.
        confidence: Optional measure of how sure the detector was.
        metadata: Optional extra information about the event.

    The ``sample_index``/``signal_name``/``sample_frequency`` trio mirrors
    the same fields on :class:`BreathEvent`, for the same reason.
    """

    name: str
    modality: str
    time: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex, compare=False)
    sample_index: int | None = None
    signal_name: str | None = None
    sample_frequency: float | None = None
    label: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BreathEvent:
    """A respiratory event represented on a common time axis.

    Unlike :class:`Event`, which occurs at a single ``time``, a
    ``BreathEvent`` spans an interval (``start_time`` to ``end_time``) - one
    breath, not an instant. The sample-index fields and :attr:`duration` were
    added here rather than introducing a parallel ``Breath`` dataclass, so
    EIT, EMG, and ventilator breath detectors keep sharing one type.

    Attributes:
        modality: The device or technique the breath was detected in, e.g.
            ``'eit'`` or ``'emg'``.
        start_time: Real-world time at which the breath starts.
        end_time: Real-world time at which the breath ends.
        id: Per-process, in-memory identifier, generated automatically and
            excluded from equality. See :class:`Event` for the full
            explanation.
        peak_time: Real-world time of the turning point between inhalation
            and exhalation. ``None`` when the detector didn't report one.
        start_index: Position of ``start_time`` in the signal this breath was
            detected in. ``None`` when the breath wasn't derived from
            indexing into a signal.
        peak_index: Position of ``peak_time`` in that signal, or ``None``.
        end_index: Position of ``end_time`` in that signal, or ``None``.
        sample_frequency: Sampling rate of that signal, in Hz. Together with
            ``signal_name`` it says which time axis the ``*_index`` fields
            are relative to, since different signals have different start
            times, durations, and sampling rates.
        signal_name: Which signal the ``*_index`` fields refer to.
        source: Optional name of the detector that produced this breath.
        confidence: Optional measure of how sure that detector was.
        metadata: Optional extra information about the breath.

    Raises:
        ValueError: If ``end_time`` is before ``start_time``.

    ``start_time``/``end_time``/``peak_time`` are always the authoritative,
    real-world times - they don't need to be recomputed from an index.
    """

    modality: str
    start_time: float
    end_time: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex, compare=False)
    peak_time: float | None = None
    start_index: int | None = None
    peak_index: int | None = None
    end_index: int | None = None
    sample_frequency: float | None = None
    signal_name: str | None = None
    source: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end_time < self.start_time:
            raise ValueError(
                f"BreathEvent.end_time ({self.end_time}) must not be before "
                f"start_time ({self.start_time})"
            )

    @property
    def duration(self) -> float:
        """Breath duration in the same time units as ``start_time``/``end_time``."""

        return self.end_time - self.start_time


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
        kwargs: dict[str, Any] = {}
        if value.get("id") is not None:
            kwargs["id"] = str(value["id"])
        return Event(
            name=_required_str(value.get("name", name), "name"),
            modality=_required_str(value.get("modality", modality), "modality"),
            time=float(value["time"]),
            sample_index=value.get("sample_index"),
            signal_name=value.get("signal_name"),
            sample_frequency=_optional_float(value.get("sample_frequency")),
            label=value.get("label", label),
            confidence=value.get("confidence"),
            metadata=dict(value.get("metadata") or {}),
            **kwargs,
        )

    if hasattr(value, "time"):
        kwargs = {}
        value_id = getattr(value, "id", None)
        if value_id is not None:
            kwargs["id"] = str(value_id)
        return Event(
            name=_required_str(getattr(value, "name", name), "name"),
            modality=_required_str(getattr(value, "modality", modality), "modality"),
            time=float(value.time),
            sample_index=getattr(value, "sample_index", None),
            signal_name=getattr(value, "signal_name", None),
            sample_frequency=_optional_float(getattr(value, "sample_frequency", None)),
            label=getattr(value, "label", label),
            confidence=getattr(value, "confidence", None),
            metadata=dict(getattr(value, "metadata", {}) or {}),
            **kwargs,
        )

    event_name, event_modality, time, *rest = _coerce_positional_sequence(
        value, min_length=3, max_length=4, target="Event"
    )
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
        kwargs: dict[str, Any] = {}
        if value.get("id") is not None:
            kwargs["id"] = str(value["id"])
        peak_time = value.get("peak_time")
        if peak_time is None:
            peak_time = value.get("middle_time")
        return BreathEvent(
            modality=_required_str(value.get("modality", modality), "modality"),
            start_time=float(value["start_time"]),
            end_time=float(value["end_time"]),
            peak_time=_optional_float(peak_time),
            start_index=value.get("start_index"),
            peak_index=value.get("peak_index"),
            end_index=value.get("end_index"),
            sample_frequency=_optional_float(value.get("sample_frequency")),
            signal_name=value.get("signal_name"),
            source=value.get("source", source),
            confidence=value.get("confidence"),
            metadata=dict(value.get("metadata") or {}),
            **kwargs,
        )

    if hasattr(value, "start_time") and hasattr(value, "end_time"):
        peak_time = getattr(value, "peak_time", None)
        if peak_time is None:
            peak_time = getattr(value, "middle_time", None)
        kwargs = {}
        value_id = getattr(value, "id", None)
        if value_id is not None:
            kwargs["id"] = str(value_id)
        return BreathEvent(
            modality=_required_str(
                getattr(value, "modality", modality),
                "modality",
            ),
            start_time=float(value.start_time),
            end_time=float(value.end_time),
            peak_time=_optional_float(peak_time),
            start_index=getattr(value, "start_index", None),
            peak_index=getattr(value, "peak_index", None),
            end_index=getattr(value, "end_index", None),
            sample_frequency=_optional_float(getattr(value, "sample_frequency", None)),
            signal_name=getattr(value, "signal_name", None),
            source=getattr(value, "source", source),
            confidence=getattr(value, "confidence", None),
            metadata=dict(getattr(value, "metadata", {}) or {}),
            **kwargs,
        )

    start_time, end_time, *rest = _coerce_positional_sequence(
        value, min_length=2, max_length=3, target="BreathEvent"
    )
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


def _coerce_positional_sequence(
    value: Any, *, min_length: int, max_length: int, target: str
) -> tuple[Any, ...]:
    """Validate the positional fallback `coerce_event`/`coerce_breath_event` use
    for a `detector=callable` that returns a bare sequence rather than a dict
    or one of our own types.

    A string/bytes value or anything without a length is rejected outright -
    without this, a short string would silently iterate character by
    character. Anything else is checked against ``min_length``/``max_length``
    before being unpacked, so a same-length-but-different-meaning sequence
    (e.g. a ``(confidence, snr)`` pair reaching `coerce_breath_event`) raises
    immediately naming what was expected, rather than silently becoming a
    plausible but wrong `Event`/`BreathEvent`.
    """

    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise TypeError(
            f"Cannot coerce {value!r} into a {target}: expected a mapping, an "
            f"object with the right attributes, or a sequence of "
            f"{min_length} to {max_length} values."
        )
    length = len(value)
    if not min_length <= length <= max_length:
        raise ValueError(
            f"Cannot coerce a length-{length} sequence into a {target}: "
            f"expected {min_length} to {max_length} values, got {tuple(value)!r}."
        )
    return tuple(value)
