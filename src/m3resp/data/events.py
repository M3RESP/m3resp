"""Events, intervals and breaths shared by all modalities, and helpers that
turn other libraries' outputs into them."""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any, NoReturn, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


@dataclass
class Event:
    """A timestamped event from one modality.

    Attributes:
        name: What kind of event this is, e.g. ``'arterial_blood_gas_draw'``
            or ``'baydur_maneuver'``.
        modality: The device or technique the event came from, e.g.
            ``'ventilator'`` or ``'eit'``.
        time: Time at which the event occurred, in seconds on the recording's
            time axis or the shared clock after alignment.
        id: Random identifier, generated automatically. It stays the same
            when the event is shifted onto a common clock and when it is
            saved to a dictionary and read back. Results can use it to point
            at this event through ``ParameterResult.event_id``. Two events
            that differ only in ``id`` count as equal.
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
        label: Optional free-text note that tells this occurrence apart from
            others of the same kind, e.g. ``name='arterial_blood_gas_draw'``
            with ``label='before PEEP step 2'``.
        confidence: Optional measure of how sure the detector was.
        metadata: Optional extra information about the event.

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
class Interval:
    """A stretch of time from one modality, with a start and an end.

    Use it for an occlusion, a period of noise, an expiratory hold, or an
    intervention. Times are in seconds on the recording's time axis or the
    shared clock after alignment. Zero duration is allowed when only one
    time point is known, as with EMG breaths detected from their peaks.
    :class:`BreathEvent` is an ``Interval`` with one more point inside it,
    the signal's turning point.

    Attributes:
        modality: The device or technique the interval came from, e.g.
            ``'ventilator'`` or ``'emg'``.
        start_time: Start of the interval, in seconds on its time axis.
        end_time: End of the interval, in seconds on the same time axis.
        name: What kind of interval this is, e.g. ``'occlusion'`` or
            ``'noise'``. Required, and given by keyword.
        id: Random identifier that stays the same through time shifts and
            saving. See :class:`Event` for the full explanation.
        label: Optional name for this particular occurrence of the interval.
        start_index: Position of ``start_time`` in the signal this interval
            was found in. ``None`` when the interval wasn't derived from
            indexing into a signal.
        end_index: Position of ``end_time`` in that signal, or ``None``.
        sample_frequency: Sampling rate of that signal, in Hz. Together with
            ``signal_name`` it says which time axis the ``*_index`` fields
            are relative to, since different signals have different start
            times, durations, and sampling rates.
        signal_name: Which signal the ``*_index`` fields refer to.
        source: Optional name of the method that produced this interval.
        confidence: Optional measure of how sure that method was.
        metadata: Optional extra information about the interval.

    Everything after ``end_time`` is given by keyword, e.g.
    ``Interval("ventilator", 10.0, 12.5, name="occlusion")``.

    Raises:
        ValueError: If ``end_time`` is before ``start_time``.
    """

    modality: str
    start_time: float
    end_time: float
    _: KW_ONLY
    name: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex, compare=False)
    label: str | None = None
    start_index: int | None = None
    end_index: int | None = None
    sample_frequency: float | None = None
    signal_name: str | None = None
    source: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check that the interval ends at or after its start."""

        if self.end_time < self.start_time:
            raise ValueError(
                f"{type(self).__name__}.end_time ({self.end_time}) must not be "
                f"before start_time ({self.start_time})"
            )

    @property
    def duration(self) -> float:
        """Return the interval duration in seconds."""

        return self.end_time - self.start_time


@dataclass(kw_only=True)
class BreathEvent(Interval):
    """One breath from one modality, with a signal turning point.

    EIT, EMG, and ventilator breath detectors all produce this one type, so
    breaths from different modalities can be compared and matched. It has
    every field of :class:`Interval`, with ``name`` set to ``'breath'``
    automatically, plus the two below.

    Attributes:
        extremum_time: Time of the detected turning point, in seconds on the
            same time axis as ``start_time`` and ``end_time``. This is the
            inhalation-to-exhalation transition for EIT and ventilator volume,
            the EMG peak, or the pressure minimum for an occluded breath.
            ``None`` when the detector didn't report one.
        extremum_index: Position of ``extremum_time`` in the signal named by
            ``signal_name``, or ``None``.

    Give fields after ``end_time`` by keyword, e.g.
    ``BreathEvent("eit", 1.0, 2.0, extremum_time=1.5)``.
    """

    name: str = field(default="breath", init=False)
    extremum_time: float | None = None
    extremum_index: int | None = None


def coerce_event(
    value: Any,
    *,
    name: str | None = None,
    modality: str | None = None,
    label: str | None = None,
) -> Event:
    """Convert one timestamped event into an :class:`Event`.

    Args:
        value: An ``Event``, a dictionary with ``time``, an object with a
            ``time`` attribute, or a ``(name, modality, time)`` sequence with
            an optional fourth entry, ``sample_index``. Time is in seconds.
        name: Event name used when a dictionary or object omits it. For a
            sequence, this overrides its name.
        modality: Device or technique used when a dictionary or object omits
            it. For a sequence, this overrides its modality.
        label: Note used when a dictionary or object omits its label, or
            assigned to an event created from a sequence.

    Returns:
        Event: The existing ``Event`` unchanged, or a new event with the
            supplied fields. An input identifier is preserved; otherwise a
            new identifier is generated.

    Raises:
        KeyError: If a dictionary omits ``time``.
        ValueError: If the name or modality is missing, the sequence has
            other than three or four entries, or a time or sampling rate
            cannot be converted to a number.
        TypeError: If the input has an unsupported form or a time or sampling
            rate has an unsupported type.
    """

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


def coerce_events(
    values: Iterable[Any],
    *,
    name: str | None = None,
    modality: str | None = None,
    label: str | None = None,
) -> list[Event]:
    """Convert an iterable of events into a list of :class:`Event` objects.

    Args:
        values: Events in any form accepted by :func:`coerce_event`.
        name: Event name passed to ``coerce_event`` for each item.
        modality: Device or technique passed to ``coerce_event`` for each item.
        label: Note passed to ``coerce_event`` for each item.

    Returns:
        list[Event]: Events in input order. Each item is converted by
            ``coerce_event``, with the same field handling and errors.
    """

    return [
        coerce_event(value, name=name, modality=modality, label=label)
        for value in values
    ]


def coerce_breath_event(
    value: Any,
    *,
    modality: str | None = None,
    source: str | None = None,
) -> BreathEvent:
    """Convert one detected breath into a :class:`BreathEvent`.

    The turning point is read from ``extremum_time`` and ``extremum_index``,
    then from the older ``peak_time`` and ``peak_index``, then from
    eitprocessing's ``middle_time``. The first pair with a value is used, and
    both time and index come from that pair. Reading ``peak_time`` or
    ``peak_index`` gives a ``UserWarning`` naming the current fields.

    Args:
        value: A ``BreathEvent``, a dictionary or object with ``start_time``
            and ``end_time``, or a ``(start_time, end_time)`` sequence with an
            optional third entry, ``extremum_time``. All times are in seconds.
        modality: Device or technique used when the input's modality is
            missing or empty. Required for a sequence.
        source: Method name used when the input's source is missing or empty.

    Returns:
        BreathEvent: The existing breath unchanged, or a new breath with the
            input's timing, sample indices, label, identifier and metadata.
            A new identifier is generated when the input omits one.

    Raises:
        ValueError: If the input has a name other than ``'breath'``, omits a
            required modality or dictionary time, has other than two or
            three sequence entries, or ends before it starts. Also raised
            when a time or sampling rate cannot be converted to a number.
        TypeError: If the input has an unsupported form or a time or sampling
            rate has an unsupported type.
    """

    if isinstance(value, BreathEvent):
        return value

    if _has_start_and_end(value):
        name = _read(value, "name")
        if name is not None and name != "breath":
            raise ValueError(
                f"Cannot use a {name!r} interval as a breath. Only intervals "
                f"named 'breath' (or with no name) can become a BreathEvent."
            )
        turning_time, turning_index, _ = _turning_point(value, warn=True)
        return BreathEvent(
            **_interval_fields(value, modality=modality, source=source),
            extremum_time=_optional_float(turning_time),
            extremum_index=turning_index,
        )

    start_time, end_time, *rest = _coerce_positional_sequence(
        value, min_length=2, max_length=3, target="BreathEvent"
    )
    extremum_time = rest[0] if rest else None
    return BreathEvent(
        modality=_required_str(modality, "modality"),
        start_time=float(start_time),
        end_time=float(end_time),
        extremum_time=_optional_float(extremum_time),
        source=source,
    )


def coerce_breath_events(
    values: Iterable[Any],
    *,
    modality: str | None = None,
    source: str | None = None,
) -> list[BreathEvent]:
    """Convert an iterable of breaths into a list of :class:`BreathEvent` objects.

    Args:
        values: Breaths in any form accepted by :func:`coerce_breath_event`.
        modality: Device or technique used for items with a missing or empty
            modality.
        source: Method name used for items with a missing or empty source.

    Returns:
        list[BreathEvent]: Breaths in input order. Each item is converted by
            ``coerce_breath_event``, with the same field handling and errors.
    """

    return [
        coerce_breath_event(value, modality=modality, source=source) for value in values
    ]


def coerce_interval(
    value: Any,
    *,
    name: str | None = None,
    modality: str | None = None,
    source: str | None = None,
) -> Interval:
    """Convert one timed interval into an :class:`Interval`.

    Args:
        value: An ``Interval``, a dictionary or object with ``start_time``
            and ``end_time`` (including eitprocessing's ``Interval``), or a
            ``(start_time, end_time)`` pair. Times are in seconds.
        name: Interval kind used when the input's name is missing or empty.
            Required for a pair. ``'breath'`` selects ``BreathEvent``.
        modality: Device or technique used when the input's modality is
            missing or empty. Required for a pair.
        source: Method name used when the input's source is missing or empty.

    Returns:
        Interval: The existing interval unchanged, or a new interval with the
            input's fields. A converted input named ``'breath'`` or carrying
            a turning point becomes a ``BreathEvent``; the turning point is
            read as in :func:`coerce_breath_event`. Input identifiers are
            preserved; a new identifier is generated when omitted.

    Raises:
        ValueError: If a required name, modality or dictionary time is
            missing, a pair has other than two entries, or the end precedes
            the start. Also raised when a time or sampling rate cannot be
            converted to a number, or an input with a turning point has a
            name other than ``'breath'``.
        TypeError: If the input has an unsupported form or a time or sampling
            rate has an unsupported type.
    """

    if isinstance(value, Interval):
        return value

    if _has_start_and_end(value):
        resolved_name = _read(value, "name") or name
        turning_key = _turning_point(value)[2]
        if turning_key is not None:
            if resolved_name not in (None, "breath"):
                raise ValueError(
                    f"This {resolved_name!r} interval has a turning point "
                    f"({turning_key!r}), which an Interval cannot keep. Only "
                    f"breaths keep a turning point."
                )
            return coerce_breath_event(value, modality=modality, source=source)
        if resolved_name == "breath":
            return coerce_breath_event(value, modality=modality, source=source)
        return Interval(
            **_interval_fields(value, modality=modality, source=source),
            name=_required_str(resolved_name, "name"),
        )

    start_time, end_time = _coerce_positional_sequence(
        value, min_length=2, max_length=2, target="Interval"
    )
    if name == "breath":
        return BreathEvent(
            modality=_required_str(modality, "modality"),
            start_time=float(start_time),
            end_time=float(end_time),
            source=source,
        )
    return Interval(
        _required_str(modality, "modality"),
        float(start_time),
        float(end_time),
        name=_required_str(name, "name"),
        source=source,
    )


def coerce_intervals(
    values: Iterable[Any],
    *,
    name: str | None = None,
    modality: str | None = None,
    source: str | None = None,
) -> list[Interval]:
    """Convert an iterable of intervals into a list of :class:`Interval` objects.

    Args:
        values: Intervals in any form accepted by :func:`coerce_interval`.
        name: Interval kind used for items with a missing or empty name.
        modality: Device or technique used for items with a missing or empty
            modality.
        source: Method name used for items with a missing or empty source.

    Returns:
        list[Interval]: Intervals in input order, including ``BreathEvent``
            objects for converted breaths. Each item is converted by
            ``coerce_interval``, with the same field handling and errors.
    """

    return [
        coerce_interval(value, name=name, modality=modality, source=source)
        for value in values
    ]


def _read(value: Any, key: str) -> Any:
    """Read a dictionary entry or object attribute, returning ``None`` if absent."""

    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _has_start_and_end(value: Any) -> bool:
    """Return whether both interval time fields are present."""

    if isinstance(value, Mapping):
        return "start_time" in value and "end_time" in value
    return hasattr(value, "start_time") and hasattr(value, "end_time")


#: The names a breath's turning point can be given under, as (time key,
#: index key), most preferred first. ``peak_*`` is what older m3resp
#: versions and other detectors use; eitprocessing has only ``middle_time``.
_TURNING_POINT_KEYS: tuple[tuple[str, str | None], ...] = (
    ("extremum_time", "extremum_index"),
    ("peak_time", "peak_index"),
    ("middle_time", None),
)


def _turning_point(value: Any, *, warn: bool = False) -> tuple[Any, Any, str | None]:
    """Read the first available turning-point pair as (time, index, key).

    Pairs follow `_TURNING_POINT_KEYS` order. A pair is selected when either
    value is non-None; both values come from that pair. The returned key is
    the time key when its value is present, otherwise the index key.
    Returns ``(None, None, None)`` when all values are None. With ``warn=True``,
    selecting the older ``peak_*`` fields emits a UserWarning.
    """

    for time_key, index_key in _TURNING_POINT_KEYS:
        time = _read(value, time_key)
        index = None if index_key is None else _read(value, index_key)
        if time is None and index is None:
            continue
        if warn and time_key == "peak_time":
            warnings.warn(
                "Breath turning point given as 'peak_time'/'peak_index'; these "
                "are now called 'extremum_time'/'extremum_index'.",
                UserWarning,
                stacklevel=3,
            )
        return time, index, time_key if time is not None else index_key
    return None, None, None


def _interval_fields(
    value: Any, *, modality: str | None, source: str | None
) -> dict[str, Any]:
    """Read interval fields from a dictionary or object.

    ``modality`` and ``source`` fill missing or empty fields. Times and
    sampling rate are converted to floats, and metadata is copied.
    """

    fields: dict[str, Any] = {
        "modality": _required_str(_read(value, "modality") or modality, "modality"),
        "start_time": float(_read(value, "start_time")),
        "end_time": float(_read(value, "end_time")),
        "label": _read(value, "label"),
        "start_index": _read(value, "start_index"),
        "end_index": _read(value, "end_index"),
        "sample_frequency": _optional_float(_read(value, "sample_frequency")),
        "signal_name": _read(value, "signal_name"),
        "source": _read(value, "source") or source,
        "confidence": _read(value, "confidence"),
        "metadata": dict(_read(value, "metadata") or {}),
    }
    value_id = _read(value, "id")
    if value_id is not None:
        fields["id"] = str(value_id)
    return fields


def event_to_dict(
    value: Event | Interval | Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Return the fields of an event, interval or breath as a dictionary.

    Args:
        value: An ``Event``, ``Interval`` or ``BreathEvent``, another dataclass
            instance, a dictionary, or an iterable of key-value pairs.

    Returns:
        dict[str, Any]: All fields, including timing, identifier and metadata.
            Dataclass fields are copied recursively; dictionary inputs are
            copied at the top level. Field values keep their existing types.

    Raises:
        TypeError: If the input cannot be read as a dictionary.
        ValueError: If a key-value pair has an invalid length.
    """

    if isinstance(value, Mapping):
        return dict(value)
    if _is_dataclass_instance(value):
        return asdict(cast("DataclassInstance", value))
    return dict(cast(Any, value))


def _is_dataclass_instance(value: Any) -> bool:
    """Return whether the value is an instance of a dataclass."""

    return is_dataclass(value) and not isinstance(value, type)


def _optional_float(value: Any) -> float | None:
    """Convert a supplied value to a float, preserving ``None``."""

    if value is None:
        return None
    return float(value)


def _required_str(value: Any, field_name: str) -> str:
    """Convert a supplied field to text, raising ``ValueError`` for ``None``."""

    if value is None:
        raise ValueError(f"{field_name} is required")
    return str(value)


def _coerce_positional_sequence(
    value: Any, *, min_length: int, max_length: int, target: str
) -> tuple[Any, ...]:
    """Check the number of sequence entries and return them as a tuple.

    Strings, bytes and inputs without a length raise ``TypeError``. A length
    outside ``min_length`` to ``max_length`` raises ``ValueError``, as does a
    dictionary missing interval times. The conversion functions assign
    meanings to the entries.
    """

    if isinstance(value, Mapping):
        _raise_missing_times(value, target)
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


def _raise_missing_times(value: Mapping[str, Any], target: str) -> NoReturn:
    """Raise ``ValueError`` naming the missing interval time fields."""

    missing = [key for key in ("start_time", "end_time") if key not in value]
    raise ValueError(
        f"Cannot make {target} from this dictionary: it has no "
        f"{' or '.join(missing)} entry."
    )


def reuse_matching_breaths(
    breaths: Sequence[BreathEvent], stored: Iterable[Any] | None
) -> list[BreathEvent]:
    """Reuse stored breaths with the same modality, start time and end time.

    Matching uses exact equality of the modality and both times, in seconds
    on the same time axis. A match uses the stored object's identifier, peak
    timing and metadata. When several stored breaths match, the last one is
    used. Only ``BreathEvent`` objects in ``stored`` are considered.

    Args:
        breaths: Breaths to match, in the order required by the result.
        stored: Breaths already stored, e.g. ``session.events["eit_breaths"]``,
            or None.

    Returns:
        list[BreathEvent]: Stored matches and unmatched input breaths, in
            input order. The list holds the original objects, so subsequent
            changes to a matched breath also affect its stored result.
    """

    by_times = {
        (item.modality, item.start_time, item.end_time): item
        for item in stored or ()
        if isinstance(item, BreathEvent)
    }
    return [
        by_times.get((breath.modality, breath.start_time, breath.end_time), breath)
        for breath in breaths
    ]
