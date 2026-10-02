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
    the same fields on :class:`Interval`, for the same reason.
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

    Use it for anything that lasts rather than happens at one instant: an
    occlusion, a period of noise, an expiratory hold, an intervention.
    :class:`BreathEvent` is an ``Interval`` with one more point inside it, the
    moment the signal turns from inhalation to exhalation.

    Attributes:
        modality: The device or technique the interval came from, e.g.
            ``'ventilator'`` or ``'emg'``.
        start_time: Real-world time at which the interval starts, in seconds.
        end_time: Real-world time at which the interval ends, in seconds.
        name: What kind of interval this is, e.g. ``'occlusion'`` or
            ``'noise'``. Required, and given by keyword.
        id: Per-process, in-memory identifier, generated automatically and
            excluded from equality. See :class:`Event` for the full
            explanation.
        label: Optional name for this particular interval, as opposed to
            ``name``, which says what kind of interval it is.
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

    ``start_time``/``end_time`` are always the authoritative, real-world
    times - they don't need to be recomputed from an index.
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
        if self.end_time < self.start_time:
            raise ValueError(
                f"{type(self).__name__}.end_time ({self.end_time}) must not be "
                f"before start_time ({self.start_time})"
            )

    @property
    def duration(self) -> float:
        """Length of the interval in the same time units as ``start_time``/``end_time``."""

        return self.end_time - self.start_time


@dataclass(kw_only=True)
class BreathEvent(Interval):
    """One breath from one modality: an :class:`Interval` with a turning point.

    EIT, EMG, and ventilator breath detectors all produce this one type, so
    breaths from different modalities can be compared and matched. It has
    every field of :class:`Interval`, with ``name`` always ``'breath'`` (it
    cannot be set to anything else), plus the two below.

    Attributes:
        extremum_time: Real-world time of the turning point between
            inhalation and exhalation. It is called an extremum, not a peak,
            because the signal can turn at a maximum (impedance, volume, EMG
            envelope) or at a minimum (esophageal pressure, an occlusion's
            deepest pressure). ``None`` when the detector didn't report one.
        extremum_index: Position of ``extremum_time`` in the signal named by
            ``signal_name``, or ``None``.

    Only ``modality``, ``start_time`` and ``end_time`` may be given by
    position, e.g. ``BreathEvent("eit", 1.0, 2.0, extremum_time=1.5)``.
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
    """Turn one breath, written in any of the forms below, into a `BreathEvent`.

    Accepts a `BreathEvent` (returned as it is), a dictionary, an object with
    ``start_time``/``end_time`` attributes (such as eitprocessing's
    ``Breath``, whose ``middle_time`` becomes ``extremum_time``), or a
    ``(start_time, end_time)`` or ``(start_time, end_time, extremum_time)``
    sequence. ``modality`` and ``source`` fill in whatever the input does
    not carry.

    The turning point is read from the first of these that is given:
    ``extremum_time``/``extremum_index``, then ``peak_time``/``peak_index``
    (the names older m3resp versions and other detectors use; a warning
    names the new keys), then eitprocessing's ``middle_time``. The time and
    its position always come from the same pair, so they point at the same
    moment.

    Raises:
        ValueError: If the input names itself as something other than a
            breath, e.g. an ``Interval`` with ``name="occlusion"``.
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
    """Turn a list of breaths, in any form `coerce_breath_event` accepts, into
    `BreathEvent` objects."""

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
    """Turn one interval, written in any of the forms below, into an `Interval`.

    Accepts an `Interval` (returned as it is), a dictionary, an object with
    ``start_time``/``end_time`` attributes (such as eitprocessing's
    ``Interval``), or a ``(start_time, end_time)`` pair. ``name``,
    ``modality`` and ``source`` fill in whatever the input does not carry.

    An input that is a breath (named ``'breath'``, or carrying a turning
    point in ``extremum_time``, ``extremum_index``, ``peak_time``,
    ``peak_index`` or ``middle_time``) comes back as
    a `BreathEvent`, which is also an `Interval`, so the turning point is
    kept.

    Raises:
        ValueError: If the input carries a turning point but is named
            something other than ``'breath'``: a plain `Interval` has no
            place to keep that point.
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
    """Turn a list of intervals, in any form `coerce_interval` accepts, into
    `Interval` objects."""

    return [
        coerce_interval(value, name=name, modality=modality, source=source)
        for value in values
    ]


def _read(value: Any, key: str) -> Any:
    """Read ``key`` from a dictionary, or the attribute of that name from an
    object. ``None`` when it is not there."""

    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _has_start_and_end(value: Any) -> bool:
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
    """The turning point of ``value`` as (time, index, key it was read from).

    The first pair in `_TURNING_POINT_KEYS` with a time or an index is used,
    and both come from that pair, so they never describe different moments.
    Returns (None, None, None) when there is no turning point. With
    ``warn=True``, reading the old ``peak_*`` names gives a warning.
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
    """The fields every `Interval` has, read from a dictionary or an object.

    Shared by `coerce_interval` and `coerce_breath_event`, so both keep the
    same fields. ``modality`` and ``source`` are used only when the input
    does not carry its own.
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
    """Turn an event, interval or breath into a plain dictionary of its fields,
    ready to write to a table or a JSON file."""

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
    missing = [key for key in ("start_time", "end_time") if key not in value]
    raise ValueError(
        f"Cannot make {target} from this dictionary: it has no "
        f"{' or '.join(missing)} entry."
    )


def reuse_matching_breaths(
    breaths: Sequence[BreathEvent], stored: Iterable[Any] | None
) -> list[BreathEvent]:
    """Swap each breath for an already stored breath that is the same breath.

    A stored breath counts as the same breath when it is a `BreathEvent`
    with the same modality and exactly the same start and end time. Results
    computed over a breath (a TIV value, say) can then point to the breath
    object that is already stored, instead of a copy of it. Breaths with no
    match are kept as they are, and stored items that are not a
    `BreathEvent` are ignored.

    Args:
        breaths: The breaths a result was computed over.
        stored: Breaths already stored, e.g. ``session.events["eit_breaths"]``,
            or None.

    Returns:
        The breaths, in the same order, with matches swapped for the stored
        ones.
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
