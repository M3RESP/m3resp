"""``IntervalData`` and ``EventData``: values that belong to intervals or events.

Some results have one value per breath, per occlusion or per heartbeat.
Examples include tidal impedance variation per breath, a pixel map of breath
timing per breath, or the amplitude of each ECG peak. These two types keep
each value next to the interval or event it belongs to.

They take the place of eitprocessing's ``IntervalData`` (values per interval)
and ``SparseData`` (values at single times) inside m3resp.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from m3resp.data.categories import normalize_category
from m3resp.data.events import Event, Interval, event_to_dict
from m3resp.data.units import normalize_unit


class _TimedData:
    """Shared checks, comparison and dictionary conversion for timed values."""

    _items_field: ClassVar[str]
    _item_type: ClassVar[type]
    _convert_with: ClassVar[str]

    name: str
    modality: str
    values: Sequence[Any] | np.ndarray | None
    category: str | None
    unit: str | None
    method: str | None
    metadata: dict[str, Any]

    def _items(self) -> list[Any]:
        """Return the intervals or events associated with the values."""

        items: list[Any] = getattr(self, self._items_field)
        return items

    def _check_and_tidy(self) -> None:
        """Check item types and value count, then normalize unit and category names."""

        owner = type(self).__name__
        items = list(self._items())
        setattr(self, self._items_field, items)
        for item in items:
            if not isinstance(item, self._item_type):
                raise TypeError(
                    f"{owner}.{self._items_field} must hold "
                    f"{self._item_type.__name__} objects, got "
                    f"{type(item).__name__}. Use {self._convert_with}() to "
                    f"convert other types first."
                )
        _check_one_value_each(self.values, len(items), owner)
        self.unit = normalize_unit(self.unit)
        self.category = normalize_category(self.category) or self.category

    def __len__(self) -> int:
        """Return the number of intervals or events."""

        return len(self._items())

    def __eq__(self, other: object) -> bool:
        """Compare fields and values, treating matching floating-point NaNs as equal."""

        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, _TimedData)
        return (
            self.name == other.name
            and self.modality == other.modality
            and self._items() == other._items()
            and _values_equal(self.values, other.values)
            and self.category == other.category
            and self.unit == other.unit
            and self.method == other.method
            and self.metadata == other.metadata
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the timed values and their descriptive fields as a dictionary.

        Arrays in ``values`` become lists, NumPy scalars become Python
        scalars, and objects such as ``ParameterResult`` are converted with
        their own ``to_dict()`` method. The metadata dictionary is copied at
        the top level.

        Returns:
            dict[str, Any]: Name, modality, intervals or events, values,
                category, unit, method and metadata. Items and values keep
                their order; omitted values remain ``None``.
        """

        return {
            "name": self.name,
            "modality": self.modality,
            self._items_field: [event_to_dict(item) for item in self._items()],
            "values": _values_to_lists(self.values),
            "category": self.category,
            "unit": self.unit,
            "method": self.method,
            "metadata": dict(self.metadata),
        }


@dataclass(eq=False)
class IntervalData(_TimedData):
    """One value for each interval, in the same order as the intervals.

    Attributes:
        name: What the values are, e.g. ``'tidal_impedance_variation'``.
        modality: The device the values were measured with. This can differ
            from the device the intervals were found in: EIT values can be
            computed over breaths found in ventilator data.
        intervals: The intervals the values belong to. A ``BreathEvent`` is
            an ``Interval``, so a list of breaths works here too.
        values: One value per interval, in the same order: a list, a tuple,
            or an array whose first axis runs over the intervals. A value can
            be a number, an array (for example a pixel map) or an m3resp
            object such as a ``ParameterResult``. ``None`` when only the
            intervals themselves are the result.
        category: The physical quantity the values are derived from (see
            :mod:`m3resp.data.categories`).
        unit: Unit of the values, normalized to a recognized spelling.
        method: Name of the method that produced the values.
        metadata: Optional extra information.

    Raises:
        TypeError: If an item of ``intervals`` is not an ``Interval``, or
            supplied ``values`` are other than a list, tuple or array with
            at least one dimension.
        ValueError: If supplied ``values`` have a different number of entries
            than ``intervals``.
    """

    _items_field: ClassVar[str] = "intervals"
    _item_type: ClassVar[type] = Interval
    _convert_with: ClassVar[str] = "coerce_intervals"

    name: str
    modality: str
    intervals: list[Interval]
    values: Sequence[Any] | np.ndarray | None = None
    category: str | None = None
    unit: str | None = None
    method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check intervals and values, then normalize units and categories."""

        self._check_and_tidy()


@dataclass(eq=False)
class EventData(_TimedData):
    """One value for each event, in the same order as the events.

    Each value belongs to something that happens at one instant, such as an
    ECG peak represented by an :class:`Event`.

    Attributes:
        name: What the values are, e.g. ``'ecg_peak_amplitude'``.
        modality: The device the values were measured with. This can differ
            from the device the events came from.
        events: The events the values belong to.
        values: One value per event, in the same order, or ``None``. The
            same forms as for :class:`IntervalData` are accepted.
        category: The physical quantity the values are derived from.
        unit: Unit of the values, normalized to a recognized spelling.
        method: Name of the method that produced the values.
        metadata: Optional extra information.

    Raises:
        TypeError: If an item of ``events`` is not an ``Event``, or supplied
            ``values`` are other than a list, tuple or array with at least
            one dimension.
        ValueError: If supplied ``values`` have a different number of entries
            than ``events``.
    """

    _items_field: ClassVar[str] = "events"
    _item_type: ClassVar[type] = Event
    _convert_with: ClassVar[str] = "coerce_events"

    name: str
    modality: str
    events: list[Event]
    values: Sequence[Any] | np.ndarray | None = None
    category: str | None = None
    unit: str | None = None
    method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check events and values, then normalize units and categories."""

        self._check_and_tidy()


def _check_one_value_each(values: Any, expected: int, owner: str) -> None:
    """Check that supplied values form a list, tuple or array of the expected length."""

    if values is None:
        return
    is_list_or_array = (
        isinstance(values, np.ndarray) and values.ndim > 0
    ) or isinstance(values, (list, tuple))
    if not is_list_or_array:
        raise TypeError(
            f"{owner}.values must be a list, tuple or array with one value per "
            f"item, got {type(values).__name__}"
            + (" with a single number" if isinstance(values, np.ndarray) else "")
            + "."
        )
    if len(values) != expected:
        raise ValueError(
            f"{owner}.values has {len(values)} entries but there are {expected} "
            f"items; there must be exactly one value per item."
        )


def _values_equal(first: Any, second: Any) -> bool:
    """Compare value sequences, treating matching floating-point NaNs as equal."""

    if first is None or second is None:
        return first is None and second is None
    if len(first) != len(second):
        return False
    return all(
        np.array_equal(a, b, equal_nan=_is_float_like(a) and _is_float_like(b))
        for a, b in zip(first, second, strict=True)
    )


def _is_float_like(value: Any) -> bool:
    """Return whether the value has a floating-point or complex NumPy dtype."""

    return np.asarray(value).dtype.kind in "fc"


def _values_to_lists(values: Any) -> list[Any] | None:
    """Convert each value to a list, scalar or dictionary, preserving ``None``."""

    if values is None:
        return None
    return [_value_to_plain(value) for value in values]


def _value_to_plain(value: Any) -> Any:
    """Convert one value using its ``to_dict()`` method or NumPy conversion."""

    if callable(getattr(value, "to_dict", None)):
        return value.to_dict()
    if np.ndim(value) > 0:
        return np.asarray(value).tolist()
    return _scalar(value)


def _scalar(value: Any) -> Any:
    """Convert a NumPy scalar to its Python value."""

    return value.item() if isinstance(value, np.generic) else value
