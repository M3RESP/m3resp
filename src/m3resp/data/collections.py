"""Lists of `Signal`, `ParameterResult`, `QualityFlag`, `IntervalData` and
`PixelMask` objects, with helpers to look items up. `M3Session` keeps one of
each.

Events and breaths are kept in `session.events` (see
`M3Session.add_events`/`get_events`).

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/roi/pixelmaskcollection.py::
                PixelMaskCollection.combine; PixelMaskCollection.apply
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The collection is a list of masks with names. `apply` gives a
      dictionary with the mask names as keys, and refuses two masks with the
      same name (eitprocessing: a dictionary of masks with labels, or of
      masks without labels, keyed 0, 1, 2, ...).
    - `combine` takes a `name` for the combined mask, and `apply` takes a
      `name_format` with `{mask_name}` (eitprocessing: `label` and
      `label_format` with `{mask_label}`).
    - `apply` works on a NumPy array, a `Signal` or a `PixelMap`.
    - Error messages are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import Any, Literal

import numpy as np

from m3resp.data.categories import normalize_category
from m3resp.data.event_data import IntervalData
from m3resp.data.parameters import ParameterResult
from m3resp.data.pixel_maps import PixelMap, PixelMask
from m3resp.data.quality import QualityFlag
from m3resp.data.signals import Signal


@dataclass
class SignalCollection:
    """An ordered collection of `Signal` objects, queryable by modality."""

    items: list[Signal] = field(default_factory=list)

    def add(self, signal: Signal) -> Signal:
        """Append `signal`, unless this exact object is already present.

        Identity (not value) equality: re-adding the same `Signal` instance -
        e.g. if it gets assembled twice on an export path - is a no-op, while
        two distinct results that happen to compare equal both get added.
        """

        if not any(signal is existing for existing in self.items):
            self.items.append(signal)
        return signal

    def for_modality(self, modality: str) -> list[Signal]:
        """Filter by producing device/technique (e.g. ``"ventilator"``)."""

        return [signal for signal in self.items if signal.modality == modality]

    def for_category(self, category: str) -> list[Signal]:
        """Filter by physical quantity (e.g. ``"airway_pressure"``).

        Independent of `for_modality`: a ventilator contributes several
        categories, and one category can come from several devices. `category`
        is matched after canonicalization, so a known alias (``"paw"``) finds
        signals stored under the canonical name.
        """

        resolved = normalize_category(category) or category
        return [signal for signal in self.items if signal.category == resolved]

    def to_manifest_rows(self) -> list[dict[str, Any]]:
        return [signal.to_manifest_row() for signal in self.items]

    def __iter__(self) -> Iterator[Signal]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class ParameterResultCollection:
    """An ordered collection of `ParameterResult` objects."""

    items: list[ParameterResult] = field(default_factory=list)

    def add(self, parameter: ParameterResult) -> ParameterResult:
        """Append `parameter`, unless this exact object is already present
        (see `SignalCollection.add`)."""

        if not any(parameter is existing for existing in self.items):
            self.items.append(parameter)
        return parameter

    def for_modality(self, modality: str) -> list[ParameterResult]:
        """Filter by producing device/technique (e.g. ``"ventilator"``)."""

        return [p for p in self.items if p.modality == modality]

    def for_category(self, category: str) -> list[ParameterResult]:
        """Filter by source physical quantity (see `SignalCollection.for_category`)."""

        resolved = normalize_category(category) or category
        return [p for p in self.items if p.category == resolved]

    def for_name(self, name: str) -> list[ParameterResult]:
        return [p for p in self.items if p.name == name]

    def to_rows(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.items]

    def __iter__(self) -> Iterator[ParameterResult]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class QualityReport:
    """An ordered collection of `QualityFlag` results."""

    items: list[QualityFlag] = field(default_factory=list)

    def add(self, flag: QualityFlag) -> QualityFlag:
        self.items.append(flag)
        return flag

    def failed(self) -> list[QualityFlag]:
        return [flag for flag in self.items if not flag.passed]

    def for_modality(self, modality: str) -> list[QualityFlag]:
        """Filter by producing device/technique (e.g. ``"ventilator"``)."""

        return [flag for flag in self.items if flag.modality == modality]

    def for_category(self, category: str) -> list[QualityFlag]:
        """Filter by checked physical quantity (see `SignalCollection.for_category`)."""

        resolved = normalize_category(category) or category
        return [flag for flag in self.items if flag.category == resolved]

    def for_signal(self, signal_name: str) -> list[QualityFlag]:
        return [flag for flag in self.items if flag.signal_name == signal_name]

    def to_rows(self) -> list[dict[str, Any]]:
        return [flag.to_dict() for flag in self.items]

    def __iter__(self) -> Iterator[QualityFlag]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class IntervalDataCollection:
    """An ordered collection of results with values per interval or breath.

    Attributes:
        items: ``IntervalData`` results in insertion order.
    """

    items: list[IntervalData] = field(default_factory=list)

    def add(self, interval_data: IntervalData) -> IntervalData:
        """Add and return the result, keeping each object once in insertion order."""

        if not any(interval_data is existing for existing in self.items):
            self.items.append(interval_data)
        return interval_data

    def for_modality(self, modality: str) -> list[IntervalData]:
        """Return results whose modality exactly matches the given device name."""

        return [item for item in self.items if item.modality == modality]

    def for_name(self, name: str) -> list[IntervalData]:
        """Return results with this exact name, in insertion order."""

        return [item for item in self.items if item.name == name]

    def __iter__(self) -> Iterator[IntervalData]:
        """Iterate over results in insertion order."""

        return iter(self.items)

    def __len__(self) -> int:
        """Return the number of interval results."""

        return len(self.items)


@dataclass
class PixelMaskCollection:
    """An ordered collection of pixel masks, such as lung-space masks.

    Attributes:
        items: ``PixelMask`` objects in insertion order.
    """

    items: list[PixelMask] = field(default_factory=list)

    def add(self, mask: PixelMask) -> PixelMask:
        """Add and return the mask, keeping each object once in insertion order."""

        if not any(mask is existing for existing in self.items):
            self.items.append(mask)
        return mask

    def for_name(self, name: str) -> list[PixelMask]:
        """Return masks with this exact name, in insertion order."""

        return [mask for mask in self.items if mask.name == name]

    def combine(
        self, method: Literal["sum", "product"] = "sum", *, name: str
    ) -> PixelMask:
        """Combine all masks into one mask.

        ``"sum"`` adds the masks one after the other (``a + b + ...``): the
        pixels in at least one mask, weights added up to 1. ``"product"``
        multiplies them (``a * b * ...``): the pixels in every mask.

        The result runs the checks of :class:`PixelMask` again with the
        default settings, as in eitprocessing, also for a collection of one
        mask. So a weight of 0 becomes NaN (with a warning), and a mask made
        with ``suppress_value_range_error=True`` and values outside 0 to 1
        raises ValueError.

        Args:
            method (str): ``"sum"`` or ``"product"``.
            name (str): Name of the combined mask.

        Returns:
            PixelMask: The combined mask, with the modality of the first
                mask.

        Raises:
            ValueError: If the collection is empty, ``method`` is unknown,
                the masks differ in shape, or a value of the result is
                outside 0 to 1.

        Warns:
            UserWarning: If the result holds a 0 (it becomes NaN) or is NaN
                everywhere.
        """

        if not self.items:
            raise ValueError("Cannot combine masks: the collection is empty.")
        if method not in ("sum", "product"):
            raise ValueError(
                f"Unknown method {method!r} to combine masks; use 'sum' or 'product'."
            )
        combine_two = PixelMask.__add__ if method == "sum" else PixelMask.__mul__
        combined = reduce(combine_two, self.items)
        if combined is self.items[0]:
            combined = replace(combined, metadata=dict(combined.metadata))
        combined.name = name
        return combined

    def apply(
        self,
        data: Any,
        *,
        name_format: str | None = None,
        **changes: Any,
    ) -> dict[str, Any]:
        """Apply each mask to the same data.

        Args:
            data (numpy.ndarray | Signal | PixelMap): The pixels; the last two
                axes are rows and columns, with the size of the masks.
            name_format (str | None): Name of each result, with
                ``{mask_name}`` for the name of the mask, e.g.
                ``"impedance in {mask_name}"``. When None, each result keeps
                the name of ``data``. Only for a ``Signal`` or a ``PixelMap``.
            **changes: Other fields of each result to change. Only for a
                ``Signal`` or a ``PixelMap``.

        Returns:
            dict[str, Any]: For each mask name, in the order of the
                collection, the masked data, of the same kind as ``data``
                (see :meth:`PixelMask.apply`).

        Raises:
            ValueError: If the collection is empty, two masks have the same
                name, ``changes`` holds ``name``, ``name_format`` has no
                ``{mask_name}``, or ``name_format`` or ``changes`` are given
                with a NumPy array.
            TypeError: If ``data`` is not an array, a ``Signal`` or a
                ``PixelMap``.
        """

        if not self.items:
            raise ValueError("Cannot apply masks: the collection is empty.")
        names = [mask.name for mask in self.items]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(
                f"Cannot apply masks: more than one mask has the name {repeated}; "
                "each mask needs its own name."
            )
        if "name" in changes:
            raise ValueError(
                "Pass the names of the results with 'name_format', not 'name'."
            )
        if name_format is not None and "{mask_name}" not in name_format:
            raise ValueError(
                f"'name_format' must hold '{{mask_name}}', got {name_format!r}."
            )
        if isinstance(data, np.ndarray):
            if name_format is not None or changes:
                raise ValueError(
                    "'name_format' and other changes work only on a Signal or "
                    "a PixelMap, not on a NumPy array."
                )
            return {mask.name: mask.apply(data) for mask in self.items}
        if not isinstance(data, (Signal, PixelMap)):
            raise TypeError(
                "PixelMaskCollection.apply needs a NumPy array, a Signal or a "
                f"PixelMap, got {type(data).__name__}."
            )

        results: dict[str, Any] = {}
        for mask in self.items:
            fields = {
                "values": mask.apply(np.asarray(data.values)),
                "metadata": dict(data.metadata),
                **changes,
            }
            if name_format is not None:
                fields["name"] = name_format.format(mask_name=mask.name)
            results[mask.name] = replace(data, **fields)
        return results

    def __iter__(self) -> Iterator[PixelMask]:
        """Iterate over masks in insertion order."""

        return iter(self.items)

    def __len__(self) -> int:
        """Return the number of masks."""

        return len(self.items)
