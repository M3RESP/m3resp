"""``PixelMap`` and ``PixelMask``: one number for each EIT pixel.

An EIT image is a grid of pixels (rows by columns). Some results give one
number for every pixel: the tidal
impedance variation of each pixel in one breath, for example. A
:class:`PixelMap` holds such a grid. Its ``kind`` says what the numbers are
(TIV, amplitude, a difference of two maps, region numbers, ...).

A :class:`PixelMask` is a grid that says which pixels belong to a region,
such as the functional lung space. A pixel that is not part of the region is
NaN; a pixel that is part of it is 1, or a weight from 0 to 1.

Maps can be added, subtracted, multiplied and divided pixel by pixel, scaled
(``normalize``), averaged over many maps (``from_aggregate``) and turned
into a mask with a threshold (``create_mask_from_threshold``). Masks combine
with ``*`` (pixels in both masks), ``+`` (pixels in one mask or both) and
``-`` (pixels of the first mask that are not in the second).

Both follow eitprocessing's ``PixelMap`` and ``PixelMask``, so values can be
handed back to eitprocessing without being changed.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/roi/__init__.py::PixelMask.__init__;
                PixelMask.apply; PixelMask.is_weighted; PixelMask.__mul__;
                PixelMask.__add__; PixelMask.__sub__
                eitprocessing/datahandling/pixelmap.py::PixelMap.__init__;
                PixelMap.normalize; PixelMap._check_normalization_reference;
                PixelMap.create_mask_from_threshold; PixelMap.convert_to;
                PixelMap._validate_other; PixelMap.__add__; PixelMap.__sub__;
                PixelMap.__rsub__; PixelMap.__mul__; PixelMap.__truediv__;
                PixelMap.__rtruediv__; PixelMap.from_aggregate;
                PixelMap.to_boolean_array; PixelMap.to_integer_array;
                PixelMap.to_non_nan_array; TIVMap; AmplitudeMap; ODCLMap;
                DifferenceMap; PerfusionMap; PendelluftMap;
                SignedPendelluftMap; IntegerMap
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The checks of `PixelMask.__init__` (2D grid, true/false grids, values
      from 0 to 1, zeros turned into NaN, all-NaN warning) and their four
      switches (`keep_zeros`, `suppress_value_range_error`,
      `suppress_zero_conversion_warning`, `suppress_all_nan_warning`) run in
      `PixelMask.__post_init__`.
    - The mask has the fields `name`, `values`, `modality`, `method` and
      `metadata`, the same as `PixelMap`; plot settings stay in
      eitprocessing.
    - `PixelMask.apply` takes a NumPy array, a `Signal` or a `PixelMap` and
      gives the same kind back; a `Signal` or `PixelMap` keeps all its other
      fields.
    - The 8 subclasses of `PixelMap` are one field, `kind`: "tiv",
      "amplitude", "odcl", "difference", "perfusion", "pendelluft",
      "signed_pendelluft" and "integer" (None for a plain `PixelMap`).
      `convert_to` takes a kind.
    - A map of kind "integer" holds whole numbers, stored as floats; NaN or
      infinity raises TypeError, as in eitprocessing `IntegerMap`.
    - A `PixelMap` that is NaN everywhere gives no warning.
    - A map keeps its `suppress_negative_warning` choice in its copies
      (`with_values`, `dataclasses.replace`, `normalize`, `convert_to`,
      `+ - * /` with a number or a grid, `PixelMask.apply`); eitprocessing
      checks each copy with the default setting.
    - Warnings point to the line of the user's code that made the map or
      the mask.
    - `normalize(mode="reference")`, `create_mask_from_threshold` and
      `+ - * /` take NumPy numbers and refuse True and False.
    - A map combined with a number or a grid keeps its name; a map combined
      with another map gets the name "a + b", "a - b", "a * b" or "a / b",
      and keeps the category and the unit when both maps have the same
      (for + and - only).
    - Masks combined with `*`, `+` and `-` get the name "a and b", "a or b"
      and "a without b". Masks of different shapes raise ValueError.
    - NumPy numbers (for example `numpy.int64`) work like Python numbers in
      `+ - * /`.
    - Dividing by the number 0 gives the "Dividing by 0" warning, the same
      as dividing by a grid with a 0 (eitprocessing warns for a grid only).
    - The division fills the result with NaN first, so NumPy gives no
      "'where' used without 'out'" warning (eitprocessing gives it).
    - `create_mask_from_threshold` takes a `name` for the mask.
    - Error and warning messages are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import dataclasses
import os
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import InitVar, dataclass, field, replace
from types import CodeType, FrameType
from typing import Any, Literal, overload

import numpy as np

from m3resp.data.categories import normalize_category
from m3resp.data.event_data import IntervalData
from m3resp.data.events import BreathEvent
from m3resp.data.pixel_breaths import PixelBreaths
from m3resp.data.signals import Signal
from m3resp.data.units import normalize_unit

#: The kinds of ``PixelMap``. A plain map has kind None.
PIXEL_MAP_KINDS = (
    "tiv",
    "amplitude",
    "odcl",
    "difference",
    "perfusion",
    "pendelluft",
    "signed_pendelluft",
    "integer",
)

#: Kinds whose values are normally 0 or more; a negative value gives a warning.
_NON_NEGATIVE_KINDS = frozenset({"amplitude", "perfusion", "pendelluft"})

NormalizeMode = Literal["zero-based", "symmetric", "maximum", "reference"]


@dataclass(eq=False)
class PixelMap:
    """One number for each pixel of an EIT image.

    The ``kind`` says what the numbers are. It follows the subclasses of
    eitprocessing's ``PixelMap``:

    - ``"tiv"``: tidal impedance variation.
    - ``"amplitude"``: amplitude; values are normally 0 or more.
    - ``"odcl"``: overdistention (0 to 1) and collapse (-1 to 0).
    - ``"difference"``: the difference of two maps. Subtracting one map from
      another gives this kind.
    - ``"perfusion"``: perfusion; values are normally 0 or more.
    - ``"pendelluft"``: pendelluft size; values are normally 0 or more.
    - ``"signed_pendelluft"``: pendelluft size, with a sign for early or late
      inflation.
    - ``"integer"``: whole numbers, such as region numbers, in every pixel.
    - None: a plain map.

    A map of kind ``"amplitude"``, ``"perfusion"`` or ``"pendelluft"`` with
    a negative value gives a warning, unless
    ``suppress_negative_warning=True``. Copies of the map (``with_values``,
    ``dataclasses.replace``, ``normalize``, ``+ - * /`` with a number or a
    grid, ``PixelMask.apply``) keep this choice.

    A map of kind ``"integer"`` keeps its kind in ``+ - * /`` with a number
    or a grid, so a result that is not whole (``regions / 2``) or that holds
    NaN (``regions / 0``) raises TypeError. Use ``regions.convert_to(None)``
    first for a plain map.

    Attributes:
        name: What the numbers are, e.g. ``'pixel_tiv'``.
        values: The numbers as a 2D grid of shape (row, column). They are
            stored as floats; NaN means the pixel has no value.
        modality: The device the values were measured with.
        category: The physical quantity the values are derived from (see
            :mod:`m3resp.data.categories`).
        unit: Unit of the values, normalized to a recognized spelling.
        method: Name of the method that produced the values.
        metadata: Optional extra information.
        kind: One of :data:`PIXEL_MAP_KINDS`, or None for a plain map.
        suppress_negative_warning: True skips the warning for negative
            values of the kinds ``"amplitude"``, ``"perfusion"`` and
            ``"pendelluft"``.

    Raises:
        ValueError: If ``values`` is not a 2D grid, or ``kind`` is unknown.
        TypeError: If ``values`` cannot be stored as floats without changing
            them (text, for example), or a map of kind ``"integer"`` holds a
            number that is not whole, NaN or infinity.
    """

    name: str
    values: np.ndarray
    modality: str = "eit"
    category: str | None = None
    unit: str | None = None
    method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: str | None = None
    suppress_negative_warning: bool = field(default=False, kw_only=True, repr=False)

    #: A NumPy number or array on the left of ``+ - * /`` gives the
    #: calculation to the map, so the result is a ``PixelMap``.
    __array_priority__ = 1000

    def __post_init__(self) -> None:
        """Check the 2D numeric grid and the kind, then normalize unit and category names."""

        self.values = _as_float_grid(self.values, owner="PixelMap")
        self.unit = normalize_unit(self.unit)
        self.category = normalize_category(self.category) or self.category
        if self.kind is not None and self.kind not in PIXEL_MAP_KINDS:
            raise ValueError(
                f"PixelMap.kind must be None or one of {PIXEL_MAP_KINDS}, "
                f"got {self.kind!r}."
            )
        if self.kind == "integer" and (
            not np.all(np.isfinite(self.values))
            or np.any(self.values != np.trunc(self.values))
        ):
            raise TypeError(
                f"PixelMap '{self.name}' of kind 'integer' must hold a whole "
                "number in every pixel (no NaN or infinity). Use "
                "convert_to(None) first to calculate with a plain map."
            )
        if (
            self.kind in _NON_NEGATIVE_KINDS
            and not self.suppress_negative_warning
            and np.any(self.values < 0)
        ):
            _warn(
                f"PixelMap '{self.name}' of kind '{self.kind}' holds negative "
                "values; values of this kind are normally 0 or more. Pass "
                "suppress_negative_warning=True to skip this warning."
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the number of rows and columns, in that order."""

        return self.values.shape

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Return the grid as a NumPy array with the requested dtype.

        Conversion follows ``numpy.asarray``; the ``copy`` argument is accepted
        but does not change how the array is returned.
        """

        return np.asarray(self.values, dtype=dtype)

    def __eq__(self, other: object) -> bool:
        """Compare fields and grids, treating matching NaN pixels as equal."""

        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, PixelMap)
        return (
            self.name == other.name
            and np.array_equal(self.values, other.values, equal_nan=True)
            and self.modality == other.modality
            and self.category == other.category
            and self.unit == other.unit
            and self.method == other.method
            and self.metadata == other.metadata
            and self.kind == other.kind
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the grid and its descriptive fields as a dictionary.

        Returns:
            dict[str, Any]: The grid as a list of rows, together with name,
                modality, category, unit, method, metadata and kind. NaN
                pixels remain NaN. Metadata is copied at the top level.
        """

        return {
            "name": self.name,
            "values": self.values.tolist(),
            "modality": self.modality,
            "category": self.category,
            "unit": self.unit,
            "method": self.method,
            "metadata": dict(self.metadata),
            "kind": self.kind,
        }

    def with_values(self, values: Any, **changes: Any) -> PixelMap:
        """Return a copy of the map with new values and the same other fields.

        Args:
            values (array-like): The new 2D grid.
            **changes: Other fields to change, e.g. ``name`` or ``kind``.

        Returns:
            PixelMap: The copy, with a copy of the metadata and the same
                ``suppress_negative_warning``. The checks of the kind run
                again.
        """

        fields: dict[str, Any] = {"values": values, "metadata": dict(self.metadata)}
        fields.update(changes)
        return replace(self, **fields)

    @classmethod
    def from_aggregate(
        cls,
        maps: Sequence[Any] | np.ndarray,
        aggregator: Callable[..., np.ndarray],
        **fields: Any,
    ) -> PixelMap:
        """Combine several maps into one, pixel by pixel.

        The maps are stacked, and ``aggregator`` runs over the stack for each
        pixel, for example ``numpy.nanmean`` for the mean over the breaths.
        A pixel that is NaN in every map stays NaN, with no warning.

        Args:
            maps (Sequence | numpy.ndarray): The maps: ``PixelMap`` objects or
                2D grids, all of the same shape, or one (map, row, column)
                array.
            aggregator (Callable): A function that takes the stacked values
                and ``axis=0``, e.g. ``numpy.nanmean``, ``numpy.nanmedian``
                or ``numpy.nanmax``.
            **fields: Fields of the new map; ``name`` is required.

        Returns:
            PixelMap: The combined map.

        Raises:
            ValueError: If ``maps`` is empty, a grid is not 2D, or the maps
                differ in shape.
        """

        def grid_of(map_: Any) -> np.ndarray:
            if isinstance(map_, PixelMap):
                return map_.values
            grid = np.asarray(map_, dtype=float)
            if grid.ndim != 2:
                raise ValueError(
                    "PixelMap.from_aggregate needs 2D grids (row, column), got "
                    f"a grid with {grid.ndim} dimension(s)."
                )
            return grid

        if len(maps) == 0:
            raise ValueError("PixelMap.from_aggregate needs at least one map.")
        stacked = np.stack([grid_of(map_) for map_ in maps])
        included = ~np.all(np.isnan(stacked), axis=0)
        aggregated = np.full(stacked.shape[1:], np.nan)
        aggregated[included] = aggregator(stacked[:, included], axis=0)
        return cls(values=aggregated, **fields)

    def normalize(
        self,
        *,
        mode: NormalizeMode = "zero-based",
        reference: float | None = None,
        **changes: Any,
    ) -> PixelMap:
        """Return a copy of the map with the values divided by a reference.

        The four modes:

        - ``"zero-based"`` (default): subtract the smallest value, then
          divide by the new largest value. The values go from 0 to 1.
        - ``"symmetric"``: divide by the largest absolute value. The values
          go from -1 to 1.
        - ``"maximum"``: divide by the largest value. Positive values go from
          0 to 1 and keep their ratios; negative values can go below -1.
        - ``"reference"``: divide by ``reference``. The values can be
          outside -1 to 1.

        NaN pixels stay NaN and do not count for the smallest or largest
        value.

        Args:
            mode (str): One of the four modes above.
            reference (float | None): The number to divide by; only with
                ``mode="reference"``.
            **changes: Other fields of the new map to change, e.g. ``name``.

        Returns:
            PixelMap: The copy, with the same other fields.

        Raises:
            NotImplementedError: If the map has kind ``"integer"``.
            ValueError: If ``reference`` is given with another mode, is
                missing with ``mode="reference"`` or is NaN; if the map is
                NaN everywhere; or if ``mode`` is unknown.
            TypeError: If ``reference`` is not a number.
            ZeroDivisionError: If the number to divide by is 0.

        Warns:
            UserWarning: If the number to divide by is negative; this turns
                the sign of the values around.
        """

        if self.kind == "integer":
            raise NotImplementedError(
                "A PixelMap of kind 'integer' cannot be normalized."
            )
        if reference is not None and mode != "reference":
            raise ValueError("'reference' can only be used with mode='reference'.")
        if mode == "reference":
            if reference is None:
                raise ValueError("'reference' is required with mode='reference'.")
            if not _is_number(reference):
                raise TypeError(
                    f"'reference' must be a number, got {type(reference).__name__}."
                )
            _check_normalization_reference(reference)

        if np.all(np.isnan(self.values)):
            raise ValueError(
                f"PixelMap '{self.name}' is NaN everywhere, so it cannot be normalized."
            )

        values = self.values
        match mode:
            case "symmetric":
                divisor = float(np.nanmax(np.abs(values)))
            case "zero-based":
                values = values - np.nanmin(values)
                divisor = float(np.nanmax(values))
            case "maximum":
                divisor = float(np.nanmax(values))
            case "reference":
                divisor = float(reference)  # type: ignore[arg-type]
            case _:
                raise ValueError(
                    f"Unknown normalization mode {mode!r}; use 'zero-based', "
                    "'symmetric', 'maximum' or 'reference'."
                )
        _check_normalization_reference(divisor)
        return self.with_values(values / divisor, **changes)

    def create_mask_from_threshold(
        self,
        threshold: float,
        *,
        comparator: Callable[[np.ndarray, Any], np.ndarray] = np.greater_equal,
        use_magnitude: bool = False,
        fraction_of_max: bool = False,
        name: str | None = None,
        captures: dict[str, Any] | None = None,
    ) -> PixelMask:
        """Make a mask of the pixels whose value passes a threshold.

        By default, a pixel is in the mask when its value is at least the
        threshold (``numpy.greater_equal``). Another comparison, such as
        ``numpy.less``, can be given with ``comparator``.

        Args:
            threshold (float): The threshold, or a fraction of the largest
                value with ``fraction_of_max=True``.
            comparator (Callable): A function that takes the values and the
                threshold and gives a true/false grid.
            use_magnitude (bool): If True, compare the absolute values.
            fraction_of_max (bool): If True, the threshold is ``threshold``
                times the largest value (of the absolute values with
                ``use_magnitude=True``). NaN pixels do not count.
            name (str | None): Name of the mask; ``"<map name>_mask"`` when
                None.
            captures (dict | None): When a dictionary, the threshold that was
                used is stored in it under ``"actual threshold"``.

        Returns:
            PixelMask: 1 where the comparison is true, NaN elsewhere, with
                the modality of the map.

        Raises:
            TypeError: If ``threshold`` is not a number or ``comparator`` is
                not a function.
            ValueError: If the map is NaN everywhere.
        """

        if not _is_number(threshold):
            raise TypeError(
                f"'threshold' must be a number, got {type(threshold).__name__}."
            )
        if not callable(comparator):
            raise TypeError("'comparator' must be a function.")

        compare_values = np.abs(self.values) if use_magnitude else self.values
        if np.all(np.isnan(compare_values)):
            raise ValueError(
                f"PixelMap '{self.name}' is NaN everywhere, so no pixel can "
                "pass a threshold."
            )
        actual_threshold = (
            threshold * np.nanmax(compare_values) if fraction_of_max else threshold
        )
        if captures is not None:
            captures["actual threshold"] = actual_threshold
        return PixelMask(
            name=name if name is not None else f"{self.name}_mask",
            values=comparator(compare_values, actual_threshold),
            modality=self.modality,
        )

    def convert_to(self, kind: str | None, **changes: Any) -> PixelMap:
        """Return a copy of the map with another kind.

        Args:
            kind (str | None): One of :data:`PIXEL_MAP_KINDS`, or None for a
                plain map.
            **changes: Other fields to change, e.g. ``name``.

        Returns:
            PixelMap: The copy, with the same values and other fields. The
                checks of the new kind run.

        Raises:
            ValueError: If ``kind`` is unknown.
            TypeError: If ``kind`` is ``"integer"`` and a value is not whole,
                NaN or infinity.
        """

        return self.with_values(self.values, kind=kind, **changes)

    def to_boolean_array(self, *, zero: bool = False) -> np.ndarray:
        """Return a true/false grid: true for pixels with a value other than 0.

        Args:
            zero (bool): If True, a pixel with value 0 is true too.

        Returns:
            numpy.ndarray: The 2D grid. NaN pixels are false.
        """

        values = self.values.copy()
        if zero:
            values[values == 0] = 1
        np.nan_to_num(values, nan=0, copy=False)
        return values.astype(bool)

    def to_integer_array(self) -> np.ndarray:
        """Return the values as a grid of integers, with 0 for NaN pixels.

        Returns:
            numpy.ndarray: The 2D grid; values that are not whole are cut
                towards 0.
        """

        return self.to_non_nan_array(nan=0, dtype=int)

    def to_non_nan_array(
        self, *, nan: float = 0.0, dtype: type | np.dtype = float
    ) -> np.ndarray:
        """Return the values as a grid with a number in place of NaN.

        Args:
            nan (float): The number for NaN pixels.
            dtype (type | numpy.dtype): Type of the numbers of the grid.

        Returns:
            numpy.ndarray: The 2D grid.
        """

        return np.nan_to_num(self.values, nan=nan).astype(dtype)

    def _other_values(self, other: Any) -> Any:
        """Return the values of ``other`` for a pixel-by-pixel calculation."""

        if isinstance(other, (bool, np.bool_)):
            raise TypeError(
                f"PixelMap '{self.name}' can be combined with a number, a grid or "
                "a PixelMap, not with True or False."
            )
        if _is_number(other):
            return other
        other_values = np.asarray(
            other.values if isinstance(other, PixelMap) else other
        )
        if other_values.shape != self.values.shape:
            raise ValueError(
                f"PixelMap '{self.name}' has shape {self.values.shape} and the "
                f"other value has shape {other_values.shape}; they must have "
                "the same shape."
            )
        return other_values

    def _result(self, other: Any, values: np.ndarray, symbol: str) -> PixelMap:
        """Return the map of ``self <symbol> other``."""

        if not isinstance(other, PixelMap):
            return self.with_values(values)
        additive = symbol in ("+", "-")
        return PixelMap(
            name=f"{self.name} {symbol} {other.name}",
            values=values,
            modality=self.modality,
            category=self.category
            if additive and self.category == other.category
            else None,
            unit=self.unit if additive and self.unit == other.unit else None,
            kind="difference" if symbol == "-" else None,
        )

    def __add__(self, other: Any) -> PixelMap:
        """Add pixel by pixel."""

        return self._result(other, self.values + self._other_values(other), "+")

    __radd__ = __add__

    def __sub__(self, other: Any) -> PixelMap:
        """Subtract pixel by pixel; a map minus a map has kind ``"difference"``."""

        return self._result(other, self.values - self._other_values(other), "-")

    def __rsub__(self, other: Any) -> PixelMap:
        """Subtract this map from a number or a grid; the kind stays."""

        return self.with_values(self._other_values(other) - self.values)

    def __mul__(self, other: Any) -> PixelMap:
        """Multiply pixel by pixel."""

        return self._result(other, self.values * self._other_values(other), "*")

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> PixelMap:
        """Divide pixel by pixel; a pixel divided by 0 or by NaN is NaN.

        Warns:
            UserWarning: If the divisor holds a 0.
        """

        other_values = self._other_values(other)
        if np.any(np.asarray(other_values) == 0):
            _warn_division_by_zero()
        return self._result(other, _divide(self.values, other_values), "/")

    def __rtruediv__(self, other: Any) -> PixelMap:
        """Divide a number or a grid by this map; the kind stays.

        Warns:
            UserWarning: If this map holds a 0.
        """

        other_values = self._other_values(other)
        if np.any(self.values == 0):
            _warn_division_by_zero()
        return self.with_values(_divide(other_values, self.values))


@dataclass(eq=False)
class PixelMask:
    """Which pixels of an EIT image belong to a region.

    Each pixel holds NaN when it is outside the region, 1 when it is included,
    or a weight from 0 to 1 when it counts partly. Applying the mask
    multiplies each pixel by this number, so a NaN pixel drops out and a
    pixel with weight 0 adds nothing to a sum.

    The checks follow eitprocessing's ``PixelMask``:

    - A 0 becomes NaN, with a warning. Pass ``keep_zeros=True`` to keep it as
      weight 0, or ``suppress_zero_conversion_warning=True`` to skip the
      warning.
    - A grid of true/false values is accepted: true becomes 1 and false
      becomes 0, which then becomes NaN without a warning (or stays 0 with
      ``keep_zeros=True``).
    - A number below 0 or above 1 raises ``ValueError``, unless
      ``suppress_value_range_error=True``.
    - A grid that is NaN everywhere gives a warning, unless
      ``suppress_all_nan_warning=True``.

    Masks of the same shape combine pixel by pixel:

    - ``a * b`` (name ``"a and b"``): the pixels in both masks; weights are
      multiplied.
    - ``a + b`` (name ``"a or b"``): the pixels in one mask or both; weights
      are added, up to 1.
    - ``a - b`` (name ``"a without b"``): the pixels of ``a`` that are not
      in ``b``; a weight of ``b`` is subtracted, down to 0, and a pixel that
      ends at 0 is NaN.

    The result runs the checks above again, so a 0 that ``a * b`` gives
    becomes NaN with a warning, and a result that is NaN everywhere gives a
    warning.

    Attributes:
        name: What the mask is, e.g. ``'tiv_lungspace_mask'``.
        values: The mask as a 2D grid of shape (row, column), stored as
            floats.
        modality: The device the mask belongs to.
        method: Name of the method that made the mask.
        metadata: Optional extra information.

    Raises:
        ValueError: If ``values`` is not a 2D grid, or holds a number below 0
            or above 1 while ``suppress_value_range_error`` is False.
        TypeError: If ``values`` cannot be stored as floats without changing
            them.
    """

    name: str
    values: np.ndarray
    modality: str = "eit"
    method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    keep_zeros: InitVar[bool] = field(default=False, kw_only=True)
    suppress_value_range_error: InitVar[bool] = field(default=False, kw_only=True)
    suppress_zero_conversion_warning: InitVar[bool] = field(default=False, kw_only=True)
    suppress_all_nan_warning: InitVar[bool] = field(default=False, kw_only=True)

    def __post_init__(
        self,
        keep_zeros: bool,
        suppress_value_range_error: bool,
        suppress_zero_conversion_warning: bool,
        suppress_all_nan_warning: bool,
    ) -> None:
        """Check the grid, then turn zeros into NaN unless they are kept."""

        raw = np.asarray(self.values)
        is_boolean_mask = raw.dtype == bool
        if is_boolean_mask:
            grid = raw.astype(float)
            if grid.ndim != 2:
                raise ValueError(
                    f"PixelMask.values must be a 2D grid (row, column), got "
                    f"{grid.ndim} dimension(s)."
                )
        else:
            grid = _as_float_grid(raw, owner="PixelMask")

        all_nan = bool(np.all(np.isnan(grid)))
        if all_nan and not suppress_all_nan_warning:
            _warn(
                f"PixelMask '{self.name}' is NaN everywhere, so applying it "
                "gives NaN for every pixel."
            )
        if (
            not all_nan
            and not suppress_value_range_error
            and (np.nanmin(grid) < 0 or np.nanmax(grid) > 1)
        ):
            raise ValueError(
                "PixelMask.values must be NaN, or a number from 0 to 1 "
                f"(found values from {np.nanmin(grid)} to {np.nanmax(grid)}). "
                "Pass suppress_value_range_error=True to allow other values."
            )
        if not keep_zeros and np.any(grid == 0):
            if not is_boolean_mask and not suppress_zero_conversion_warning:
                _warn(
                    f"PixelMask '{self.name}' holds 0 values, which are turned "
                    "into NaN (outside the region). Pass keep_zeros=True to "
                    "keep them as weight 0, or "
                    "suppress_zero_conversion_warning=True to skip this warning."
                )
            grid = np.where(grid == 0, np.nan, grid)
        self.values = grid

    @property
    def shape(self) -> tuple[int, int]:
        """Return the number of rows and columns, in that order."""

        return self.values.shape

    @property
    def included_pixel_count(self) -> int:
        """Return the count of pixels that are not NaN, weight 0 included."""

        return int(np.count_nonzero(~np.isnan(self.values)))

    @property
    def is_weighted(self) -> bool:
        """Return True when a pixel holds a number other than NaN or 1."""

        return not bool(np.all(np.isnan(self.values) | (self.values == 1.0)))

    def _check_same_shape(self, other: PixelMask) -> None:
        if other.values.shape != self.values.shape:
            raise ValueError(
                f"The masks '{self.name}' and '{other.name}' have shapes "
                f"{self.values.shape} and {other.values.shape}; they must "
                "have the same shape."
            )

    def __mul__(self, other: object) -> PixelMask:
        """Return the mask of the pixels in both masks, named ``"a and b"``."""

        if not isinstance(other, PixelMask):
            return NotImplemented
        self._check_same_shape(other)
        return PixelMask(
            name=f"{self.name} and {other.name}",
            values=self.values * other.values,
            modality=self.modality,
        )

    def __add__(self, other: object) -> PixelMask:
        """Return the mask of the pixels in one mask or both, named ``"a or b"``."""

        if not isinstance(other, PixelMask):
            return NotImplemented
        self._check_same_shape(other)
        values = np.clip(np.nansum([self.values, other.values], axis=0), None, 1)
        values[values == 0] = np.nan
        return PixelMask(
            name=f"{self.name} or {other.name}",
            values=values,
            modality=self.modality,
        )

    def __sub__(self, other: object) -> PixelMask:
        """Return the pixels of this mask that are not in ``other``, named ``"a without b"``."""

        if not isinstance(other, PixelMask):
            return NotImplemented
        self._check_same_shape(other)
        values = (
            np.nan_to_num(self.values, nan=0) - np.nan_to_num(other.values, nan=0)
        ).clip(min=0)
        return PixelMask(
            name=f"{self.name} without {other.name}",
            values=values,
            modality=self.modality,
            suppress_zero_conversion_warning=True,
        )

    @overload
    def apply(self, data: np.ndarray) -> np.ndarray: ...

    @overload
    def apply(self, data: Signal) -> Signal: ...

    @overload
    def apply(self, data: PixelMap) -> PixelMap: ...

    def apply(
        self, data: np.ndarray | Signal | PixelMap
    ) -> np.ndarray | Signal | PixelMap:
        """Multiply each pixel of ``data`` by its mask value.

        The mask applies to the last two axes of the data, so it works on one
        image and on a series of images alike. A NaN in the mask makes the
        pixel NaN; a weight scales the pixel.

        Args:
            data (numpy.ndarray | Signal | PixelMap): The pixels. The last two
                axes are rows and columns, with the same size as the mask.

        Returns:
            numpy.ndarray | Signal | PixelMap: The same kind as ``data``, with
                the masked pixels. A ``Signal`` or ``PixelMap`` is a copy with
                the same other fields and a copy of the metadata.

        Raises:
            ValueError: If the last two axes of the data differ in size from
                the mask.
            TypeError: If ``data`` is not an array, a ``Signal`` or a
                ``PixelMap``.
        """

        if isinstance(data, Signal):
            return replace(
                data,
                values=self._multiply(np.asarray(data.values)),
                metadata=dict(data.metadata),
            )
        if isinstance(data, PixelMap):
            return data.with_values(self._multiply(data.values))
        if isinstance(data, np.ndarray):
            return self._multiply(data)
        raise TypeError(
            "PixelMask.apply needs a NumPy array, a Signal or a PixelMap, "
            f"got {type(data).__name__}."
        )

    def _multiply(self, data: np.ndarray) -> np.ndarray:
        """Multiply the last two axes of ``data`` by the mask grid."""

        if data.shape[-2:] != self.values.shape:
            raise ValueError(
                f"The data has shape {data.shape} and the mask "
                f"'{self.name}' has shape {self.values.shape}; the last two "
                "axes of the data must have the size of the mask."
            )
        return data * self.values[(np.newaxis,) * (data.ndim - 2) + (...,)]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Return the grid as a NumPy array with the requested dtype.

        Conversion follows ``numpy.asarray``; the ``copy`` argument is accepted
        but does not change how the array is returned.
        """

        return np.asarray(self.values, dtype=dtype)

    def __eq__(self, other: object) -> bool:
        """Compare fields and grids, treating matching NaN pixels as equal."""

        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, PixelMask)
        return (
            self.name == other.name
            and np.array_equal(self.values, other.values, equal_nan=True)
            and self.modality == other.modality
            and self.method == other.method
            and self.metadata == other.metadata
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the mask and its descriptive fields as a dictionary.

        Returns:
            dict[str, Any]: The grid as a list of rows, together with name,
                modality, method and metadata. Excluded pixels remain NaN.
                Metadata is copied at the top level.
        """

        return {
            "name": self.name,
            "values": self.values.tolist(),
            "modality": self.modality,
            "method": self.method,
            "metadata": dict(self.metadata),
        }


def pixel_breath_events(
    pixel_breaths: IntervalData, row: int, column: int
) -> list[BreathEvent]:
    """Return the breaths of one pixel as ``BreathEvent`` objects.

    ``eit.pixel_breaths`` stores, for each global breath, a grid of each
    pixel's breath start, end of inspiration and end time. This function
    picks one pixel from that grid and turns its breaths into ``BreathEvent``
    objects, so analyses written for global breaths (durations, timing
    checks, linking) can run on one pixel too. It gives the same breaths as
    :meth:`PixelBreaths.pixel <m3resp.data.pixel_breaths.PixelBreaths.pixel>`.

    Args:
        pixel_breaths (IntervalData): The ``IntervalData`` written by
            ``eit.pixel_breaths`` (``pixel_breath_timing_result``): one
            (row, column, 3) grid of start, end of inspiration and end times
            in seconds per global breath.
        row (int): Row of the pixel, counted from 0.
        column (int): Column of the pixel, counted from 0.

    Returns:
        list[BreathEvent]: One breath per global breath in which the pixel
            has a breath, in time order. ``start_time`` and ``end_time`` are
            the pixel's own breath start and end, ``extremum_time`` is its middle
            time (end of inspiration), all in seconds. A global breath where
            the pixel has no breath (NaN) gives no entry. ``metadata`` holds
            ``row``, ``column`` and ``global_breath_id``, the ``id`` of the
            global breath it belongs to. No global breaths give an empty
            list.

    Raises:
        TypeError: If an interval of ``pixel_breaths`` is not a
            ``BreathEvent``.
        ValueError: If the grids are not (row, column, 3), a pixel breath
            has NaN for only some of its three times or its times are not
            in the order start, end of inspiration, end, or ``row`` or ``column`` is
            outside the grid.
    """

    if not pixel_breaths.intervals:
        return []
    return PixelBreaths.from_interval_data(pixel_breaths).pixel(row, column)


def _as_float_grid(values: Any, *, owner: str) -> np.ndarray:
    """Return a 2D float grid whose values equal the input, including NaNs.

    Conversion that changes a value raises ``TypeError``. A grid with other
    than two dimensions raises ``ValueError``.
    """

    raw = np.asarray(values)
    try:
        grid = np.asarray(raw, dtype=float)
        unchanged = np.array_equal(grid, raw, equal_nan=True)
    except (TypeError, ValueError):
        unchanged = False
    if not unchanged:
        raise TypeError(
            f"{owner}.values must be numbers that can be stored as floats "
            f"without change, got an array of {raw.dtype}."
        )
    if grid.ndim != 2:
        raise ValueError(
            f"{owner}.values must be a 2D grid (row, column), got "
            f"{grid.ndim} dimension(s)."
        )
    return grid


def _check_normalization_reference(divisor: float) -> None:
    """Check the number that ``PixelMap.normalize`` divides by."""

    if divisor == 0:
        raise ZeroDivisionError(
            "PixelMap.normalize cannot divide by 0: the reference is 0, or the "
            "map has no value other than 0."
        )
    if np.isnan(divisor):
        raise ValueError(
            "PixelMap.normalize cannot divide by NaN: the reference is NaN, or "
            "the map has no value other than NaN."
        )
    if divisor < 0:
        _warn(
            f"PixelMap.normalize divides by a negative number ({divisor}), so "
            "the sign of the values turns around."
        )


def _warn_division_by_zero() -> None:
    _warn("Dividing by 0 gives NaN for those pixels.")


def _is_number(value: Any) -> bool:
    """Return True for a Python or NumPy number; True and False do not count."""

    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


#: Folder of the m3resp package; warnings point to the first line outside it.
_PACKAGE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATACLASSES_FILE = os.path.abspath(dataclasses.__file__)


def _warn(message: str) -> None:
    """Give a UserWarning that points to the line that called m3resp.

    The warning shows the first line outside the m3resp package and the
    Python ``dataclasses`` code. So the warning of a copy such as ``m * 2``
    points to the line with ``m * 2``, and a warning of a map that an m3resp
    function makes points to the line that called that function.

    Args:
        message (str): The text of the warning.
    """

    frame: FrameType | None = sys._getframe(1)
    stacklevel = 2
    while frame is not None and _is_package_code(frame.f_code):
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, UserWarning, stacklevel=stacklevel)


def _is_package_code(code: CodeType) -> bool:
    """Return True for code of the m3resp package or of ``dataclasses``."""

    filename = code.co_filename
    if filename == "<string>":
        # The __init__ that dataclasses writes for each class.
        return code.co_name == "__init__"
    filename = os.path.abspath(filename)
    return filename == _DATACLASSES_FILE or filename.startswith(
        _PACKAGE_FOLDER + os.sep
    )


def _divide(numerator: Any, denominator: Any) -> np.ndarray:
    """Divide pixel by pixel; NaN where either side is NaN or the divisor is 0."""

    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    invalid = np.isnan(numerator) | np.isnan(denominator) | (denominator == 0)
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=~invalid)
    return result
