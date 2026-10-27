"""``PixelMap`` and ``PixelMask``: one number for each EIT pixel.

An EIT image is a grid of pixels (rows by columns). Some results give one
number for every pixel: the tidal
impedance variation of each pixel in one breath, for example. A
:class:`PixelMap` holds such a grid.

A :class:`PixelMask` is a grid that says which pixels belong to a region,
such as the functional lung space. A pixel that is not part of the region is
NaN; a pixel that is part of it is 1, or a weight from 0 to 1.

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
                PixelMask.apply
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The checks of `PixelMask.__init__` (2D grid, true/false grids, values
      from 0 to 1, zeros turned into NaN, all-NaN warning) and their four
      switches (`keep_zeros`, `suppress_value_range_error`,
      `suppress_zero_conversion_warning`, `suppress_all_nan_warning`) run in
      `PixelMask.__post_init__`.
    - The mask has the fields `name`, `values`, `modality`, `method` and
      `metadata`, the same as `PixelMap`; plot settings and adding,
      subtracting or multiplying masks stay in eitprocessing.
    - `PixelMask.apply` takes a NumPy array, a `Signal` or a `PixelMap` and
      gives the same kind back; a `Signal` or `PixelMap` keeps all its other
      fields.
    - Error and warning messages are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
from dataclasses import InitVar, dataclass, field, replace
from typing import Any, overload

import numpy as np

from m3resp.data.categories import normalize_category
from m3resp.data.event_data import IntervalData
from m3resp.data.events import BreathEvent
from m3resp.data.signals import Signal
from m3resp.data.units import normalize_unit


@dataclass(eq=False)
class PixelMap:
    """One number for each pixel of an EIT image.

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

    Raises:
        ValueError: If ``values`` is not a 2D grid.
        TypeError: If ``values`` cannot be stored as floats without changing
            them (text, for example).
    """

    name: str
    values: np.ndarray
    modality: str = "eit"
    category: str | None = None
    unit: str | None = None
    method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the 2D numeric grid, then normalize unit and category names."""

        self.values = _as_float_grid(self.values, owner="PixelMap")
        self.unit = normalize_unit(self.unit)
        self.category = normalize_category(self.category) or self.category

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
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the grid and its descriptive fields as a dictionary.

        Returns:
            dict[str, Any]: The grid as a list of rows, together with name,
                modality, category, unit, method and metadata. NaN pixels
                remain NaN. Metadata is copied at the top level.
        """

        return {
            "name": self.name,
            "values": self.values.tolist(),
            "modality": self.modality,
            "category": self.category,
            "unit": self.unit,
            "method": self.method,
            "metadata": dict(self.metadata),
        }


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
            warnings.warn(
                f"PixelMask '{self.name}' is NaN everywhere, so applying it "
                "gives NaN for every pixel.",
                UserWarning,
                stacklevel=3,
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
                warnings.warn(
                    f"PixelMask '{self.name}' holds 0 values, which are turned "
                    "into NaN (outside the region). Pass keep_zeros=True to "
                    "keep them as weight 0, or "
                    "suppress_zero_conversion_warning=True to skip this warning.",
                    UserWarning,
                    stacklevel=3,
                )
            grid[grid == 0] = np.nan
        self.values = grid

    @property
    def shape(self) -> tuple[int, int]:
        """Return the number of rows and columns, in that order."""

        return self.values.shape

    @property
    def included_pixel_count(self) -> int:
        """Return the count of pixels that are not NaN, weight 0 included."""

        return int(np.count_nonzero(~np.isnan(self.values)))

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
            return replace(
                data,
                values=self._multiply(data.values),
                metadata=dict(data.metadata),
            )
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
    pixel's breath start, middle and end time. This function picks one pixel
    from that grid and turns its breaths into ``BreathEvent`` objects, so
    analyses written for global breaths (durations, timing checks, linking)
    can run on one pixel too.

    Args:
        pixel_breaths: The ``IntervalData`` written by ``eit.pixel_breaths``
            (name ``'pixel_breaths'`` by default): one (row, column, 3) grid
            of start, middle and end times in seconds per global breath.
        row: Row of the pixel, counted from 0.
        column: Column of the pixel, counted from 0.

    Returns:
        list[BreathEvent]: One breath per global breath in which the pixel
            has a breath, in time order. ``start_time`` and ``end_time`` are
            the pixel's own breath start and end, ``extremum_time`` is its middle
            time (end of inspiration), all in seconds. A global breath where
            the pixel has no breath (NaN) gives no entry. ``metadata`` holds
            ``row``, ``column`` and ``global_breath_id``, the ``id`` of the
            global breath it belongs to.

    Raises:
        ValueError: If the grids are not (row, column, 3), or ``row`` or
            ``column`` is outside the grid.
    """

    values = pixel_breaths.values if pixel_breaths.values is not None else []
    events: list[BreathEvent] = []
    for interval, grid in zip(pixel_breaths.intervals, values, strict=True):
        timings = np.asarray(grid, dtype=float)
        if timings.ndim != 3 or timings.shape[-1] != 3:
            raise ValueError(
                f"'{pixel_breaths.name}' must hold (row, column, 3) grids of "
                f"start, middle and end times, got shape {timings.shape}."
            )
        n_rows, n_columns, _ = timings.shape
        if not (0 <= row < n_rows and 0 <= column < n_columns):
            raise ValueError(
                f"Pixel ({row}, {column}) is outside the {n_rows} x {n_columns} grid."
            )
        start, middle, end = timings[row, column]
        if np.isnan([start, middle, end]).any():
            continue
        events.append(
            BreathEvent(
                pixel_breaths.modality,
                float(start),
                float(end),
                extremum_time=float(middle),
                source=pixel_breaths.method,
                metadata={
                    "row": row,
                    "column": column,
                    "global_breath_id": interval.id,
                },
            )
        )
    return events


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
