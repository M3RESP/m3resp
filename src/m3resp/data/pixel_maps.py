"""``PixelMap`` and ``PixelMask``: one number for each EIT pixel.

An EIT image is a grid of pixels (rows by columns). Some results give one
number for every pixel rather than one number for the whole lung: the tidal
impedance variation of each pixel in one breath, for example. A
:class:`PixelMap` holds such a grid.

A :class:`PixelMask` is a grid that says which pixels belong to a region,
such as the functional lung space. A pixel that is not part of the region is
NaN; a pixel that is part of it is 1, or a weight between 0 and 1.

Both follow eitprocessing's ``PixelMap`` and ``PixelMask``, so values can be
handed back to eitprocessing without being changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from m3resp.data.categories import normalize_category
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
        unit: Unit of the values.
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
        self.values = _as_float_grid(self.values, owner="PixelMap")
        self.unit = normalize_unit(self.unit)
        self.category = normalize_category(self.category) or self.category

    @property
    def shape(self) -> tuple[int, int]:
        """Number of (rows, columns)."""

        rows, columns = self.values.shape
        return rows, columns

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        # Lets numpy read a PixelMap as its grid, e.g. np.nanmean(pixel_map).
        return np.asarray(self.values, dtype=dtype)

    def __eq__(self, other: object) -> bool:
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
        """The same content as plain lists and dictionaries, ready to write to
        a JSON file. The grid becomes a list of rows."""

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

    Each pixel holds NaN when it is not part of the region, 1 when it is, or
    a number between 0 and 1 when it counts only partly (a weight). Applying
    the mask multiplies each pixel by this number, so a NaN pixel drops out.

    A grid of true/false values is accepted too: true becomes 1 and false
    becomes NaN. A grid of numbers must already use NaN for the pixels left
    out; a 0 is refused, because it is unclear whether it means "left out"
    or "weight 0".

    Attributes:
        name: What the mask is, e.g. ``'tiv_lungspace_mask'``.
        values: The mask as a 2D grid of shape (row, column), stored as
            floats.
        modality: The device the mask belongs to.
        method: Name of the method that made the mask.
        metadata: Optional extra information.

    Raises:
        ValueError: If ``values`` is not a 2D grid, holds a number below 0
            or above 1, or holds a 0.
        TypeError: If ``values`` cannot be stored as floats without changing
            them.
    """

    name: str
    values: np.ndarray
    modality: str = "eit"
    method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = np.asarray(self.values)
        if raw.dtype == bool:
            grid = np.where(raw, 1.0, np.nan)
            if grid.ndim != 2:
                raise ValueError(
                    f"PixelMask.values must be a 2D grid (row, column), got "
                    f"{grid.ndim} dimension(s)."
                )
        else:
            grid = _as_float_grid(raw, owner="PixelMask")
            if np.any(grid == 0):
                raise ValueError(
                    "PixelMask.values holds 0. Use NaN for pixels that are not "
                    "part of the region, or pass a true/false grid."
                )
            if np.any((grid < 0) | (grid > 1)):
                raise ValueError(
                    "PixelMask.values must be NaN, or a number from 0 to 1 "
                    f"(found values from {np.nanmin(grid)} to {np.nanmax(grid)})."
                )
        self.values = grid

    @property
    def shape(self) -> tuple[int, int]:
        """Number of (rows, columns)."""

        rows, columns = self.values.shape
        return rows, columns

    @property
    def included_pixel_count(self) -> int:
        """Number of pixels that are part of the region (not NaN)."""

        return int(np.count_nonzero(~np.isnan(self.values)))

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        # Lets numpy read a PixelMask as its grid.
        return np.asarray(self.values, dtype=dtype)

    def __eq__(self, other: object) -> bool:
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
        """The same content as plain lists and dictionaries, ready to write to
        a JSON file. The grid becomes a list of rows."""

        return {
            "name": self.name,
            "values": self.values.tolist(),
            "modality": self.modality,
            "method": self.method,
            "metadata": dict(self.metadata),
        }


def _as_float_grid(values: Any, *, owner: str) -> np.ndarray:
    """Return ``values`` as a 2D float array, refusing anything that would
    change on the way (text, or numbers too large for a float)."""

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
