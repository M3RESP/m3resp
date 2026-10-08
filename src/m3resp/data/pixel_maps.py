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
"""

from __future__ import annotations

import warnings
from dataclasses import InitVar, dataclass, field
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

        rows, columns = self.values.shape
        return rows, columns

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

        rows, columns = self.values.shape
        return rows, columns

    @property
    def included_pixel_count(self) -> int:
        """Return the count of pixels that are not NaN, weight 0 included."""

        return int(np.count_nonzero(~np.isnan(self.values)))

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
