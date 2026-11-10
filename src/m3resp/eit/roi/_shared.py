"""Helpers shared by the lung-space functions of ``m3resp.eit.roi``."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.data import IntervalData, PixelMap, PixelMask


def check_threshold(value: Any, *, name: str, function: str) -> None:
    """Raise an error unless ``value`` is a float between 0 and 1 (both left out).

    Args:
        value (Any): The threshold.
        name (str): Name of the argument, for the error message.
        function (str): Name of the function, for the error message.

    Raises:
        TypeError: If ``value`` is not a float (an int such as ``1`` too).
        ValueError: If ``value`` is not between 0 and 1.
    """

    if not isinstance(value, float):
        raise TypeError(
            f"{function} '{name}' must be a float, got {type(value).__name__}."
        )
    if not 0 < value < 1:
        raise ValueError(f"{function} '{name}' must be between 0 and 1, got {value!r}.")


def pixel_maps_of(per_breath: IntervalData) -> list[PixelMap]:
    """Return the ``PixelMap`` of each breath of a pixel TIV result."""

    return list(per_breath.values or [])


def per_breath_grids(per_breath: IntervalData) -> np.ndarray:
    """Return the maps of each breath as one (breath, row, column) array."""

    return np.stack([np.asarray(m.values) for m in pixel_maps_of(per_breath)])


def check_per_breath(per_breath: IntervalData, *, quantity: str) -> np.ndarray:
    """Return the maps per breath, or raise ValueError when there are no values.

    Args:
        per_breath (IntervalData): One ``PixelMap`` per breath.
        quantity (str): ``"TIV"`` or ``"amplitude"``, for the error message.

    Returns:
        numpy.ndarray: The maps as one (breath, row, column) array.

    Raises:
        ValueError: If there are no breaths, or every value is NaN.
    """

    if not per_breath.intervals:
        raise ValueError(f"No breaths were detected. Cannot compute {quantity}.")
    grids = per_breath_grids(per_breath)
    if np.all(np.isnan(grids)):
        raise ValueError(
            f"No {quantity} values other than NaN were found. This can happen "
            "when too few breaths were detected in the data."
        )
    return grids


def finish_mask(
    mask: PixelMask, *, name: str, method: str, settings: dict[str, Any]
) -> PixelMask:
    """Give a lung-space mask its name, method and metadata.

    Args:
        mask (PixelMask): The mask to finish; it is changed in place.
        name (str): Name of the mask.
        method (str): Name of the function that made the mask.
        settings (dict[str, Any]): The settings of that function.

    Returns:
        PixelMask: ``mask``, with ``metadata`` holding ``axes``, the
            settings, ``included_pixel_count`` and
            ``included_pixel_fraction``.
    """

    included = ~np.isnan(mask.values)
    mask.name = name
    mask.method = method
    mask.metadata = {
        "axes": ["row", "column"],
        **settings,
        "included_pixel_count": int(included.sum()),
        "included_pixel_fraction": float(included.mean()) if included.size else 0.0,
    }
    return mask
