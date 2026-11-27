"""Helpers shared by the lung-space functions of ``m3resp.eit.roi``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from m3resp.data import IntervalData, PixelMap, PixelMask, Signal
from m3resp.data.pixel_maps import MASKS_KEY
from m3resp.eit.processing import global_impedance


def breath_waveform(pixels: Signal, *, signal: Signal | None, function: str) -> Signal:
    """Return ``signal``, or the sum of all pixels of each image when it is None.

    Args:
        pixels (Signal): Pixel impedance, values with shape (time, row, column).
        signal (Signal | None): The waveform of the breaths, or None.
        function (str): Name of the function, for the error message.

    Returns:
        Signal: ``signal``, or the global impedance of ``pixels``.

    Raises:
        ValueError: If ``signal`` is None and the pixel impedance is masked
            (``metadata["masks"]``).
    """

    if signal is not None:
        return signal
    if pixels.metadata.get(MASKS_KEY):
        raise ValueError(
            f"{function}: the pixel impedance {pixels.name!r} is masked, so its sum is "
            "not the global impedance. Give the waveform of the breaths as "
            "signal=..., for example the global impedance of the pixel impedance before "
            "masking."
        )
    return global_impedance(pixels)


def pixel_maps_of(per_breath: IntervalData) -> list[PixelMap]:
    """Return the ``PixelMap`` of each breath of a pixel TIV result."""

    return list(per_breath.values or [])


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
    grids = np.stack([np.asarray(m.values) for m in pixel_maps_of(per_breath)])
    if np.all(np.isnan(grids)):
        raise ValueError(
            f"No {quantity} values other than NaN were found. This can happen "
            "when too few breaths were detected in the data."
        )
    return grids


def finish_mask(
    mask: PixelMask, *, name: str, method: str, settings: dict[str, Any]
) -> PixelMask:
    """Return a copy of a lung-space mask with its name, method and metadata.

    Args:
        mask (PixelMask): The mask.
        name (str): Name of the mask.
        method (str): Name of the function that made the mask.
        settings (dict[str, Any]): The settings of that function.

    Returns:
        PixelMask: The values of ``mask``, with ``metadata`` holding
            ``axes`` and the settings.
    """

    # The values were checked when ``mask`` was made, so the copy keeps
    # weight 0 and gives no second warning for a mask that is NaN everywhere.
    return replace(
        mask,
        name=name,
        method=method,
        metadata={"axes": ["row", "column"], **settings},
        keep_zeros=True,
        suppress_all_nan_warning=True,
    )
