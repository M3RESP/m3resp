"""Geometric masks: fixed parts of the EIT image, such as the ventral half.

``geometric_mask`` gives a ``PixelMask`` that cuts the image into halves,
quarter layers or quadrants. Rows go from ventral (top of the image, row 0)
to dorsal (bottom). The image shows the patient as seen from the feet, so the
anatomical right of the patient is the left side of the image (the first
columns), and the anatomical left is the right side.

======================  ==============  ==========================================
Region                  Short name      Pixels
======================  ==============  ==========================================
``"ventral"``           ``"V"``         the first half of the rows
``"dorsal"``            ``"D"``         the last half of the rows
``"anatomical right"``  ``"R"``         the first half of the columns
``"anatomical left"``   ``"L"``         the last half of the columns
``"layer 1"`` to 4      ``"L1"`` to 4   the first to the last quarter of the rows
``"quadrant 1"``        ``"Q1"``        ventral and anatomical right
``"quadrant 2"``        ``"Q2"``        ventral and anatomical left
``"quadrant 3"``        ``"Q3"``        dorsal and anatomical right
``"quadrant 4"``        ``"Q4"``        dorsal and anatomical left
======================  ==============  ==========================================

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/roi/__init__.py::get_geometric_mask
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The function is `geometric_mask(region, shape)`; the result is an
      m3resp `PixelMask` with the full region name as `name`, and `region`
      and `shape` in `metadata`.
    - Each mask is set from a range of rows and a range of columns
      (eitprocessing makes the dorsal, anatomical left and quadrant masks by
      flipping and multiplying other masks). The grids are the same.
    - The error for an image height that does not divide shows the number
      (eitprocessing issue I19: "{height_divisor}" in the message).
    - A shape that is not two positive whole numbers raises ValueError.
    - Error messages are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import re

import numpy as np

from m3resp.data import PixelMask

#: The full names of the 12 regions, in the order of the table above.
GEOMETRIC_REGIONS = (
    "ventral",
    "dorsal",
    "anatomical right",
    "anatomical left",
    "layer 1",
    "layer 2",
    "layer 3",
    "layer 4",
    "quadrant 1",
    "quadrant 2",
    "quadrant 3",
    "quadrant 4",
)

_SHORT_NAMES = {
    "V": "ventral",
    "D": "dorsal",
    "R": "anatomical right",
    "L": "anatomical left",
}

#: For each quadrant: the half of the rows and the half of the columns.
_QUADRANT_HALVES = {
    "quadrant 1": ("ventral", "anatomical right"),
    "quadrant 2": ("ventral", "anatomical left"),
    "quadrant 3": ("dorsal", "anatomical right"),
    "quadrant 4": ("dorsal", "anatomical left"),
}


def geometric_mask(region: str, shape: tuple[int, int] = (32, 32)) -> PixelMask:
    """Return the mask of a fixed part of the EIT image.

    Args:
        region (str): One of the full names of :data:`GEOMETRIC_REGIONS`, or
            its short name (``"V"``, ``"D"``, ``"R"``, ``"L"``, ``"L1"`` to
            ``"L4"``, ``"Q1"`` to ``"Q4"``). Upper and lower case count.
        shape (tuple[int, int]): Number of rows and columns of the image.

    Returns:
        PixelMask: 1 for the pixels of the region, NaN elsewhere. ``name`` is
            the full region name; ``metadata`` holds ``region`` and ``shape``.

    Raises:
        ValueError: If ``region`` is unknown, ``shape`` is not two positive
            whole numbers, or the number of rows (ventral, dorsal, layers,
            quadrants) or columns (anatomical right and left, quadrants)
            does not divide into the parts of the region.
        TypeError: If ``region`` is not text.
    """

    name = _full_name(region)
    height, width = _check_shape(shape)

    if name.startswith("layer"):
        layer_height = _part_size(height, 4, name="layer", side="height")
        number = int(name.split(" ")[1])
        rows = slice((number - 1) * layer_height, number * layer_height)
        columns = slice(0, width)
    else:
        vertical, lateral = _QUADRANT_HALVES.get(name, (name, name))
        rows = slice(0, height)
        columns = slice(0, width)
        if vertical in ("ventral", "dorsal"):
            half = _part_size(height, 2, name=name, side="height")
            rows = slice(0, half) if vertical == "ventral" else slice(half, height)
        if lateral in ("anatomical right", "anatomical left"):
            half = _part_size(width, 2, name=name, side="width")
            columns = (
                slice(0, half) if lateral == "anatomical right" else slice(half, width)
            )

    values = np.full((height, width), np.nan)
    values[rows, columns] = 1.0
    return PixelMask(
        name=name,
        values=values,
        modality="eit",
        method="m3resp.eit.roi.geometric_mask",
        metadata={"region": name, "shape": [height, width]},
    )


def _full_name(region: str) -> str:
    """Return the full region name for a full or short name."""

    if not isinstance(region, str):
        raise TypeError(f"The region must be a name, got {region!r}.")
    name = _SHORT_NAMES.get(region, region)
    name = re.sub(r"^Q([1-4])$", r"quadrant \1", name)
    name = re.sub(r"^L([1-4])$", r"layer \1", name)
    if name not in GEOMETRIC_REGIONS:
        raise ValueError(
            f"Unknown region {region!r}; use one of {GEOMETRIC_REGIONS} or "
            "their short names V, D, R, L, L1-L4, Q1-Q4."
        )
    return name


def _check_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Return the number of rows and columns of ``shape``."""

    try:
        height, width = shape
    except (TypeError, ValueError):
        raise ValueError(f"The shape must be (rows, columns), got {shape!r}.") from None
    for size in (height, width):
        if (
            isinstance(size, bool)
            or not isinstance(size, (int, np.integer))
            or size < 1
        ):
            raise ValueError(
                f"The shape must be two positive whole numbers, got {shape!r}."
            )
    return int(height), int(width)


def _part_size(total: int, parts: int, *, name: str, side: str) -> int:
    """Return ``total // parts``; ValueError when ``total`` does not divide."""

    if total % parts != 0:
        raise ValueError(
            f"A {name} mask needs an image {side} that is a multiple of "
            f"{parts}; the {side} is {total}."
        )
    return total // parts
