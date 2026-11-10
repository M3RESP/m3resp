"""Keep the connected regions of a mask that have a minimum number of pixels.

``filter_mask_by_size`` finds the groups of connected pixels of a mask (all
pixels that are not NaN), and keeps the groups with at least
``min_region_size`` pixels. The connectivity says which pixels touch:

- ``1``: pixels that share an edge (up, down, left, right). Also called
  4-connectivity.
- ``2``: pixels that share an edge or a corner. Also called 8-connectivity.
- A true/false or 0/1 array of 3 x 3 for other neighbourhoods; see
  ``scipy.ndimage.label``.

The result has 1 for the pixels of the kept groups and NaN elsewhere, so the
weights of a weighted mask are not kept.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/roi/filter_by_size.py::FilterROIBySize
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The class `FilterROIBySize` and its method `apply` are one function,
      `filter_mask_by_size`.
    - The kept groups are put together in one step (eitprocessing makes one
      mask per group and adds them with `PixelMaskCollection.combine`). The
      result is the same.
    - The mask has a `name`, and `metadata` with the settings and the count
      and fraction of included pixels. With a `captures` dictionary, the
      group numbers and sizes are stored.
    - The weights of a weighted mask are lost, as in eitprocessing.
    - Error messages are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import ndimage

from m3resp.data import PixelMap, PixelMask
from m3resp.eit.roi._shared import finish_mask

METHOD = "m3resp.eit.roi.filter_mask_by_size"


def filter_mask_by_size(
    mask: PixelMask,
    *,
    min_region_size: int = 10,
    connectivity: Literal[1, 2] | np.ndarray = 1,
    name: str | None = None,
    captures: dict[str, Any] | None = None,
) -> PixelMask:
    """Return the groups of connected pixels with at least ``min_region_size`` pixels.

    Args:
        mask (PixelMask): The mask; every pixel that is not NaN counts.
        min_region_size (int): Smallest number of pixels of a kept group.
        connectivity (int | numpy.ndarray): ``1`` (pixels that share an
            edge), ``2`` (an edge or a corner) or a custom structure array
            for ``scipy.ndimage.label``.
        name (str | None): Name of the result; ``"<mask name>_size_filtered"``
            when None.
        captures (dict | None): When a dictionary, it receives
            ``"regions"`` (``PixelMap`` of kind ``"integer"`` with the group
            number of each pixel, 0 outside the mask), ``"region sizes"``
            (pixel count of each group, in group order) and
            ``"kept regions"`` (numbers of the kept groups).

    Returns:
        PixelMask: 1 for the pixels of the kept groups, NaN elsewhere.
            ``metadata`` holds ``min_region_size``, ``connectivity``,
            ``source_mask``, ``included_pixel_count`` and
            ``included_pixel_fraction``.

    Raises:
        TypeError: If ``mask`` is not a ``PixelMask``.
        ValueError: If ``connectivity`` is not 1, 2 or an array.
        RuntimeError: If no group has ``min_region_size`` pixels or more
            (for example an empty mask, only small groups, or a connectivity
            that joins too few pixels).
    """

    if not isinstance(mask, PixelMask):
        raise TypeError(
            f"filter_mask_by_size needs a PixelMask, got {type(mask).__name__}."
        )
    structure = _structure(connectivity)
    labels, count = ndimage.label(~np.isnan(mask.values), structure=structure)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:].tolist()
    kept = [
        number
        for number, size in zip(range(1, count + 1), sizes, strict=True)
        if size >= min_region_size
    ]
    if captures is not None:
        captures["regions"] = PixelMap(name="regions", values=labels, kind="integer")
        captures["region sizes"] = sizes
        captures["kept regions"] = kept
    if not kept:
        raise RuntimeError(
            f"No group of connected pixels has {min_region_size} pixels or "
            "more. This happens when the mask is empty, all groups are too "
            "small, or the connectivity joins too few pixels."
        )

    result = PixelMask(name="kept regions", values=np.isin(labels, kept))
    return finish_mask(
        result,
        name=name if name is not None else f"{mask.name}_size_filtered",
        method=METHOD,
        settings={
            "min_region_size": min_region_size,
            "connectivity": np.asarray(connectivity).tolist(),
            "source_mask": mask.name,
        },
    )


def _structure(connectivity: Any) -> np.ndarray:
    """Return the structure array of ``scipy.ndimage.label`` for ``connectivity``."""

    match connectivity:
        case np.ndarray():
            return connectivity
        case 1 | 2:
            return ndimage.generate_binary_structure(2, connectivity)
        case int():
            raise ValueError(
                f"Unknown connectivity {connectivity}; use 1, 2 or a structure array."
            )
        case _:
            raise ValueError(
                f"Unknown connectivity of type {type(connectivity).__name__}; use "
                "1, 2 or a structure array."
            )
