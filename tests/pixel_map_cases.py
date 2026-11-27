"""Seeded pixel maps and masks, and the eitprocessing classes, for the tests of
`m3resp.visualization.pixel_map_style` and `m3resp.visualization.pixel_maps`."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from m3resp.data.pixel_maps import PIXEL_MAP_KINDS
from m3resp.visualization.pixel_map_style import MASK

STYLES = (None, *PIXEL_MAP_KINDS, MASK)


def upstream_classes() -> dict:
    """The eitprocessing class of each style."""

    pixelmap = pytest.importorskip("eitprocessing.datahandling.pixelmap")
    roi = pytest.importorskip("eitprocessing.roi")
    return {
        None: pixelmap.PixelMap,
        "tiv": pixelmap.TIVMap,
        "amplitude": pixelmap.AmplitudeMap,
        "odcl": pixelmap.ODCLMap,
        "difference": pixelmap.DifferenceMap,
        "perfusion": pixelmap.PerfusionMap,
        "pendelluft": pixelmap.PendelluftMap,
        "signed_pendelluft": pixelmap.SignedPendelluftMap,
        "integer": pixelmap.IntegerMap,
        MASK: roi.PixelMask,
    }


def seeded_grids(kind, seed: int = 0) -> dict[str, np.ndarray]:
    """Seeded 6 x 5 grids for a style: the cases of the module text."""

    rng = np.random.default_rng(seed)
    shape = (6, 5)
    if kind == "integer":
        return {
            "numbers 0 to 24": rng.integers(0, 25, shape),
            "one number": np.full(shape, 3),
            "negative numbers": rng.integers(-3, 4, shape),
        }
    if kind == MASK:
        return {
            "binary": np.where(rng.random(shape) > 0.4, 1.0, np.nan),
            "weighted": np.where(
                rng.random(shape) > 0.4, rng.uniform(0.1, 1, shape), np.nan
            ),
            "all NaN": np.full(shape, np.nan),
        }
    grid = rng.normal(size=shape) * 3
    grid[0, 0] = np.nan
    grid[3, 2] = np.nan
    return {
        "mixed signs with NaN": grid,
        "positive with NaN": np.abs(grid),
        "negative with NaN": -np.abs(grid),
        "small, below 1": np.abs(grid) / 20,
        "zeros": np.zeros(shape),
        "flat": np.full(shape, 2.0),
        "all NaN": np.full(shape, np.nan),
    }


def upstream_object(kind, values):
    """The eitprocessing map or mask of a style, warnings switched off."""

    cls = upstream_classes()[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == MASK:
            return cls(values, suppress_all_nan_warning=True)
        return cls(values)
