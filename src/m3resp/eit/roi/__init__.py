"""Regions of interest (ROI) of EIT images, as ``PixelMask`` objects.

``geometric_mask`` gives a fixed part of the image: the ventral or dorsal
half, the anatomical right or left half, one of four layers, or one of four
quadrants.

Three functions find the functional lung space from the breaths:
``tiv_lungspace`` (mean TIV of each pixel), ``amplitude_lungspace`` (mean
amplitude of each pixel, with the pixel's own breaths) and
``watershed_lungspace`` (both, with watershed regions around the maxima of
the amplitude). ``filter_mask_by_size`` keeps the groups of connected pixels
of a mask that are large enough.
"""

from m3resp.eit.roi.amplitude import amplitude_lungspace
from m3resp.eit.roi.filter_by_size import filter_mask_by_size
from m3resp.eit.roi.geometric import GEOMETRIC_REGIONS, geometric_mask
from m3resp.eit.roi.tiv import tiv_lungspace
from m3resp.eit.roi.watershed import watershed_lungspace

__all__ = [
    "GEOMETRIC_REGIONS",
    "amplitude_lungspace",
    "filter_mask_by_size",
    "geometric_mask",
    "tiv_lungspace",
    "watershed_lungspace",
]
