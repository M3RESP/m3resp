"""Impedance waveforms made from EIT images.

The global impedance is the sum of all pixels of each image. It follows the
change of air volume in the slice of the chest under the electrode belt.
Pixels without a value (NaN, for example outside the chest in Sentec and
Timpel images) are left out of the sum; an image with no value at all gives
0.

The regional impedance is the same sum over the pixels of a region only: each
pixel is first multiplied by its value in a ``PixelMask``. The functional
impedance is the regional impedance of the functional (ventilated) lung space.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/datahandling/eitdata.py::
                EITData.get_summed_impedance;
                EITData.calculate_global_impedance;
                eitprocessing/roi/__init__.py::PixelMask.apply
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The input and the result are `Signal`s.
    - The result keeps the device maker of the images in `source` and names
      this function in `method`.
    - `regional_impedance` and `functional_impedance` join
      `PixelMask.apply` and the sum in one function.
    - Docstrings are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
"""

from __future__ import annotations

import numpy as np

from m3resp.data.pixel_maps import PixelMask
from m3resp.data.signals import Signal


def global_impedance(
    pixel_impedance: Signal, *, name: str = "global_impedance"
) -> Signal:
    """Return the global impedance: the sum of all pixels of each image.

    Args:
        pixel_impedance (Signal): EIT images, values with shape (frames,
            rows, columns) and time in seconds.
        name (str): The name of the result.

    Returns:
        Signal: One value per frame, in arbitrary units (``"a.u."``), with a
            copy of the time of ``pixel_impedance`` and its frames per
            second, device maker (``source``) and processing state. Modality
            ``"eit"``, category ``"impedance"``, channel
            ``"global_impedance"``.

    Raises:
        ValueError: If the values of ``pixel_impedance`` do not have three
            axes.
    """

    return _summed(
        pixel_impedance,
        np.asarray(pixel_impedance.values),
        name=name,
        channel="global_impedance",
        function="global_impedance",
    )


def regional_impedance(
    pixel_impedance: Signal,
    mask: PixelMask,
    *,
    name: str = "regional_impedance",
    channel: str = "regional_impedance",
) -> Signal:
    """Return the regional impedance: the sum of the pixels inside a mask.

    Each pixel is multiplied by its mask value (``PixelMask.apply``): a NaN
    pixel of the mask leaves the pixel out of the sum and a weight from 0 to 1
    counts it partly. Then the pixels of each image are summed.

    Args:
        pixel_impedance (Signal): EIT images, values with shape (frames,
            rows, columns) and time in seconds.
        mask (PixelMask): The region, with the shape (rows, columns) of the
            images.
        name (str): The name of the result.
        channel (str): The channel of the result.

    Returns:
        Signal: One value per frame, in arbitrary units (``"a.u."``), with a
            copy of the time of ``pixel_impedance`` and its frames per
            second, device maker (``source``) and processing state. Modality
            ``"eit"``, category ``"impedance"``.

    Raises:
        ValueError: If the values of ``pixel_impedance`` do not have three
            axes, or if the image size differs from the mask size.
    """

    values = np.asarray(pixel_impedance.values)
    _check_images(pixel_impedance, values, function="regional_impedance")
    return _summed(
        pixel_impedance,
        mask.apply(values),
        name=name,
        channel=channel,
        function="regional_impedance",
    )


def functional_impedance(
    pixel_impedance: Signal,
    lung_space: PixelMask,
    *,
    name: str = "functional_impedance",
) -> Signal:
    """Return the functional impedance: the sum of the pixels in the lung space.

    This is ``regional_impedance`` with the mask of the functional
    (ventilated) lung space, for example the result of
    ``eit.roi_tiv_lungspace``.

    Args:
        pixel_impedance (Signal): EIT images, values with shape (frames,
            rows, columns) and time in seconds.
        lung_space (PixelMask): The functional lung space, with the shape
            (rows, columns) of the images.
        name (str): The name of the result.

    Returns:
        Signal: As ``regional_impedance``, with channel
            ``"functional_impedance"``.

    Raises:
        ValueError: If the values of ``pixel_impedance`` do not have three
            axes, or if the image size differs from the mask size.
    """

    values = np.asarray(pixel_impedance.values)
    _check_images(pixel_impedance, values, function="functional_impedance")
    return _summed(
        pixel_impedance,
        lung_space.apply(values),
        name=name,
        channel="functional_impedance",
        function="functional_impedance",
    )


def _check_images(
    pixel_impedance: Signal, values: np.ndarray, *, function: str
) -> None:
    """Raise ``ValueError`` if ``values`` is not a series of images."""

    if values.ndim != 3:
        raise ValueError(
            f"{function} needs EIT images with shape (frames, rows, "
            f"columns); {pixel_impedance.name!r} has shape {values.shape}."
        )


def _summed(
    pixel_impedance: Signal,
    values: np.ndarray,
    *,
    name: str,
    channel: str,
    function: str,
) -> Signal:
    """Return the sum of the pixels of each image as a ``Signal``."""

    _check_images(pixel_impedance, values, function=function)
    return Signal(
        values=np.nansum(values, axis=(1, 2)),
        time=np.copy(pixel_impedance.time),
        sample_frequency=pixel_impedance.sample_frequency,
        unit="AU",
        name=name,
        modality="eit",
        category="impedance",
        channel=channel,
        source=pixel_impedance.source,
        processing_state=pixel_impedance.processing_state,
        method=f"m3resp.eit.processing.{function}",
    )
