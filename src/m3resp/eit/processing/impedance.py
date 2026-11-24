"""Impedance waveforms made from EIT images.

The global impedance is the sum of all pixels of each image. It follows the
change of air volume in the slice of the chest under the electrode belt.
Pixels without a value (NaN, for example outside the chest in Sentec and
Timpel images) are left out of the sum; an image with no value at all gives
0.

The regional impedance is the same sum over the pixels of a region only: each
pixel is first multiplied by its value in a ``PixelMask``. The functional
impedance is the regional impedance of the functional (ventilated) lung space.

The usual order of the steps is: filter the images, make a mask, then apply the
mask to the images (``PixelMask.apply``) and use the masked images in the next
steps. ``PixelMask.apply`` lists the name of the mask under
``metadata["masks"]`` of the result. ``regional_impedance(masked_images)``
sums the pixels of masked images. ``regional_impedance(images, mask=mask)``
applies the mask and sums the pixels in one call. ``global_impedance`` raises
an error for masked images, because their sum covers only the region of the
mask.

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
      `PixelMask.apply` and the sum in one function, and also sum images
      that are already masked.
    - `global_impedance` raises an error for masked images, and names the
      result after the label of the images.
    - Docstrings are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
"""

from __future__ import annotations

import numpy as np

from m3resp.data.pixel_maps import MASKS_KEY, PixelMask
from m3resp.data.signals import Signal


def global_impedance(pixel_impedance: Signal, *, name: str | None = None) -> Signal:
    """Return the global impedance: the sum of all pixels of each image.

    Args:
        pixel_impedance (Signal): EIT images, values with shape (frames,
            rows, columns) and time in seconds.
        name (str | None): The name of the result. None gives
            ``"Global impedance (<label>)"``, with the label of the images in
            ``metadata["label"]`` (for example ``"mdn_filtered"``), else the
            name of the images.

    Returns:
        Signal: One value per frame, in arbitrary units (``"AU"``), with a
            copy of the time of ``pixel_impedance`` and its frames per
            second, device maker (``source``) and processing state. Modality
            ``"eit"``, category ``"impedance"``, channel
            ``"global_impedance"``.

    Raises:
        ValueError: If the values of ``pixel_impedance`` do not have three
            axes, or if the images are masked (``metadata["masks"]``). Use
            ``regional_impedance`` for masked images.
    """

    masks = pixel_impedance.metadata.get(MASKS_KEY)
    if masks:
        raise ValueError(
            f"global_impedance: the images {pixel_impedance.name!r} are masked "
            f"with {masks}, so their sum covers only the region of the mask. "
            "Use regional_impedance(images) for masked images."
        )
    if name is None:
        label = pixel_impedance.metadata.get("label", pixel_impedance.name)
        name = f"Global impedance ({label})"
    return _summed(
        pixel_impedance,
        np.asarray(pixel_impedance.values),
        name=name,
        channel="global_impedance",
        function="global_impedance",
    )


def regional_impedance(
    pixel_impedance: Signal,
    *,
    mask: PixelMask | None = None,
    name: str | None = None,
    channel: str = "regional_impedance",
) -> Signal:
    """Return the regional impedance: the sum of the pixels inside a mask.

    With ``mask``, each pixel is first multiplied by its mask value
    (``PixelMask.apply``): a NaN pixel of the mask leaves the pixel out of the
    sum and a weight from 0 to 1 counts it partly. Without ``mask``, the
    images are masked already with ``PixelMask.apply`` and their pixels are
    summed as they are.

    Args:
        pixel_impedance (Signal): EIT images, values with shape (frames,
            rows, columns) and time in seconds.
        mask (PixelMask | None): The region, with the shape (rows, columns)
            of the images. None for images that are masked already.
        name (str | None): The name of the result. None gives
            ``"<mask name>_impedance"``, for example ``"ventral_impedance"``.
            For masked images without ``mask``, the names of their masks are
            joined with ``"_"``.
        channel (str): The channel of the result.

    Returns:
        Signal: One value per frame, in arbitrary units (``"AU"``), with a
            copy of the time of ``pixel_impedance`` and its frames per
            second, device maker (``source``) and processing state. Modality
            ``"eit"``, category ``"impedance"``.

    Raises:
        ValueError: If the values of ``pixel_impedance`` do not have three
            axes, if the image size differs from the mask size, if ``mask``
            is None and the images are not masked, or if the images are
            masked already with a mask of the same name as ``mask``.
    """

    values = np.asarray(pixel_impedance.values)
    _check_images(pixel_impedance, values, function="regional_impedance")
    masked, mask_name = _masked(
        pixel_impedance, values, mask, function="regional_impedance"
    )
    return _summed(
        pixel_impedance,
        masked,
        name=name if name is not None else f"{mask_name}_impedance",
        channel=channel,
        function="regional_impedance",
    )


def functional_impedance(
    pixel_impedance: Signal,
    *,
    lung_space: PixelMask | None = None,
    name: str = "functional_impedance",
) -> Signal:
    """Return the functional impedance: the sum of the pixels in the lung space.

    This is ``regional_impedance`` with the mask of the functional
    (ventilated) lung space, for example the result of
    ``eit.roi_tiv_lungspace``. Without ``lung_space``, the images are masked
    already with the lung space (``PixelMask.apply``).

    Args:
        pixel_impedance (Signal): EIT images, values with shape (frames,
            rows, columns) and time in seconds.
        lung_space (PixelMask | None): The functional lung space, with the
            shape (rows, columns) of the images. None for images that are
            masked already.
        name (str): The name of the result.

    Returns:
        Signal: As ``regional_impedance``, with channel
            ``"functional_impedance"``.

    Raises:
        ValueError: As ``regional_impedance``.
    """

    values = np.asarray(pixel_impedance.values)
    _check_images(pixel_impedance, values, function="functional_impedance")
    masked, _ = _masked(
        pixel_impedance, values, lung_space, function="functional_impedance"
    )
    return _summed(
        pixel_impedance,
        masked,
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


def _masked(
    pixel_impedance: Signal,
    values: np.ndarray,
    mask: PixelMask | None,
    *,
    function: str,
) -> tuple[np.ndarray, str]:
    """Return the masked pixels and the name of the mask.

    Args:
        pixel_impedance (Signal): The EIT images, with the names of the masks
            applied to them in ``metadata["masks"]``.
        values (numpy.ndarray): The pixels of ``pixel_impedance``.
        mask (PixelMask | None): The mask to apply, or None for images that
            are masked already.
        function (str): The name of the calling function, for the errors.

    Returns:
        tuple[numpy.ndarray, str]: The masked pixels, and the name of
            ``mask``, or the names of the masks of the images joined with
            ``"_"``.

    Raises:
        ValueError: If ``mask`` is None and the images are not masked, or if
            the images are masked already with a mask of the name of ``mask``.
    """

    applied = list(pixel_impedance.metadata.get(MASKS_KEY, []))
    if mask is None:
        if not applied:
            raise ValueError(
                f"{function}: the images {pixel_impedance.name!r} are not "
                "masked. Give the mask, or mask the images with "
                "PixelMask.apply first."
            )
        return values, "_".join(applied)
    if mask.name in applied:
        raise ValueError(
            f"{function}: the images {pixel_impedance.name!r} are masked "
            f"already with {mask.name!r}. Use {function}(images) without a "
            "mask to sum them."
        )
    return mask.apply(values), mask.name


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
