"""Impedance waveforms made from the pixel impedance.

The global impedance is the sum of all pixels of each image. It follows the
change of air volume in the slice of the chest under the electrode belt.
Pixels without a value (NaN, for example outside the chest in Sentec and
Timpel images) are left out of the sum; an image with no value at all gives
0.

The regional impedance is the same sum over the pixels of a region only: each
pixel is first multiplied by its value in a ``PixelMask``. The functional
impedance is the regional impedance of the functional (ventilated) lung space.

The usual order of the steps is: filter the pixel impedance, make a mask,
then apply the mask to the pixel impedance (``PixelMask.apply``) and use the
masked pixel impedance in the next steps. ``PixelMask.apply`` lists the name
of the mask under ``metadata["masks"]`` of the result.
``regional_impedance(masked_pixels)`` sums the pixels of masked pixel
impedance. ``regional_impedance(pixel_impedance, mask=mask)`` applies the mask
and sums the pixels in one call. ``global_impedance`` raises an error for
masked pixel impedance, because its sum covers only the region of the mask.

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
    - The result keeps the device maker of the pixel impedance in `source` and names
      this function in `method`.
    - `regional_impedance` and `functional_impedance` join
      `PixelMask.apply` and the sum in one function, and also sum pixel impedance
      that are already masked.
    - `global_impedance` raises an error for masked pixel impedance, and names the
      result after the label of the pixel impedance.
    - Docstrings are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
"""

from __future__ import annotations

import numpy as np

from m3resp.data.pixel_maps import MASKS_KEY, PixelMask
from m3resp.data.signals import Signal

#: Public names of the three functions, stored as the method of each result.
GLOBAL_METHOD = "m3resp.eit.processing.global_impedance"
REGIONAL_METHOD = "m3resp.eit.processing.regional_impedance"
FUNCTIONAL_METHOD = "m3resp.eit.processing.functional_impedance"


def global_impedance(pixel_impedance: Signal, *, name: str | None = None) -> Signal:
    """Return the global impedance: the sum of all pixels of each image.

    Args:
        pixel_impedance (Signal): Pixel impedance, values with shape (frames,
            rows, columns) and time in seconds.
        name (str | None): The name of the result. None gives
            ``"Global impedance (<label>)"``, with the label of the pixel impedance in
            ``metadata["label"]`` (for example ``"mdn_filtered"``), else the
            name of the pixel impedance.

    Returns:
        Signal: One value per frame, in arbitrary units (``"AU"``), with a
            copy of the time of ``pixel_impedance`` and its frames per
            second, device maker (``source``) and processing state. Modality
            ``"eit"``, category ``"impedance"``, channel
            ``"global_impedance"``.

    Raises:
        ValueError: If the values of ``pixel_impedance`` do not have three
            axes, or if the pixel impedance is masked (``metadata["masks"]``). Use
            ``regional_impedance`` for masked pixel impedance.
    """

    masks = pixel_impedance.metadata.get(MASKS_KEY)
    if masks:
        raise ValueError(
            f"global_impedance: the pixel impedance {pixel_impedance.name!r} is masked "
            f"with {masks}, so its sum covers only the region of the mask. "
            "Use regional_impedance(pixel_impedance) for masked pixel impedance."
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
        method=GLOBAL_METHOD,
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
    the pixel impedance is masked already with ``PixelMask.apply`` and its pixels are
    summed as they are.

    Args:
        pixel_impedance (Signal): Pixel impedance, values with shape (frames,
            rows, columns) and time in seconds.
        mask (PixelMask | None): The region, with the shape (rows, columns)
            of the pixel impedance. None for pixel impedance that is masked
            already.
        name (str | None): The name of the result. None gives
            ``"<mask name>_impedance"``, for example ``"ventral_impedance"``.
            For masked pixel impedance without ``mask``, the names of its masks
            are
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
            is None and the pixel impedance is not masked, or if the pixel impedance is
            masked already with a mask of the same name as ``mask``.
    """

    values = np.asarray(pixel_impedance.values)
    _check_pixels(pixel_impedance, values, function="regional_impedance")
    masked, mask_name = _masked(
        pixel_impedance, values, mask, function="regional_impedance"
    )
    return _summed(
        pixel_impedance,
        masked,
        name=name if name is not None else f"{mask_name}_impedance",
        channel=channel,
        function="regional_impedance",
        method=REGIONAL_METHOD,
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
    ``eit.roi_tiv_lungspace``. Without ``lung_space``, the pixel impedance is masked
    already with the lung space (``PixelMask.apply``).

    Args:
        pixel_impedance (Signal): Pixel impedance, values with shape (frames,
            rows, columns) and time in seconds.
        lung_space (PixelMask | None): The functional lung space, with the
            shape (rows, columns) of the pixel impedance. None for pixel
            impedance that is masked already.
        name (str): The name of the result.

    Returns:
        Signal: As ``regional_impedance``, with channel
            ``"functional_impedance"``.

    Raises:
        ValueError: As ``regional_impedance``.
    """

    values = np.asarray(pixel_impedance.values)
    _check_pixels(pixel_impedance, values, function="functional_impedance")
    masked, _ = _masked(
        pixel_impedance, values, lung_space, function="functional_impedance"
    )
    return _summed(
        pixel_impedance,
        masked,
        name=name,
        channel="functional_impedance",
        function="functional_impedance",
        method=FUNCTIONAL_METHOD,
    )


def _check_pixels(
    pixel_impedance: Signal, values: np.ndarray, *, function: str
) -> None:
    """Raise ``ValueError`` if ``values`` does not have the shape (frames, rows, columns)."""

    if values.ndim != 3:
        raise ValueError(
            f"{function} needs pixel impedance with shape (frames, rows, "
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
        pixel_impedance (Signal): The pixel impedance, with the names of the masks
            applied to them in ``metadata["masks"]``.
        values (numpy.ndarray): The pixels of ``pixel_impedance``.
        mask (PixelMask | None): The mask to apply, or None for pixel impedance that
            are masked already.
        function (str): The name of the calling function, for the errors.

    Returns:
        tuple[numpy.ndarray, str]: The masked pixels, and the name of
            ``mask``, or the names of the masks of the pixel impedance joined with
            ``"_"``.

    Raises:
        ValueError: If ``mask`` is None and the pixel impedance is not masked, or if
            the pixel impedance is masked already with a mask of the name of ``mask``.
    """

    applied = list(pixel_impedance.metadata.get(MASKS_KEY, []))
    if mask is None:
        if not applied:
            raise ValueError(
                f"{function}: the pixel impedance {pixel_impedance.name!r} is not "
                "masked. Give the mask, or mask the pixel impedance with "
                "PixelMask.apply first."
            )
        return values, "_".join(applied)
    if mask.name in applied:
        raise ValueError(
            f"{function}: the pixel impedance {pixel_impedance.name!r} is masked "
            f"already with {mask.name!r}. Use {function}(pixel_impedance) without a "
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
    method: str,
) -> Signal:
    """Return the sum of the pixels of each image as a ``Signal``."""

    _check_pixels(pixel_impedance, values, function=function)
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
        method=method,
    )
