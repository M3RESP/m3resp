"""Plot data of EIT results for the GUI.

Each function returns a dictionary that holds only numbers, text, lists,
dictionaries, True/False and None, so ``json.dumps`` writes it as it is and
the browser draws it:

- :func:`pixel_map_plot_data`: a ``PixelMap`` with its style (colours and
  colour scale).
- :func:`pixel_mask_plot_data`: a ``PixelMask``.
- :func:`eit_image_plot_data`: the EIT image at one time (the dynamic image).
- :func:`eit_images_plot_data`: the EIT images from one time to another, one
  per time step, with one shared colour scale (to play them).
- :func:`lungspace_plot_data`: the mean maps, masks, watershed regions and
  peaks of a lung-space method of :mod:`m3resp.eit.roi`.
- :func:`rate_spectrum_plot_data`: the spectra and rates of
  :func:`~m3resp.eit.processing.detect_rates`.

A grid is a list of rows, each a list of numbers, with None for a NaN pixel.
Colours are red, green, blue (and alpha) from 0 to 1. Times are in seconds
and frequencies in Hz. The colours are those of the PNG figures
(:mod:`m3resp.visualization.pixel_map_style`).

:func:`region_label_positions` gives a point inside each region of a map of
region numbers, where a label goes.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/plotting/pixelmap.py::
                PixelMapPlotting._find_point_inside_region
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - `region_label_positions` gives the point of every region of a map of
      kind "integer", by region number (eitprocessing finds the point of one
      region inside `add_region_markers`). Region 0, the background, gets
      no point (eitprocessing labels it).
    - Docstrings are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage

from m3resp.data.event_data import IntervalData
from m3resp.data.pixel_maps import PixelMap, PixelMask
from m3resp.data.signals import Signal
from m3resp.visualization.pixel_map_style import (
    PixelMapStyle,
    normalized_values,
    pixel_map_style,
    scale_limits,
    style_of,
)

#: Capture names of the lung-space methods, by the part of the result they
#: go to.
_LUNGSPACE_MAPS = (
    "mean TIV",
    "mean amplitude",
    "mean TIV of shared breaths",
    "mean amplitude of shared breaths",
    "included watershed regions",
)
_LUNGSPACE_MASKS = (
    "functional tiv mask",
    "functional amplitude mask",
    "included region",
)
_LUNGSPACE_REGIONS = ("watershed regions", "regions")
_LUNGSPACE_POINTS = ("local peaks", "included peaks", "excluded peaks")
_LUNGSPACE_NUMBERS = ("included marker indices", "region sizes", "kept regions")
_LUNGSPACE_PER_BREATH = ("TIV per breath", "amplitude per breath")


def pixel_map_plot_data(
    pixel_map: PixelMap,
    *,
    style: PixelMapStyle | None = None,
    normalize: bool | None = None,
) -> dict[str, Any]:
    """Return a pixel map and its style as plot data.

    Args:
        pixel_map (PixelMap): The map.
        style (PixelMapStyle | None): The style. None takes the style of the
            map kind.
        normalize (bool | None): Whether the values are divided by the
            largest value (or by 1 when the largest value is below 1), as in
            the figure. None takes the value of the style.

    Returns:
        dict[str, Any]: ``name``, ``kind``, ``unit``, ``category``,
            ``shape`` (rows, columns), ``values`` (the grid, divided when
            ``normalize``), ``normalized`` and ``style`` (see
            :func:`style_plot_data`).

    Raises:
        TypeError: If ``pixel_map`` is not a ``PixelMap``.
        ValueError: If a value is infinite.
    """

    if not isinstance(pixel_map, PixelMap):
        raise TypeError(
            f"pixel_map_plot_data needs a PixelMap, got {type(pixel_map).__name__}."
        )
    style = style_of(pixel_map) if style is None else style
    normalize = style.normalize if normalize is None else normalize
    values = normalized_values(pixel_map.values) if normalize else pixel_map.values
    return {
        "name": pixel_map.name,
        "kind": pixel_map.kind,
        "unit": pixel_map.unit,
        "category": pixel_map.category,
        "shape": list(pixel_map.shape),
        "values": grid_plot_data(values, owner=pixel_map.name),
        "normalized": bool(normalize),
        "style": style_plot_data(style, values),
    }


def pixel_mask_plot_data(mask: PixelMask) -> dict[str, Any]:
    """Return a pixel mask as plot data.

    Args:
        mask (PixelMask): The mask.

    Returns:
        dict[str, Any]: ``name``, ``shape``, ``values`` (1 or a weight for
            the pixels in the mask, None elsewhere), ``included_pixel_count``,
            ``included_pixel_fraction`` (of all pixels), ``is_weighted`` and
            ``style`` (the mask style).

    Raises:
        TypeError: If ``mask`` is not a ``PixelMask``.
    """

    if not isinstance(mask, PixelMask):
        raise TypeError(
            f"pixel_mask_plot_data needs a PixelMask, got {type(mask).__name__}."
        )
    count = mask.included_pixel_count
    return {
        "name": mask.name,
        "shape": list(mask.shape),
        "values": grid_plot_data(mask.values, owner=mask.name),
        "included_pixel_count": int(count),
        "included_pixel_fraction": float(count / mask.values.size),
        "is_weighted": bool(mask.is_weighted),
        "style": style_plot_data(pixel_map_style("mask"), mask.values),
    }


def eit_image_plot_data(
    pixels: Signal, time: float, *, kind: str | None = None
) -> dict[str, Any]:
    """Return the EIT image at the sample nearest to a time.

    Args:
        pixels (Signal): EIT images, values with shape (time, row, column),
            with time in seconds.
        time (float): The time in seconds, between the first and the last
            sample. Of two samples at the same distance, the first is taken.
        kind (str | None): The style of the image (a ``PixelMap.kind``, or
            None for a plain map).

    Returns:
        dict[str, Any]: ``name``, ``unit``, ``time`` (of the sample),
            ``sample`` (its number, from 0), ``shape``, ``values`` and
            ``style`` with the scale of this image.

    Raises:
        ValueError: If ``pixels`` does not hold images, the time axis does
            not rise, ``time`` lies outside the time axis, or a value is
            infinite.
    """

    function = "eit_image_plot_data"
    images, sample_times = _images(pixels, function=function)
    sample = int(_nearest_samples(sample_times, [time], function=function)[0])
    image = np.asarray(images[sample], dtype=float)
    return {
        "name": pixels.name,
        "unit": pixels.unit,
        "time": float(sample_times[sample]),
        "sample": int(sample),
        "shape": list(image.shape),
        "values": grid_plot_data(image, owner=str(pixels.name)),
        "style": style_plot_data(pixel_map_style(kind), image),
    }


def eit_images_plot_data(
    pixels: Signal,
    start: float,
    end: float,
    step: float,
    *,
    kind: str | None = None,
) -> dict[str, Any]:
    """Return the EIT images from ``start`` to ``end``, one per ``step``.

    The requested times are ``start``, ``start + step``, ... up to ``end``;
    each gives the image at the nearest sample. Of two samples at the same
    distance, the first is taken. All images share one colour
    scale, set by the values of all of them.

    Args:
        pixels (Signal): EIT images, values with shape (time, row, column),
            with time in seconds.
        start (float): First time in seconds.
        end (float): Last time in seconds, at or after ``start``.
        step (float): Time between the images in seconds, above 0.
        kind (str | None): The style of the images (a ``PixelMap.kind``, or
            None for a plain map).

    Returns:
        dict[str, Any]: ``name``, ``unit``, ``shape``, ``requested_times``,
            ``times`` and ``samples`` (time and number of each sample),
            ``images`` (one grid per time) and ``style``.

    Raises:
        ValueError: If ``pixels`` does not hold images, the time axis does
            not rise, ``step`` is not above 0, ``start`` is after ``end``, a
            time lies outside the time axis, or a value is infinite.
    """

    function = "eit_images_plot_data"
    if not step > 0:
        raise ValueError(f"{function} needs step > 0, got {step}.")
    if start > end:
        raise ValueError(f"{function} needs start <= end, got {start} and {end}.")
    images, sample_times = _images(pixels, function=function)
    count = int(np.floor((end - start) / step + 1e-9)) + 1
    # The rounding of start + step * k can put the last time just after end.
    requested = np.minimum(start + step * np.arange(count), end)
    samples = _nearest_samples(sample_times, requested, function=function)
    chosen = np.asarray(images[samples], dtype=float)
    return {
        "name": pixels.name,
        "unit": pixels.unit,
        "shape": list(chosen.shape[1:]),
        "requested_times": [float(value) for value in requested],
        "times": [float(sample_times[sample]) for sample in samples],
        "samples": [int(sample) for sample in samples],
        "images": [grid_plot_data(image, owner=str(pixels.name)) for image in chosen],
        "style": style_plot_data(pixel_map_style(kind), chosen),
    }


def lungspace_plot_data(
    captures: Mapping[str, Any], *, mask: PixelMask | None = None
) -> dict[str, Any]:
    """Return the captures of a lung-space method as plot data.

    The captures are those of :func:`~m3resp.eit.roi.tiv_lungspace`,
    :func:`~m3resp.eit.roi.amplitude_lungspace`,
    :func:`~m3resp.eit.roi.watershed_lungspace` or
    :func:`~m3resp.eit.roi.filter_mask_by_size`. The maps per breath
    (``"TIV per breath"``, ``"amplitude per breath"``) give only their
    number of breaths.

    Args:
        captures (Mapping[str, Any]): The captures.
        mask (PixelMask | None): The lung-space mask that the method
            returned, to add to the result.

    Returns:
        dict[str, Any]: ``maps`` (the mean maps and the included
            watershed regions, see :func:`pixel_map_plot_data`), ``masks`` (see
            :func:`pixel_mask_plot_data`), ``regions`` (maps of region
            numbers, each with ``label_positions``), ``points`` (row and
            column of each peak), ``numbers`` (lists of numbers, such as
            ``"included marker indices"``), ``breath_counts`` (number of
            breaths of each per-breath capture) and ``mask`` (or None).

    Raises:
        ValueError: If a capture name is unknown.
        TypeError: If a capture does not have the type of its name.
    """

    result: dict[str, Any] = {
        "maps": {},
        "masks": {},
        "regions": {},
        "points": {},
        "numbers": {},
        "breath_counts": {},
        "mask": None if mask is None else pixel_mask_plot_data(mask),
    }
    for key, value in captures.items():
        if key in _LUNGSPACE_MAPS:
            result["maps"][key] = pixel_map_plot_data(_checked(key, value, PixelMap))
        elif key in _LUNGSPACE_MASKS:
            result["masks"][key] = pixel_mask_plot_data(_checked(key, value, PixelMask))
        elif key in _LUNGSPACE_REGIONS:
            regions = _checked(key, value, PixelMap)
            data = pixel_map_plot_data(regions)
            data["label_positions"] = [
                {"region": number, "row": row, "column": column}
                for number, (row, column) in region_label_positions(regions).items()
            ]
            result["regions"][key] = data
        elif key in _LUNGSPACE_POINTS:
            points = np.asarray(value)
            result["points"][key] = [[int(row), int(col)] for row, col in points]
        elif key in _LUNGSPACE_NUMBERS:
            result["numbers"][key] = _number_list(value, owner=key)
        elif key in _LUNGSPACE_PER_BREATH:
            result["breath_counts"][key] = len(
                _checked(key, value, IntervalData).intervals
            )
        else:
            known = (
                _LUNGSPACE_MAPS
                + _LUNGSPACE_MASKS
                + _LUNGSPACE_REGIONS
                + _LUNGSPACE_POINTS
                + _LUNGSPACE_NUMBERS
                + _LUNGSPACE_PER_BREATH
            )
            raise ValueError(
                f"lungspace_plot_data does not know the capture {key!r}; "
                f"known captures: {', '.join(known)}."
            )
    return result


def rate_spectrum_plot_data(
    captures: Mapping[str, Any], settings: Any = None
) -> dict[str, Any]:
    """Return the spectra and the rates of the rate detection as plot data.

    Args:
        captures (Mapping[str, Any]): The captures of
            :func:`~m3resp.eit.processing.detect_rates`.
        settings (RateDetectionSettings | None): The settings the rates were
            found with, for the rate ranges. None gives no ranges.

    Returns:
        dict[str, Any]: ``frequencies`` (Hz), ``normalized_total_power``,
            ``average_normalized_pixel_power`` and
            ``diff_total_averaged_power`` (one number per frequency),
            ``respiratory_rate`` and ``heart_rate`` (Hz),
            ``respiratory_rate_per_minute`` and ``heart_rate_per_minute``,
            and ``respiratory_rate_range`` and ``heart_rate_range`` (lowest
            and highest rate in Hz, or None).

    Raises:
        KeyError: If a capture is missing.
        ValueError: If a value is infinite.
    """

    respiratory_rate = float(captures["estimated_respiratory_rate"])
    heart_rate = float(captures["estimated_heart_rate"])
    data: dict[str, Any] = {
        "frequency_unit": "Hz",
        "frequencies": _number_list(captures["frequencies"], owner="frequencies"),
    }
    for key in (
        "normalized_total_power",
        "average_normalized_pixel_power",
        "diff_total_averaged_power",
    ):
        data[key] = _number_list(captures[key], owner=key)
    data.update(
        {
            "respiratory_rate": respiratory_rate,
            "heart_rate": heart_rate,
            "respiratory_rate_per_minute": respiratory_rate * 60,
            "heart_rate_per_minute": heart_rate * 60,
            "respiratory_rate_range": _range(settings, "respiratory"),
            "heart_rate_range": _range(settings, "heart"),
        }
    )
    return data


def style_plot_data(style: PixelMapStyle, values: Any) -> dict[str, Any]:
    """Return a style, with the scale ends for some values, as plot data.

    Args:
        style (PixelMapStyle): The style.
        values (Any): The values that are drawn with it.

    Returns:
        dict[str, Any]: The fields of :meth:`PixelMapStyle.to_dict` and
            ``limits``: the low end and the high end of the colour scale
            (see :func:`~m3resp.visualization.pixel_map_style.scale_limits`).
    """

    data = style.to_dict()
    data["limits"] = list(scale_limits(style, values))
    return data


def grid_plot_data(values: Any, *, owner: str) -> list:
    """Return a grid of numbers as nested lists, with None for NaN.

    Args:
        values (Any): A grid of numbers.
        owner (str): Name of the data, for the error message.

    Returns:
        list: The grid as lists of Python numbers and None.

    Raises:
        ValueError: If a value is infinite.
    """

    grid = np.asarray(values, dtype=float)
    if np.isinf(grid).any():
        raise ValueError(f"{owner!r} holds an infinite value, which JSON can not hold.")
    as_objects = grid.astype(object)
    as_objects[np.isnan(grid)] = None
    return as_objects.tolist()


def region_label_positions(regions: PixelMap) -> dict[int, tuple[float, float]]:
    """Return a point inside each region, where its label goes.

    The point is the centre of mass of the region. A region made of separate
    parts uses its largest part. When the centre of mass lies outside the
    region after one erosion (one pixel removed from its edge), the point is
    the pixel of the eroded region farthest from its edge. A region of one
    or two pixels wide, which erosion removes, uses its middle pixel.

    Region 0 is the background, the pixels outside all regions (as in the
    captures ``"watershed regions"`` and ``"regions"``), and gets no point.

    Args:
        regions (PixelMap): A map of whole numbers, one per region (kind
            ``"integer"``).

    Returns:
        dict[int, tuple[float, float]]: Row and column of the point of each
            region number other than 0, in rising order.

    Raises:
        TypeError: If ``regions`` is not a ``PixelMap`` of kind ``"integer"``.
    """

    if not isinstance(regions, PixelMap) or regions.kind != "integer":
        raise TypeError("region_label_positions needs a PixelMap of kind 'integer'.")
    numbers = regions.to_integer_array()
    return {
        int(number): _point_inside_region(numbers == number)
        for number in np.unique(numbers)
        if number != 0
    }


def _point_inside_region(region: np.ndarray) -> tuple[float, float]:
    """Row and column of a point inside a region of true pixels."""

    def middle_pixel(part: np.ndarray) -> tuple[float, float]:
        rows, columns = np.where(part)
        middle = len(rows) // 2
        return float(rows[middle]), float(columns[middle])

    labeled, count = ndimage.label(region)
    if count == 0:
        return middle_pixel(region)
    if count > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        region = labeled == np.argmax(sizes)

    eroded = ndimage.binary_erosion(region)
    if not eroded.any():
        return middle_pixel(region)

    row, column = ndimage.center_of_mass(region)
    near_row, near_column = round(row), round(column)
    if (
        0 <= near_row < eroded.shape[0]
        and 0 <= near_column < eroded.shape[1]
        and eroded[near_row, near_column]
    ):
        return float(row), float(column)

    distance = ndimage.distance_transform_edt(eroded)
    far_row, far_column = np.unravel_index(np.argmax(distance), distance.shape)
    return float(far_row), float(far_column)


def _images(pixels: Signal, *, function: str) -> tuple[np.ndarray, np.ndarray]:
    """The (time, row, column) values and the rising time axis of EIT images.

    The values keep their number type; the caller turns the chosen images
    into floats.
    """

    if not isinstance(pixels, Signal):
        raise TypeError(f"{function} needs a Signal, got {type(pixels).__name__}.")
    images = np.asarray(pixels.values)
    times = np.asarray(pixels.time, dtype=float)
    if images.ndim != 3 or images.shape[0] == 0:
        raise ValueError(
            f"{function} needs EIT images with shape (time, row, column), got "
            f"shape {images.shape}."
        )
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{function} needs a time axis that rises at every sample.")
    return images, times


def _nearest_samples(times: np.ndarray, requested: Any, *, function: str) -> np.ndarray:
    """Numbers of the samples nearest to times inside a rising time axis.

    Of two samples at the same distance, the first is taken.
    """

    requested = np.asarray(requested, dtype=float)
    inside = (times[0] <= requested) & (requested <= times[-1])
    if not np.all(inside):
        time = requested[~inside][0]
        raise ValueError(
            f"{function}: time {time} s lies outside the data "
            f"({times[0]} s to {times[-1]} s)."
        )
    after = np.searchsorted(times, requested)
    before = np.maximum(after - 1, 0)
    take_before = (after > 0) & (requested - times[before] <= times[after] - requested)
    return np.where(take_before, before, after)


def _checked(key: str, value: Any, kind: type) -> Any:
    """The capture, after a check of its type."""

    if not isinstance(value, kind):
        raise TypeError(
            f"The capture {key!r} must be a {kind.__name__}, got "
            f"{type(value).__name__}."
        )
    return value


def _number_list(values: Any, *, owner: str) -> list:
    """A list of numbers, with None for NaN."""

    array = np.asarray(values)
    if array.dtype.kind in "iub":
        return array.tolist()
    return grid_plot_data(array, owner=owner)


def _range(settings: Any, rate: str) -> list[float] | None:
    """Lowest and highest rate in Hz of the settings, or None."""

    if settings is None:
        return None
    return [
        float(getattr(settings, f"min_{rate}_rate")),
        float(getattr(settings, f"max_{rate}_rate")),
    ]
