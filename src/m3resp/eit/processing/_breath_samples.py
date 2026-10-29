"""The samples of impedance data at the start, extremum and end of each breath.

TIV and EELI read the data at given breath times: one set of times for a
waveform, or the times of each pixel (``PixelBreaths``) for EIT pixel data.
The breaths can come from another signal or another modality, so a breath
time can lie between two samples. Each time is then read at the nearest
sample. A time more than half a sample before the first sample or after the
last sample has no sample. A gap in the time axis, for example between two
joined files, counts as outside the data in the same way.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from m3resp.data.event_data import IntervalData
from m3resp.data.events import BreathEvent, Interval
from m3resp.data.pixel_breaths import PixelBreaths
from m3resp.data.pixel_maps import PixelMap
from m3resp.data.signals import Signal
from m3resp.eit.io.join import GAP_TOLERANCE_FRAMES

#: Index used for a breath time that has no sample.
NO_SAMPLE: Final = -1

#: Categories of data that TIV and EELI accept: impedance and the impedance
#: categories of eitprocessing.
IMPEDANCE_CATEGORIES: Final = frozenset(
    ("impedance", "absolute impedance", "relative impedance", "impedance difference")
)


def check_impedance_waveform(signal: Signal, *, function: str) -> np.ndarray:
    """Return the values of an impedance waveform, after the checks.

    Args:
        signal (Signal): One value per sample. Its ``category`` must be
            None or an impedance category (``IMPEDANCE_CATEGORIES``).
        function (str): Name of the calling function, for the messages.

    Returns:
        np.ndarray: The values of ``signal``, one per sample.

    Raises:
        TypeError: If ``signal`` is not a ``Signal``.
        ValueError: If the category is not an impedance category, or the
            values do not have one axis.
    """

    values = _check_impedance(signal, function=function)
    if values.ndim != 1:
        raise ValueError(
            f"{function} needs one value per sample; {signal.name!r} has shape "
            f"{values.shape}."
        )
    return values


def check_pixel_impedance(signal: Signal, *, function: str) -> np.ndarray:
    """Return the values of EIT pixel impedance, after the checks.

    Args:
        signal (Signal): EIT images, values with shape (time, row, column).
            Its ``category`` must be None or an impedance category
            (``IMPEDANCE_CATEGORIES``).
        function (str): Name of the calling function, for the messages.

    Returns:
        np.ndarray: The values of ``signal``, shape (time, row, column).

    Raises:
        TypeError: If ``signal`` is not a ``Signal``.
        ValueError: If the category is not an impedance category, or the
            values do not have the shape (time, row, column).
    """

    values = _check_impedance(signal, function=function)
    if values.ndim != 3:
        raise ValueError(
            f"{function} needs EIT images with shape (time, row, column); "
            f"{signal.name!r} has shape {values.shape}."
        )
    return values


def _check_impedance(signal: Signal, *, function: str) -> np.ndarray:
    """Return the values of ``signal`` after the type and category checks.

    Args:
        signal (Signal): The impedance data. Its ``category`` must be None or
            an impedance category (``IMPEDANCE_CATEGORIES``).
        function (str): Name of the calling function, for the messages.

    Returns:
        np.ndarray: The values of ``signal``, in any shape.

    Raises:
        TypeError: If ``signal`` is not a ``Signal``.
        ValueError: If the category is not an impedance category.
    """

    if not isinstance(signal, Signal):
        raise TypeError(f"{function} needs a Signal, got {type(signal).__name__}.")
    if signal.category is not None and signal.category not in IMPEDANCE_CATEGORIES:
        raise ValueError(
            f"{function} works on impedance data only; {signal.name!r} has the "
            f"category {signal.category!r}."
        )
    return np.asarray(signal.values)


def check_breaths(breaths: Sequence[Any], *, function: str) -> list[BreathEvent]:
    """Return the breaths as a list, after a check of their type.

    Args:
        breaths (Sequence[Any]): The breaths.
        function (str): Name of the calling function, for the message.

    Returns:
        list[BreathEvent]: The same breaths, in the same order.

    Raises:
        TypeError: If an item is not a ``BreathEvent``.
    """

    checked = list(breaths)
    for breath in checked:
        if not isinstance(breath, BreathEvent):
            raise TypeError(
                f"{function} needs BreathEvent objects, got {type(breath).__name__}."
            )
    return checked


def check_pixel_breaths(
    breaths: Sequence[Any] | PixelBreaths, data: np.ndarray, *, function: str
) -> tuple[list[BreathEvent] | PixelBreaths, list[BreathEvent], str]:
    """Return the breaths for a pixel calculation, after the checks.

    Args:
        breaths (Sequence[Any] | PixelBreaths): A list of ``BreathEvent``s
            (the same times for every pixel) or a ``PixelBreaths`` (the
            times of each pixel).
        data (np.ndarray): The EIT images, shape (time, row, column).
        function (str): Name of the calling function, for the messages.

    Returns:
        tuple[list[BreathEvent] | PixelBreaths, list[BreathEvent], str]: The
            breaths to look up (a list, or the ``PixelBreaths``), the global
            breaths of the result, and the timing: ``"continuous"`` for a
            list, ``"pixel"`` for ``PixelBreaths``.

    Raises:
        TypeError: If an item of a list is not a ``BreathEvent``.
        ValueError: If the grid of ``PixelBreaths`` differs from the grid of
            the images.
    """

    if isinstance(breaths, PixelBreaths):
        if breaths.shape != data.shape[1:]:
            raise ValueError(
                f"{function}: the pixel breaths are on a {breaths.shape[0]} x "
                f"{breaths.shape[1]} grid, the images on a {data.shape[1]} x "
                f"{data.shape[2]} grid."
            )
        return breaths, list(breaths.breaths), "pixel"
    checked = check_breaths(breaths, function=function)
    return checked, checked, "continuous"


def nearest_samples(
    time: np.ndarray, times: np.ndarray | Sequence[float | None]
) -> np.ndarray:
    """Return the index of the nearest sample for each time.

    Args:
        time (np.ndarray): Sample times in seconds, in increasing order.
        times (np.ndarray | Sequence[float | None]): Times in seconds on the
            same time axis: an array of any shape, for example (breath,) or
            (breath, row, column), or a list. None or NaN means the time is
            unknown.

    Returns:
        np.ndarray: One sample index per time (integers), in the shape of
            ``times``. ``NO_SAMPLE`` for an unknown time, for a time more
            than half a sample before the first sample or after the last
            sample, and for a time in a gap more than half a sample from both
            sides of the gap. A sample is the median step between the sample
            times. A gap is a step
            longer than ``1 + GAP_TOLERANCE_FRAMES`` samples, as for
            ``join_eit_files``. A time exactly halfway between two samples
            gets the later sample.
    """

    sample_time = np.asarray(time, dtype=float)
    if isinstance(times, np.ndarray):
        wanted = times.astype(float)
    else:
        wanted = np.array(
            [np.nan if value is None else float(value) for value in times],
            dtype=float,
        )
    indices = np.full(wanted.shape, NO_SAMPLE, dtype=int)
    if not sample_time.size or not wanted.size:
        return indices

    half_sample = (
        0.5 * float(np.median(np.diff(sample_time))) if sample_time.size > 1 else 0.0
    )
    inside = (
        np.isfinite(wanted)
        & (wanted >= sample_time[0] - half_sample)
        & (wanted <= sample_time[-1] + half_sample)
    )
    last = sample_time.size - 1
    after = np.clip(np.searchsorted(sample_time, wanted[inside]), 0, last)
    before = np.clip(after - 1, 0, last)
    before_is_nearer = np.abs(wanted[inside] - sample_time[before]) < np.abs(
        sample_time[after] - wanted[inside]
    )
    nearest = np.where(before_is_nearer, before, after)
    in_gap = (
        sample_time[after] - sample_time[before]
        > (1 + GAP_TOLERANCE_FRAMES) * 2 * half_sample
    ) & (np.abs(wanted[inside] - sample_time[nearest]) > half_sample)
    nearest[in_gap] = NO_SAMPLE
    indices[inside] = nearest
    return indices


def breath_samples(
    signal: Signal,
    breaths: list[BreathEvent] | PixelBreaths,
    *,
    points: Sequence[str],
    function: str,
) -> dict[str, np.ndarray]:
    """Return the nearest sample of each breath point, and warn for missing ones.

    Args:
        signal (Signal): The data; its ``time`` gives the sample times.
        breaths (list[BreathEvent] | PixelBreaths): Breaths on the time axis
            of ``signal``: one set of times for all of the data, or the
            breaths of each pixel.
        points (Sequence[str]): The breath times to look up:
            ``"start_time"``, ``"extremum_time"`` and/or ``"end_time"``.
        function (str): Name of the calling function, for the warnings.

    Returns:
        dict[str, np.ndarray]: For each point, one sample index per breath:
            shape (breath,) for a list of breaths, and (breath, row, column)
            for ``PixelBreaths``. When any point of a breath has no sample,
            every point of that breath is ``NO_SAMPLE``. A pixel with no
            breath (NaN times) is ``NO_SAMPLE``.

    Warns:
        UserWarning: Once when breaths lie partly outside the data or in a
            gap of the time axis, and once when breaths have no
            ``extremum_time``. Each warning gives the number of breaths.
            A pixel with no breath gives no warning.
    """

    if isinstance(breaths, PixelBreaths):
        return _pixel_breath_samples(signal, breaths, points=points, function=function)

    time = np.asarray(signal.time)
    samples = {
        point: nearest_samples(time, [getattr(breath, point) for breath in breaths])
        for point in points
    }
    if not breaths:
        return samples

    no_extremum = np.zeros(len(breaths), dtype=bool)
    if "extremum_time" in points:
        no_extremum = np.array([breath.extremum_time is None for breath in breaths])
    missing = np.zeros(len(breaths), dtype=bool)
    for indices in samples.values():
        missing |= indices == NO_SAMPLE
    outside = missing & ~no_extremum

    if outside.any():
        warnings.warn(
            f"{function}: {int(outside.sum())} of {len(breaths)} breaths lie "
            f"partly outside the data of {signal.name!r} or in a gap of its "
            "time axis; their value is NaN.",
            UserWarning,
            stacklevel=3,
        )
    if no_extremum.any():
        warnings.warn(
            f"{function}: {int(no_extremum.sum())} of {len(breaths)} breaths "
            "have no extremum_time; their value is NaN.",
            UserWarning,
            stacklevel=3,
        )
    for indices in samples.values():
        indices[missing] = NO_SAMPLE
    return samples


def _pixel_breath_samples(
    signal: Signal,
    breaths: PixelBreaths,
    *,
    points: Sequence[str],
    function: str,
) -> dict[str, np.ndarray]:
    """Return the nearest sample of each pixel breath time.

    Args:
        signal (Signal): The EIT images; its ``time`` gives the sample times.
        breaths (PixelBreaths): The breaths of each pixel, on the time axis
            of ``signal``.
        points (Sequence[str]): The breath times to look up:
            ``"start_time"``, ``"extremum_time"`` and/or ``"end_time"``.
        function (str): Name of the calling function, for the warning.

    Returns:
        dict[str, np.ndarray]: For each point, one sample index per pixel
            breath, shape (breath, row, column). ``NO_SAMPLE`` for a pixel
            with no breath, and for every point of a pixel breath with one
            point outside the data or in a gap of the time axis.

    Warns:
        UserWarning: Once when pixel breaths lie partly outside the data or
            in a gap of the time axis, with the number of pixel breaths.
    """

    time = np.asarray(signal.time)
    samples = {
        point: nearest_samples(time, getattr(breaths, point)) for point in points
    }
    missing = ~breaths.has_breath
    for indices in samples.values():
        missing |= indices == NO_SAMPLE
    outside = missing & breaths.has_breath

    if outside.any():
        warnings.warn(
            f"{function}: {int(outside.sum())} of {int(breaths.has_breath.sum())} "
            f"pixel breaths lie partly outside the data of {signal.name!r} or in "
            "a gap of its time axis; their value is NaN.",
            UserWarning,
            stacklevel=4,
        )
    for indices in samples.values():
        indices[missing] = NO_SAMPLE
    return samples


def values_at(data: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return the data at the given samples, NaN where there is no sample.

    Args:
        data (np.ndarray): Values with time on the first axis: shape (time,)
            for a waveform, or (time, row, column) for EIT images.
        indices (np.ndarray): Sample indices from ``breath_samples``: shape
            (breath,) for one time per breath, or (breath, row, column) for
            the times of each pixel. ``NO_SAMPLE`` means no sample.

    Returns:
        np.ndarray: Floats. With indices of shape (breath,): shape (breath,)
            plus the other axes of ``data``, so each image at that sample.
            With indices of shape (breath, row, column): shape (breath, row,
            column), each pixel at its own sample. NaN where the index is
            ``NO_SAMPLE``.
    """

    found = indices != NO_SAMPLE
    safe = np.where(found, indices, 0)
    if indices.ndim == 1:
        values = np.asarray(data[safe], dtype=float)
        found = found.reshape(found.shape + (1,) * (data.ndim - 1))
    elif not indices.size:
        values = np.empty(indices.shape)
    else:
        values = np.take_along_axis(np.asarray(data, dtype=float), safe, axis=0)
    return np.where(found, values, np.nan)


def pixel_maps_per_breath(
    values: np.ndarray,
    pixels: Signal,
    breaths: list[BreathEvent],
    *,
    name: str,
    method: str,
    metadata: dict[str, Any],
) -> IntervalData:
    """Return one ``PixelMap`` per breath, as an ``IntervalData``.

    Args:
        values (np.ndarray): One grid per breath, shape (breath, row,
            column).
        pixels (Signal): The EIT images the values come from; they give the
            modality and the unit.
        breaths (list[BreathEvent]): The global breaths, one per grid.
        name (str): Name of the result and of each ``PixelMap``.
        method (str): Name of the calculation.
        metadata (dict[str, Any]): Metadata of the result.

    Returns:
        IntervalData: The breaths, with one ``PixelMap`` each (category
            ``"impedance"``, unit of ``pixels``).
    """

    maps = [
        PixelMap(
            name=name,
            values=grid,
            modality=pixels.modality,
            category="impedance",
            unit=pixels.unit,
            method=method,
        )
        for grid in values
    ]
    return IntervalData(
        name=name,
        modality=pixels.modality,
        intervals=list[Interval](breaths),
        values=maps,
        category="impedance",
        unit=pixels.unit,
        method=method,
        metadata=metadata,
    )


def breath_source_metadata(
    breaths: list[BreathEvent] | PixelBreaths,
) -> dict[str, Any]:
    """Return the modality and the source of the breaths, for the result metadata.

    Args:
        breaths (list[BreathEvent] | PixelBreaths): The breaths of the
            result. For ``PixelBreaths``, the modality and the method of the
            pixel breaths.

    Returns:
        dict[str, Any]: ``breath_modality`` and ``breath_source``. Each is a
            string when every breath has the same value, a sorted list when
            the breaths differ, and None when there are no breaths.
    """

    if isinstance(breaths, PixelBreaths):
        return {"breath_modality": breaths.modality, "breath_source": breaths.method}

    def one_or_list(values: set[Any]) -> Any:
        if not values:
            return None
        if len(values) == 1:
            return next(iter(values))
        return sorted(values, key=str)

    return {
        "breath_modality": one_or_list({breath.modality for breath in breaths}),
        "breath_source": one_or_list({breath.source for breath in breaths}),
    }
