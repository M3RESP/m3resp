"""The minute image: the mean pixel TIV over the breaths in a time window.

``minute_image`` computes the tidal impedance variation (TIV) of each pixel
for each breath that lies fully inside the window, with the times of the
global breaths for every pixel, and returns the mean over these breaths. The
Prepare page of the GUI shows it next to the EIT image at the cursor time.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from m3resp.data.events import BreathEvent
from m3resp.data.pixel_maps import PixelMap
from m3resp.data.signals import Signal
from m3resp.eit.processing.tidal_impedance_variation import (
    pixel_tidal_impedance_variation,
)

METHOD = "m3resp.eit.processing.minute_image"


def minute_image(
    pixels: Signal,
    breaths: Sequence[BreathEvent],
    *,
    start: float | None = None,
    end: float | None = None,
    tiv_method: str = "inspiratory",
    name: str = "minute_image",
) -> PixelMap:
    """Return the mean pixel TIV over the breaths inside a time window.

    A breath is inside the window when its start time is at or after
    ``start`` and its end time is at or before ``end``. A breath whose TIV is
    NaN in every pixel (for example a breath over a gap in the time axis)
    does not count. A pixel that is NaN in every breath stays NaN.

    Args:
        pixels (Signal): EIT images, values with shape (time, row, column),
            with time in seconds.
        breaths (Sequence[BreathEvent]): The global breaths, with times on
            the time axis of ``pixels``, for example the breaths of the
            global impedance.
        start (float | None): Start of the window in seconds. None is the
            first sample of ``pixels``.
        end (float | None): End of the window in seconds. None is the last
            sample of ``pixels``.
        tiv_method (str): ``"inspiratory"``, ``"expiratory"`` or ``"mean"``
            (see :func:`~m3resp.eit.processing.pixel_tidal_impedance_variation`).
        name (str): Name of the map.

    Returns:
        PixelMap: The mean TIV of each pixel, of kind ``"tiv"``, in the unit
            of ``pixels``. ``metadata`` holds ``start``, ``end`` (seconds),
            ``breath_count`` (the breaths in the mean) and ``tiv_method``.

    Raises:
        ValueError: If ``pixels`` has no samples, ``start`` is after ``end``,
            no breath lies inside the window, or every breath inside has NaN
            TIV in all pixels; also the errors of
            :func:`~m3resp.eit.processing.pixel_tidal_impedance_variation`.
        TypeError: If a breath is not a ``BreathEvent``.
    """

    time = np.asarray(pixels.time, dtype=float)
    if time.size == 0:
        raise ValueError("minute_image needs EIT images with at least one sample.")
    start = float(time[0]) if start is None else float(start)
    end = float(time[-1]) if end is None else float(end)
    if start > end:
        raise ValueError(
            f"minute_image needs start <= end, got start={start} and end={end}."
        )
    for breath in breaths:
        if not isinstance(breath, BreathEvent):
            raise TypeError(
                f"minute_image needs BreathEvent breaths, got {type(breath).__name__}."
            )
    inside = [
        breath
        for breath in breaths
        if breath.start_time >= start and breath.end_time <= end
    ]
    if not inside:
        raise ValueError(
            f"No breath lies fully inside the window {start} s to {end} s, so "
            "minute_image has no TIV to average."
        )
    per_breath = pixel_tidal_impedance_variation(pixels, inside, method=tiv_method)
    maps = [
        pixel_map
        for pixel_map in per_breath.values or []
        if not np.all(np.isnan(pixel_map.values))
    ]
    if not maps:
        raise ValueError(
            f"Every breath inside the window {start} s to {end} s has NaN TIV in "
            "all pixels, so minute_image has no TIV to average."
        )
    first = maps[0]
    return PixelMap.from_aggregate(
        maps,
        np.nanmean,
        name=name,
        modality=first.modality,
        category=first.category,
        unit=first.unit,
        method=METHOD,
        kind="tiv",
        metadata={
            "start": start,
            "end": end,
            "breath_count": len(maps),
            "tiv_method": tiv_method,
        },
    )
