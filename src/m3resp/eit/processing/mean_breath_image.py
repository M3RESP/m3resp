"""The mean pixel TIV or pixel amplitude over the breaths in a time window.

``mean_breath_image`` computes the tidal impedance variation (TIV) or the
amplitude of each pixel for each breath that lies fully inside the window,
and returns the mean over these breaths. With a window of 60 s
(``duration=60``) the result is the minute image. The Prepare page of the GUI
shows it next to the EIT image at the cursor time.

TIV is the usual measure during controlled ventilation, and amplitude during
assisted ventilation. The TIV uses the times of the global breaths for every
pixel. The amplitude uses the breaths of each pixel, found in the "phase
shift" mode, as the amplitude lung space does.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from m3resp.data.events import BreathEvent
from m3resp.data.pixel_breaths import PIXEL_BREATH_POINTS, PixelBreaths
from m3resp.data.pixel_maps import PixelMap
from m3resp.data.signals import Signal
from m3resp.eit.processing.pixel_breath import find_pixel_breaths
from m3resp.eit.processing.tidal_impedance_variation import (
    pixel_tidal_impedance_variation,
)

METHOD = "m3resp.eit.processing.mean_breath_image"
#: The two quantities that ``mean_breath_image`` averages.
MEAN_BREATH_QUANTITIES = ("tiv", "amplitude")


def mean_breath_image(
    pixels: Signal,
    *,
    breaths: Sequence[BreathEvent],
    quantity: Literal["tiv", "amplitude"] = "tiv",
    start: float | None = None,
    end: float | None = None,
    duration: float | None = None,
    signal: Signal | None = None,
    tiv_method: str = "inspiratory",
    name: str | None = None,
) -> PixelMap:
    """Return the mean pixel TIV or pixel amplitude over the breaths in a window.

    A breath is inside the window when its start time is at or after
    ``start`` and its end time is at or before the end of the window. A
    breath whose value is NaN in every pixel (for example a breath over a
    gap in the time axis) does not count. A pixel that is NaN in every
    breath stays NaN.

    The window ends at ``end``, or at ``start + duration``. For example,
    ``start=120.0, duration=60.0`` gives the minute image from 120 s to
    180 s.

    Args:
        pixels (Signal): EIT images, values with shape (time, row, column),
            with time in seconds.
        breaths (Sequence[BreathEvent]): The global breaths, with times on
            the time axis of ``pixels``, for example the breaths of the
            global impedance.
        quantity (str): ``"tiv"`` for the TIV of each pixel with the times
            of the global breaths, or ``"amplitude"`` for the TIV of each
            pixel with the times of its own breaths ("phase shift" mode, see
            :func:`~m3resp.eit.processing.find_pixel_breaths`).
        start (float | None): Start of the window in seconds. None is the
            first sample of ``pixels``.
        end (float | None): End of the window in seconds. None is the last
            sample of ``pixels``, unless ``duration`` is set.
        duration (float | None): Length of the window in seconds, from
            ``start``. Give ``end`` or ``duration``, not both.
        signal (Signal | None): For ``quantity="amplitude"`` only: the
            waveform of the breaths, for example the global impedance. None
            uses the sum of all pixels of each image.
        tiv_method (str): ``"inspiratory"``, ``"expiratory"`` or ``"mean"``
            (see :func:`~m3resp.eit.processing.pixel_tidal_impedance_variation`),
            for both quantities.
        name (str | None): Name of the map. None gives ``"mean_breath_tiv"``
            or ``"mean_breath_amplitude"``.

    Returns:
        PixelMap: The mean value of each pixel, of kind ``"tiv"`` or
            ``"amplitude"``, in the unit of ``pixels``. ``metadata`` holds
            ``quantity``, ``start``, ``end`` (seconds), ``breath_count``
            (the breaths in the mean) and ``tiv_method``.

    Raises:
        ValueError: If ``quantity`` is unknown, ``pixels`` has no samples,
            both ``end`` and ``duration`` are set, ``duration`` is negative,
            ``start`` is after the end of the window, ``signal`` is set with
            ``quantity="tiv"``, ``signal`` is None and the images are masked,
            no breath lies inside the window, or every breath inside has a NaN
            value in all pixels; also the errors of
            :func:`~m3resp.eit.processing.pixel_tidal_impedance_variation`
            and :func:`~m3resp.eit.processing.find_pixel_breaths`.
        TypeError: If a breath is not a ``BreathEvent``.

    Warns:
        UserWarning: For ``quantity="amplitude"``, if a mean amplitude is
            negative.
        RuntimeWarning: For ``quantity="amplitude"``, the warnings of the
            pixel breath search.
    """

    if quantity not in MEAN_BREATH_QUANTITIES:
        raise ValueError(
            f"mean_breath_image: quantity must be one of {MEAN_BREATH_QUANTITIES}, "
            f"got {quantity!r}."
        )
    if signal is not None and quantity == "tiv":
        raise ValueError(
            "mean_breath_image: signal is used only with quantity='amplitude'; "
            "the TIV uses the times of the global breaths."
        )
    start, end = _window(pixels, start=start, end=end, duration=duration)
    for breath in breaths:
        if not isinstance(breath, BreathEvent):
            raise TypeError(
                "mean_breath_image needs BreathEvent breaths, got "
                f"{type(breath).__name__}."
            )
    inside = [
        index
        for index, breath in enumerate(breaths)
        if breath.start_time >= start and breath.end_time <= end
    ]
    if not inside:
        raise ValueError(
            f"No breath lies fully inside the window {start} s to {end} s, so "
            "mean_breath_image has no value to average."
        )

    if quantity == "tiv":
        timing: Sequence[BreathEvent] | PixelBreaths = [breaths[i] for i in inside]
    else:
        timing = _pixel_breaths_inside(pixels, signal, breaths, inside)
    per_breath = pixel_tidal_impedance_variation(
        pixels, breaths=timing, method=tiv_method
    )
    maps = [
        pixel_map
        for pixel_map in per_breath.values or []
        if not np.all(np.isnan(pixel_map.values))
    ]
    if not maps:
        raise ValueError(
            f"Every breath inside the window {start} s to {end} s has a NaN "
            f"{quantity} in all pixels, so mean_breath_image has no value to "
            "average."
        )
    first = maps[0]
    return PixelMap.from_aggregate(
        maps,
        np.nanmean,
        name=f"mean_breath_{quantity}" if name is None else name,
        modality=first.modality,
        category=first.category,
        unit=first.unit,
        method=METHOD,
        kind=quantity,
        metadata={
            "quantity": quantity,
            "start": start,
            "end": end,
            "breath_count": len(maps),
            "tiv_method": tiv_method,
        },
    )


def _window(
    pixels: Signal,
    *,
    start: float | None,
    end: float | None,
    duration: float | None,
) -> tuple[float, float]:
    """Return the start and the end of the window, in seconds."""

    time = np.asarray(pixels.time, dtype=float)
    if time.size == 0:
        raise ValueError("mean_breath_image needs EIT images with at least one sample.")
    if end is not None and duration is not None:
        raise ValueError(
            "mean_breath_image: give end or duration, not both; "
            f"got end={end} and duration={duration}."
        )
    start = float(time[0]) if start is None else float(start)
    if duration is not None:
        if float(duration) < 0:
            raise ValueError(
                f"mean_breath_image needs duration >= 0, got duration={duration}."
            )
        end = start + float(duration)
    end = float(time[-1]) if end is None else float(end)
    if start > end:
        raise ValueError(
            f"mean_breath_image needs start <= end, got start={start} and end={end}."
        )
    return start, end


def _pixel_breaths_inside(
    pixels: Signal,
    signal: Signal | None,
    breaths: Sequence[BreathEvent],
    inside: list[int],
) -> PixelBreaths:
    """Return the pixel breaths ("phase shift" mode) of the breaths inside.

    The pixel breaths are found for all global breaths, as in the amplitude
    lung space, and then the breaths inside the window are kept.
    """

    from m3resp.eit.roi._shared import breath_waveform

    signal = breath_waveform(pixels, signal=signal, function="mean_breath_image")
    found = find_pixel_breaths(
        pixels,
        signal=signal,
        breaths=breaths,
        phase_correction_mode="phase shift",
    )
    return PixelBreaths(
        name=found.name,
        breaths=[found.breaths[i] for i in inside],
        **{point: getattr(found, point)[inside] for point in PIXEL_BREATH_POINTS},
        modality=found.modality,
        method=found.method,
        metadata=dict(found.metadata),
    )
