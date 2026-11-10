"""EIT breaths in the forms that the EIT steps take and give.

``breath_intervals_to_breath_events`` and ``given_breath_events`` return
breaths as a list of ``BreathEvent`` objects, ``breaths_to_interval_data`` holds
EIT breaths in an ``IntervalData``, and ``pixel_sign_breaths`` gives the
breaths that set the sign of each pixel in the pixel breath search.
"""

from __future__ import annotations

from typing import Any

from m3resp.data import IntervalData, Signal
from m3resp.data.events import BreathEvent, coerce_breath_events
from m3resp.eit.processing._breath_samples import check_breaths
from m3resp.eit.processing.breath_detection import (
    BreathDetectionSettings,
    find_breaths,
)


def breath_intervals_to_breath_events(breath_intervals: Any) -> list[BreathEvent]:
    """Return detected EIT breaths as ``BreathEvent`` objects.

    Args:
        breath_intervals (Any): The breaths: an ``IntervalData`` of
            ``BreathEvent`` objects (``eit.detect_breaths``) or a list of
            ``BreathEvent`` objects. Times are in seconds.

    Returns:
        list[BreathEvent]: EIT breaths in input order, on the time axis of
            the input.

    Raises:
        TypeError: If ``breath_intervals`` is not an ``IntervalData`` or a
            list.
    """

    if isinstance(breath_intervals, IntervalData):
        return coerce_breath_events(breath_intervals.intervals, modality="eit")
    if isinstance(breath_intervals, list):
        return coerce_breath_events(breath_intervals, modality="eit")
    raise TypeError(
        "EIT breaths must be an m3resp IntervalData of BreathEvents or a list "
        f"of BreathEvents, got {_type_name(breath_intervals)}."
    )


def given_breath_events(breaths: Any, *, function: str) -> list[BreathEvent]:
    """Return the breaths given to a TIV or EELI calculation as ``BreathEvent`` objects.

    Args:
        breaths (Any): A list or tuple of ``BreathEvent`` objects of any modality,
            or an ``IntervalData`` of them. Times are in seconds.
        function (str): Name of the calling function or step, for the message.

    Returns:
        list[BreathEvent]: The breaths in input order.

    Raises:
        TypeError: If ``breaths`` is not a list, tuple or ``IntervalData``,
            or if one of its items is not a ``BreathEvent``.
    """

    if isinstance(breaths, IntervalData):
        return check_breaths(breaths.intervals, function=function)
    if isinstance(breaths, (list, tuple)):
        return check_breaths(breaths, function=function)
    raise TypeError(
        f"{function} needs a list of BreathEvent objects or an m3resp "
        f"IntervalData of them, got {_type_name(breaths)}."
    )


def _type_name(value: Any) -> str:
    """Return the module and the name of the type of ``value``, for messages."""

    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def breaths_to_interval_data(
    breaths: list[BreathEvent], *, name: str = "eit_breaths"
) -> IntervalData:
    """Return EIT breaths as an ``IntervalData`` without values.

    Args:
        breaths (list[BreathEvent]): The breaths, in time order.
        name (str): The name of the result.

    Returns:
        IntervalData: The breaths in ``intervals``, modality ``"eit"``,
            category ``"breath"`` and method
            ``"m3resp.eit.processing.find_breaths"``.
    """

    return IntervalData(
        name=name,
        modality="eit",
        intervals=list(breaths),
        category="breath",
        method="m3resp.eit.processing.find_breaths",
    )


def pixel_sign_breaths(signal: Signal, detector: Any) -> list[BreathEvent] | None:
    """Return the breaths that give the sign of each pixel in the pixel breath search.

    eitprocessing 1.8.7 finds these breaths on the waveform with the default
    settings of the breath detection, whatever the settings of the global
    breaths. The result goes to ``sign_breaths`` of
    ``m3resp.eit.processing.find_pixel_breaths``, so the pixel breaths are
    the same as in eitprocessing.

    Args:
        signal (Signal): The waveform the global breaths were found in.
        detector (Any): The settings of the global breaths
            (``BreathDetectionSettings``), or an eitprocessing
            ``BreathDetection``.

    Returns:
        list[BreathEvent] | None: The breaths of ``signal`` found with the
            default settings. None when ``detector`` holds the default
            settings, because the global breaths are then the same breaths.
    """

    default = BreathDetectionSettings()
    if detector == default:
        return None
    return find_breaths(signal, default)
