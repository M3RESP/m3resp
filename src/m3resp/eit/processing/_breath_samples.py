"""The samples of an impedance waveform at the start, extremum and end of each breath.

TIV and EELI read the waveform at given breath times. The breaths can come
from another signal or another modality, so a breath time can lie between two
samples. Each time is then read at the nearest sample. A time more than half
a sample before the first sample or after the last sample has no sample. A
gap in the time axis, for example between two joined files, counts as
outside the data in the same way.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from m3resp.data.events import BreathEvent
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

    if not isinstance(signal, Signal):
        raise TypeError(f"{function} needs a Signal, got {type(signal).__name__}.")
    if signal.category is not None and signal.category not in IMPEDANCE_CATEGORIES:
        raise ValueError(
            f"{function} works on impedance data only; {signal.name!r} has the "
            f"category {signal.category!r}."
        )
    values = np.asarray(signal.values)
    if values.ndim != 1:
        raise ValueError(
            f"{function} needs one value per sample; {signal.name!r} has shape "
            f"{values.shape}."
        )
    return values


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


def nearest_samples(time: np.ndarray, times: Sequence[float | None]) -> np.ndarray:
    """Return the index of the nearest sample for each time.

    Args:
        time (np.ndarray): Sample times in seconds, in increasing order.
        times (Sequence[float | None]): Times in seconds on the same time
            axis. None or NaN means the time is unknown.

    Returns:
        np.ndarray: One sample index per time (integers). ``NO_SAMPLE`` for
            an unknown time, for a time more than half a sample before the
            first sample or after the last sample, and for a time in a gap
            more than half a sample from both sides of the gap. A sample is
            the median step between the sample times. A gap is a step
            longer than ``1 + GAP_TOLERANCE_FRAMES`` samples, as for
            ``join_eit_files``. A time exactly halfway between two samples
            gets the later sample.
    """

    sample_time = np.asarray(time, dtype=float)
    wanted = np.array(
        [np.nan if value is None else float(value) for value in times], dtype=float
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
    breaths: list[BreathEvent],
    *,
    points: Sequence[str],
    function: str,
) -> dict[str, np.ndarray]:
    """Return the nearest sample of each breath point, and warn for missing ones.

    Args:
        signal (Signal): The waveform; its ``time`` gives the sample times.
        breaths (list[BreathEvent]): Breaths on the time axis of ``signal``.
        points (Sequence[str]): The breath times to look up:
            ``"start_time"``, ``"extremum_time"`` and/or ``"end_time"``.
        function (str): Name of the calling function, for the warnings.

    Returns:
        dict[str, np.ndarray]: For each point, one sample index per breath.
            When any point of a breath has no sample, every point of that
            breath is ``NO_SAMPLE``.

    Warns:
        UserWarning: Once when breaths lie partly outside the data or in a
            gap of the time axis, and once when breaths have no
            ``extremum_time``. Each warning gives the number of breaths.
    """

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


def breath_source_metadata(breaths: list[BreathEvent]) -> dict[str, Any]:
    """Return the modality and the source of the breaths, for the result metadata.

    Args:
        breaths (list[BreathEvent]): The breaths of the result.

    Returns:
        dict[str, Any]: ``breath_modality`` and ``breath_source``. Each is a
            string when every breath has the same value, a sorted list when
            the breaths differ, and None when there are no breaths.
    """

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
