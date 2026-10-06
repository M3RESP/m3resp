"""Link breaths across modalities by their nearest representative times.

Align breath times to a shared time axis before linking them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from m3resp.data.events import BreathEvent
from m3resp.data.linked_breath import LinkedBreath


def link_breaths_by_time(
    breaths_by_modality: Mapping[str, Sequence[BreathEvent] | None] | None = None,
    *,
    time_tolerance: float = 0.5,
) -> list[LinkedBreath]:
    """Group breaths across modalities by their timing in seconds.

    Each breath is used once. Its representative time is ``extremum_time``
    when available, otherwise the midpoint of start/end. Modalities are
    visited in mapping order, with each modality's breaths sorted by this
    time. For each unassigned breath, the closest unused breath from each
    other modality is added if it falls within time_tolerance of that
    breath. For equal distances, the first match in sorted order is used.

    Args:
        breaths_by_modality: Modality names mapped to breath sequences or None.
            Breath times must share a time axis. None gives an empty result.
        time_tolerance: Maximum time difference in seconds from the breath
            that starts a group; must be at least zero.

    Returns:
        list[LinkedBreath]: Groups ordered by the representative time of their
            first breath, containing the original breath objects. Unmatched
            breaths form single-modality groups with confidence=None. Matched
            groups have confidence ``1 - largest_time_difference/tolerance``;
            exact matches with zero tolerance have confidence=1.

    Raises:
        ValueError: If time_tolerance is negative.
    """

    if time_tolerance < 0:
        raise ValueError("time_tolerance must not be negative")

    modalities = list(breaths_by_modality or {})
    sorted_breaths = {
        modality: sorted(
            (breaths_by_modality or {}).get(modality) or [], key=_anchor_time
        )
        for modality in modalities
    }
    used: dict[str, set[int]] = {modality: set() for modality in modalities}

    linked: list[LinkedBreath] = []
    for anchor_modality in modalities:
        for index, breath in enumerate(sorted_breaths[anchor_modality]):
            if index in used[anchor_modality]:
                continue
            used[anchor_modality].add(index)
            group = {anchor_modality: breath}
            anchor_time = _anchor_time(breath)
            max_time_diff = 0.0

            for other_modality in modalities:
                if other_modality == anchor_modality:
                    continue
                match_index, time_diff = _nearest_unused(
                    sorted_breaths[other_modality],
                    used[other_modality],
                    anchor_time,
                    time_tolerance,
                )
                if match_index is not None:
                    used[other_modality].add(match_index)
                    group[other_modality] = sorted_breaths[other_modality][match_index]
                    max_time_diff = max(max_time_diff, time_diff)

            confidence = (
                max(0.0, 1.0 - max_time_diff / time_tolerance)
                if len(group) > 1 and time_tolerance > 0
                else (1.0 if len(group) > 1 else None)
            )
            linked.append(
                LinkedBreath(
                    breaths=group,
                    time_tolerance=time_tolerance,
                    confidence=confidence,
                )
            )

    return sorted(linked, key=_linked_breath_anchor_time)


def _anchor_time(breath: BreathEvent) -> float:
    """Return the extremum time, or the start/end midpoint, in seconds."""

    if breath.extremum_time is not None:
        return breath.extremum_time
    return (breath.start_time + breath.end_time) / 2.0


def _nearest_unused(
    breaths: Sequence[BreathEvent],
    used_indices: set[int],
    anchor_time: float,
    tolerance: float,
) -> tuple[int | None, float]:
    best_index: int | None = None
    best_diff: float | None = None
    for index, breath in enumerate(breaths):
        if index in used_indices:
            continue
        diff = abs(_anchor_time(breath) - anchor_time)
        if diff > tolerance:
            continue
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_index = index
    return best_index, (best_diff if best_diff is not None else 0.0)


def _linked_breath_anchor_time(linked: LinkedBreath) -> float:
    if not linked.breaths:
        raise ValueError("LinkedBreath has no breath in any modality slot")
    return _anchor_time(next(iter(linked.breaths.values())))
