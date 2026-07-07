"""Nearest-neighbor breath linking across modalities (plan_stage2.md Sec 20,
Milestone 2.5).

Deliberately simple: breaths are matched by how close their representative
times are, greedily and one-to-one, with no clock-drift correction - matching
plan_stage2.md's own guidance ("avoid overbuilding clock-drift correction in
Stage 2 unless needed"). Run `align_events_by_modality_offset` first if the
modalities are not already on a common time axis.
"""

from __future__ import annotations

from collections.abc import Sequence

from m3resp.core.events import BreathEvent
from m3resp.data.linked_breath import LinkedBreath

_MODALITY_SLOTS = ("eit", "emg", "ventilator")


def link_breaths_by_time(
    *,
    eit_breaths: Sequence[BreathEvent] | None = None,
    emg_breaths: Sequence[BreathEvent] | None = None,
    ventilator_breaths: Sequence[BreathEvent] | None = None,
    time_tolerance: float = 0.5,
) -> list[LinkedBreath]:
    """Link breaths from up to three modalities into `LinkedBreath` groups.

    Each breath is assigned to at most one link. Two breaths from different
    modalities are linked when their representative times (`peak_time`,
    falling back to the start/end midpoint) are within `time_tolerance`
    seconds of each other; the closest available match wins. A breath with no
    match in the other modalities still produces a `LinkedBreath` with only
    its own slot filled, so no input breath is silently dropped.
    """

    if time_tolerance < 0:
        raise ValueError("time_tolerance must not be negative")

    breaths_by_modality = {
        "eit": sorted(eit_breaths or [], key=_anchor_time),
        "emg": sorted(emg_breaths or [], key=_anchor_time),
        "ventilator": sorted(ventilator_breaths or [], key=_anchor_time),
    }
    used: dict[str, set[int]] = {modality: set() for modality in breaths_by_modality}

    linked: list[LinkedBreath] = []
    for anchor_modality in _MODALITY_SLOTS:
        for index, breath in enumerate(breaths_by_modality[anchor_modality]):
            if index in used[anchor_modality]:
                continue
            used[anchor_modality].add(index)
            group = {anchor_modality: breath}
            anchor_time = _anchor_time(breath)
            max_time_diff = 0.0

            for other_modality in _MODALITY_SLOTS:
                if other_modality == anchor_modality:
                    continue
                match_index, time_diff = _nearest_unused(
                    breaths_by_modality[other_modality],
                    used[other_modality],
                    anchor_time,
                    time_tolerance,
                )
                if match_index is not None:
                    used[other_modality].add(match_index)
                    group[other_modality] = breaths_by_modality[other_modality][
                        match_index
                    ]
                    max_time_diff = max(max_time_diff, time_diff)

            confidence = (
                max(0.0, 1.0 - max_time_diff / time_tolerance)
                if len(group) > 1 and time_tolerance > 0
                else (1.0 if len(group) > 1 else None)
            )
            linked.append(
                LinkedBreath(
                    eit_breath=group.get("eit"),
                    emg_breath=group.get("emg"),
                    ventilator_breath=group.get("ventilator"),
                    time_tolerance=time_tolerance,
                    confidence=confidence,
                )
            )

    return sorted(linked, key=_linked_breath_anchor_time)


def _anchor_time(breath: BreathEvent) -> float:
    if breath.peak_time is not None:
        return breath.peak_time
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
    breath = linked.eit_breath or linked.emg_breath or linked.ventilator_breath
    if breath is None:
        raise ValueError("LinkedBreath has no breath in any modality slot")
    return _anchor_time(breath)
