"""Breath timing metrics computed from `LinkedBreath` objects.

Deliberately narrow, initial metrics rather than the full "coupling metric"
list: a signed timing delay between two modalities' breath anchors, a breath
duration difference, and a breath-to-breath event-agreement fraction. All
three are pure functions over `LinkedBreath`/`list[LinkedBreath]`;
`compute_breath_timing_parameters` turns them into `ParameterResult`s, and
is what `M3Session.compute_breath_timing_parameters` calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from m3resp.data.events import BreathEvent
from m3resp.data.linked_breath import LinkedBreath
from m3resp.data.parameters import ParameterResult

_ANCHORS = ("start", "extremum", "end")
_TIMING_DELAY_METHOD = "timing_delay"
_EVENT_AGREEMENT_METHOD = "event_agreement"
_DURATION_DIFFERENCE_METHOD = "breath_duration_difference"


def compute_timing_delay(
    linked: LinkedBreath,
    from_modality: str,
    to_modality: str,
    *,
    anchor: str = "start",
) -> float | None:
    """Signed delay in seconds from `from_modality` to `to_modality`.

    Positive means `to_modality`'s breath anchor occurs after
    `from_modality`'s. `anchor` selects which point on each breath to
    compare (``"start"``, ``"extremum"``, or ``"end"``). Returns `None` when
    either modality did not contribute a breath to this link, or when
    ``anchor="extremum"`` and either breath has no `extremum_time`.

    Raises:
        ValueError: If `anchor` is not one of the three names above.
    """

    _check_anchor(anchor)
    from_breath = linked.breaths.get(from_modality)
    to_breath = linked.breaths.get(to_modality)
    if from_breath is None or to_breath is None:
        return None
    from_time = _anchor_time(from_breath, anchor)
    to_time = _anchor_time(to_breath, anchor)
    if from_time is None or to_time is None:
        return None
    return to_time - from_time


def compute_breath_duration_difference(
    linked: LinkedBreath, modality_a: str, modality_b: str
) -> float | None:
    """``duration(modality_a) - duration(modality_b)`` in seconds.

    Returns `None` when either modality did not contribute a breath to this
    link.
    """

    breath_a = linked.breaths.get(modality_a)
    breath_b = linked.breaths.get(modality_b)
    if breath_a is None or breath_b is None:
        return None
    return breath_a.duration - breath_b.duration


def compute_event_agreement(
    linked_breaths: Sequence[LinkedBreath], modalities: Sequence[str]
) -> float:
    """Fraction of breaths seen by any of `modalities` that were seen by all
    of them - a coarse breath-to-breath timing agreement score.

    Only linked breaths that hold a breath from at least one of `modalities`
    are counted. A breath seen only by another modality (say, a ventilator
    breath while EIT and EMG are compared) says nothing about whether EIT
    and EMG agree, so it does not lower the score.

    Returns ``0.0`` when no linked breath holds any of `modalities`,
    including for an empty `linked_breaths` (no evidence of agreement,
    rather than an undefined ``0/0``).
    """

    matched, counted = _agreement_counts(linked_breaths, modalities)
    if counted == 0:
        return 0.0
    return matched / counted


def _agreement_counts(
    linked_breaths: Sequence[LinkedBreath], modalities: Sequence[str]
) -> tuple[int, int]:
    """(linked breaths holding all of `modalities`, linked breaths holding at
    least one of them)."""

    required = set(modalities)
    matched = 0
    counted = 0
    for linked in linked_breaths:
        present = required & set(linked.breaths)
        if present:
            counted += 1
            if present == required:
                matched += 1
    return matched, counted


def compute_breath_timing_parameters(
    linked_breaths: Sequence[LinkedBreath],
    *,
    delay_pairs: Sequence[tuple[str, str]] | None = None,
    duration_pairs: Sequence[tuple[str, str]] | None = None,
    anchor: str = "start",
) -> list[ParameterResult]:
    """Turn `linked_breaths` into `ParameterResult`s: a per-breath timing
    delay for each pair in `delay_pairs`, a per-breath duration difference
    for each pair in `duration_pairs`, and one aggregate event-agreement
    result per delay pair.

    `delay_pairs`/`duration_pairs` default to every unordered pair of
    modalities actually observed across `linked_breaths`, so a session that
    only linked EIT and EMG does not get a meaningless ventilator pairing.
    A breath missing either side of a pair is skipped for that pair rather
    than raising, so a partially-linked recording still yields parameters
    for the breaths that do have both modalities. The event-agreement result
    is left out for a pair when no linked breath holds either modality, so
    an empty `linked_breaths` always gives an empty list.

    Results carry ``modality="multimodal"``; `is_breath_timing_result` tells
    them apart from other results.

    Raises:
        ValueError: If `anchor` is not ``"start"``, ``"extremum"`` or ``"end"``.
    """

    _check_anchor(anchor)
    observed_modalities = sorted(
        {m for linked in linked_breaths for m in linked.breaths}
    )
    if delay_pairs is None:
        delay_pairs = list(combinations(observed_modalities, 2))
    if duration_pairs is None:
        duration_pairs = list(combinations(observed_modalities, 2))

    results: list[ParameterResult] = []

    for from_modality, to_modality in delay_pairs:
        for index, linked in enumerate(linked_breaths):
            delay = compute_timing_delay(
                linked, from_modality, to_modality, anchor=anchor
            )
            if delay is None:
                continue
            results.append(
                ParameterResult(
                    name=f"{from_modality}_to_{to_modality}_delay",
                    value=delay,
                    modality="multimodal",
                    unit="s",
                    breath_id=str(index),
                    method=f"{_TIMING_DELAY_METHOD}[{anchor}]",
                    metadata={
                        "from_modality": from_modality,
                        "to_modality": to_modality,
                        "confidence": linked.confidence,
                    },
                )
            )
        matched, counted = _agreement_counts(
            linked_breaths, (from_modality, to_modality)
        )
        if counted == 0:
            continue
        results.append(
            ParameterResult(
                name=f"{from_modality}_{to_modality}_event_agreement",
                value=matched / counted,
                modality="multimodal",
                method=_EVENT_AGREEMENT_METHOD,
                metadata={"modalities": [from_modality, to_modality]},
            )
        )

    for modality_a, modality_b in duration_pairs:
        for index, linked in enumerate(linked_breaths):
            difference = compute_breath_duration_difference(
                linked, modality_a, modality_b
            )
            if difference is None:
                continue
            results.append(
                ParameterResult(
                    name=f"{modality_a}_{modality_b}_duration_difference",
                    value=difference,
                    modality="multimodal",
                    unit="s",
                    breath_id=str(index),
                    method=_DURATION_DIFFERENCE_METHOD,
                    metadata={"modality_a": modality_a, "modality_b": modality_b},
                )
            )

    return results


def is_breath_timing_result(parameter: ParameterResult) -> bool:
    """True when `parameter` was made by `compute_breath_timing_parameters`."""

    if parameter.modality != "multimodal" or parameter.method is None:
        return False
    return parameter.method in (
        _EVENT_AGREEMENT_METHOD,
        _DURATION_DIFFERENCE_METHOD,
    ) or parameter.method.startswith(f"{_TIMING_DELAY_METHOD}[")


def _check_anchor(anchor: str) -> None:
    if anchor not in _ANCHORS:
        raise ValueError(
            f"anchor ({anchor!r}) must be one of 'start', 'extremum', 'end'"
        )


def _anchor_time(breath: BreathEvent, anchor: str) -> float | None:
    if anchor == "start":
        return breath.start_time
    if anchor == "end":
        return breath.end_time
    return breath.extremum_time
