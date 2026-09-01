"""Multimodal timing/agreement metrics computed from `LinkedBreath` objects.

Deliberately narrow, initial metrics rather than the full "coupling metric"
list: a signed timing delay between two modalities' breath anchors, a breath
duration difference, and a breath-to-breath event-agreement fraction. All
three are pure functions over `LinkedBreath`/`list[LinkedBreath]`;
`compute_multimodal_parameters` is the convenience entry point that turns
them into `ParameterResult`s, and is what `M3Session.compute_multimodal_parameters`
 calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from m3resp.core.events import BreathEvent
from m3resp.data.linked_breath import LinkedBreath
from m3resp.data.parameters import ParameterResult


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
    compare (``"start"``, ``"peak"``, or ``"end"``). Returns `None` when
    either modality did not contribute a breath to this link, or when
    ``anchor="peak"`` and either breath has no `peak_time`.
    """

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
    """Fraction of `linked_breaths` where every modality in `modalities`
    contributed a breath - a coarse breath-to-breath timing agreement score.

    Returns ``0.0`` for an empty `linked_breaths` (no evidence of
    agreement, rather than an undefined ``0/0``).
    """

    if not linked_breaths:
        return 0.0
    required = set(modalities)
    matched = sum(1 for linked in linked_breaths if required <= set(linked.breaths))
    return matched / len(linked_breaths)


def compute_multimodal_parameters(
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
    for the breaths that do have both modalities.
    """

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
                    method=f"timing_delay[{anchor}]",
                    metadata={
                        "from_modality": from_modality,
                        "to_modality": to_modality,
                        "confidence": linked.confidence,
                    },
                )
            )
        results.append(
            ParameterResult(
                name=f"{from_modality}_{to_modality}_event_agreement",
                value=compute_event_agreement(
                    linked_breaths, (from_modality, to_modality)
                ),
                modality="multimodal",
                method="event_agreement",
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
                    method="breath_duration_difference",
                    metadata={"modality_a": modality_a, "modality_b": modality_b},
                )
            )

    return results


def _anchor_time(breath: BreathEvent, anchor: str) -> float | None:
    if anchor == "start":
        return breath.start_time
    if anchor == "end":
        return breath.end_time
    if anchor == "peak":
        return breath.peak_time
    raise ValueError(f"anchor ({anchor!r}) must be one of 'start', 'peak', 'end'")
