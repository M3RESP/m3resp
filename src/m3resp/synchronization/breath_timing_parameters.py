"""Breath timing measures from breaths linked across modalities.

These functions compare breath anchors and durations in seconds and count
agreement between detections. Breath times must share a common time axis.
`compute_breath_timing_parameters` collects the measures as `ParameterResult`s.
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
    """Compute the signed delay between two linked breaths, in seconds.

    Args:
        linked (LinkedBreath): Breaths on a shared time axis.
        from_modality (str): Modality whose breath supplies the reference time.
        to_modality (str): Modality whose breath supplies the comparison time.
        anchor (str): Breath point to compare: "start" (default), "extremum",
            or "end".

    Returns:
        float | None: Comparison time minus reference time, in seconds.
            Positive values mean the second modality occurs later. None means
            either breath or its selected anchor is missing.

    Raises:
        ValueError: If the anchor name is invalid, including for a link with
            missing breaths.
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
    """Compute the fraction of linked breaths found by every requested modality.

    The denominator counts links containing at least one requested modality.
    The numerator counts links containing all requested modalities. For EIT
    and EMG, a link containing only a ventilator breath is excluded from both
    counts.

    Args:
        linked_breaths (Sequence[LinkedBreath]): Breaths linked on a shared
            time axis, including links with missing modalities.
        modalities (Sequence[str]): Modalities whose detections to compare.

    Returns:
        float: Agreement from 0 to 1. Returns 0.0 when the denominator is zero,
            including when either input sequence is empty.
    """

    matched, counted = _agreement_counts(linked_breaths, modalities)
    if counted == 0:
        return 0.0
    return matched / counted


def _agreement_counts(
    linked_breaths: Sequence[LinkedBreath], modalities: Sequence[str]
) -> tuple[int, int]:
    """Return counts of links with all and with any requested modalities."""

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
    """Compute timing delays, duration differences, and detection agreement.

    Args:
        linked_breaths (Sequence[LinkedBreath]): Breaths linked on a shared
            time axis. Their position in this sequence becomes each per-breath
            result's string-valued `breath_id`.
        delay_pairs (Sequence[tuple[str, str]] | None): Ordered modality pairs
            (from, to). Delays are to minus from, in seconds. Each pair also
            produces an aggregate event-agreement fraction when at least one
            link contains either modality. None selects all unordered pairs
            of observed modalities in alphabetical order; [] selects none.
        duration_pairs (Sequence[tuple[str, str]] | None): Ordered modality
            pairs (a, b). Differences are duration(a) minus duration(b), in
            seconds. None uses the same default pairs; [] selects none.
        anchor (str): Breath point for delays: "start" (default), "extremum",
            or "end".

    Returns:
        list[ParameterResult]: Results with modality "multimodal". Per-breath
            values have unit "s" and require both breaths; delays also require
            both selected anchors. Agreement counts links containing either
            requested modality and measures the fraction containing both.
            An empty linked-breath sequence returns an empty list.

    Raises:
        ValueError: If the anchor name is invalid, including for empty inputs.
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
    """Return whether a result's modality and method identify breath timing.

    Matches modality "multimodal" with method "event_agreement",
    "breath_duration_difference", or a method starting with "timing_delay[".
    These are the tags used by `compute_breath_timing_parameters`.
    """

    if parameter.modality != "multimodal" or parameter.method is None:
        return False
    return parameter.method in (
        _EVENT_AGREEMENT_METHOD,
        _DURATION_DIFFERENCE_METHOD,
    ) or parameter.method.startswith(f"{_TIMING_DELAY_METHOD}[")


def _check_anchor(anchor: str) -> None:
    """Raise ValueError unless the anchor is start, extremum, or end."""

    if anchor not in _ANCHORS:
        raise ValueError(
            f"anchor ({anchor!r}) must be one of 'start', 'extremum', 'end'"
        )


def _anchor_time(breath: BreathEvent, anchor: str) -> float | None:
    """Return a breath's selected time in seconds for a validated anchor."""

    if anchor == "start":
        return breath.start_time
    if anchor == "end":
        return breath.end_time
    return breath.extremum_time
