"""``Signal``: the modality-tagged runtime signal type (plan_stage2.md Sec 9).

Adapters convert whatever ``eitprocessing``/``resurfemg`` return into
``Signal`` instances at the public boundary (Milestone 2.3); everything
downstream - session storage, pipeline steps, export - operates on this type
instead of vendor-specific objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from m3resp.data.categories import Category, normalize_category
from m3resp.data.timeseries import TimeSeries

#: Common values for ``Signal.modality`` (data model doc Sec 2.5): the
#: *device/technique* that produced the data. This is an open vocabulary, not
#: an enum - a loader/adapter can use any string here for a new device. These
#: are listed for documentation/IDE-completion convenience and are not enforced
#: at runtime.
#:
#: What the numbers physically *are* is a separate axis: see
#: :data:`m3resp.data.categories.Category` and ``Signal.category``. Physical
#: quantities such as pressure/flow/volume used to appear in this vocabulary
#: too, which made "ventilator device, volume quantity" inexpressible; they now
#: belong in ``category``. The values here line up with Layer 2's
#: ``Device.device_type`` (``m3resp.datamodel.entities``).
_KNOWN_MODALITIES_LITERAL = Literal["eit", "emg", "ventilator", "monitor", "unknown"]
Modality = _KNOWN_MODALITIES_LITERAL | str
KNOWN_MODALITIES = frozenset(get_args(_KNOWN_MODALITIES_LITERAL))

#: Controlled vocabulary for ``Signal.processing_state``:
#:
#: - ``raw``: untouched, straight from the source/loader.
#: - ``filtered``: an intermediate signal-processing step has been applied
#:   (e.g. noise/frequency filtering) - work in progress, not yet the final
#:   signal a downstream parameter computation should use.
#: - ``processed``: the final signal for this channel, ready to compute
#:   parameters/results from.
#: - ``derived``: computed from another signal (e.g. a difference between
#:   two signals, or a transform), rather than a step in that signal's own
#:   raw -> filtered -> processed pipeline. Use ``Signal.derived_from`` to
#:   record which processing_state it was derived from (raw or filtered),
#:   since a channel can have derived signals from either.
#:
#: Multiple differently produced signals can share the same ``channel`` and
#: ``processing_state`` (e.g. two "filtered" variants using different filter
#: methods) - use ``Signal.method`` to tell them apart, not a new state.
ProcessingState = Literal["raw", "filtered", "processed", "derived"]
_VALID_PROCESSING_STATES = frozenset(get_args(ProcessingState))


@dataclass
class Signal(TimeSeries):
    """A :class:`TimeSeries` tagged with modality and provenance context.

    Provenance fields (``source`` vs ``method`` are distinct and often
    confused):

    - ``modality``: *which device/technique produced this* - ``"eit"``,
      ``"emg"``, ``"ventilator"``. Orthogonal to ``category``.
    - ``category``: *what physical quantity these numbers are* - an impedance,
      an airway pressure, a flow. Known aliases are canonicalized via
      :func:`m3resp.data.categories.normalize_category`; an unrecognized value
      is kept as-is, since the vocabulary is open. ``None`` means unspecified.
    - ``channel``: which physical/logical channel this is (e.g. an EMG
      lead name, ``"global_impedance"`` for EIT).
    - ``source``: *where the data came from* - the upstream producer/origin,
      typically the loader or library that emitted it (e.g. ``"resurfemg"``,
      ``"eitprocessing"``). Provenance of origin, not of transformation.
    - ``method``: *what was done to it* - the specific algorithm/function that
      produced this particular signal (e.g.
      ``"resurfemg.postprocessing.moving_baseline"``). This is what
      distinguishes two signals sharing the same ``channel`` and
      ``processing_state`` (e.g. two differently-filtered variants).
    - ``processing_state``: how far along the raw -> filtered -> processed
      pipeline this signal is, or ``"derived"`` if computed from another
      signal. See :data:`ProcessingState`.
    - ``derived_from``: for a ``"derived"`` signal, the ``processing_state``
      it was computed from (e.g. a baseline derived from the ``"processed"``
      envelope has ``derived_from="processed"``). ``None`` otherwise.
    """

    modality: Modality = "unknown"
    category: Category | None = None
    channel: str | None = None
    source: str | None = None
    processing_state: ProcessingState = "raw"
    derived_from: ProcessingState | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        # Canonicalize a known alias; pass an unrecognized value through
        # unchanged, since `Category` is an open vocabulary and there is no
        # separate field preserving the caller's original label.
        self.category = normalize_category(self.category) or self.category
        if self.processing_state not in _VALID_PROCESSING_STATES:
            raise ValueError(
                f"Unknown Signal.processing_state {self.processing_state!r}; "
                f"expected one of {sorted(_VALID_PROCESSING_STATES)}"
            )
        if (
            self.derived_from is not None
            and self.derived_from not in _VALID_PROCESSING_STATES
        ):
            raise ValueError(
                f"Unknown Signal.derived_from {self.derived_from!r}; expected "
                f"one of {sorted(_VALID_PROCESSING_STATES)}"
            )

    def __eq__(self, other: object) -> bool:
        # TimeSeries.__eq__ already handles `values`/`time` array comparison
        # safely (np.array_equal); extend it with Signal's own fields rather
        # than letting the dataclass default reintroduce the ambiguous-array
        # `==` crash for this subclass.
        if type(other) is not type(self):
            return NotImplemented
        base_equal = super().__eq__(other)
        if base_equal is NotImplemented:
            return NotImplemented
        return (
            base_equal
            and self.modality == other.modality
            and self.category == other.category
            and self.channel == other.channel
            and self.source == other.source
            and self.processing_state == other.processing_state
            and self.derived_from == other.derived_from
            and self.method == other.method
        )

    def to_manifest_row(self) -> dict[str, object]:
        row = super().to_manifest_row()
        row.update(
            {
                "modality": self.modality,
                "category": self.category,
                "channel": self.channel,
                "source": self.source,
                "processing_state": self.processing_state,
                "derived_from": self.derived_from,
                "method": self.method,
            }
        )
        return row
