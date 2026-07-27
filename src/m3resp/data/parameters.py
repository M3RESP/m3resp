"""``ParameterResult``: a computed respiratory metric (plan_stage2.md Sec 11).

Covers both scalar metrics (EIT TIV, EMG amplitude, respiratory rate) and
array-valued ones (regional ventilation maps), computed by a pipeline step
from one or more source signals/breaths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from m3resp.data.categories import normalize_category
from m3resp.data.metrics import normalize_metric_name
from m3resp.data.units import normalize_unit


@dataclass
class ParameterResult:
    """A named, unit-tagged metric produced by a processing step.

    A parameter can be scoped to whichever of these apply (all are
    optional and independent, so combinations - e.g. one metric per
    breath within a time period - are possible):

    - a single breath (``breath_id``);
    - multiple breaths, e.g. a metric computed over a rolling window of
      breaths (``breath_ids``);
    - a single timepoint (``start_time`` set, ``end_time`` left ``None``);
    - a time period, e.g. during an intervention or every 30 seconds
      (``start_time`` and ``end_time`` both set);
    - a specific ``m3resp.core.events.Event`` (``event_id``), e.g. a
      blood-gas draw used for a P/F ratio, or a labeled intervention like a
      Baydur maneuver;
    - the whole signal, when none of the above are set.

    ``name`` stays a free-form, human-readable label. ``metric_type`` is the
    standardized metric this instance represents, used to group/compare across
    runs regardless of what a step happened to call it: it's auto-derived from
    ``name`` via :func:`m3resp.data.metrics.normalize_metric_name` when left
    unset, and stays ``None`` for a custom name that isn't a known standard
    metric (so nothing is mislabelled). Pass ``metric_type`` explicitly to
    override the derivation.

    ``modality`` names the device/technique this metric came from; ``category``
    names the physical quantity it is derived from (see
    :mod:`m3resp.data.categories`). They are independent axes - a Pocc
    pressure-time product is ``modality="ventilator"``,
    ``category="airway_pressure"``.

    ``unit`` is normalized via :func:`m3resp.data.units.normalize_unit`.
    """

    name: str
    value: float | np.ndarray
    modality: str
    category: str | None = None
    unit: str | None = None
    metric_type: str | None = None
    breath_id: str | None = None
    breath_ids: list[str] | None = None
    event_id: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    region: str | None = None
    channel: str | None = None
    method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.unit = normalize_unit(self.unit)
        self.category = normalize_category(self.category) or self.category
        if self.metric_type is None:
            self.metric_type = normalize_metric_name(self.name)

    def __eq__(self, other: object) -> bool:
        # The dataclass-generated __eq__ compares `value` with plain `==`,
        # which raises ValueError ("truth value of an array with more than
        # one element is ambiguous") for array-valued parameters - use
        # np.array_equal instead, which collapses to a single bool.
        if type(other) is not type(self):
            return NotImplemented
        return (
            self.name == other.name
            and np.array_equal(self.value, other.value)
            and self.modality == other.modality
            and self.category == other.category
            and self.unit == other.unit
            and self.metric_type == other.metric_type
            and self.breath_id == other.breath_id
            and self.breath_ids == other.breath_ids
            and self.event_id == other.event_id
            and self.start_time == other.start_time
            and self.end_time == other.end_time
            and self.region == other.region
            and self.channel == other.channel
            and self.method == other.method
            and self.metadata == other.metadata
        )

    @property
    def is_scalar(self) -> bool:
        """Whether :attr:`value` is a single number rather than an array."""

        return np.ndim(self.value) == 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (``value`` becomes a list if array)."""

        value = self.value
        serialized_value = (
            float(value) if self.is_scalar else np.asarray(value).tolist()
        )
        return {
            "name": self.name,
            "value": serialized_value,
            "modality": self.modality,
            "category": self.category,
            "unit": self.unit,
            "metric_type": self.metric_type,
            "breath_id": self.breath_id,
            "breath_ids": self.breath_ids,
            "event_id": self.event_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "region": self.region,
            "channel": self.channel,
            "method": self.method,
            "metadata": dict(self.metadata),
        }
