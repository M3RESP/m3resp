"""``ParameterResult``: a computed respiratory metric (plan_stage2.md Sec 11).

Covers both scalar metrics (EIT TIV, EMG amplitude, respiratory rate) and
array-valued ones (regional ventilation maps), computed by a pipeline step
from one or more source signals/breaths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
    - the whole signal, when none of the above are set.
    """

    name: str
    value: float | np.ndarray
    modality: str
    unit: str | None = None
    breath_id: str | None = None
    breath_ids: list[str] | None = None
    start_time: float | None = None
    end_time: float | None = None
    region: str | None = None
    channel: str | None = None
    method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
            "unit": self.unit,
            "breath_id": self.breath_id,
            "breath_ids": self.breath_ids,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "region": self.region,
            "channel": self.channel,
            "method": self.method,
            "metadata": dict(self.metadata),
        }
