"""Shared time-series container (plan_stage2.md Sec 9, Milestone 2.1).

``TimeSeries`` is the base runtime type every continuous signal in m3resp
should be represented as, whatever its modality. It intentionally mirrors
only what EIT, EMG, and ventilator signals actually have in common: values,
a time vector, sampling rate, unit, and free-form metadata. Modality-specific
fields live on :class:`~m3resp.data.signals.Signal`, which subclasses this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TimeSeries:
    """A value array paired with its time axis."""

    values: np.ndarray
    time: np.ndarray
    sample_frequency: float | None = None
    unit: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        self.time = np.asarray(self.time)
        if self.values.shape[0] != self.time.shape[0]:
            raise ValueError(
                "values and time must have the same length along the time "
                f"axis (got {self.values.shape[0]} and {self.time.shape[0]})"
            )

    @property
    def n_samples(self) -> int:
        """Number of samples along the time axis."""

        return int(self.values.shape[0])

    @property
    def duration(self) -> float:
        """Elapsed time covered by :attr:`time`, in the same units as ``time``."""

        if self.time.size == 0:
            return 0.0
        return float(self.time[-1] - self.time[0])

    def to_manifest_row(self) -> dict[str, Any]:
        """Lightweight, JSON-serializable metadata row (no raw sample arrays).

        Matches the data model doc's separation of raw arrays (kept in
        files) from relational/manifest metadata (Sec 8): a
        ``signals_manifest.csv``/``.json`` row should describe a stream, not
        embed it.
        """

        return {
            "name": self.name,
            "unit": self.unit,
            "sample_frequency": self.sample_frequency,
            "n_samples": self.n_samples,
            "duration": self.duration,
            "metadata": dict(self.metadata),
        }
