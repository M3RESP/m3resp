"""Shared time-series container (plan_stage2.md Sec 9, Milestone 2.1).

``TimeSeries`` is the base runtime type every continuous signal in m3resp
should be represented as, whatever its modality. It intentionally mirrors
only what EIT, EMG, and ventilator signals actually have in common: values,
a time vector, sampling rate, unit, and free-form metadata. Modality-specific
fields live on :class:`~m3resp.data.signals.Signal`, which subclasses this.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from m3resp.data.units import normalize_unit


@dataclass
class TimeSeries:
    """A value array paired with its time axis.

    ``values``'s leading axis must line up with ``time`` (one entry per
    timepoint), but there's no constraint on its remaining dimensions - a 1D
    array (one scalar per timepoint, e.g. global impedance), a 2D array (one
    value per timepoint per channel/region), or higher-dimensional arrays
    (e.g. EIT pixel-impedance frames: time x rows x cols) are all valid.

    ``unit`` is normalized via :func:`m3resp.data.units.normalize_unit` (e.g.
    ``"uV"`` -> ``"µV"``), so equivalent units from different loaders end up
    comparable instead of drifting into incompatible spellings.
    """

    values: np.ndarray
    time: np.ndarray
    sample_frequency: float | None = None
    unit: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        self.time = np.asarray(self.time)
        self.unit = normalize_unit(self.unit)
        if self.time.ndim != 1:
            raise ValueError(
                f"time must be 1-dimensional (one timestamp per sample), got "
                f"shape {self.time.shape}"
            )
        if self.values.shape[0] != self.time.shape[0]:
            raise ValueError(
                "values and time must have the same length along the time "
                f"axis (got {self.values.shape[0]} and {self.time.shape[0]})"
            )
        other_axes_matching_time = [
            axis
            for axis, length in enumerate(self.values.shape[1:], start=1)
            if length == self.time.shape[0]
        ]
        if other_axes_matching_time:
            warnings.warn(
                "values has another axis "
                f"({other_axes_matching_time}) whose length also matches "
                f"len(time) ({self.time.shape[0]}); this is allowed (axis 0 "
                "is always taken as the time axis), but double-check that "
                "the array wasn't transposed, since a mixup here can't be "
                "distinguished from a coincidental shape match.",
                stacklevel=2,
            )

    def __eq__(self, other: object) -> bool:
        # The dataclass-generated __eq__ compares fields with plain `==`,
        # which raises ValueError ("truth value of an array with more than
        # one element is ambiguous") the moment `values`/`time` hold more
        # than one element - np.array_equal compares element-wise and
        # collapses to a single bool instead.
        if type(other) is not type(self):
            return NotImplemented
        return (
            np.array_equal(self.values, other.values)
            and np.array_equal(self.time, other.time)
            and self.sample_frequency == other.sample_frequency
            and self.unit == other.unit
            and self.name == other.name
            and self.metadata == other.metadata
        )

    @property
    def n_samples(self) -> int:
        """Number of samples along the time axis."""

        return int(self.values.shape[0])

    def __len__(self) -> int:
        """Same as :attr:`n_samples`, so ``len(signal)`` works as expected."""

        return self.n_samples

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
