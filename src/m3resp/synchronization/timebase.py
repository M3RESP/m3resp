"""Timebase helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Timebase:
    """Description of a modality time axis."""

    sample_rate_hz: float | None = None
    start_time_seconds: float = 0.0

    def sample_to_seconds(self, sample_index: int) -> float:
        """Convert a sample index to seconds."""

        if self.sample_rate_hz is None:
            raise ValueError("sample_rate_hz is required to convert samples to seconds")
        return self.start_time_seconds + sample_index / self.sample_rate_hz
