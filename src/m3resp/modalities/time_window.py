"""The part of a recording to keep when cutting it to a time window."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeWindow:
    """Which samples of a recording to keep.

    The kept samples run from `start_index` up to, but not including,
    `end_index` (sample numbers in the recording as it was before cutting).
    """

    #: Start and end of the window, in seconds from the recording's first
    #: sample. `end_seconds` is the recording's length when no end was given.
    start_seconds: float
    end_seconds: float
    start_index: int
    end_index: int
    #: Number of samples in the recording before cutting.
    n_samples: int
    #: Time (s) from the recording's first sample to the first kept sample:
    #: how much later the recording starts after cutting.
    shift_seconds: float
    #: Samples per second, when the recording has one fixed rate.
    sample_frequency: float | None = None
