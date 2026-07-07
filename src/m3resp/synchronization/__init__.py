"""Synchronization helpers for multimodal recordings."""

from m3resp.synchronization.alignment import (
    align_events_by_modality_offset,
    align_events_manual_offset,
    compute_offsets_from_timestamps,
)
from m3resp.synchronization.linking import link_breaths_by_time
from m3resp.synchronization.resampling import resample_signal
from m3resp.synchronization.timebase import Timebase

__all__ = [
    "Timebase",
    "align_events_by_modality_offset",
    "align_events_manual_offset",
    "compute_offsets_from_timestamps",
    "link_breaths_by_time",
    "resample_signal",
]
