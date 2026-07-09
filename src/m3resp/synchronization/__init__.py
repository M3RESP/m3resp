"""Synchronization helpers for multimodal recordings."""

from m3resp.synchronization.alignment import (
    align_events_by_modality_offset,
    align_events_manual_offset,
    compute_offsets_from_timestamps,
)
from m3resp.synchronization.linking import link_breaths_by_time
from m3resp.synchronization.offset_estimation import (
    CrossCorrelationOffsetResult,
    InterferenceOffsetResult,
    SyncOffsetResult,
    estimate_offset_from_interference,
    estimate_offset_from_interference_signal,
    estimate_sync_offset,
    interference_power,
    refine_offset_by_crosscorrelation,
    refine_offset_by_crosscorrelation_signals,
)
from m3resp.synchronization.resampling import resample_signal
from m3resp.synchronization.timebase import Timebase

__all__ = [
    "CrossCorrelationOffsetResult",
    "InterferenceOffsetResult",
    "SyncOffsetResult",
    "Timebase",
    "align_events_by_modality_offset",
    "align_events_manual_offset",
    "compute_offsets_from_timestamps",
    "estimate_offset_from_interference",
    "estimate_offset_from_interference_signal",
    "estimate_sync_offset",
    "interference_power",
    "link_breaths_by_time",
    "refine_offset_by_crosscorrelation",
    "refine_offset_by_crosscorrelation_signals",
    "resample_signal",
]
