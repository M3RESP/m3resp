"""Synchronization helpers for multimodal recordings."""

from m3resp.synchronization.alignment import (
    align_events_by_modality_offset,
    align_events_manual_offset,
    compute_offsets_from_timestamps,
)
from m3resp.synchronization.linking import link_breaths_by_time
from m3resp.synchronization.multimodal_parameters import (
    compute_breath_duration_difference,
    compute_event_agreement,
    compute_multimodal_parameters,
    compute_timing_delay,
)
from m3resp.synchronization.offset_estimation import (
    SyncOffsetResult,
    estimate_sync_offset,
)
from m3resp.synchronization.resampling import resample_signal
from m3resp.synchronization.timebase import Timebase
from m3resp.synchronization.ventilator import (
    iter_ventilator_detections,
    normalize_ventilator_breath,
)

__all__ = [
    "SyncOffsetResult",
    "Timebase",
    "align_events_by_modality_offset",
    "align_events_manual_offset",
    "compute_breath_duration_difference",
    "compute_event_agreement",
    "compute_multimodal_parameters",
    "compute_offsets_from_timestamps",
    "compute_timing_delay",
    "estimate_sync_offset",
    "iter_ventilator_detections",
    "link_breaths_by_time",
    "normalize_ventilator_breath",
    "resample_signal",
]
