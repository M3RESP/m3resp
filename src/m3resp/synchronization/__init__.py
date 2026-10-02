"""Synchronization helpers for multimodal recordings."""

# Turning ventilator detections into breaths is not synchronization; these two
# live in `m3resp.adapters.ventilator_adapter` and are imported here only so
# older code that imports them from `m3resp.synchronization` keeps working.
from m3resp.adapters.ventilator_adapter import (
    iter_ventilator_detections,
    normalize_ventilator_breath,
)
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

__all__ = [
    "SyncOffsetResult",
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
