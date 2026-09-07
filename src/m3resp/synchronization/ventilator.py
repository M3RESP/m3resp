"""Ventilator breath detection normalization into common `BreathEvent`s."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from m3resp.core.events import BreathEvent, Event, coerce_breath_event
from m3resp.synchronization.cropping import VENTILATOR


def iter_ventilator_detections(detections: Any) -> list[Any]:
    if isinstance(detections, (BreathEvent, Event, Mapping)):
        return [detections]
    if hasattr(detections, "tolist"):
        detections = detections.tolist()
    return list(detections)


def normalize_ventilator_breath(
    detection: Any,
    *,
    fs: float | None,
    width_seconds: float,
    duration_seconds: float | None = None,
) -> BreathEvent:
    """Turn one ventilator breath detection into a `BreathEvent`.

    A detection given as a sample index becomes a window of `width_seconds`
    centred on that sample. `duration_seconds` is the length of the ventilator
    recording: when it is known, a breath detected near the end of the
    recording has its window trimmed so it does not run past the data.
    """

    if isinstance(detection, BreathEvent):
        return replace(detection, modality=VENTILATOR)
    if isinstance(detection, Mapping):
        breath = coerce_breath_event(
            detection, modality=VENTILATOR, source="ventilator"
        )
        return replace(breath, modality=VENTILATOR)

    if hasattr(detection, "start_time") and hasattr(detection, "end_time"):
        breath = coerce_breath_event(
            detection, modality=VENTILATOR, source="ventilator"
        )
        return replace(breath, modality=VENTILATOR)

    if fs is None:
        raise ValueError(
            "Ventilator breath indices require a ventilator sampling rate. "
            "Pass ventilator_fs or include metadata['fs'] in the ventilator input."
        )

    sample_index = int(detection)
    peak_time = sample_index / float(fs)
    half_width = width_seconds / 2
    start_time = max(0.0, peak_time - half_width)
    end_time = peak_time + half_width
    if duration_seconds is not None:
        end_time = max(start_time, min(end_time, float(duration_seconds)))
    return BreathEvent(
        modality=VENTILATOR,
        start_time=start_time,
        end_time=end_time,
        peak_time=peak_time,
        source="resurfemg.detect_ventilator_breath",
        metadata={
            "sample_index": sample_index,
            "fs": float(fs),
            "width_seconds": width_seconds,
        },
    )


def infer_ventilator_fs(
    ventilator: Any | None,
    ventilator_fs: float | None,
) -> float | None:
    if ventilator_fs is not None:
        return float(ventilator_fs)
    if isinstance(ventilator, Mapping):
        metadata = ventilator.get("metadata", {})
        if isinstance(metadata, Mapping) and metadata.get("fs") is not None:
            return float(metadata["fs"])
    return None


def infer_ventilator_duration(
    ventilator: Any | None,
    fs: float | None,
) -> float | None:
    """Length of a ventilator recording in seconds, or `None` if unknown."""

    if fs is None or not fs:
        return None
    array = ventilator.get("array") if isinstance(ventilator, Mapping) else ventilator
    if array is None:
        return None
    shape = np.asarray(array).shape
    if not shape:
        return None
    # Same convention as the cropping helpers: channels first unless the
    # array is stored one row per sample.
    axis = 1 if len(shape) > 1 and shape[1] >= shape[0] else 0
    n_samples = shape[axis]
    return n_samples / float(fs) if n_samples else None
