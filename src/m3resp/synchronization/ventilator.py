"""Ventilator breath detection normalization into common `BreathEvent`s."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from m3resp.core.events import BreathEvent, Event, coerce_breath_event


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
) -> BreathEvent:
    if isinstance(detection, BreathEvent):
        return replace(detection, modality="vent")
    if isinstance(detection, Mapping):
        breath = coerce_breath_event(detection, modality="vent", source="ventilator")
        return replace(breath, modality="vent")

    if hasattr(detection, "start_time") and hasattr(detection, "end_time"):
        breath = coerce_breath_event(detection, modality="vent", source="ventilator")
        return replace(breath, modality="vent")

    if fs is None:
        raise ValueError(
            "Ventilator breath indices require a ventilator sampling rate. "
            "Pass ventilator_fs or include metadata['fs'] in the ventilator input."
        )

    sample_index = int(detection)
    peak_time = sample_index / float(fs)
    half_width = width_seconds / 2
    return BreathEvent(
        modality="vent",
        start_time=max(0.0, peak_time - half_width),
        end_time=peak_time + half_width,
        peak_time=peak_time,
        source="resurfemg.detect_ventilator_breath",
        metadata={
            "sample_index": sample_index,
            "fs": float(fs),
            "width_seconds": width_seconds,
        },
    )


def _infer_ventilator_fs(
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
