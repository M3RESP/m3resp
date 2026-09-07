"""Tests for `m3resp.synchronization.ventilator`, split out of `core/session.py`
in the raw-modality synchronization refactor. Covers normalizing ventilator
breath detections (sample indices, mappings, and already-native BreathEvents)
into common `BreathEvent`s.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.core.events import BreathEvent
from m3resp.synchronization.ventilator import (
    infer_ventilator_duration,
    infer_ventilator_fs,
    iter_ventilator_detections,
    normalize_ventilator_breath,
)


class TestIterVentilatorDetections:
    def test_wraps_a_single_breath_event(self):
        breath = BreathEvent(modality="vent", start_time=0.0, end_time=1.0)
        assert iter_ventilator_detections(breath) == [breath]

    def test_wraps_a_single_mapping(self):
        detection = {"start_time": 0.0, "end_time": 1.0}
        assert iter_ventilator_detections(detection) == [detection]

    def test_converts_numpy_array_to_a_list(self):
        detections = iter_ventilator_detections(np.array([1, 2, 3]))
        assert detections == [1, 2, 3]

    def test_passes_through_a_plain_list(self):
        assert iter_ventilator_detections([1, 2, 3]) == [1, 2, 3]


class TestNormalizeVentilatorBreath:
    def test_breath_event_is_retagged_with_vent_modality(self):
        breath = BreathEvent(modality="eit", start_time=0.0, end_time=1.0)
        normalized = normalize_ventilator_breath(breath, fs=None, width_seconds=0.5)
        assert normalized.modality == "ventilator"
        assert normalized.start_time == 0.0

    def test_sample_index_derives_start_end_peak_from_fs(self):
        normalized = normalize_ventilator_breath(100, fs=10.0, width_seconds=0.4)

        assert normalized.modality == "ventilator"
        assert normalized.peak_time == pytest.approx(10.0)
        assert normalized.start_time == pytest.approx(9.8)
        assert normalized.end_time == pytest.approx(10.2)
        assert normalized.metadata["sample_index"] == 100

    def test_sample_index_without_fs_raises(self):
        with pytest.raises(ValueError, match="sampling rate"):
            normalize_ventilator_breath(100, fs=None, width_seconds=0.5)

    def test_window_is_trimmed_at_the_end_of_the_recording(self):
        # Breath detected 0.1 s before the recording stops, with a 0.4 s
        # window: the second half of the window has no data behind it.
        normalized = normalize_ventilator_breath(
            100, fs=10.0, width_seconds=0.4, duration_seconds=10.1
        )

        assert normalized.start_time == pytest.approx(9.8)
        assert normalized.end_time == pytest.approx(10.1)

    def test_window_is_untouched_well_inside_the_recording(self):
        normalized = normalize_ventilator_breath(
            100, fs=10.0, width_seconds=0.4, duration_seconds=60.0
        )

        assert normalized.end_time == pytest.approx(10.2)


class TestInferVentilatorDuration:
    def test_duration_uses_the_sample_axis_of_a_channel_array(self):
        ventilator = {"metadata": {"fs": 100.0}, "array": [[0.0] * 300]}
        assert infer_ventilator_duration(ventilator, 100.0) == pytest.approx(3.0)

    def test_duration_of_a_single_channel_trace(self):
        assert infer_ventilator_duration([0.0] * 50, 10.0) == pytest.approx(5.0)

    def test_duration_is_unknown_without_a_rate_or_an_array(self):
        assert infer_ventilator_duration({"array": [[0.0]]}, None) is None
        assert infer_ventilator_duration(None, 100.0) is None


class TestInferVentilatorFs:
    def test_explicit_fs_wins(self):
        assert (
            infer_ventilator_fs({"metadata": {"fs": 10.0}}, ventilator_fs=20.0) == 20.0
        )

    def test_falls_back_to_ventilator_metadata(self):
        assert (
            infer_ventilator_fs({"metadata": {"fs": 10.0}}, ventilator_fs=None) == 10.0
        )

    def test_returns_none_when_unavailable(self):
        assert infer_ventilator_fs(None, ventilator_fs=None) is None
