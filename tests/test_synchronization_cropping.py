"""Tests for `m3resp.synchronization.cropping`, split out of `core/session.py`
in the raw-modality synchronization refactor. Covers the module boundary
`M3Session.synchronize_raw_modalities` relies on: offset resolution/
normalization and in-place cropping of loaded EIT/EMG/ventilator recordings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from m3resp.core.session import M3Session
from m3resp.modalities.emg import EMGRecording
from m3resp.synchronization.cropping import (
    crop_loaded_modality,
    normalize_modality,
    offsets_relative_to_reference,
    raw_synchronization_traces,
    resolve_alignment_offsets,
)


class TestNormalizeModality:
    @pytest.mark.parametrize(
        "modality, expected",
        [
            ("EMG", "emg"),
            ("eit", "eit"),
            ("ventilator", "ventilator"),
            ("Ventilation", "ventilator"),
            ("vent", "ventilator"),
        ],
    )
    def test_normalizes_case_and_ventilator_aliases(self, modality, expected):
        assert normalize_modality(modality) == expected


class TestResolveAlignmentOffsets:
    def test_scalar_offset_applies_only_to_emg(self):
        offsets = resolve_alignment_offsets(1.5)
        assert offsets == {"eit": 0.0, "emg": 1.5, "ventilator": 0.0}

    def test_mapping_offset_normalizes_modality_keys(self):
        offsets = resolve_alignment_offsets({"EIT": 0.5, "Ventilation": -0.25})
        assert offsets == {"eit": 0.5, "emg": 0.0, "ventilator": -0.25}


class TestOffsetsRelativeToReference:
    def test_subtracts_reference_offset_from_every_modality(self):
        offsets = offsets_relative_to_reference(
            {"eit": 1.0, "emg": 1.5, "ventilator": 0.5}, reference_modality="eit"
        )
        assert offsets == {"eit": 0.0, "emg": 0.5, "ventilator": -0.5}


class TestCropLoadedModality:
    def test_zero_offset_is_a_no_op(self):
        session = M3Session()
        assert crop_loaded_modality(session, "emg", 0.0) == 0

    def test_crops_emg_recording_in_place_and_returns_sample_count(self):
        session = M3Session()
        array = np.arange(100.0).reshape(1, 100)
        session.emg = EMGRecording(
            data={"array": array, "metadata": {"fs": 10.0}},
            path=Path("synthetic.npy"),
        )

        n_samples = crop_loaded_modality(session, "emg", offset=1.0)

        assert n_samples == 10
        assert session.emg.data["array"].shape[1] == 90
        assert session.emg.raw is session.emg.data["array"]

    def test_rejects_an_offset_that_would_remove_every_emg_sample(self):
        session = M3Session()
        array = np.arange(100.0).reshape(1, 100)
        session.emg = EMGRecording(
            data={"array": array, "metadata": {"fs": 10.0}},
            path=Path("synthetic.npy"),
        )

        with pytest.raises(ValueError, match="would remove every sample"):
            crop_loaded_modality(session, "emg", offset=10.0)

        np.testing.assert_array_equal(session.emg.data["array"], array)

    def test_missing_recording_is_a_no_op(self):
        session = M3Session()
        assert crop_loaded_modality(session, "eit", 1.0) == 0


class TestSynchronizeRawModalities:
    def test_rejects_an_oversized_offset_without_recording_alignment(self):
        session = M3Session()
        session.emg = EMGRecording(
            data={
                "array": np.arange(100.0).reshape(1, 100),
                "metadata": {"fs": 10.0},
            },
            path=Path("synthetic.npy"),
        )

        with pytest.raises(ValueError, match="would remove every sample"):
            session.synchronize_raw_modalities(offset_seconds={"emg": 10.0})

        assert "raw_alignment" not in session.parameters
        assert "raw_synchronization" not in session.processed


class TestRawSynchronizationTraces:
    def test_no_loaded_modality_returns_empty_traces(self):
        session = M3Session()
        assert raw_synchronization_traces(session, "emg") == {}

    def test_emg_trace_includes_time_and_values(self):
        session = M3Session()
        array = np.array([[0.0, 1.0, 2.0, 3.0]])
        session.emg = EMGRecording(
            data={
                "array": array,
                "metadata": {"fs": 2.0, "labels": ["ch0"], "units": ["mV"]},
            },
            path=Path("synthetic.npy"),
        )

        traces = raw_synchronization_traces(session, "emg")

        assert traces["emg"]["values"] == [0.0, 1.0, 2.0, 3.0]
        assert traces["emg"]["time"] == [0.0, 0.5, 1.0, 1.5]
