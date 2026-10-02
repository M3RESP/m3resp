"""Modality names and the offsets `M3Session.synchronize_raw_modalities` and
`synchronize_multimodal_breaths` resolve (`m3resp.modalities.names`,
`m3resp.synchronization.alignment`)."""

from __future__ import annotations

import pytest

from m3resp.core.events import BreathEvent
from m3resp.core.session import M3Session
from m3resp.modalities.names import normalize_modality
from m3resp.synchronization.alignment import (
    breath_reference_modality,
    normalize_offset_key,
    offsets_relative_to_reference,
    raw_reference_modality,
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


class TestNormalizeOffsetKey:
    def test_a_modality_key_is_normalized(self):
        assert normalize_offset_key("Vent") == "ventilator"

    def test_a_recording_name_keeps_its_case(self):
        assert normalize_offset_key("VENT:Monitor") == "ventilator:Monitor"


class TestRawReferenceModality:
    def test_the_requested_reference_wins(self):
        session = M3Session()
        session.raw["emg"] = object()
        assert raw_reference_modality(session, "Vent:Monitor") == "ventilator:Monitor"

    @pytest.mark.parametrize(
        "loaded, expected",
        [
            (("eit", "emg", "ventilator"), "ventilator"),
            (("eit", "emg", "vent"), "ventilator"),
            (("eit", "emg"), "eit"),
            (("emg",), "emg"),
            ((), "eit"),
        ],
    )
    def test_without_a_request_the_first_loaded_one_in_order(self, loaded, expected):
        session = M3Session()
        for key in loaded:
            session.raw[key] = object()
        assert raw_reference_modality(session, None) == expected


class TestBreathReferenceModality:
    def test_the_requested_reference_wins(self):
        assert breath_reference_modality(M3Session(), "EMG") == ("emg", None)

    def test_ventilator_breaths_make_the_ventilator_the_reference(self):
        session = M3Session()
        session.add_events("ventilator_breaths", [BreathEvent("ventilator", 1.0, 2.0)])
        assert breath_reference_modality(session, None) == ("ventilator", None)

    def test_without_ventilator_breaths_it_falls_back_to_eit(self):
        assert breath_reference_modality(M3Session(), None) == ("eit", "eit")
