"""The ventilator modality canonicalizes to `"ventilator"`, with `"vent"` kept
working as a legacy alias.

Stage 1 shipped `"vent"` as the internal spelling (a `session.raw` key, a
`BreathEvent.modality` value, a pipeline-spec parameter value) while the docs,
`M3Session.link_breaths`, and `Signal.modality` all used `"ventilator"` - so a
breath could carry `modality="vent"` while the `LinkedBreath` holding it was
keyed `"ventilator"`. These tests pin the canonical spelling and, just as
importantly, that the old one still works everywhere it was accepted.
"""

from __future__ import annotations

import numpy as np

from m3resp.core.events import BreathEvent
from m3resp.core.session import M3Session, set_ventilator_raw
from m3resp.synchronization.alignment import align_events_by_modality_offset
from m3resp.synchronization.cropping import (
    VENTILATOR,
    normalize_modality,
    resolve_alignment_offsets,
    ventilator_raw,
)
from m3resp.synchronization.ventilator import normalize_ventilator_breath


def _recording(n_samples: int = 100, fs: float = 10.0) -> dict:
    return {
        "array": np.zeros((3, n_samples), dtype=float),
        "metadata": {"fs": fs},
    }


class TestCanonicalSpelling:
    def test_every_alias_normalizes_to_ventilator(self):
        for alias in ("vent", "ventilator", "Ventilation", "VENT"):
            assert normalize_modality(alias) == "ventilator"

    def test_detected_breaths_are_tagged_canonically(self):
        breath = normalize_ventilator_breath(5, fs=10.0, width_seconds=0.5)
        assert breath.modality == "ventilator"

    def test_an_existing_breath_is_retagged_canonically(self):
        legacy = BreathEvent(modality="vent", start_time=0.0, end_time=1.0)
        assert normalize_ventilator_breath(
            legacy, fs=None, width_seconds=0.5
        ).modality == ("ventilator")

    def test_offsets_resolve_under_the_canonical_key(self):
        offsets = resolve_alignment_offsets({"vent": 2.0})
        assert offsets[VENTILATOR] == 2.0
        assert "vent" not in offsets

    def test_breath_modality_matches_the_linked_breath_key(self):
        # The inconsistency this rename exists to fix: link_breaths has always
        # keyed its dict "ventilator", while detected breaths said "vent".
        session = M3Session()
        breath = normalize_ventilator_breath(5, fs=10.0, width_seconds=0.5)
        session.add_events("ventilator_breaths", [breath])

        linked = session.link_breaths()
        assert linked
        assert breath.modality in linked[0].breaths


class TestLegacyAliasStillWorks:
    def test_raw_is_readable_under_both_keys(self):
        session = M3Session()
        recording = _recording()
        set_ventilator_raw(session.raw, recording)

        assert session.raw["ventilator"] is recording
        assert session.raw["vent"] is recording
        assert ventilator_raw(session) is recording

    def test_raw_assigned_under_the_legacy_key_alone_is_still_found(self):
        # A user notebook doing `session.raw["vent"] = ...` directly.
        session = M3Session()
        recording = _recording()
        session.raw["vent"] = recording
        assert ventilator_raw(session) is recording

    def test_cropping_through_one_key_is_visible_through_the_other(self):
        # Both keys reference the same object and cropping mutates in place,
        # so the two views can never drift apart.
        session = M3Session()
        recording = _recording(n_samples=100, fs=10.0)
        set_ventilator_raw(session.raw, recording)

        session.synchronize_raw_modalities(
            offset_seconds={"ventilator": 1.0}, reference_modality="eit"
        )

        assert session.raw["vent"]["array"].shape[1] == 90
        assert session.raw["ventilator"]["array"] is session.raw["vent"]["array"]

    def test_legacy_events_still_shift_under_a_canonical_offset_key(self):
        # Guards the silent-no-op failure mode: a mismatch between the event's
        # spelling and the offset's spelling would apply 0.0 rather than raise.
        events = [BreathEvent(modality="vent", start_time=1.0, end_time=2.0)]
        aligned = align_events_by_modality_offset(events, {"ventilator": 0.5})
        assert aligned[0].start_time == 1.5

    def test_canonical_events_still_shift_under_a_legacy_offset_key(self):
        events = [BreathEvent(modality="ventilator", start_time=1.0, end_time=2.0)]
        aligned = align_events_by_modality_offset(events, {"vent": 0.5})
        assert aligned[0].start_time == 1.5

    def test_alignment_reference_is_found_from_a_legacy_raw_key(self):
        session = M3Session()
        session.raw["vent"] = _recording()
        assert session._resolve_raw_alignment_reference(None) == VENTILATOR
