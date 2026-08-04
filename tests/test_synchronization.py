"""Synchronization and breath linking.

Acceptance criterion under test: EIT and EMG breaths can be linked into
common multimodal events.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp import M3Session
from m3resp.core.events import BreathEvent
from m3resp.data import Signal
from m3resp.data.linked_breath import LinkedBreath
from m3resp.synchronization import (
    compute_breath_duration_difference,
    compute_event_agreement,
    compute_multimodal_parameters,
    compute_offsets_from_timestamps,
    compute_timing_delay,
    link_breaths_by_time,
    resample_signal,
)


class TestComputeOffsetsFromTimestamps:
    def test_offsets_are_relative_to_the_reference_modality(self):
        offsets = compute_offsets_from_timestamps(
            "eit", {"eit": 100.0, "emg": 105.0, "vent": 98.0}
        )

        assert offsets == {"eit": 0.0, "emg": 5.0, "vent": -2.0}

    def test_raises_when_reference_modality_is_missing(self):
        with pytest.raises(KeyError):
            compute_offsets_from_timestamps("eit", {"emg": 1.0})


class TestResampleSignal:
    def test_upsamples_a_linear_ramp(self):
        signal = Signal(
            values=[0.0, 1.0, 2.0],
            time=[0.0, 1.0, 2.0],
            modality="eit",
            sample_frequency=1.0,
        )

        resampled = resample_signal(signal, 2.0)

        assert resampled.sample_frequency == 2.0
        assert resampled.n_samples == 5
        np.testing.assert_allclose(resampled.values, [0.0, 0.5, 1.0, 1.5, 2.0])

    def test_rejects_non_positive_target_frequency(self):
        signal = Signal(values=[0.0], time=[0.0], modality="eit")
        with pytest.raises(ValueError):
            resample_signal(signal, 0.0)

    def test_handles_empty_signal(self):
        signal = Signal(values=[], time=[], modality="eit")

        resampled = resample_signal(signal, 10.0)

        assert resampled.n_samples == 0
        assert resampled.sample_frequency == 10.0


class TestLinkBreathsByTime:
    def test_matches_close_breaths_across_modalities(self):
        eit_breath = BreathEvent(
            modality="eit", start_time=1.0, end_time=2.0, peak_time=1.5
        )
        emg_breath = BreathEvent(
            modality="emg", start_time=1.1, end_time=2.1, peak_time=1.6
        )

        linked = link_breaths_by_time(
            {"eit": [eit_breath], "emg": [emg_breath]}, time_tolerance=0.5
        )

        assert len(linked) == 1
        assert linked[0].breaths["eit"] is eit_breath
        assert linked[0].breaths["emg"] is emg_breath
        assert linked[0].confidence == pytest.approx(0.8)  # 1 - 0.1/0.5

    def test_keeps_out_of_tolerance_breaths_as_separate_links(self):
        eit_breath = BreathEvent(
            modality="eit", start_time=1.0, end_time=2.0, peak_time=1.5
        )
        emg_breath = BreathEvent(
            modality="emg", start_time=3.0, end_time=4.0, peak_time=3.5
        )

        linked = link_breaths_by_time(
            {"eit": [eit_breath], "emg": [emg_breath]}, time_tolerance=0.5
        )

        assert len(linked) == 2
        assert sorted(link.modalities for link in linked) == [["eit"], ["emg"]]
        assert all(link.confidence is None for link in linked)

    def test_links_all_three_modalities(self):
        eit_breath = BreathEvent(
            modality="eit", start_time=1.0, end_time=2.0, peak_time=1.5
        )
        emg_breath = BreathEvent(
            modality="emg", start_time=1.05, end_time=2.05, peak_time=1.55
        )
        vent_breath = BreathEvent(
            modality="ventilator", start_time=0.9, end_time=1.9, peak_time=1.45
        )

        linked = link_breaths_by_time(
            {
                "eit": [eit_breath],
                "emg": [emg_breath],
                "ventilator": [vent_breath],
            },
            time_tolerance=0.5,
        )

        assert len(linked) == 1
        assert linked[0].modalities == ["eit", "emg", "ventilator"]

    def test_does_not_double_assign_a_breath_to_two_links(self):
        eit_breath = BreathEvent(
            modality="eit", start_time=1.0, end_time=2.0, peak_time=1.5
        )
        emg_near = BreathEvent(
            modality="emg", start_time=1.05, end_time=2.05, peak_time=1.55
        )
        emg_far = BreathEvent(
            modality="emg", start_time=1.2, end_time=2.2, peak_time=1.7
        )

        linked = link_breaths_by_time(
            {"eit": [eit_breath], "emg": [emg_near, emg_far]},
            time_tolerance=0.5,
        )

        assert len(linked) == 2
        matched = next(link for link in linked if "eit" in link.breaths)
        assert matched.breaths["emg"] is emg_near
        unmatched = next(link for link in linked if "eit" not in link.breaths)
        assert unmatched.breaths["emg"] is emg_far

    def test_rejects_negative_tolerance(self):
        with pytest.raises(ValueError):
            link_breaths_by_time(time_tolerance=-1.0)

    def test_no_breaths_returns_empty_list(self):
        assert link_breaths_by_time() == []


class TestSessionLinkBreaths:
    def test_link_breaths_uses_raw_event_lists_by_default(self):
        session = M3Session()
        eit_breath = BreathEvent(
            modality="eit", start_time=1.0, end_time=2.0, peak_time=1.5
        )
        emg_breath = BreathEvent(
            modality="emg", start_time=1.1, end_time=2.1, peak_time=1.6
        )
        session.add_events("eit_breaths", [eit_breath])
        session.add_events("emg_breaths", [emg_breath])

        linked = session.link_breaths(time_tolerance=0.5)

        assert linked is session.linked_breaths
        assert len(linked) == 1
        assert linked[0].modalities == ["eit", "emg"]
        assert session.provenance[-1].action == "link_breaths"

    def test_link_breaths_prefers_aligned_events_over_raw_ones(self):
        session = M3Session()
        eit_breath = BreathEvent(
            modality="eit", start_time=1.0, end_time=2.0, peak_time=1.5
        )
        emg_breath = BreathEvent(
            modality="emg", start_time=1.0, end_time=2.0, peak_time=1.5
        )
        session.add_events("eit_breaths", [eit_breath])
        session.add_events("emg_breaths", [emg_breath])

        # Shift emg 5s later relative to eit: after alignment the two breaths
        # are far apart and should no longer link together.
        session.synchronize_multimodal_breaths(offset_seconds={"emg": 5.0})
        linked = session.link_breaths(time_tolerance=0.5)

        assert len(linked) == 2


class TestComputeTimingDelay:
    def test_positive_delay_when_target_is_later(self):
        linked = LinkedBreath(
            breaths={
                "emg": BreathEvent(
                    modality="emg", start_time=1.0, end_time=2.0, peak_time=1.5
                ),
                "eit": BreathEvent(
                    modality="eit", start_time=1.2, end_time=2.2, peak_time=1.9
                ),
            }
        )

        assert compute_timing_delay(linked, "emg", "eit") == pytest.approx(0.2)
        assert compute_timing_delay(
            linked, "emg", "eit", anchor="peak"
        ) == pytest.approx(0.4)

    def test_returns_none_when_a_modality_is_missing(self):
        linked = LinkedBreath(
            breaths={"emg": BreathEvent(modality="emg", start_time=1.0, end_time=2.0)}
        )

        assert compute_timing_delay(linked, "emg", "eit") is None

    def test_rejects_unknown_anchor(self):
        linked = LinkedBreath(
            breaths={
                "emg": BreathEvent(modality="emg", start_time=1.0, end_time=2.0),
                "eit": BreathEvent(modality="eit", start_time=1.0, end_time=2.0),
            }
        )

        with pytest.raises(ValueError):
            compute_timing_delay(linked, "emg", "eit", anchor="middle")


class TestComputeBreathDurationDifference:
    def test_computes_signed_difference(self):
        linked = LinkedBreath(
            breaths={
                "eit": BreathEvent(modality="eit", start_time=0.0, end_time=1.5),
                "emg": BreathEvent(modality="emg", start_time=0.0, end_time=1.0),
            }
        )

        assert compute_breath_duration_difference(
            linked, "eit", "emg"
        ) == pytest.approx(0.5)

    def test_returns_none_when_a_modality_is_missing(self):
        linked = LinkedBreath(
            breaths={"eit": BreathEvent(modality="eit", start_time=0.0, end_time=1.5)}
        )

        assert compute_breath_duration_difference(linked, "eit", "emg") is None


class TestComputeEventAgreement:
    def test_fraction_of_fully_linked_breaths(self):
        both = LinkedBreath(
            breaths={
                "eit": BreathEvent(modality="eit", start_time=0.0, end_time=1.0),
                "emg": BreathEvent(modality="emg", start_time=0.0, end_time=1.0),
            }
        )
        eit_only = LinkedBreath(
            breaths={"eit": BreathEvent(modality="eit", start_time=2.0, end_time=3.0)}
        )

        assert compute_event_agreement([both, eit_only], ("eit", "emg")) == 0.5

    def test_empty_list_returns_zero(self):
        assert compute_event_agreement([], ("eit", "emg")) == 0.0


class TestComputeMultimodalParameters:
    def test_default_pairs_come_from_observed_modalities(self):
        linked = [
            LinkedBreath(
                breaths={
                    "eit": BreathEvent(
                        modality="eit", start_time=1.0, end_time=2.0, peak_time=1.5
                    ),
                    "emg": BreathEvent(
                        modality="emg", start_time=1.1, end_time=2.1, peak_time=1.6
                    ),
                }
            )
        ]

        results = compute_multimodal_parameters(linked)

        names = {p.name for p in results}
        assert "eit_to_emg_delay" in names
        assert "eit_emg_event_agreement" in names
        assert "eit_emg_duration_difference" in names
        assert all(p.modality == "multimodal" for p in results)

    def test_no_linked_breaths_returns_empty_list(self):
        assert compute_multimodal_parameters([]) == []


class TestSessionComputeMultimodalParameters:
    def test_adds_results_to_parameter_results_and_provenance(self):
        session = M3Session()
        session.add_events(
            "eit_breaths",
            [BreathEvent(modality="eit", start_time=1.0, end_time=2.0, peak_time=1.5)],
        )
        session.add_events(
            "emg_breaths",
            [BreathEvent(modality="emg", start_time=1.1, end_time=2.1, peak_time=1.6)],
        )
        session.link_breaths(time_tolerance=0.5)

        results = session.compute_multimodal_parameters()

        assert results
        assert all(r in session.parameter_results.items for r in results)
        assert session.provenance[-1].action == "compute_multimodal_parameters"

    def test_empty_linked_breaths_returns_empty_list(self):
        session = M3Session()

        assert session.compute_multimodal_parameters() == []
