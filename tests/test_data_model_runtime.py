"""Milestone 2.1 - Layer 1 runtime data model (plan_stage2.md Sec 8-13).

Acceptance criterion under test: all core objects can be created, validated,
and serialized (plan_stage2.md Milestone 2.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.core.events import BreathEvent
from m3resp.data import (
    Breath,
    ParameterResult,
    ProcessingHistory,
    ProcessingStep,
    QualityFlag,
    Signal,
    TimeSeries,
)


def test_breath_is_the_same_type_as_breath_event():
    assert Breath is BreathEvent


def test_breath_event_exposes_duration_and_sample_indices():
    breath = BreathEvent(
        modality="eit",
        start_time=1.0,
        end_time=2.5,
        peak_time=1.5,
        start_index=10,
        peak_index=15,
        end_index=25,
    )

    assert breath.duration == pytest.approx(1.5)
    assert (breath.start_index, breath.peak_index, breath.end_index) == (10, 15, 25)


class TestTimeSeries:
    def test_create_and_derived_properties(self):
        series = TimeSeries(
            values=[1.0, 2.0, 3.0],
            time=[0.0, 0.5, 1.0],
            sample_frequency=2.0,
            unit="a.u.",
            name="demo",
        )

        assert series.n_samples == 3
        assert series.duration == pytest.approx(1.0)
        assert isinstance(series.values, np.ndarray)

    def test_rejects_mismatched_values_and_time_length(self):
        with pytest.raises(ValueError, match="same length"):
            TimeSeries(values=[1.0, 2.0], time=[0.0])

    def test_to_manifest_row_excludes_raw_arrays(self):
        series = TimeSeries(values=[1.0, 2.0], time=[0.0, 1.0], unit="mV")

        row = series.to_manifest_row()

        assert row == {
            "name": None,
            "unit": "mV",
            "sample_frequency": None,
            "n_samples": 2,
            "duration": 1.0,
            "metadata": {},
        }


class TestSignal:
    def test_create_with_modality_and_channel(self):
        signal = Signal(
            values=[1.0, 2.0],
            time=[0.0, 1.0],
            modality="eit",
            channel="global",
            source="subject.eit",
            unit="a.u.",
            sample_frequency=50.0,
        )

        assert signal.n_samples == 2
        row = signal.to_manifest_row()
        assert row["modality"] == "eit"
        assert row["source"] == "subject.eit"

    def test_accepts_any_modality(self):
        # Modality is an open vocabulary: new loaders/quantity types aren't
        # blocked by a fixed enum (only processing_state is still enforced).
        signal = Signal(values=[1.0], time=[0.0], modality="not_a_known_modality")
        assert signal.modality == "not_a_known_modality"

    def test_rejects_unknown_processing_state(self):
        with pytest.raises(ValueError, match="processing_state"):
            Signal(
                values=[1.0],
                time=[0.0],
                modality="emg",
                processing_state="not_a_state",
            )


class TestParameterResult:
    def test_scalar_value_serializes_to_float(self):
        result = ParameterResult(
            name="tidal_impedance_variation", value=1.23, modality="eit", unit="a.u."
        )

        assert result.is_scalar
        assert result.to_dict()["value"] == pytest.approx(1.23)

    def test_array_value_serializes_to_list(self):
        result = ParameterResult(
            name="regional_ventilation",
            value=np.array([0.1, 0.2, 0.3]),
            modality="eit",
        )

        assert not result.is_scalar
        assert result.to_dict()["value"] == [0.1, 0.2, 0.3]


class TestQualityFlag:
    def test_create_with_valid_severity(self):
        flag = QualityFlag(name="snr_check", passed=True, severity="info")
        assert flag.passed is True

    def test_rejects_unknown_severity(self):
        with pytest.raises(ValueError, match="severity"):
            QualityFlag(name="snr_check", passed=False, severity="catastrophic")


class TestProcessingHistoryAndStep:
    def test_record_appends_and_serializes_steps(self):
        history = ProcessingHistory()

        step = history.record(
            "load_eit",
            input_keys=[],
            output_keys=["eit_signal"],
            parameters={"vendor": "sentec"},
        )

        assert isinstance(step, ProcessingStep)
        assert len(history) == 1
        assert list(history) == [step]
        serialized = history.to_list()
        assert serialized[0]["name"] == "load_eit"
        assert serialized[0]["parameters"] == {"vendor": "sentec"}
        assert serialized[0]["software"] == "m3resp"
