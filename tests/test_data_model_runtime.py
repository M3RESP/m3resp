"""Milestone 2.1 - Layer 1 runtime data model (plan_stage2.md Sec 8-13).

Acceptance criterion under test: all core objects can be created, validated,
and serialized (plan_stage2.md Milestone 2.1).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from m3resp.data import (
    ParameterResult,
    ProcessingHistory,
    ProcessingStep,
    QualityFlag,
    Signal,
    TimeSeries,
)
from m3resp.data.events import BreathEvent


def test_breath_event_exposes_duration_and_sample_indices():
    breath = BreathEvent(
        modality="eit",
        start_time=1.0,
        end_time=2.5,
        extremum_time=1.5,
        start_index=10,
        extremum_index=15,
        end_index=25,
    )

    assert breath.duration == pytest.approx(1.5)
    assert (breath.start_index, breath.extremum_index, breath.end_index) == (10, 15, 25)


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

    def test_rejects_multidimensional_time(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            TimeSeries(values=np.zeros((2, 3)), time=np.zeros((2, 3)))

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

    def test_nan_at_the_same_places_counts_as_equal(self):
        values = np.array([[1.0, np.nan], [np.nan, 4.0]])

        first = TimeSeries(values=values, time=[0.0, 1.0])
        second = TimeSeries(values=values.copy(), time=[0.0, 1.0])

        assert first == second

    def test_nan_at_other_places_is_not_equal(self):
        first = TimeSeries(values=[1.0, np.nan], time=[0.0, 1.0])
        second = TimeSeries(values=[np.nan, 2.0], time=[0.0, 1.0])

        assert first != second
        assert first != TimeSeries(values=[1.0, 2.0], time=[0.0, 1.0])

    def test_text_values_compare(self):
        first = TimeSeries(values=np.array(["a", "b"]), time=[0.0, 1.0])

        assert first == TimeSeries(values=np.array(["a", "b"]), time=[0.0, 1.0])
        assert first != TimeSeries(values=np.array(["a", "c"]), time=[0.0, 1.0])


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

    def test_eit_images_with_missing_pixels_are_equal(self):
        frames = np.full((3, 2, 2), np.nan)
        frames[:, 0, 0] = [1.0, 2.0, 3.0]

        first = Signal(values=frames, time=[0.0, 0.05, 0.1], modality="eit")
        second = Signal(values=frames.copy(), time=[0.0, 0.05, 0.1], modality="eit")

        assert first == second
        assert first != Signal(values=frames, time=[0.0, 0.05, 0.1], modality="emg")

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
    def test_nan_value_counts_as_equal(self):
        first = ParameterResult(name="tiv", value=float("nan"), modality="eit")
        grid = np.array([[1.0, np.nan], [np.nan, 4.0]])
        image = ParameterResult(name="map", value=grid, modality="eit")

        assert first == ParameterResult(name="tiv", value=float("nan"), modality="eit")
        assert first != ParameterResult(name="tiv", value=1.0, modality="eit")
        assert image == ParameterResult(name="map", value=grid.copy(), modality="eit")

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

    def test_refuses_both_one_breath_and_several_breaths(self):
        with pytest.raises(ValueError, match="both breath_id"):
            ParameterResult(
                name="tiv", value=1.0, modality="eit", breath_id="0", breath_ids=["0"]
            )

    def test_refuses_an_empty_breath_list(self):
        with pytest.raises(ValueError, match="empty list"):
            ParameterResult(name="tiv", value=1.0, modality="eit", breath_ids=[])


class TestQualityFlag:
    def test_create_with_valid_severity(self):
        flag = QualityFlag(name="snr_check", passed=True, severity="info")
        assert flag.passed is True

    def test_rejects_unknown_severity(self):
        with pytest.raises(ValueError, match="severity"):
            QualityFlag(name="snr_check", passed=False, severity="catastrophic")

    def test_one_flag_covers_several_breaths(self):
        flag = QualityFlag(
            name="stable_stretch",
            passed=True,
            severity="info",
            modality="eit",
            breath_ids=["3", "4", "7"],
        )

        row = flag.to_dict()

        assert row["breath_id"] is None
        assert row["breath_ids"] == ["3", "4", "7"]

    def test_to_dict_has_one_key_per_field(self):
        flag = QualityFlag(
            name="snr_check",
            passed=False,
            severity="warning",
            breath_id="2",
            start_time=1.5,
            end_time=2.5,
            value=3.0,
            threshold=5.0,
            metadata={"channel": 1},
        )

        row = flag.to_dict()

        assert list(row) == [f.name for f in dataclasses.fields(QualityFlag)]
        assert row["breath_id"] == "2"
        assert (row["start_time"], row["end_time"]) == (1.5, 2.5)
        assert row["metadata"] == {"channel": 1}

    def test_refuses_both_one_breath_and_several_breaths(self):
        with pytest.raises(ValueError, match="both breath_id"):
            QualityFlag(
                name="check",
                passed=True,
                severity="info",
                breath_id="0",
                breath_ids=["0", "1"],
            )

    def test_refuses_an_empty_breath_list(self):
        with pytest.raises(ValueError, match="empty list"):
            QualityFlag(name="check", passed=True, severity="info", breath_ids=[])


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
