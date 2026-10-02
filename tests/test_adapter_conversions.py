"""Milestone 2.3 - adapter conversion boundary: legacy output -> Layer 1 objects.

Uses lightweight fakes (not the optional `eitprocessing`/`resurfemg`
dependencies) shaped like the upstream objects the adapters duck-type on, per
`plan/stage2_consolidation.md`.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from m3resp.adapters import EITProcessingAdapter, ReSurfEMGAdapter
from m3resp.adapters.eitprocessing_adapter import sparse_data_to_interval_data
from m3resp.core.exceptions import UnsupportedWorkflowError
from m3resp.data import (
    BreathEvent,
    IntervalData,
    ParameterResult,
    PixelMap,
    QualityFlag,
    Signal,
)


def _continuous(values, *, name="signal", unit="a.u.", sample_frequency=50.0):
    return SimpleNamespace(
        values=values,
        time=list(range(len(values))),
        unit=unit,
        name=name,
        sample_frequency=sample_frequency,
    )


def _sparse(values, *, name="parameter", unit="a.u."):
    return SimpleNamespace(
        values=values, time=list(range(len(values))), unit=unit, name=name
    )


def _breaths(start_middle_end):
    """Fake eitprocessing breaths: an object whose ``values`` are breaths
    with a start, middle and end time."""

    return SimpleNamespace(
        values=[
            SimpleNamespace(start_time=start, middle_time=middle, end_time=end)
            for start, middle, end in start_middle_end
        ]
    )


class TestEITAdapterConversions:
    def test_to_signals_converts_raw_and_filtered_global_impedance(self):
        adapter = EITProcessingAdapter()
        preprocessed = {
            "raw_global_impedance": _continuous([1.0, 2.0, 3.0], name="raw gi"),
            "filtered_global_impedance": _continuous(
                [1.1, 2.1, 3.1], name="filtered gi"
            ),
        }

        signals = adapter.to_signals(preprocessed)

        assert len(signals) == 2
        assert all(isinstance(s, Signal) for s in signals)
        assert [s.processing_state for s in signals] == ["raw", "intermediate"]
        assert all(
            s.modality == "eit" and s.channel == "global_impedance" for s in signals
        )
        assert signals[0].sample_frequency == 50.0

    def test_to_signals_does_not_duplicate_unfiltered_data(self):
        adapter = EITProcessingAdapter()
        raw = _continuous([1.0, 2.0])
        preprocessed = {
            "raw_global_impedance": raw,
            "filtered_global_impedance": raw,
        }

        signals = adapter.to_signals(preprocessed)

        assert len(signals) == 1
        assert signals[0].processing_state == "raw"

    def test_to_parameters_converts_only_the_rates(self):
        adapter = EITProcessingAdapter()
        preprocessed = {
            "respiratory_rate_hz": 0.3,
            "heart_rate_hz": 1.2,
            "breath_intervals": _breaths([(0.0, 1.0, 2.0)]),
            "continuous_tiv": _sparse([1.0], name="continuous_tivs"),
            "eeli": None,
            "pixel_tiv": None,
        }

        parameters = adapter.to_parameters(preprocessed)

        assert all(isinstance(p, ParameterResult) for p in parameters)
        assert [p.name for p in parameters] == ["respiratory_rate", "heart_rate"]

    def test_to_interval_data_keeps_each_value_with_its_breath(self):
        adapter = EITProcessingAdapter()
        preprocessed = {
            "breath_intervals": _breaths(
                [(0.0, 1.0, 2.0), (2.0, 3.0, 4.0), (4.0, 5.0, 6.0)]
            ),
            "continuous_tiv": _sparse([1.0, math.nan, 2.0], name="continuous_tivs"),
            "eeli": None,
            "pixel_tiv": None,
        }

        [tiv] = adapter.to_interval_data(preprocessed)

        assert isinstance(tiv, IntervalData)
        assert tiv.name == "continuous_tivs"
        # A NaN value stays next to its breath instead of being dropped.
        np.testing.assert_array_equal(tiv.values, [1.0, math.nan, 2.0])
        assert all(isinstance(b, BreathEvent) for b in tiv.intervals)
        assert [b.start_time for b in tiv.intervals] == [0.0, 2.0, 4.0]
        assert [b.peak_time for b in tiv.intervals] == [1.0, 3.0, 5.0]

    def test_to_interval_data_needs_the_breaths(self):
        adapter = EITProcessingAdapter()
        preprocessed = {"continuous_tiv": _sparse([1.0], name="continuous_tivs")}

        with pytest.raises(ValueError, match="breath_intervals"):
            adapter.to_interval_data(preprocessed)

    def test_to_interval_data_is_empty_without_tiv_or_eeli(self):
        adapter = EITProcessingAdapter()

        assert adapter.to_interval_data({"breath_intervals": object()}) == []

    def test_to_quality_flags_reports_missing_optional_outputs(self):
        adapter = EITProcessingAdapter()

        flags = adapter.to_quality_flags(
            {"respiratory_rate_hz": None, "breath_intervals": None}
        )

        assert all(isinstance(f, QualityFlag) for f in flags)
        assert all(f.passed is False for f in flags)


def test_as_pixel_mask_names_the_wrong_type_clearly():
    adapter = EITProcessingAdapter()
    not_a_mask = ParameterResult(name="mask", value=np.ones((2, 2)), modality="eit")

    with pytest.raises(UnsupportedWorkflowError, match="got ParameterResult"):
        adapter.as_pixel_mask(not_a_mask)


class TestSparseDataToIntervalData:
    """`sparse_data_to_interval_data()` stores one value per breath next to
    the breath, for numbers and for pixel maps alike."""

    def test_numbers_become_one_value_per_breath(self):
        obj = _sparse([1.0, 2.0], name="continuous_tivs")
        breaths = [BreathEvent("eit", 0.0, 1.0), BreathEvent("eit", 1.0, 2.0)]

        result = sparse_data_to_interval_data(obj, breaths, modality="eit", method="m")

        assert result.intervals == breaths
        np.testing.assert_array_equal(result.values, [1.0, 2.0])
        assert result.unit == "a.u."
        assert result.category == "impedance"

    def test_maps_become_one_pixel_map_per_breath(self):
        map_value = np.full((2, 3), 5.5)
        obj = SimpleNamespace(
            values=[map_value],
            time=[np.full((2, 3), 1.25)],
            unit="a.u.",
            name="pixel_tivs",
        )

        result = sparse_data_to_interval_data(
            obj,
            [BreathEvent("eit", 0.0, 2.0)],
            modality="eit",
            method="m",
            as_pixel_maps=True,
        )

        [pixel_map] = result.values
        assert isinstance(pixel_map, PixelMap)
        np.testing.assert_array_equal(pixel_map.values, map_value)
        assert pixel_map.name == "pixel_tivs"

    def test_all_nan_map_is_kept_not_dropped(self):
        obj = SimpleNamespace(
            values=[np.full((2, 2), np.nan)],
            time=[np.zeros((2, 2))],
            unit="a.u.",
            name="pixel_tivs",
        )

        result = sparse_data_to_interval_data(
            obj,
            [BreathEvent("eit", 0.0, 2.0)],
            modality="eit",
            method="m",
            as_pixel_maps=True,
        )

        assert len(result) == 1
        assert np.all(np.isnan(result.values[0].values))

    def test_a_different_number_of_breaths_is_refused(self):
        obj = _sparse([1.0, 2.0, 3.0], name="continuous_tivs")

        with pytest.raises(ValueError, match="3 per-breath values but 1 breaths"):
            sparse_data_to_interval_data(
                obj, [BreathEvent("eit", 0.0, 1.0)], modality="eit", method="m"
            )


class TestReSurfEMGAdapterConversions:
    def test_to_signals_converts_raw_filtered_and_envelope(self):
        adapter = ReSurfEMGAdapter()
        processed_emg = {
            "fs": 100.0,
            "channel": 0,
            "raw_channel": [0.0, 1.0, 2.0],
            "filtered": [0.1, 0.2, 0.3],
            "envelope": [0.05, 0.15, 0.25],
        }

        signals = adapter.to_signals(processed_emg)

        assert [s.processing_state for s in signals] == [
            "raw",
            "intermediate",
            "processed",
        ]
        assert all(s.modality == "emg" and s.channel == "0" for s in signals)
        assert all(s.sample_frequency == 100.0 for s in signals)

    def test_to_parameters_converts_feature_values_scalar_and_array(self):
        adapter = ReSurfEMGAdapter()
        postprocessed = {
            "computed": {
                "features": {
                    "amplitude": 1.5,
                    "time_to_peak": np.array([0.1, 0.2]),
                }
            }
        }

        parameters = adapter.to_parameters(postprocessed)

        by_name = {p.name: p for p in parameters}
        assert by_name["amplitude"].value == 1.5
        assert by_name["amplitude"].is_scalar
        assert by_name["amplitude"].method == "resurfemg.amplitude"
        assert by_name["amplitude"].metadata == {
            "source_method": "resurfemg.amplitude",
            "implementation": "m3resp.processing.metrics",
        }
        assert not by_name["time_to_peak"].is_scalar

    def test_to_parameters_splits_respiratory_rate_into_median_and_per_breath(self):
        """The respiratory rate arrives as (median rate, rate of each breath).
        The two parts have different shapes, so they become two results."""

        adapter = ReSurfEMGAdapter()
        postprocessed = {
            "computed": {
                "features": {
                    "respiratory_rate": (12.0, np.array([11.5, 12.0, np.nan])),
                }
            }
        }

        parameters = adapter.to_parameters(postprocessed)

        by_name = {p.name: p for p in parameters}
        assert set(by_name) == {
            "respiratory_rate",
            "respiratory_rate_breath_to_breath",
        }
        median = by_name["respiratory_rate"]
        assert median.is_scalar
        assert median.value == 12.0
        assert median.unit == "breaths/min"
        assert median.metric_type == "respiratory_rate"
        per_breath = by_name["respiratory_rate_breath_to_breath"]
        assert not per_breath.is_scalar
        # NaN marks an outlier breath and must be kept, not dropped.
        np.testing.assert_array_equal(per_breath.value, [11.5, 12.0, np.nan])
        assert per_breath.unit == "breaths/min"
        assert per_breath.method == "resurfemg.respiratory_rate"

    def test_to_quality_flags_converts_results_and_skipped_functions(self):
        adapter = ReSurfEMGAdapter()
        postprocessed = {
            "computed": {
                "quality_assessment": {
                    "snr_pseudo": 12.5,
                    "interpeak_dist": True,
                }
            },
            "skipped": {"pocc_quality": "missing ventilator pressure"},
        }

        flags = adapter.to_quality_flags(postprocessed)

        by_name = {f.name: f for f in flags}
        assert by_name["snr_pseudo"].passed is False
        assert by_name["snr_pseudo"].metadata["measurement_only"] is True
        assert by_name["snr_pseudo"].value == 12.5
        assert by_name["snr_pseudo"].metadata == {
            "source_method": "resurfemg.snr_pseudo",
            "measurement_only": True,
        }
        assert by_name["interpeak_dist"].passed is True
        assert by_name["pocc_quality"].passed is False
        assert by_name["pocc_quality"].severity == "warning"
        assert by_name["pocc_quality"].message == "missing ventilator pressure"
        assert by_name["pocc_quality"].metadata == {"skipped": True}
