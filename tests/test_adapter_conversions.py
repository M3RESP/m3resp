"""Milestone 2.3 - adapter conversion boundary: legacy output -> Layer 1 objects.

Uses lightweight fakes (not the optional `eitprocessing`/`resurfemg`
dependencies) shaped like the upstream objects the adapters duck-type on, per
`plan/stage2_consolidation.md`.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from m3resp.adapters import EITProcessingAdapter, ReSurfEMGAdapter
from m3resp.data import ParameterResult, QualityFlag, Signal


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
        assert [s.processing_state for s in signals] == ["raw", "filtered"]
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

    def test_to_parameters_converts_rates_and_skips_nan_sparse_samples(self):
        adapter = EITProcessingAdapter()
        preprocessed = {
            "respiratory_rate_hz": 0.3,
            "heart_rate_hz": 1.2,
            "continuous_tiv": _sparse([1.0, math.nan, 2.0], name="continuous_tivs"),
            "eeli": None,
            "pixel_tiv": None,
        }

        parameters = adapter.to_parameters(preprocessed)

        assert all(isinstance(p, ParameterResult) for p in parameters)
        names = [p.name for p in parameters]
        assert names.count("respiratory_rate") == 1
        assert names.count("heart_rate") == 1
        tiv_params = [p for p in parameters if p.name == "continuous_tivs"]
        assert len(tiv_params) == 2  # the NaN sample is skipped
        assert {p.breath_id for p in tiv_params} == {"0", "2"}

    def test_to_quality_flags_reports_missing_optional_outputs(self):
        adapter = EITProcessingAdapter()

        flags = adapter.to_quality_flags(
            {"respiratory_rate_hz": None, "breath_intervals": None}
        )

        assert all(isinstance(f, QualityFlag) for f in flags)
        assert all(f.passed is False for f in flags)


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

        assert [s.processing_state for s in signals] == ["raw", "filtered", "processed"]
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
        assert not by_name["time_to_peak"].is_scalar

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
        assert by_name["snr_pseudo"].passed is True
        assert by_name["snr_pseudo"].value == 12.5
        assert by_name["interpeak_dist"].passed is True
        assert by_name["pocc_quality"].passed is False
        assert by_name["pocc_quality"].severity == "warning"
        assert by_name["pocc_quality"].message == "missing ventilator pressure"
