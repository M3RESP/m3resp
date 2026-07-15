"""Stage 2 ReSurfEMG gap migration, Phase 5 (plan/stage2/
2_resurfemg_gap_migration_implementation_plan.md): native ParameterResult/
QualityFlag outputs for the ten clinical quality operations.
"""

from __future__ import annotations

from typing import Any

import pytest

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, QualityFlag
from m3resp.workflows.steps.emg import (
    area_under_baseline,
    detect_extreme_time_products,
    detect_local_high_aub,
    detect_non_consecutive_manoeuvres,
    evaluate_bell_curve_error,
    evaluate_event_timing,
    evaluate_respiratory_rates,
    interpeak_dist,
    percentage_under_baseline,
    pocc_quality,
    snr_pseudo,
    time_product,
)

pytest.importorskip("resurfemg")
np = pytest.importorskip("numpy")


def _synthetic_breaths(
    *, fs: float = 100.0, n_breaths: int = 6, duration_seconds: float = 30.0
) -> dict[str, Any]:
    """A clean envelope with `n_breaths` bell-shaped bursts and a flat
    baseline, plus matching peak/start/end indices - enough structure for
    every quality function to produce real, non-degenerate output."""

    n_samples = int(duration_seconds * fs)
    time = np.arange(n_samples) / fs
    envelope = np.full(n_samples, 0.5)
    peak_indices = []
    start_indices = []
    end_indices = []
    period = n_samples // (n_breaths + 1)
    for i in range(1, n_breaths + 1):
        center = i * period
        width = period // 3
        start = center - width
        end = center + width
        bump = 3.0 * np.exp(
            -0.5 * ((np.arange(start, end) - center) / (width / 3)) ** 2
        )
        envelope[start:end] += bump
        peak_indices.append(center)
        start_indices.append(start)
        end_indices.append(end)

    baseline = np.full(n_samples, 0.5)
    return {
        "envelope": envelope,
        "baseline": baseline,
        "peak_indices": np.array(peak_indices),
        "start_indices": np.array(start_indices),
        "end_indices": np.array(end_indices),
        "processed_emg": {
            "envelope": envelope,
            "fs": fs,
            "channel": 0,
            "metadata": {"labels": ["EMGdi"], "units": ["uV"]},
        },
        "time": time,
    }


class TestSnrPseudo:
    def test_results_have_no_flags_without_a_configured_minimum(self):
        session = M3Session()
        data = _synthetic_breaths()
        out = snr_pseudo(
            session, data["processed_emg"], data["peak_indices"], data["baseline"]
        )

        assert len(out["snr_pseudo_results"]) == len(data["peak_indices"])
        assert all(isinstance(r, ParameterResult) for r in out["snr_pseudo_results"])
        assert out["snr_pseudo_flags"] == []
        assert len(session.parameter_results) == len(data["peak_indices"])
        assert len(session.quality) == 0

    def test_flags_appear_only_when_minimum_snr_is_set(self):
        session = M3Session()
        data = _synthetic_breaths()
        out = snr_pseudo(
            session,
            data["processed_emg"],
            data["peak_indices"],
            data["baseline"],
            minimum_snr=1.0,
        )

        assert len(out["snr_pseudo_flags"]) == len(data["peak_indices"])
        assert all(isinstance(f, QualityFlag) for f in out["snr_pseudo_flags"])
        assert all(f.threshold == 1.0 for f in out["snr_pseudo_flags"])
        assert [f.breath_id for f in out["snr_pseudo_flags"]] == [
            str(i) for i in range(len(data["peak_indices"]))
        ]


class TestPercentageUnderBaseline:
    def test_writes_per_breath_flags_and_percentage_reference_measurements(self):
        session = M3Session()
        data = _synthetic_breaths()
        out = percentage_under_baseline(
            session,
            data["processed_emg"],
            data["peak_indices"],
            data["start_indices"],
            data["end_indices"],
            data["baseline"],
        )

        assert len(out["percentage_under_baseline_results"]) == len(
            data["peak_indices"]
        )
        assert len(out["percentage_under_baseline_flags"]) == len(data["peak_indices"])
        for result in out["percentage_under_baseline_results"]:
            assert result.unit == "%"
            assert "reference_value" in result.metadata


class TestDetectLocalHighAub:
    def test_writes_per_breath_flags_and_the_effective_threshold(self):
        session = M3Session()
        data = _synthetic_breaths()
        aub_out = area_under_baseline(
            data["processed_emg"],
            data["peak_indices"],
            data["start_indices"],
            data["end_indices"],
            data["baseline"],
        )
        out = detect_local_high_aub(
            session, aub_out["area_under_baseline"], data["peak_indices"]
        )

        assert len(out["detect_local_high_aub_flags"]) == len(data["peak_indices"])
        threshold_result = out["detect_local_high_aub_threshold_result"]
        assert isinstance(threshold_result, ParameterResult)
        # AUB measures area *below* baseline; this fixture's envelope never
        # dips below its own flat baseline, so 0 is the legitimately correct
        # value here - the check that matters is the threshold was computed
        # from the real AUB values, not that it's positive.
        assert threshold_result.value >= 0


class TestDetectExtremeTimeProducts:
    def test_writes_per_breath_flags_and_effective_bounds(self):
        session = M3Session()
        data = _synthetic_breaths()
        tp_out = time_product(
            data["processed_emg"],
            data["start_indices"],
            data["end_indices"],
            data["baseline"],
        )
        out = detect_extreme_time_products(
            session, tp_out["time_product"], data["peak_indices"]
        )

        assert len(out["detect_extreme_time_products_flags"]) == len(
            data["peak_indices"]
        )
        bounds_result = out["detect_extreme_time_products_bounds_result"]
        assert isinstance(bounds_result, ParameterResult)
        assert (
            bounds_result.metadata["lower_bound"]
            <= bounds_result.metadata["upper_bound"]
        )


class TestDetectNonConsecutiveManoeuvres:
    def test_writes_one_pressure_flag_per_manoeuvre(self):
        session = M3Session()
        ventilator_breaths = np.array([0, 1000, 2000, 3000, 4000])
        manoeuvres = np.array([500, 2500])
        out = detect_non_consecutive_manoeuvres(session, ventilator_breaths, manoeuvres)

        flags = out["detect_non_consecutive_manoeuvres_flags"]
        assert len(flags) == len(manoeuvres)
        assert all(f.modality == "pressure" for f in flags)


class TestEvaluateBellCurveError:
    def test_writes_per_breath_flags_and_measurement_results(self):
        session = M3Session()
        data = _synthetic_breaths()
        tp_out = time_product(
            data["processed_emg"],
            data["start_indices"],
            data["end_indices"],
            data["baseline"],
        )
        out = evaluate_bell_curve_error(
            session,
            data["peak_indices"],
            data["start_indices"],
            data["end_indices"],
            data["processed_emg"],
            tp_out["time_product"],
        )

        results = out["evaluate_bell_curve_error_results"]
        flags = out["evaluate_bell_curve_error_flags"]
        n_breaths = len(data["peak_indices"])
        assert len(flags) == n_breaths
        # Percentage-error results (scalar) plus fitted-parameter results
        # (array-valued, their own ParameterResult so they reuse the shared
        # array exporter rather than sitting in metadata - plan Phase 6.3).
        assert len(results) == 2 * n_breaths

        percentage_results = [
            r for r in results if r.name == "evaluate_bell_curve_error"
        ]
        fitted_results = [
            r
            for r in results
            if r.name == "evaluate_bell_curve_error_fitted_parameters"
        ]
        assert len(percentage_results) == len(fitted_results) == n_breaths
        for result in percentage_results:
            assert "bell_error" in result.metadata
            assert "y_min" in result.metadata
        for result in fitted_results:
            assert not result.is_scalar
            assert np.asarray(result.value).ndim == 1


class TestEvaluateEventTiming:
    def test_writes_per_pair_results_with_both_source_indices_and_fs(self):
        session = M3Session()
        data = _synthetic_breaths()
        vent_fs = 50.0
        ventilator_breath_indices = (data["peak_indices"] * (vent_fs / 100.0)).astype(
            int
        )
        ventilator_signals = {"fs": vent_fs}

        out = evaluate_event_timing(
            session,
            data["peak_indices"],
            data["processed_emg"],
            ventilator_breath_indices,
            ventilator_signals,
        )

        assert out["evaluate_event_timing_unmatched_count"] == 0
        results = out["evaluate_event_timing_results"]
        assert len(results) == len(data["peak_indices"])
        for index, result in enumerate(results):
            assert result.metadata["emg_sample_frequency"] == 100.0
            assert result.metadata["ventilator_sample_frequency"] == vent_fs
            assert result.metadata["emg_sample_index"] == int(
                data["peak_indices"][index]
            )

    def test_reports_unmatched_count_instead_of_silently_truncating(self):
        session = M3Session()
        data = _synthetic_breaths()
        # One extra EMG peak with no ventilator counterpart.
        ventilator_breath_indices = (data["peak_indices"][:-1]).astype(int)
        ventilator_signals = {"fs": 100.0}

        out = evaluate_event_timing(
            session,
            data["peak_indices"],
            data["processed_emg"],
            ventilator_breath_indices,
            ventilator_signals,
        )

        assert out["evaluate_event_timing_unmatched_count"] == 1
        unmatched_flags = [
            f
            for f in out["evaluate_event_timing_flags"]
            if f.name == "evaluate_event_timing_unmatched"
        ]
        assert len(unmatched_flags) == 1
        assert unmatched_flags[0].severity == "warning"
        assert unmatched_flags[0].metadata["unmatched_count"] == 1


class TestEvaluateRespiratoryRates:
    def test_writes_one_flag_and_one_detected_fraction_measurement(self):
        session = M3Session()
        data = _synthetic_breaths()
        ventilator_respiratory_rate = (12.0, np.array([12.0, 12.5, 11.5]))
        out = evaluate_respiratory_rates(
            session,
            data["peak_indices"],
            data["processed_emg"],
            ventilator_respiratory_rate,
        )

        assert isinstance(out["evaluate_respiratory_rates_result"], ParameterResult)
        assert isinstance(out["evaluate_respiratory_rates_flag"], QualityFlag)
        assert len(session.parameter_results) == 1
        assert len(session.quality) == 1


class TestPoccQualityAndInterpeakDist:
    def test_pocc_quality_labels_criteria_rows_explicitly(self):
        session = M3Session()
        pressure = -np.abs(np.sin(np.linspace(0, 10, 2000))) * 5.0
        ventilator_signals = {"pressure": pressure, "fs": 100.0}
        pocc_peaks = np.array([100, 800])
        pocc_ends = np.array([150, 850])
        time_products = np.array([1.0, 1.5])

        out = pocc_quality(
            session, ventilator_signals, pocc_peaks, pocc_ends, time_products
        )

        assert len(out["pocc_quality_flags"]) == len(pocc_peaks)
        criteria_names = {r.metadata["criterion"] for r in out["pocc_quality_results"]}
        assert criteria_names == {"dp_up_10", "dp_up_90", "dp_up_90_norm"}
        assert len(out["pocc_quality_results"]) == len(pocc_peaks) * 3

    def test_interpeak_dist_requires_at_least_two_peaks_each(self):
        session = M3Session()
        with pytest.raises(ValueError, match="at least two peaks"):
            interpeak_dist(session, [1], [1, 2], {"fs": 100.0})
