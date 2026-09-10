"""Stage 2 ReSurfEMG gap migration, Phase 8 (plan/stage2/
2_resurfemg_gap_migration_implementation_plan.md) - characterization tests
for empty, NaN, zero-denominator, and mismatched-input behavior in the
quality steps. Per the plan: "When upstream behavior appears unsafe or
incorrect, capture it as a named compatibility case ... do not hide a
correction inside an unrelated port" - the one real bug found here
(interpeak_dist crashing with ZeroDivisionError instead of matching
upstream's own inf/nan-with-warning behavior) was fixed as its own change,
documented by the test below rather than silently folded into something else.
"""

from __future__ import annotations

import pytest

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult
from m3resp.workflows.steps.emg import (
    detect_extreme_time_products,
    interpeak_dist,
    percentage_under_baseline,
    snr_pseudo,
)

pytest.importorskip("resurfemg")
np = pytest.importorskip("numpy")


class TestEmptyInputs:
    def test_snr_pseudo_with_no_peaks_produces_no_results_not_a_crash(self):
        session = M3Session()
        processed_emg = {
            "envelope": np.zeros(100),
            "fs": 100.0,
            "channel": 0,
            "metadata": {"labels": ["x"], "units": ["uV"]},
        }

        out = snr_pseudo(session, processed_emg, np.array([], dtype=int), np.zeros(100))

        assert out["snr_pseudo_results"] == []
        assert out["snr_pseudo_flags"] == []
        assert len(session.parameter_results) == 0


class TestNaNPropagation:
    def test_nan_envelope_produces_nan_measurements_not_silent_passing(self):
        session = M3Session()
        processed_emg = {
            "envelope": np.full(100, np.nan),
            "fs": 100.0,
            "channel": 0,
            "metadata": {"labels": ["x"], "units": ["uV"]},
        }

        out = snr_pseudo(
            session, processed_emg, np.array([10, 20]), np.full(100, np.nan)
        )

        results = out["snr_pseudo_results"]
        assert len(results) == 2
        for result in results:
            assert isinstance(result, ParameterResult)
            assert np.isnan(result.value)


class TestMismatchedLengths:
    def test_percentage_under_baseline_step_rejects_mismatched_arrays(self):
        session = M3Session()
        processed_emg = {
            "envelope": np.zeros(100),
            "fs": 100.0,
            "channel": 0,
            "metadata": {"labels": ["x"], "units": ["uV"]},
        }

        with pytest.raises(ValueError, match="equal length"):
            percentage_under_baseline(
                session,
                processed_emg,
                np.array([10, 20]),
                np.array([5]),  # one too few
                np.array([15, 25]),
                np.zeros(100),
            )


class TestZeroDenominator:
    def test_interpeak_dist_with_degenerate_ecg_peaks_returns_inf_not_a_crash(self):
        # Regression: this previously raised a raw ZeroDivisionError from
        # Python float division; fixed to use NumPy division, matching
        # upstream interpeak_dist's own behavior (a RuntimeWarning plus an
        # inf/nan ratio, not a raised exception) for a degenerate peak set
        # (three identical ECG "peaks" -> a zero median interval).
        session = M3Session()
        processed_emg = {"fs": 100.0}

        out = interpeak_dist(
            session,
            np.array([100, 100, 100]),
            np.array([50, 150, 250]),
            processed_emg,
        )

        ratio_result = next(
            r for r in out["interpeak_dist_result"] if r.name == "interpeak_dist_ratio"
        )
        assert np.isinf(ratio_result.value)
        # inf >= threshold is True, matching upstream's own comparison.
        assert out["interpeak_dist_flag"].passed is True


class TestFailedFit:
    def test_extreme_time_products_with_a_single_value_does_not_crash(self):
        # A single-element input degenerates the percentile-based bounds
        # (upper == lower == the one value), which is a legitimate "cannot
        # meaningfully evaluate extremeness" case, not a crash.
        session = M3Session()
        out = detect_extreme_time_products(session, np.array([5.0]), np.array([42]))

        assert len(out["detect_extreme_time_products_flags"]) == 1
        bounds = out["detect_extreme_time_products_bounds_result"]
        assert np.isfinite(bounds.metadata["lower_bound"])
        assert np.isfinite(bounds.metadata["upper_bound"])
