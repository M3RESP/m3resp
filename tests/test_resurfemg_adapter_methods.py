"""Stage 2 ReSurfEMG gap migration, Phase 8 (plan/stage2/
2_resurfemg_gap_migration_implementation_plan.md) - dedicated tests for the
Phase 1 `ReSurfEMGAdapter` methods (ECG, baseline, and quality operations):
`OptionalDependencyError` behavior when `resurfemg` can't be imported, and
validation of shapes/finite sample frequency/indices/array lengths before
calling upstream. Happy-path behavior against real synthetic data is already
exercised indirectly through the workflow-step tests
(test_emg_ecg_removal.py, test_emg_pocc_prerequisites.py,
test_emg_quality_steps.py), which all call these same adapter methods.
"""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from m3resp.adapters import ReSurfEMGAdapter
from m3resp.core.exceptions import OptionalDependencyError

pytest.importorskip("resurfemg")
np = pytest.importorskip("numpy")


class TestOptionalDependencyBehavior:
    """Every Phase 1 method must raise `OptionalDependencyError` - not a bare
    `ImportError` - the moment `resurfemg` can't be imported, and only when
    actually invoked (not at `m3resp` import time)."""

    @pytest.fixture
    def blocked_adapter(self, monkeypatch: pytest.MonkeyPatch) -> ReSurfEMGAdapter:
        real_import = builtins.__import__

        def _blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "resurfemg" or name.startswith("resurfemg."):
                raise ImportError(f"blocked for test: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)
        return ReSurfEMGAdapter()

    def test_detect_ecg_peaks(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.detect_ecg_peaks(np.zeros(100), sample_frequency=100.0)

    def test_gate_ecg(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.gate_ecg(np.zeros(100), [10, 20])

    def test_wavelet_denoise_ecg(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.wavelet_denoise_ecg(
                np.zeros(100), [10, 20], sample_frequency=100.0
            )

    def test_moving_baseline(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.moving_baseline(
                np.zeros(100), window_samples=10, step_samples=5
            )

    def test_slopesum_baseline(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.slopesum_baseline(
                np.zeros(100), window_samples=10, step_samples=5, sample_frequency=100.0
            )

    def test_snr_pseudo(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.snr_pseudo(
                np.zeros(100), [10, 20], np.zeros(100), sample_frequency=100.0
            )

    def test_pocc_quality(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.pocc_quality(np.zeros(100), [10, 20], [15, 25], [1.0, 1.0])

    def test_interpeak_distance(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.interpeak_distance([10, 20], [10, 20])

    def test_percentage_under_baseline(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.percentage_under_baseline(
                np.zeros(100),
                [10, 20],
                [5, 15],
                [15, 25],
                np.zeros(100),
                sample_frequency=100.0,
            )

    def test_detect_local_high_aub(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.detect_local_high_aub(np.array([1.0, 2.0, 3.0]))

    def test_detect_extreme_time_products(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.detect_extreme_time_products(np.array([1.0, 2.0, 3.0]))

    def test_detect_non_consecutive_manoeuvres(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.detect_non_consecutive_manoeuvres([0, 100], [50])

    def test_evaluate_bell_curve_error(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.evaluate_bell_curve_error(
                [10, 20],
                [5, 15],
                [15, 25],
                np.zeros(100),
                [1.0, 1.0],
                sample_frequency=100.0,
            )

    def test_evaluate_event_timing(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.evaluate_event_timing([1.0, 2.0], [1.1, 2.1])

    def test_evaluate_respiratory_rates(self, blocked_adapter: ReSurfEMGAdapter):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.evaluate_respiratory_rates([10, 20], 10.0, 12.0)


class TestValidation:
    """Adapter methods validate shapes, finite sample frequency, indices,
    thresholds, and array lengths before calling upstream (Phase 1
    acceptance criterion)."""

    @pytest.fixture
    def adapter(self) -> ReSurfEMGAdapter:
        return ReSurfEMGAdapter()

    def test_detect_ecg_peaks_rejects_non_integer_valued_frequency(self, adapter):
        with pytest.raises(ValueError, match="integer"):
            adapter.detect_ecg_peaks(np.zeros(100), sample_frequency=100.5)

    def test_detect_ecg_peaks_rejects_non_positive_frequency(self, adapter):
        with pytest.raises(ValueError, match="finite"):
            adapter.detect_ecg_peaks(np.zeros(100), sample_frequency=0.0)

    def test_detect_ecg_peaks_rejects_2d_signal(self, adapter):
        with pytest.raises(ValueError, match="1D"):
            adapter.detect_ecg_peaks(np.zeros((2, 100)), sample_frequency=100.0)

    def test_gate_ecg_rejects_unknown_fill_method(self, adapter):
        with pytest.raises(ValueError, match="fill_method"):
            adapter.gate_ecg(np.zeros(100), [10], fill_method=7)

    def test_gate_ecg_rejects_non_positive_gate_width(self, adapter):
        with pytest.raises(ValueError, match="gate_width_samples"):
            adapter.gate_ecg(np.zeros(100), [10], gate_width_samples=0)

    def test_wavelet_denoise_ecg_rejects_non_positive_levels(self, adapter):
        with pytest.raises(ValueError, match="levels"):
            adapter.wavelet_denoise_ecg(
                np.zeros(100), [10], sample_frequency=100.0, levels=0
            )

    def test_moving_baseline_rejects_non_positive_window(self, adapter):
        with pytest.raises(ValueError, match="window_samples"):
            adapter.moving_baseline(np.zeros(100), window_samples=0, step_samples=5)

    def test_moving_baseline_rejects_out_of_range_percentile(self, adapter):
        with pytest.raises(ValueError, match="percentile"):
            adapter.moving_baseline(
                np.zeros(100), window_samples=10, step_samples=5, percentile=150.0
            )

    def test_slopesum_baseline_rejects_non_integer_valued_frequency(self, adapter):
        with pytest.raises(ValueError, match="integer"):
            adapter.slopesum_baseline(
                np.zeros(100), window_samples=10, step_samples=5, sample_frequency=100.5
            )

    def test_snr_pseudo_rejects_non_integer_valued_frequency(self, adapter):
        with pytest.raises(ValueError, match="integer"):
            adapter.snr_pseudo(
                np.zeros(100), [10, 20], np.zeros(100), sample_frequency=100.5
            )

    def test_pocc_quality_rejects_mismatched_array_lengths(self, adapter):
        with pytest.raises(ValueError, match="equal length"):
            adapter.pocc_quality(np.zeros(100), [10, 20], [15], [1.0, 1.0])

    def test_interpeak_distance_requires_at_least_two_peaks_each(self, adapter):
        with pytest.raises(ValueError, match="at least two peaks"):
            adapter.interpeak_distance([10], [10, 20])

    def test_percentage_under_baseline_rejects_mismatched_lengths(self, adapter):
        with pytest.raises(ValueError, match="equal length"):
            adapter.percentage_under_baseline(
                np.zeros(100),
                [10, 20],
                [5, 15],
                [15],
                np.zeros(100),
                sample_frequency=100.0,
            )

    def test_evaluate_bell_curve_error_rejects_mismatched_lengths(self, adapter):
        with pytest.raises(ValueError, match="equal length"):
            adapter.evaluate_bell_curve_error(
                [10, 20],
                [5, 15],
                [15],
                np.zeros(100),
                [1.0, 1.0],
                sample_frequency=100.0,
            )

    def test_evaluate_event_timing_rejects_mismatched_lengths(self, adapter):
        with pytest.raises(ValueError, match="equal length"):
            adapter.evaluate_event_timing([1.0, 2.0], [1.1])

    def test_evaluate_respiratory_rates_rejects_non_positive_duration(self, adapter):
        with pytest.raises(ValueError, match="finite"):
            adapter.evaluate_respiratory_rates([10, 20], 0.0, 12.0)
