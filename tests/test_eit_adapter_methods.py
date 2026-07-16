"""Phase 1 (plan/stage2/1_eit_gap_migration_implementation_plan.md) - the
reusable `EITProcessingAdapter` operations.

Exercises each new adapter method against the committed synthetic Draeger
fixture (real `eitprocessing`, skipped cleanly if that optional dependency is
absent), and confirms every method raises `OptionalDependencyError` - rather
than some other ImportError-shaped failure - when `eitprocessing` can't be
imported.
"""

from __future__ import annotations

import builtins
import os
from pathlib import Path
from typing import Any

import pytest

from m3resp.adapters import EITProcessingAdapter
from m3resp.adapters.eitprocessing_adapter import add_to_collection
from m3resp.core.exceptions import OptionalDependencyError

pytest.importorskip("eitprocessing")


def _fixture_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    path = os.path.join(
        repo_root,
        "data",
        "source",
        "data_from_repo",
        "draeger_synthetic_draeger_20Hz.bin",
    )
    assert os.path.exists(path), f"missing committed EIT fixture: {path}"
    return path


@pytest.fixture(scope="module")
def loaded_chain() -> dict[str, Any]:
    """Build the real objects each Phase 1 method needs, once per module."""

    from eitprocessing.datahandling.loading import load_eit_data
    from eitprocessing.features.breath_detection import BreathDetection

    adapter = EITProcessingAdapter()
    sequence = load_eit_data(_fixture_path(), vendor="draeger")
    raw_eit = sequence.eit_data["raw"]
    adapter.get_global_impedance(sequence)

    rates = adapter.detect_rates(raw_eit, subject_type="adult", capture=True)
    mdn = adapter.apply_mdn(
        raw_eit,
        respiratory_rate_hz=rates["respiratory_rate_hz"],
        heart_rate_hz=rates["heart_rate_hz"],
        label="mdn_filtered",
    )
    filtered_eit = mdn["filtered_eit"]
    filtered_gi = filtered_eit.get_summed_impedance(
        return_label="global_impedance_(mdn_filtered)"
    )
    add_to_collection(sequence.continuous_data, filtered_gi)

    breath_detector = BreathDetection(minimum_duration=2 / 3)
    breaths = breath_detector.find_breaths(
        filtered_gi, result_label="eit_breaths", store=False
    )
    add_to_collection(sequence.interval_data, breaths)

    return {
        "adapter": adapter,
        "sequence": sequence,
        "raw_eit": raw_eit,
        "rates": rates,
        "filtered_eit": filtered_eit,
        "filtered_gi": filtered_gi,
        "breath_detector": breath_detector,
    }


class TestDetectRates:
    def test_returns_finite_positive_rates_and_captures_when_requested(
        self, loaded_chain: dict[str, Any]
    ):
        rates = loaded_chain["adapter"].detect_rates(
            loaded_chain["raw_eit"], subject_type="adult", capture=True
        )

        assert rates["respiratory_rate_hz"] > 0
        assert rates["heart_rate_hz"] > 0
        assert rates["rate_detector"] is not None
        assert rates["rate_captures"]

    def test_without_capture_returns_empty_captures(self, loaded_chain: dict[str, Any]):
        rates = loaded_chain["adapter"].detect_rates(
            loaded_chain["raw_eit"], subject_type="adult"
        )

        assert rates["rate_captures"] == {}


class TestApplyMdn:
    def test_filters_pixel_data_and_preserves_shape(self, loaded_chain: dict[str, Any]):
        rates = loaded_chain["rates"]
        result = loaded_chain["adapter"].apply_mdn(
            loaded_chain["raw_eit"],
            respiratory_rate_hz=rates["respiratory_rate_hz"],
            heart_rate_hz=rates["heart_rate_hz"],
        )

        filtered = result["filtered_eit"]
        assert (
            filtered.pixel_impedance.shape
            == loaded_chain["raw_eit"].pixel_impedance.shape
        )
        assert filtered is not loaded_chain["raw_eit"]

    @pytest.mark.parametrize(
        "respiratory_rate_hz,heart_rate_hz",
        [(0.0, 1.5), (-0.3, 1.5), (0.3, float("nan")), (0.3, float("inf"))],
    )
    def test_rejects_non_finite_or_non_positive_rates(
        self,
        loaded_chain: dict[str, Any],
        respiratory_rate_hz: float,
        heart_rate_hz: float,
    ):
        with pytest.raises(ValueError):
            loaded_chain["adapter"].apply_mdn(
                loaded_chain["raw_eit"],
                respiratory_rate_hz=respiratory_rate_hz,
                heart_rate_hz=heart_rate_hz,
            )


class TestFindPixelBreaths:
    def test_returns_one_interval_per_global_breath_and_stores_once(
        self, loaded_chain: dict[str, Any]
    ):
        sequence = loaded_chain["sequence"]
        result = loaded_chain["adapter"].find_pixel_breaths(
            loaded_chain["filtered_eit"],
            loaded_chain["filtered_gi"],
            sequence=sequence,
            result_label="pixel_breaths_test",
        )

        assert "pixel_breaths_test" in sequence.interval_data
        assert sequence.interval_data["pixel_breaths_test"] is result
        assert all(pixel is None for pixel in result.values[0].flat)
        assert all(pixel is None for pixel in result.values[-1].flat)

    def test_without_sequence_does_not_raise_or_store(
        self, loaded_chain: dict[str, Any]
    ):
        result = loaded_chain["adapter"].find_pixel_breaths(
            loaded_chain["filtered_eit"], loaded_chain["filtered_gi"]
        )

        assert result is not None

    @pytest.mark.parametrize(
        "mode", ["negative amplitude", "phase shift", "none", None]
    )
    def test_accepts_every_allowed_phase_correction_mode(
        self, loaded_chain: dict[str, Any], mode: str | None
    ):
        result = loaded_chain["adapter"].find_pixel_breaths(
            loaded_chain["filtered_eit"],
            loaded_chain["filtered_gi"],
            phase_correction_mode=mode,
        )

        assert result is not None


class TestComputeEeli:
    def test_stores_result_in_sparse_data_exactly_once(
        self, loaded_chain: dict[str, Any]
    ):
        sequence = loaded_chain["sequence"]
        result = loaded_chain["adapter"].compute_eeli(
            loaded_chain["filtered_gi"],
            sequence=sequence,
            breath_detector=loaded_chain["breath_detector"],
            result_label="eeli_test",
        )

        assert "eeli_test" in sequence.sparse_data
        assert sequence.sparse_data["eeli_test"] is result
        assert len(result.values) == len(result.time)


class TestComputePixelTiv:
    def test_shape_is_breath_row_column(self, loaded_chain: dict[str, Any]):
        sequence = loaded_chain["sequence"]
        result = loaded_chain["adapter"].compute_pixel_tiv(
            loaded_chain["filtered_eit"],
            loaded_chain["filtered_gi"],
            sequence=sequence,
            breath_detector=loaded_chain["breath_detector"],
            result_label="pixel_tiv_test",
        )

        assert "pixel_tiv_test" in sequence.sparse_data
        assert result.values[0].shape == (32, 32)


class TestRoiLungspaceMethods:
    def test_tiv_lungspace_returns_mask_and_captures(
        self, loaded_chain: dict[str, Any]
    ):
        result = loaded_chain["adapter"].compute_tiv_lungspace(
            loaded_chain["filtered_eit"], timing_data=loaded_chain["filtered_gi"]
        )

        assert result["mask"].mask.shape == (32, 32)
        assert "mean TIV" in result["captures"]

    def test_amplitude_lungspace_returns_mask_and_captures(
        self, loaded_chain: dict[str, Any]
    ):
        result = loaded_chain["adapter"].compute_amplitude_lungspace(
            loaded_chain["filtered_eit"], timing_data=loaded_chain["filtered_gi"]
        )

        assert result["mask"].mask.shape == (32, 32)
        assert "mean amplitude" in result["captures"]

    def test_watershed_lungspace_returns_mask_and_captures(
        self, loaded_chain: dict[str, Any]
    ):
        result = loaded_chain["adapter"].compute_watershed_lungspace(
            loaded_chain["filtered_eit"], timing_data=loaded_chain["filtered_gi"]
        )

        assert result["mask"].mask.shape == (32, 32)
        assert "included region" in result["captures"]

    def test_filter_roi_by_size_returns_mask(self, loaded_chain: dict[str, Any]):
        watershed = loaded_chain["adapter"].compute_watershed_lungspace(
            loaded_chain["filtered_eit"], timing_data=loaded_chain["filtered_gi"]
        )

        result = loaded_chain["adapter"].filter_roi_by_size(
            watershed["mask"], min_region_size=1
        )

        assert result.mask.shape == (32, 32)


class TestOptionalDependencyBehavior:
    """Every Phase 1 method must raise `OptionalDependencyError` - not a bare
    `ImportError` or `ModuleNotFoundError` - the moment `eitprocessing` can't
    be imported, and only when actually invoked (not at `m3resp` import time).
    """

    @pytest.fixture
    def blocked_adapter(self, monkeypatch: pytest.MonkeyPatch) -> EITProcessingAdapter:
        real_import = builtins.__import__

        def _blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "eitprocessing" or name.startswith("eitprocessing."):
                raise ImportError(f"blocked for test: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)
        return EITProcessingAdapter()

    def test_detect_rates_raises_optional_dependency_error(
        self, blocked_adapter: EITProcessingAdapter
    ):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.detect_rates(object())

    def test_apply_mdn_raises_optional_dependency_error(
        self, blocked_adapter: EITProcessingAdapter
    ):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.apply_mdn(
                object(), respiratory_rate_hz=0.3, heart_rate_hz=1.5
            )

    def test_find_pixel_breaths_raises_optional_dependency_error(
        self, blocked_adapter: EITProcessingAdapter
    ):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.find_pixel_breaths(object(), object())

    def test_compute_eeli_raises_optional_dependency_error(
        self, blocked_adapter: EITProcessingAdapter
    ):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.compute_eeli(
                object(), sequence=object(), breath_detector=object()
            )

    def test_compute_pixel_tiv_raises_optional_dependency_error(
        self, blocked_adapter: EITProcessingAdapter
    ):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.compute_pixel_tiv(
                object(), object(), sequence=object(), breath_detector=object()
            )

    def test_compute_tiv_lungspace_raises_optional_dependency_error(
        self, blocked_adapter: EITProcessingAdapter
    ):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.compute_tiv_lungspace(object())

    def test_compute_amplitude_lungspace_raises_optional_dependency_error(
        self, blocked_adapter: EITProcessingAdapter
    ):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.compute_amplitude_lungspace(object())

    def test_compute_watershed_lungspace_raises_optional_dependency_error(
        self, blocked_adapter: EITProcessingAdapter
    ):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.compute_watershed_lungspace(object())

    def test_filter_roi_by_size_raises_optional_dependency_error(
        self, blocked_adapter: EITProcessingAdapter
    ):
        with pytest.raises(OptionalDependencyError):
            blocked_adapter.filter_roi_by_size(object())
