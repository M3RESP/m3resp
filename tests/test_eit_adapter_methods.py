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

import numpy as np
import pytest

from m3resp.adapters import EITProcessingAdapter
from m3resp.adapters.eitprocessing_adapter import (
    add_to_collection,
    as_signal,
    breaths_to_interval_data,
    signal_to_eit_data,
)
from m3resp.data import BreathEvent, IntervalData, PixelBreaths, PixelMask, Signal
from m3resp.eit.processing import (
    BreathDetectionSettings,
    RateDetectionSettings,
    end_expiratory_lung_impedance,
    find_breaths,
    find_pixel_breaths,
    global_impedance,
    pixel_end_expiratory_lung_impedance,
    pixel_tidal_impedance_variation,
    tidal_impedance_variation,
)
from m3resp.eit.roi import (
    amplitude_lungspace,
    filter_mask_by_size,
    tiv_lungspace,
    watershed_lungspace,
)

pytest.importorskip("eitprocessing")


class _ModernCollection:
    def __init__(self) -> None:
        self.calls: list[tuple[object, bool]] = []

    def add(self, value: object, *, overwrite: bool = False) -> None:
        self.calls.append((value, overwrite))


class _LegacyCollection:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def add(self, value: object) -> None:
        self.calls.append(value)


class _InvalidValueCollection:
    def add(self, value: object, *, overwrite: bool = False) -> None:
        raise TypeError("value must be a data container")


def test_add_to_collection_uses_overwrite_when_supported():
    collection = _ModernCollection()

    add_to_collection(collection, "value")

    assert collection.calls == [("value", True)]


def test_add_to_collection_supports_legacy_collections_without_overwrite():
    collection = _LegacyCollection()

    add_to_collection(collection, "value")

    assert collection.calls == ["value"]


def test_add_to_collection_propagates_an_internal_type_error():
    with pytest.raises(TypeError, match="value must be a data container"):
        add_to_collection(_InvalidValueCollection(), "wrong-value")


def _fixture_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    path = os.path.join(
        repo_root,
        "tests",
        "data",
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
    filtered_eit = signal_to_eit_data(mdn["filtered_eit"])
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
        assert isinstance(filtered, Signal)
        assert filtered.values.shape == loaded_chain["raw_eit"].pixel_impedance.shape
        assert filtered.method == "m3resp.eit.processing.mdn_filter"
        assert result["filter_captures"]["filtered_data"] is filtered

    def test_the_eitprocessing_images_equal_mdn_filter_of_eitprocessing(
        self, loaded_chain: dict[str, Any]
    ):
        from eitprocessing.filters.mdn import MDNFilter

        rates = loaded_chain["rates"]
        names = {
            "label": "mdn_filtered",
            "name": "MDN-filtered EIT data",
            "description": "EIT data filtered with MDN heart-rate noise removal.",
        }
        expected = MDNFilter(
            respiratory_rate=rates["respiratory_rate_hz"],
            heart_rate=rates["heart_rate_hz"],
        ).apply(loaded_chain["raw_eit"], **names)

        result = loaded_chain["adapter"].apply_mdn(
            loaded_chain["raw_eit"],
            respiratory_rate_hz=rates["respiratory_rate_hz"],
            heart_rate_hz=rates["heart_rate_hz"],
            **names,
        )
        filtered = signal_to_eit_data(result["filtered_eit"])

        assert filtered == expected
        for field in ("label", "name", "description", "path", "vendor"):
            assert getattr(filtered, field) == getattr(expected, field), field

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


def _chain_breaths(
    loaded_chain: dict[str, Any], minimum_duration: float = 2 / 3
) -> list[BreathEvent]:
    """The breaths of the filtered global impedance of the test file."""

    return find_breaths(
        as_signal(loaded_chain["filtered_gi"]),
        BreathDetectionSettings(minimum_duration=minimum_duration),
    )


class TestFindPixelBreaths:
    def test_gives_m3resp_pixel_breaths_and_leaves_the_sequence_unchanged(
        self, loaded_chain: dict[str, Any]
    ):
        sequence = loaded_chain["sequence"]
        stored_before = dict(sequence.interval_data)
        result = loaded_chain["adapter"].find_pixel_breaths(
            loaded_chain["filtered_eit"],
            loaded_chain["filtered_gi"],
            result_label="pixel_breaths_test",
        )

        assert dict(sequence.interval_data) == stored_before
        assert isinstance(result, PixelBreaths)
        assert result == find_pixel_breaths(
            as_signal(loaded_chain["filtered_eit"]),
            as_signal(loaded_chain["filtered_gi"]),
            _chain_breaths(loaded_chain),
            name="pixel_breaths_test",
        )
        assert not result.has_breath[0].any()
        assert not result.has_breath[-1].any()
        assert result.has_breath[1:-1].any()

    def test_uses_the_given_breaths(self, loaded_chain: dict[str, Any]):
        breaths = _chain_breaths(loaded_chain)

        result = loaded_chain["adapter"].find_pixel_breaths(
            loaded_chain["filtered_eit"],
            loaded_chain["filtered_gi"],
            breaths=breaths_to_interval_data(breaths),
        )

        assert all(a is b for a, b in zip(result.breaths, breaths, strict=True))

    def test_other_minimum_duration_takes_the_sign_from_the_default_breaths(
        self, loaded_chain: dict[str, Any]
    ):
        # eitprocessing takes the sign of each pixel from the breaths found
        # with the default settings, whatever the minimum duration.
        result = loaded_chain["adapter"].find_pixel_breaths(
            loaded_chain["filtered_eit"],
            loaded_chain["filtered_gi"],
            minimum_duration_seconds=1.0,
        )

        assert result == find_pixel_breaths(
            as_signal(loaded_chain["filtered_eit"]),
            as_signal(loaded_chain["filtered_gi"]),
            _chain_breaths(loaded_chain, minimum_duration=1.0),
            sign_breaths=_chain_breaths(loaded_chain),
        )

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

        assert isinstance(result, PixelBreaths)


class TestComputeTivAndEeli:
    def test_give_m3resp_values_and_leave_the_sequence_unchanged(
        self, loaded_chain: dict[str, Any]
    ):
        sequence = loaded_chain["sequence"]
        adapter = loaded_chain["adapter"]
        waveform = as_signal(loaded_chain["filtered_gi"])
        breaths = find_breaths(
            waveform, BreathDetectionSettings(minimum_duration=2 / 3)
        )
        sparse_before = dict(sequence.sparse_data)

        eeli = adapter.compute_eeli(
            loaded_chain["filtered_gi"], breaths=breaths, result_label="eeli_test"
        )
        tiv = adapter.compute_continuous_tiv(
            loaded_chain["filtered_gi"],
            breaths=breaths_to_interval_data(breaths),
            tiv_method="mean",
        )

        assert dict(sequence.sparse_data) == sparse_before
        assert eeli == end_expiratory_lung_impedance(
            waveform, breaths, name="eeli_test"
        )
        assert tiv == tidal_impedance_variation(waveform, breaths, method="mean")
        assert len(eeli) == len(tiv) == len(breaths) > 0


class TestComputePixelTivAndEeli:
    def test_global_timing_gives_m3resp_maps_and_leaves_the_sequence_unchanged(
        self, loaded_chain: dict[str, Any]
    ):
        sequence = loaded_chain["sequence"]
        sparse_before = dict(sequence.sparse_data)
        breaths = _chain_breaths(loaded_chain)
        pixels = as_signal(loaded_chain["filtered_eit"])

        tiv = loaded_chain["adapter"].compute_pixel_tiv(
            loaded_chain["filtered_eit"],
            breaths=breaths_to_interval_data(breaths),
            tiv_method="expiratory",
            result_label="pixel_tiv_test",
        )
        eeli = loaded_chain["adapter"].compute_pixel_eeli(
            loaded_chain["filtered_eit"], breaths=breaths
        )

        assert dict(sequence.sparse_data) == sparse_before
        assert tiv == pixel_tidal_impedance_variation(
            pixels, breaths, method="expiratory", name="pixel_tiv_test"
        )
        assert eeli == pixel_end_expiratory_lung_impedance(pixels, breaths)
        assert tiv.values[0].shape == (32, 32)
        assert len(tiv) == len(eeli) == len(breaths)

    def test_pixel_breaths_give_the_pixel_timing(self, loaded_chain: dict[str, Any]):
        adapter = loaded_chain["adapter"]
        pixel_breaths = adapter.find_pixel_breaths(
            loaded_chain["filtered_eit"], loaded_chain["filtered_gi"]
        )
        pixels = as_signal(loaded_chain["filtered_eit"])

        tiv = adapter.compute_pixel_tiv(
            loaded_chain["filtered_eit"], breaths=pixel_breaths
        )
        eeli = adapter.compute_pixel_eeli(
            loaded_chain["filtered_eit"], breaths=pixel_breaths
        )

        assert tiv == pixel_tidal_impedance_variation(pixels, pixel_breaths)
        assert eeli == pixel_end_expiratory_lung_impedance(pixels, pixel_breaths)
        assert tiv.metadata["tiv_timing"] == "pixel"
        assert eeli.metadata["eeli_timing"] == "pixel"


class TestRoiLungspaceMethods:
    """The adapter methods run the m3resp lung-space functions on the breaths
    found with the default settings, as eitprocessing does."""

    def test_lungspace_methods_equal_the_m3resp_functions(
        self, loaded_chain: dict[str, Any]
    ):
        adapter = loaded_chain["adapter"]
        pixels = as_signal(loaded_chain["filtered_eit"])
        waveform = as_signal(loaded_chain["filtered_gi"])
        breaths = find_breaths(waveform, BreathDetectionSettings())

        tiv = adapter.compute_tiv_lungspace(
            loaded_chain["filtered_eit"], timing_data=loaded_chain["filtered_gi"]
        )
        amplitude = adapter.compute_amplitude_lungspace(
            loaded_chain["filtered_eit"], timing_data=loaded_chain["filtered_gi"]
        )
        watershed = adapter.compute_watershed_lungspace(
            loaded_chain["filtered_eit"], timing_data=loaded_chain["filtered_gi"]
        )

        for result, expected in (
            (tiv, tiv_lungspace(pixels, breaths)),
            (amplitude, amplitude_lungspace(pixels, waveform, breaths)),
            (watershed, watershed_lungspace(pixels, waveform, breaths)),
        ):
            assert isinstance(result["mask"], PixelMask)
            assert result["mask"].shape == (32, 32)
            np.testing.assert_array_equal(result["mask"].values, expected.values)
        assert set(tiv["captures"]) == {"TIV per breath", "mean TIV"}
        assert set(amplitude["captures"]) == {"amplitude per breath", "mean amplitude"}
        assert "included region" in watershed["captures"]

    def test_given_breaths_equal_the_found_breaths(self, loaded_chain: dict[str, Any]):
        adapter = loaded_chain["adapter"]
        waveform = as_signal(loaded_chain["filtered_gi"])
        breaths = find_breaths(waveform, BreathDetectionSettings())

        found = adapter.compute_watershed_lungspace(
            loaded_chain["filtered_eit"], timing_data=waveform
        )
        given = adapter.compute_watershed_lungspace(
            loaded_chain["filtered_eit"],
            timing_data=waveform,
            breaths=breaths_to_interval_data(breaths),
        )

        np.testing.assert_array_equal(given["mask"].values, found["mask"].values)

    def test_no_timing_data_uses_the_sum_of_all_pixels(
        self, loaded_chain: dict[str, Any]
    ):
        adapter = loaded_chain["adapter"]
        pixels = as_signal(loaded_chain["filtered_eit"])

        without = adapter.compute_tiv_lungspace(loaded_chain["filtered_eit"])
        summed = adapter.compute_tiv_lungspace(
            loaded_chain["filtered_eit"], timing_data=global_impedance(pixels)
        )

        np.testing.assert_array_equal(without["mask"].values, summed["mask"].values)

    def test_filter_roi_by_size_returns_an_m3resp_mask(
        self, loaded_chain: dict[str, Any]
    ):
        watershed = loaded_chain["adapter"].compute_watershed_lungspace(
            loaded_chain["filtered_eit"], timing_data=loaded_chain["filtered_gi"]
        )

        result = loaded_chain["adapter"].filter_roi_by_size(
            watershed["mask"], min_region_size=1
        )

        assert isinstance(result, PixelMask)
        np.testing.assert_array_equal(
            result.values,
            filter_mask_by_size(watershed["mask"], min_region_size=1).values,
        )


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

    def test_detect_rates_runs_without_eitprocessing(
        self, blocked_adapter: EITProcessingAdapter
    ):
        fs = 20.0
        time = np.arange(int(40 * fs)) / fs
        breathing = np.sin(2 * np.pi * 0.25 * time)
        images = Signal(
            values=breathing[:, None, None] * np.ones((len(time), 4, 4)) + 5.0,
            time=time,
            sample_frequency=fs,
            name="images",
            modality="eit",
        )
        rates = blocked_adapter.detect_rates(images, welch_window_seconds=20.0)
        assert rates["respiratory_rate_hz"] == pytest.approx(0.25, abs=0.01)
        assert isinstance(rates["rate_detector"], RateDetectionSettings)

    def test_find_breaths_runs_without_eitprocessing(
        self, blocked_adapter: EITProcessingAdapter
    ):
        fs = 20.0
        time = np.arange(int(30 * fs)) / fs
        waveform = Signal(
            values=np.sin(2 * np.pi * 0.25 * time),
            time=time,
            sample_frequency=fs,
            name="global_impedance",
            modality="eit",
        )
        breaths = blocked_adapter.find_breaths(waveform)
        assert isinstance(breaths, IntervalData)
        assert len(breaths.intervals) == 6

    def test_apply_mdn_runs_without_eitprocessing(
        self, blocked_adapter: EITProcessingAdapter
    ):
        fs = 20.0
        time = np.arange(int(30 * fs)) / fs
        images = Signal(
            values=np.sin(2 * np.pi * 1.5 * time)[:, None, None]
            * np.ones((len(time), 4, 4)),
            time=time,
            sample_frequency=fs,
            name="images",
            modality="eit",
        )
        result = blocked_adapter.apply_mdn(
            images, respiratory_rate_hz=0.3, heart_rate_hz=1.5
        )
        filtered = result["filtered_eit"]
        assert filtered.values.shape == images.values.shape
        assert filtered.method == "m3resp.eit.processing.mdn_filter"

    def test_pixel_methods_run_without_eitprocessing(
        self, blocked_adapter: EITProcessingAdapter
    ):
        fs = 20.0
        time = np.arange(int(40 * fs)) / fs
        breathing = np.sin(2 * np.pi * 0.25 * time)
        images = Signal(
            values=breathing[:, None, None] * np.ones((len(time), 2, 2)) + 5.0,
            time=time,
            sample_frequency=fs,
            name="images",
            modality="eit",
        )
        waveform = Signal(
            values=4 * breathing + 20.0,
            time=time,
            sample_frequency=fs,
            name="global_impedance",
            modality="eit",
        )

        pixel_breaths = blocked_adapter.find_pixel_breaths(images, waveform)
        tiv = blocked_adapter.compute_pixel_tiv(images, breaths=pixel_breaths)
        eeli = blocked_adapter.compute_pixel_eeli(images, breaths=pixel_breaths.breaths)

        assert pixel_breaths.has_breath[1:-1].all()
        np.testing.assert_allclose(tiv.values[1].values, 2.0, atol=1e-2)
        np.testing.assert_allclose(eeli.values[1].values, 4.0, atol=1e-2)

    def test_compute_tiv_and_eeli_run_without_eitprocessing(
        self, blocked_adapter: EITProcessingAdapter
    ):
        waveform = Signal(
            values=np.array([1.0, 3.0, 2.0, 5.0, 1.0]),
            time=np.arange(5.0),
            sample_frequency=1.0,
            name="global_impedance",
            modality="eit",
        )
        breaths = [BreathEvent("eit", 0.0, 2.0, extremum_time=1.0)]

        tiv = blocked_adapter.compute_continuous_tiv(waveform, breaths=breaths)
        eeli = blocked_adapter.compute_eeli(waveform, breaths=breaths)

        assert tiv.values[0] == 2.0 and eeli.values[0] == 2.0

    def test_lungspace_methods_run_without_eitprocessing(
        self, blocked_adapter: EITProcessingAdapter
    ):
        fs = 20.0
        time = np.arange(int(60 * fs)) / fs
        rows, columns = np.mgrid[0:8, 0:8]
        amplitude = np.exp(-((rows - 3.5) ** 2 + (columns - 3.5) ** 2) / 6.0)
        breathing = 0.5 - 0.5 * np.cos(2 * np.pi * 0.25 * time)
        images = Signal(
            values=10.0 + breathing[:, None, None] * amplitude,
            time=time,
            sample_frequency=fs,
            name="images",
            modality="eit",
        )

        tiv = blocked_adapter.compute_tiv_lungspace(images)
        amplitude_mask = blocked_adapter.compute_amplitude_lungspace(images)
        watershed = blocked_adapter.compute_watershed_lungspace(images)
        filtered = blocked_adapter.filter_roi_by_size(watershed["mask"])

        for mask in (tiv["mask"], amplitude_mask["mask"], watershed["mask"], filtered):
            assert isinstance(mask, PixelMask)
            assert mask.values[3, 3] == 1.0
