"""Regional and functional impedance: pixel impedance summed inside a mask.

Each pixel is multiplied by its mask value (NaN leaves it out, a weight counts
it partly, weight 0 adds nothing) and the pixels are summed per frame.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("eitprocessing")

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData, Vendor

import m3resp.workflows.steps  # noqa: F401 - registers the built-in steps
from m3resp import M3Session
from m3resp.data import PixelMap, PixelMask, Signal
from m3resp.eit.io import read_eit_file
from m3resp.eit.processing import regional_impedance
from m3resp.workflows.registry import describe_step, get_step

FS = 20.0
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")
with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as _handle:
    MANIFEST = json.load(_handle)
DATASETS = sorted(MANIFEST["inputs"])

# Three frames of a 2 x 2 image; each pixel has its own size so the sums are
# easy to check by hand.
PIXELS = np.array(
    [
        [[1.0, 10.0], [100.0, 1000.0]],
        [[2.0, 20.0], [200.0, 2000.0]],
        [[3.0, 30.0], [300.0, 3000.0]],
    ]
)


def _eit_data() -> EITData:
    return EITData(
        path="synthetic.bin",
        nframes=PIXELS.shape[0],
        time=np.arange(PIXELS.shape[0]) / FS,
        sample_frequency=FS,
        vendor=Vendor.DRAEGER,
        label="filtered",
        pixel_impedance=PIXELS.copy(),
    )


def _sequence() -> SimpleNamespace:
    return SimpleNamespace(continuous_data=DataCollection(ContinuousData))


def test_regional_impedance_sums_the_pixels_inside_the_mask():
    session = M3Session()
    sequence = _sequence()
    # Top-left pixel fully, bottom-left pixel at half weight, the rest left out.
    mask = PixelMask(name="left_half", values=[[1.0, np.nan], [0.5, np.nan]])

    result = get_step("eit.regional_impedance").func(
        _eit_data(), mask=mask, eit_sequence=sequence, session=session
    )

    expected = PIXELS[:, 0, 0] + 0.5 * PIXELS[:, 1, 0]
    np.testing.assert_allclose(result["regional_impedance"].values, expected)
    np.testing.assert_allclose(result["regional_impedance"].time, np.arange(3) / FS)
    assert "regional_impedance" in sequence.continuous_data

    native = result["regional_impedance_signal"]
    np.testing.assert_allclose(native.values, expected)
    assert native.channel == "regional_impedance"
    assert native.category == "impedance"
    assert native in list(session.signals)

    assert session.provenance[-1].action == "eit.regional_impedance"


def test_functional_impedance_keeps_weight_zero_apart_from_nan():
    session = M3Session()
    sequence = _sequence()
    # NaN: outside the lung space. 0: inside, adds nothing to the sum.
    mask = PixelMask(
        name="tiv_lungspace_mask",
        values=[[1.0, 0.0], [1.0, np.nan]],
        keep_zeros=True,
    )

    result = get_step("eit.functional_impedance").func(
        _eit_data(), mask=mask, eit_sequence=sequence, session=session
    )

    expected = PIXELS[:, 0, 0] + PIXELS[:, 1, 0]
    np.testing.assert_allclose(result["functional_impedance"].values, expected)
    assert "functional_impedance" in sequence.continuous_data
    assert result["functional_impedance_signal"].channel == "functional_impedance"
    assert session.provenance[-1].action == "eit.functional_impedance"


def test_functional_impedance_reads_the_tiv_lung_space_mask_by_default():
    description = describe_step("eit.functional_impedance")
    mask_input = next(a for a in description.input_artifacts if a.name == "mask")

    assert mask_input.default_context_key == "tiv_lungspace_result"


def test_both_outputs_can_feed_the_waveform_steps():
    for step, output in (
        ("eit.regional_impedance", "regional_impedance"),
        ("eit.functional_impedance", "functional_impedance"),
    ):
        outputs = {a.name: a for a in describe_step(step).output_artifacts}
        assert outputs[output].artifact_type == "eit_impedance_waveform"
        assert outputs[f"{output}_signal"].artifact_type == "eit_impedance_waveform"


def test_mask_with_the_wrong_shape_is_refused():
    session = M3Session()
    mask = PixelMask(name="too_small", values=[[1.0]])

    with pytest.raises(ValueError, match="shape"):
        get_step("eit.regional_impedance").func(
            _eit_data(), mask=mask, eit_sequence=_sequence(), session=session
        )


def test_upstream_mask_and_plain_grid_are_read_as_eitprocessing_reads_them():
    from eitprocessing.roi import PixelMask as UpstreamPixelMask

    session = M3Session()
    upstream = UpstreamPixelMask([[1.0, 0.0], [0.5, np.nan]], keep_zeros=True)

    from_upstream = get_step("eit.regional_impedance").func(
        _eit_data(), mask=upstream, eit_sequence=_sequence(), session=session
    )
    with pytest.warns(UserWarning, match="0 values"):
        from_grid = get_step("eit.regional_impedance").func(
            _eit_data(),
            mask=[[1.0, 0.0], [0.5, np.nan]],
            eit_sequence=_sequence(),
            session=session,
        )

    expected = PIXELS[:, 0, 0] + 0.5 * PIXELS[:, 1, 0]
    np.testing.assert_array_equal(from_upstream["regional_impedance"].values, expected)
    np.testing.assert_array_equal(from_grid["regional_impedance"].values, expected)


def test_a_mask_of_another_type_is_refused():
    from m3resp.core.exceptions import UnsupportedWorkflowError

    with pytest.raises(UnsupportedWorkflowError, match="got str"):
        get_step("eit.regional_impedance").func(
            _eit_data(), mask="lungs", eit_sequence=_sequence(), session=M3Session()
        )


def test_text_mask_values_are_refused():
    from m3resp.core.exceptions import UnsupportedWorkflowError

    with pytest.raises(UnsupportedWorkflowError, match="without changing"):
        get_step("eit.regional_impedance").func(
            _eit_data(),
            mask=[["1", "0"], ["0", "1"]],
            eit_sequence=_sequence(),
            session=M3Session(),
        )


def test_plain_boolean_mask_is_read_without_a_zero_conversion_warning():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = get_step("eit.regional_impedance").func(
            _eit_data(),
            mask=[[True, False], [False, True]],
            eit_sequence=_sequence(),
            session=M3Session(),
        )

    assert not caught
    np.testing.assert_array_equal(
        result["regional_impedance"].values, PIXELS[:, 0, 0] + PIXELS[:, 1, 1]
    )


GOLDEN_MASKS = (
    "tiv_lungspace/mask",
    "amplitude_lungspace/mask",
    "watershed_lungspace/mask",
    "filter_by_size/mask",
)


def _golden_inputs(dataset: str) -> tuple[Any, dict[str, np.ndarray]]:
    entry = MANIFEST["inputs"][dataset]
    pixels = read_eit_file(os.path.join(REPO_ROOT, entry["file"]), entry["vendor"])
    with np.load(os.path.join(GOLDEN_DIR, dataset, "roi.npz")) as data:
        masks = {key: data[key] for key in GOLDEN_MASKS}
    return pixels.pixel_impedance, masks


@pytest.mark.parametrize("dataset", DATASETS)
def test_masks_and_sums_equal_eitprocessing_on_the_golden_masks(dataset):
    from eitprocessing.roi import PixelMask as UpstreamPixelMask

    from m3resp.adapters.eitprocessing_adapter import signal_to_eit_data

    pixels, masks = _golden_inputs(dataset)
    eit_data = signal_to_eit_data(pixels)
    rng = np.random.default_rng(7)
    weights = rng.uniform(0.0, 1.0, size=pixels.values.shape[1:])
    weights[rng.uniform(size=weights.shape) < 0.3] = np.nan
    weights[0, 0] = 0.0
    masks["weighted"] = weights

    for key, grid in masks.items():
        mask = PixelMask(name=key, values=grid, keep_zeros=True)
        upstream = UpstreamPixelMask(grid, keep_zeros=True).apply(eit_data)

        masked = mask.apply(pixels)
        summed = regional_impedance(pixels, mask)

        np.testing.assert_array_equal(masked.values, upstream.pixel_impedance)
        np.testing.assert_array_equal(
            summed.values, upstream.get_summed_impedance().values
        )
        np.testing.assert_array_equal(summed.time, eit_data.time)


@pytest.mark.parametrize("dataset", DATASETS)
def test_step_waveform_on_the_sequence_equals_eitprocessing(dataset):
    from eitprocessing.roi import PixelMask as UpstreamPixelMask

    from m3resp.adapters.eitprocessing_adapter import signal_to_eit_data

    pixels, masks = _golden_inputs(dataset)
    grid = masks["tiv_lungspace/mask"]
    sequence = _sequence()

    result = get_step("eit.functional_impedance").func(
        pixels,
        mask=PixelMask(name="tiv_lungspace_mask", values=grid),
        eit_sequence=sequence,
        session=M3Session(),
    )

    expected = (
        UpstreamPixelMask(grid)
        .apply(signal_to_eit_data(pixels))
        .get_summed_impedance(
            return_label="functional_impedance",
            name="Functional impedance",
            description="Impedance summed over the pixels of a mask.",
        )
    )
    stored = sequence.continuous_data["functional_impedance"]
    assert stored == expected
    for field in ("label", "name", "description", "unit", "category", "parameters"):
        assert getattr(stored, field) == getattr(expected, field), field
    assert stored.locked == expected.locked
    np.testing.assert_array_equal(
        result["functional_impedance"].values, expected.values
    )
    assert result["functional_impedance"].method == (
        "m3resp.eit.processing.regional_impedance"
    )


class TestPixelMaskApply:
    """``PixelMask.apply`` multiplies the last two axes by the mask."""

    MASK = PixelMask(name="m", values=[[1.0, np.nan], [0.5, 0.0]], keep_zeros=True)

    def test_array_of_any_number_of_axes(self):
        images = np.arange(16.0).reshape(2, 2, 2, 2)

        masked = self.MASK.apply(images)

        np.testing.assert_array_equal(
            masked[1, 0], images[1, 0] * np.array([[1.0, np.nan], [0.5, 0.0]])
        )
        np.testing.assert_array_equal(self.MASK.apply(images[0, 0]), masked[0, 0])

    def test_signal_keeps_its_other_fields(self):
        signal = Signal(
            values=PIXELS.copy(),
            time=np.arange(3) / FS,
            sample_frequency=FS,
            name="images",
            modality="eit",
            source="draeger",
            metadata={"note": 1},
        )

        masked = self.MASK.apply(signal)

        assert masked is not signal
        np.testing.assert_array_equal(masked.values[:, 0, 0], PIXELS[:, 0, 0])
        assert np.isnan(masked.values[:, 0, 1]).all()
        np.testing.assert_array_equal(masked.values[:, 1, 1], 0.0)
        assert (masked.name, masked.source, masked.sample_frequency) == (
            "images",
            "draeger",
            FS,
        )
        assert masked.metadata == signal.metadata
        assert masked.metadata is not signal.metadata
        np.testing.assert_array_equal(signal.values, PIXELS)

    def test_pixel_map(self):
        pixel_map = PixelMap(name="tiv", values=[[2.0, 2.0], [2.0, 2.0]], unit="AU")

        masked = self.MASK.apply(pixel_map)

        np.testing.assert_array_equal(masked.values, [[2.0, np.nan], [1.0, 0.0]])
        assert (masked.name, masked.unit) == ("tiv", pixel_map.unit)

    def test_wrong_size_and_wrong_type_are_refused(self):
        with pytest.raises(ValueError, match="size of the mask"):
            self.MASK.apply(np.zeros((3, 2, 3)))
        with pytest.raises(TypeError, match="got list"):
            self.MASK.apply([[1.0, 2.0], [3.0, 4.0]])
