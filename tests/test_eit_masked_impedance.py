"""Regional and functional impedance from native EIT images."""

import numpy as np
import pytest

from m3resp.data import PixelMask, Signal
from m3resp.eit.processing import functional_impedance, regional_impedance


@pytest.mark.parametrize("calculate", [regional_impedance, functional_impedance])
def test_weighted_sum_keeps_time_units_and_source(calculate):
    values = np.array([[[2.0, 10.0], [4.0, 8.0]], [[np.nan] * 2] * 2])
    pixels = Signal(
        values=values.copy(),
        time=[5.0, 5.05],
        sample_frequency=20.0,
        name="filtered",
        modality="eit",
        source="sentec",
        processing_state="intermediate",
    )
    mask = PixelMask(name="lung", values=[[1.0, np.nan], [0.5, 0.0]], keep_zeros=True)

    result = calculate(pixels, mask, name="weighted")

    np.testing.assert_array_equal(result.values, [4.0, 0.0])
    np.testing.assert_array_equal(result.time, pixels.time)
    np.testing.assert_array_equal(pixels.values, values)
    assert not np.shares_memory(result.time, pixels.time)
    assert (result.name, result.unit, result.sample_frequency) == (
        "weighted",
        "a.u.",
        20.0,
    )
    assert (result.modality, result.category, result.source) == (
        "eit",
        "impedance",
        "sentec",
    )
    assert result.processing_state == "intermediate"
    assert result.channel == calculate.__name__
    assert result.method == f"m3resp.eit.processing.{calculate.__name__}"


@pytest.mark.parametrize("calculate", [regional_impedance, functional_impedance])
def test_empty_region_gives_zero_for_each_frame(calculate):
    pixels = Signal(values=np.ones((2, 2, 2)), time=[0.0, 0.05])
    mask = PixelMask(
        name="empty", values=np.full((2, 2), np.nan), suppress_all_nan_warning=True
    )

    np.testing.assert_array_equal(calculate(pixels, mask).values, [0.0, 0.0])


@pytest.mark.parametrize("calculate", [regional_impedance, functional_impedance])
def test_waveform_is_refused_as_image_input(calculate):
    pixels = Signal(values=[1.0, 2.0], time=[0.0, 0.05])
    mask = PixelMask(name="lung", values=np.ones((2, 2)))

    with pytest.raises(ValueError, match="shape \\(frames, rows, columns\\)"):
        calculate(pixels, mask)


@pytest.mark.parametrize("calculate", [regional_impedance, functional_impedance])
def test_image_and_mask_sizes_must_match(calculate):
    pixels = Signal(values=np.ones((2, 2, 2)), time=[0.0, 0.05])
    mask = PixelMask(name="lung", values=np.ones((1, 2)))

    with pytest.raises(ValueError, match="size of the mask"):
        calculate(pixels, mask)
