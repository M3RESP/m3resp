"""Regional and functional impedance: pixel impedance summed inside a mask.

Each pixel is multiplied by its mask value (NaN leaves it out, a weight counts
it partly, weight 0 adds nothing) and the pixels are summed per frame.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("eitprocessing")

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData, Vendor

import m3resp.workflows.steps  # noqa: F401 - registers the built-in steps
from m3resp import M3Session
from m3resp.data import PixelMask
from m3resp.workflows.registry import describe_step, get_step

FS = 20.0

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


def test_mask_with_the_wrong_shape_is_refused():
    session = M3Session()
    mask = PixelMask(name="too_small", values=[[1.0]])

    with pytest.raises(ValueError, match="shape"):
        get_step("eit.regional_impedance").func(
            _eit_data(), mask=mask, eit_sequence=_sequence(), session=session
        )
