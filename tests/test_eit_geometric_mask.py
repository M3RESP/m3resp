"""Geometric masks: the function `geometric_mask` and the step `eit.geometric_mask`.

The masks must equal eitprocessing 1.8.7 `get_geometric_mask` for the 12
region names and their short names on 32 x 32 and other image shapes, with
the same errors for shapes that do not divide. The set rules of the
eitprocessing docstring hold (ventral and anatomical right = quadrant 1, and
so on).
"""

from __future__ import annotations

import dataclasses
import os
import warnings

import numpy as np
import pytest

import m3resp.workflows.steps  # noqa: F401 - registers the steps
from m3resp.core.session import M3Session
from m3resp.data import PixelMask, PixelMaskCollection, Signal
from m3resp.eit.processing import regional_impedance
from m3resp.eit.roi import GEOMETRIC_REGIONS, geometric_mask
from m3resp.workflows import run_workflow
from m3resp.workflows.registry import describe_step
from m3resp.workflows.spec import StepSpec, load_spec
from m3resp.workflows.steps.eit import geometric_mask as geometric_mask_step

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EIT_FULL_SPEC = os.path.join(
    REPO_ROOT, "examples", "eit_full_preprocessing", "eit-full.workflow.yaml"
)

SHORT_NAMES = {
    "V": "ventral",
    "D": "dorsal",
    "R": "anatomical right",
    "L": "anatomical left",
    **{f"L{n}": f"layer {n}" for n in range(1, 5)},
    **{f"Q{n}": f"quadrant {n}" for n in range(1, 5)},
}
SHAPES = [
    (32, 32),
    (64, 32),
    (8, 8),
    (4, 12),
    (12, 4),
    (2, 2),
    (4, 2),
    (6, 4),
    (4, 6),
    (30, 32),
    (32, 30),
    (1, 2),
    (2, 1),
    (5, 5),
]


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("region", [*GEOMETRIC_REGIONS, *SHORT_NAMES])
def test_masks_and_errors_equal_eitprocessing(region, shape):
    upstream_roi = pytest.importorskip("eitprocessing.roi")
    try:
        expected = upstream_roi.get_geometric_mask(region, shape)
    except ValueError:
        with pytest.raises(ValueError, match="multiple of"):
            geometric_mask(region, shape)
        return

    mask = geometric_mask(region, shape)

    np.testing.assert_array_equal(mask.values, expected.mask)
    assert mask.name == expected.label == SHORT_NAMES.get(region, region)
    assert mask.metadata == {"region": mask.name, "shape": list(shape)}
    assert mask.modality == "eit"


def test_short_names_give_the_same_mask_as_the_full_names():
    for short, full in SHORT_NAMES.items():
        assert geometric_mask(short) == geometric_mask(full)


@pytest.mark.parametrize(
    ("first", "second", "quadrant"),
    [
        ("ventral", "anatomical right", "quadrant 1"),
        ("ventral", "anatomical left", "quadrant 2"),
        ("dorsal", "anatomical right", "quadrant 3"),
        ("dorsal", "anatomical left", "quadrant 4"),
    ],
)
def test_a_half_and_a_side_make_a_quadrant(first, second, quadrant):
    combined = geometric_mask(first) * geometric_mask(second)

    np.testing.assert_array_equal(combined.values, geometric_mask(quadrant).values)
    assert combined.name == f"{first} and {second}"


def test_the_parts_cover_the_image_once():
    shape = (8, 12)
    for parts in (
        ("ventral", "dorsal"),
        ("anatomical right", "anatomical left"),
        ("layer 1", "layer 2", "layer 3", "layer 4"),
        ("quadrant 1", "quadrant 2", "quadrant 3", "quadrant 4"),
    ):
        counts = sum(
            np.nan_to_num(geometric_mask(part, shape).values) for part in parts
        )
        np.testing.assert_array_equal(counts, np.ones(shape))
        collection = PixelMaskCollection(
            items=[geometric_mask(part, shape) for part in parts]
        )
        whole = collection.combine("sum", name="whole image")
        np.testing.assert_array_equal(whole.values, np.ones(shape))


def test_layers_one_and_two_are_the_ventral_half():
    ventral = geometric_mask("layer 1") + geometric_mask("layer 2")

    np.testing.assert_array_equal(ventral.values, geometric_mask("ventral").values)
    np.testing.assert_array_equal(
        (geometric_mask("ventral") - geometric_mask("layer 1")).values,
        geometric_mask("layer 2").values,
    )


def test_anatomical_right_is_the_left_side_of_the_image():
    right = geometric_mask("R", (2, 4))

    np.testing.assert_array_equal(
        right.values, [[1.0, 1.0, np.nan, np.nan], [1.0, 1.0, np.nan, np.nan]]
    )


@pytest.mark.parametrize(
    "region", ["Ventral", "layer 5", "Q0", "quadrant", "", "global"]
)
def test_unknown_regions_are_refused(region):
    with pytest.raises(ValueError, match="Unknown region"):
        geometric_mask(region)


def test_the_height_error_names_the_number():
    # I19: eitprocessing shows "{height_divisor}" in this message.
    with pytest.raises(
        ValueError, match="height that is a multiple of 4; the height is 30"
    ):
        geometric_mask("layer 1", (30, 32))


@pytest.mark.parametrize(
    "shape", [(0, 32), (32,), (32, 32, 1), (2.0, 2), (True, 2), "32"]
)
def test_shapes_that_are_not_two_positive_whole_numbers_are_refused(shape):
    with pytest.raises(ValueError, match="shape"):
        geometric_mask("ventral", shape)


# --- step ---------------------------------------------------------------------


def _images(shape: tuple[int, int] = (32, 32)) -> Signal:
    rng = np.random.default_rng(5)
    return Signal(
        name="pixels",
        values=rng.normal(size=(5, *shape)),
        time=np.arange(5) / 20.0,
        sample_frequency=20.0,
        unit="AU",
        modality="eit",
        category="impedance",
    )


@pytest.mark.parametrize("region", GEOMETRIC_REGIONS)
def test_step_gives_the_function_mask_and_stores_it(region):
    session = M3Session()

    result = geometric_mask_step(eit_data=_images(), session=session, region=region)

    mask = result["geometric_mask_result"]
    assert isinstance(mask, PixelMask)
    np.testing.assert_array_equal(mask.values, geometric_mask(region).values)
    assert mask.name == region
    assert mask.metadata["source_package"] == "m3resp"
    assert mask.metadata["parameters"] == {"region": region}
    assert session.pixel_masks.for_name(region) == [mask]


def test_step_takes_the_shape_of_the_images():
    result = geometric_mask_step(
        eit_data=_images((8, 4)), session=M3Session(), region="quadrant 3"
    )

    np.testing.assert_array_equal(
        result["geometric_mask_result"].values, geometric_mask("Q3", (8, 4)).values
    )


def test_step_refuses_data_that_are_not_images():
    signal = dataclasses.replace(_images(), values=np.ones((5, 32)))

    with pytest.raises(ValueError, match="time, row and column"):
        geometric_mask_step(eit_data=signal, session=M3Session(), region="ventral")


def test_step_description_lists_the_twelve_regions():
    description = describe_step("eit.geometric_mask")
    region = next(p for p in description.parameters if p.name == "region")

    assert tuple(region.choices) == GEOMETRIC_REGIONS
    assert description.optional_packages == ()
    assert [a.name for a in description.output_artifacts] == ["geometric_mask_result"]
    assert description.output_artifacts[0].artifact_type == "pixel_mask"


def test_ventral_impedance_in_a_workflow():
    spec = load_spec(EIT_FULL_SPEC)
    spec = dataclasses.replace(
        spec,
        steps=(
            *spec.steps,
            StepSpec(
                uses="eit.geometric_mask",
                id="ventral_mask",
                params={"region": "ventral"},
                outputs={"geometric_mask_result": "ventral_mask"},
            ),
            StepSpec(
                uses="eit.regional_impedance",
                id="ventral_impedance",
                inputs={"mask": "ventral_mask"},
                outputs={"regional_impedance_signal": "ventral_impedance"},
            ),
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run_workflow(spec, session=M3Session())

    assert result.status == "succeeded"
    pixels = result.value("filtered_eit")
    mask = result.value("ventral_mask")
    np.testing.assert_array_equal(mask.values, geometric_mask("ventral").values)
    np.testing.assert_array_equal(
        result.value("ventral_impedance").values,
        regional_impedance(pixels, mask).values,
    )
