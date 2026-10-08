"""`PixelMap` and `PixelMask`: one number per EIT pixel, kept as a 2D grid."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from m3resp.data import (
    BreathEvent,
    IntervalData,
    PixelMap,
    PixelMask,
    PixelMaskCollection,
    pixel_breath_events,
)


class TestPixelMap:
    def test_values_are_stored_as_a_float_grid(self):
        pixel_map = PixelMap(name="pixel_tiv", values=[[1, 2], [3, 4]], unit="a.u.")

        assert pixel_map.values.dtype == float
        assert pixel_map.shape == (2, 2)
        assert pixel_map.modality == "eit"

    def test_nan_marks_a_pixel_without_a_value(self):
        pixel_map = PixelMap(name="pixel_tiv", values=[[np.nan, 1.0]])

        assert np.isnan(pixel_map.values[0, 0])

    @pytest.mark.parametrize("values", [[1.0, 2.0], np.ones((2, 2, 2))])
    def test_anything_but_a_2d_grid_is_refused(self, values):
        with pytest.raises(ValueError, match="2D grid"):
            PixelMap(name="pixel_tiv", values=values)

    def test_values_that_are_not_numbers_are_refused(self):
        with pytest.raises(TypeError, match="stored as floats"):
            PixelMap(name="pixel_tiv", values=[["a", "b"]])

    def test_numpy_reads_a_pixel_map_as_its_grid(self):
        pixel_map = PixelMap(name="pixel_tiv", values=[[1.0, np.nan], [3.0, 5.0]])

        assert np.nanmean(pixel_map) == 3.0

    def test_maps_with_nan_in_the_same_places_are_equal(self):
        first = PixelMap(name="pixel_tiv", values=[[np.nan, 1.0]])
        second = PixelMap(name="pixel_tiv", values=[[np.nan, 1.0]])

        assert first == second
        assert first != PixelMap(name="pixel_tiv", values=[[np.nan, 2.0]])

    def test_to_dict_turns_the_grid_into_lists(self):
        pixel_map = PixelMap(name="pixel_tiv", values=[[1.0, 2.0]], method="m")

        assert pixel_map.to_dict()["values"] == [[1.0, 2.0]]
        assert pixel_map.to_dict()["method"] == "m"


class TestPixelMask:
    def test_true_false_grid_becomes_one_and_nan_without_a_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mask = PixelMask(name="lungspace", values=[[True, False], [False, True]])

        np.testing.assert_array_equal(mask.values, [[1.0, np.nan], [np.nan, 1.0]])
        assert mask.included_pixel_count == 2

    def test_true_false_grid_with_keep_zeros_keeps_false_as_zero(self):
        mask = PixelMask(name="lungspace", values=[[True, False]], keep_zeros=True)

        np.testing.assert_array_equal(mask.values, [[1.0, 0.0]])

    def test_weights_between_zero_and_one_are_kept(self):
        mask = PixelMask(name="lungspace", values=[[0.5, np.nan], [1.0, 0.25]])

        np.testing.assert_array_equal(mask.values, [[0.5, np.nan], [1.0, 0.25]])
        assert mask.included_pixel_count == 3

    def test_zero_becomes_nan_with_a_warning(self):
        with pytest.warns(UserWarning, match="keep_zeros=True"):
            mask = PixelMask(name="lungspace", values=[[0.0, 1.0]])

        np.testing.assert_array_equal(mask.values, [[np.nan, 1.0]])

    def test_zero_conversion_warning_can_be_switched_off(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mask = PixelMask(
                name="lungspace",
                values=[[0.0, 1.0]],
                suppress_zero_conversion_warning=True,
            )

        np.testing.assert_array_equal(mask.values, [[np.nan, 1.0]])

    def test_keep_zeros_keeps_weight_zero_apart_from_nan(self):
        # NaN is outside the region; 0 is inside it with no weight.
        mask = PixelMask(
            name="ventral_weights",
            values=[[np.nan, 0.0], [1.0, 0.5]],
            keep_zeros=True,
        )

        np.testing.assert_array_equal(mask.values, [[np.nan, 0.0], [1.0, 0.5]])
        assert mask.included_pixel_count == 3

        pixel_values = np.array([[10.0, 10.0], [10.0, 10.0]])
        masked = pixel_values * mask.values
        np.testing.assert_array_equal(masked, [[np.nan, 0.0], [10.0, 5.0]])
        assert np.nansum(masked) == 15.0

    @pytest.mark.parametrize("bad", [-0.5, 1.5])
    def test_numbers_outside_zero_to_one_are_refused(self, bad):
        with pytest.raises(ValueError, match="from 0 to 1"):
            PixelMask(name="lungspace", values=[[bad, 1.0]])

    def test_range_check_can_be_switched_off(self):
        mask = PixelMask(
            name="lungspace", values=[[1.5, 1.0]], suppress_value_range_error=True
        )

        np.testing.assert_array_equal(mask.values, [[1.5, 1.0]])

    def test_all_nan_grid_gives_a_warning(self):
        with pytest.warns(UserWarning, match="NaN everywhere"):
            PixelMask(name="lungspace", values=[[np.nan, np.nan]])

    def test_all_nan_warning_can_be_switched_off(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PixelMask(
                name="lungspace",
                values=[[np.nan, np.nan]],
                suppress_all_nan_warning=True,
            )

    def test_anything_but_a_2d_grid_is_refused(self):
        with pytest.raises(ValueError, match="2D grid"):
            PixelMask(name="lungspace", values=[True, False])

    def test_masks_with_nan_in_the_same_places_are_equal(self):
        first = PixelMask(name="lungspace", values=[[np.nan, 1.0]])

        assert first == PixelMask(name="lungspace", values=[[False, True]])


def test_pixel_mask_collection_keeps_order_and_skips_the_same_object():
    first = PixelMask(name="a", values=[[1.0]])
    second = PixelMask(name="b", values=[[1.0]])
    masks = PixelMaskCollection()

    masks.add(first)
    masks.add(second)
    masks.add(first)

    assert list(masks) == [first, second]
    assert masks.for_name("b") == [second]


def _pixel_breaths() -> IntervalData:
    """Two global breaths on a 1 x 2 image. Pixel (0, 1) has no breath in
    the second global breath."""

    global_breaths = [
        BreathEvent("eit", 0.0, 4.0, peak_time=1.5),
        BreathEvent("eit", 4.0, 8.0, peak_time=5.5),
    ]
    grids = [
        np.array([[[0.1, 1.6, 4.1], [0.3, 1.8, 4.2]]]),
        np.array([[[4.1, 5.6, 8.1], [np.nan, np.nan, np.nan]]]),
    ]
    return IntervalData(
        name="pixel_breaths",
        modality="eit",
        intervals=global_breaths,
        values=grids,
        unit="s",
        method="eitprocessing.PixelBreath",
    )


class TestPixelBreathEvents:
    def test_one_breath_event_per_global_breath_with_the_pixel_timings(self):
        pixel_breaths = _pixel_breaths()

        events = pixel_breath_events(pixel_breaths, row=0, column=0)

        assert [(e.start_time, e.peak_time, e.end_time) for e in events] == [
            (0.1, 1.6, 4.1),
            (4.1, 5.6, 8.1),
        ]
        assert all(isinstance(e, BreathEvent) for e in events)
        assert all(e.modality == "eit" for e in events)
        assert [e.metadata["global_breath_id"] for e in events] == [
            b.id for b in pixel_breaths.intervals
        ]
        assert events[0].metadata["row"] == 0
        assert events[0].metadata["column"] == 0

    def test_a_global_breath_without_a_pixel_breath_gives_no_entry(self):
        events = pixel_breath_events(_pixel_breaths(), row=0, column=1)

        assert [(e.start_time, e.end_time) for e in events] == [(0.3, 4.2)]

    def test_a_pixel_outside_the_grid_is_refused(self):
        with pytest.raises(ValueError, match="outside the 1 x 2 grid"):
            pixel_breath_events(_pixel_breaths(), row=1, column=0)

    def test_grids_without_three_timings_are_refused(self):
        not_timings = IntervalData(
            name="pixel_tivs",
            modality="eit",
            intervals=[BreathEvent("eit", 0.0, 4.0)],
            values=[np.ones((1, 2))],
        )

        with pytest.raises(ValueError, match="start, middle and end"):
            pixel_breath_events(not_timings, row=0, column=0)
