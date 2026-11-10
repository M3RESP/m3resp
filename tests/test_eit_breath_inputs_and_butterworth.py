"""The breath forms of the EIT steps and the Butterworth filter of EIT images.

``m3resp.eit.processing.breath_inputs`` takes m3resp breaths only, and
``m3resp.eit.processing.butterworth`` filters EIT images with the same
values as the filter that the EIT steps use.
"""

from __future__ import annotations

import numpy as np
import pytest
from reference.eitprocessing import (
    butterworth_filter_images as adapter_butterworth_filter_images,
)

from m3resp.data import BreathEvent, Signal
from m3resp.eit.processing.breath_inputs import (
    breath_intervals_to_breath_events,
    breaths_to_interval_data,
    given_breath_events,
)
from m3resp.eit.processing.butterworth import (
    MAX_EIT_BUTTERWORTH_ORDER,
    butterworth_filter_images,
)


def _breaths() -> list[BreathEvent]:
    return [
        BreathEvent(start_time=0.0, end_time=2.0, extremum_time=1.0, modality="eit"),
        BreathEvent(start_time=2.0, end_time=4.5, extremum_time=3.0, modality="eit"),
    ]


def test_breaths_in_an_interval_data_or_a_list_are_the_same_breaths() -> None:
    breaths = _breaths()
    held = breaths_to_interval_data(breaths)
    assert held.name == "eit_breaths"
    assert held.category == "breath"
    assert breath_intervals_to_breath_events(held) == breaths
    assert breath_intervals_to_breath_events(breaths) == breaths
    assert given_breath_events(held, function="test") == breaths
    assert given_breath_events(tuple(breaths), function="test") == breaths


@pytest.mark.parametrize("value", [None, 3.0, {"start_time": 0.0}, np.zeros(3)])
def test_other_breath_forms_are_refused(value) -> None:
    with pytest.raises(TypeError, match="BreathEvent"):
        breath_intervals_to_breath_events(value)
    with pytest.raises(TypeError, match="eit.tiv needs"):
        given_breath_events(value, function="eit.tiv")


def test_an_item_that_is_not_a_breath_is_refused() -> None:
    with pytest.raises(TypeError, match="BreathEvent"):
        given_breath_events([_breaths()[0], (0.0, 1.0)], function="eit.eeli")


def _images() -> Signal:
    rng = np.random.default_rng(7)
    sample_frequency = 20.0
    time = np.arange(400) / sample_frequency
    values = (
        rng.normal(size=(400, 4, 3)) + np.sin(2 * np.pi * 0.3 * time)[:, None, None]
    )
    values[:, 0, 0] = np.nan  # a pixel that was never measured
    return Signal(
        time=time,
        values=values,
        sample_frequency=sample_frequency,
        modality="eit",
        metadata={"source": "test"},
    )


@pytest.mark.parametrize(
    ("mode", "cutoff"), [("lowpass", 1.0), ("bandpass", (0.05, 1.0))]
)
def test_the_filter_gives_the_images_of_the_step_filter(mode, cutoff) -> None:
    images = _images()
    filtered, captures = butterworth_filter_images(
        images, mode=mode, cutoff_frequency=cutoff, order=4, operation="test"
    )
    expected, expected_captures = adapter_butterworth_filter_images(
        images,
        mode=mode,
        cutoff_frequency=cutoff,
        order=4,
        label=f"{mode}_filtered",
        operation="test",
    )
    np.testing.assert_array_equal(filtered.values, expected.values)
    np.testing.assert_array_equal(filtered.time, expected.time)
    assert filtered.name == expected.name == f"{mode.title()}-filtered EIT data"
    assert filtered.method == "m3resp.processing.filters.butterworth_filter"
    assert filtered.processing_state == "intermediate"
    assert filtered.metadata == images.metadata
    assert np.isnan(filtered.values[:, 0, 0]).all()
    assert set(captures) == set(expected_captures)
    assert captures["filtered_data"] is filtered
    assert captures["unfiltered_data"] is images


@pytest.mark.parametrize("order", [0, MAX_EIT_BUTTERWORTH_ORDER + 1])
def test_an_order_outside_1_to_10_is_refused(order) -> None:
    with pytest.raises(ValueError, match="from 1 to 10"):
        butterworth_filter_images(
            _images(),
            mode="lowpass",
            cutoff_frequency=1.0,
            order=order,
            operation="test",
        )
