"""Input checks of `m3resp.processing.filters.butterworth_filter`.

The filtered values themselves are compared with upstream in
`tests/regression/test_processing_filter_equivalence.py`. These tests cover
what happens with bad or unusual inputs.
"""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

import m3resp.workflows.steps  # noqa: F401 - ensure built-in steps are registered
from m3resp.processing.filters import ButterworthFilterType, butterworth_filter
from m3resp.workflows.registry import get_step

pytest.importorskip("scipy")

SAMPLE_FREQUENCY = 100.0


def _signal(n_samples: int = 1000, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n_samples)


@pytest.mark.parametrize("filter_type", ["low", "notch", "LOWPASS"])
def test_unknown_filter_type_is_named_in_the_error(filter_type: str) -> None:
    with pytest.raises(ValueError, match="filter_type must be one of"):
        butterworth_filter(
            _signal(),
            filter_type=filter_type,  # type: ignore[arg-type]
            cutoff_frequency=5.0,
            sample_frequency=SAMPLE_FREQUENCY,
            order=4,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nan_or_infinite_samples_are_refused(bad_value: float) -> None:
    values = _signal()
    values[500] = bad_value
    with pytest.raises(ValueError, match="NaN or infinite"):
        butterworth_filter(
            values,
            filter_type="lowpass",
            cutoff_frequency=5.0,
            sample_frequency=SAMPLE_FREQUENCY,
            order=4,
        )


def test_numpy_numbers_give_the_same_result_as_python_numbers() -> None:
    values = _signal()
    expected = butterworth_filter(
        values,
        filter_type="bandpass",
        cutoff_frequency=(1.0, 10.0),
        sample_frequency=SAMPLE_FREQUENCY,
        order=4,
    )
    result = butterworth_filter(
        values,
        filter_type="bandpass",
        cutoff_frequency=(np.float32(1.0), np.float64(10.0)),
        sample_frequency=np.int32(100),
        order=np.int64(4),
    )
    np.testing.assert_allclose(result, expected)

    lowpass = butterworth_filter(
        values,
        filter_type="lowpass",
        cutoff_frequency=np.float32(5.0),
        sample_frequency=SAMPLE_FREQUENCY,
        order=4,
    )
    assert np.all(np.isfinite(lowpass))


@pytest.mark.parametrize("order", [True, False, np.bool_(True), 4.0])
def test_order_must_be_a_whole_number_not_true_or_false(order: object) -> None:
    with pytest.raises(TypeError, match="order must be a whole number"):
        butterworth_filter(
            _signal(),
            filter_type="lowpass",
            cutoff_frequency=5.0,
            sample_frequency=SAMPLE_FREQUENCY,
            order=order,  # type: ignore[arg-type]
        )


def test_true_is_not_taken_as_a_sample_frequency() -> None:
    with pytest.raises(TypeError, match="sample_frequency must be numeric"):
        butterworth_filter(
            _signal(),
            filter_type="lowpass",
            cutoff_frequency=0.2,
            sample_frequency=True,  # type: ignore[arg-type]
            order=4,
        )


def test_eit_filter_step_offers_the_same_filter_types() -> None:
    step = get_step("eit.butterworth_filter")
    mode = next(p for p in step.parameters if p.name == "mode")
    assert tuple(mode.choices) == get_args(ButterworthFilterType)
