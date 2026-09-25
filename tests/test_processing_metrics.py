"""Respiratory rate from breath positions (`respiratory_rate_from_indices`)."""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.processing.metrics import respiratory_rate_from_indices


def test_rate_from_evenly_spaced_breaths():
    # One breath every 5 s at 100 Hz: 12 breaths/min.
    indices = np.arange(0, 3000, 500)

    median_rate, breath_to_breath = respiratory_rate_from_indices(indices, 100.0)

    assert median_rate == pytest.approx(12.0)
    np.testing.assert_allclose(breath_to_breath, [12.0] * 5)


@pytest.mark.parametrize("indices", [[], [250]])
def test_fewer_than_two_breaths_gives_nan_and_warns(indices):
    """No time between breaths can be measured, so there is no rate. This
    must not crash a pipeline whose recording had too few breaths."""

    with pytest.warns(UserWarning, match="at least two breaths"):
        median_rate, breath_to_breath = respiratory_rate_from_indices(
            np.asarray(indices, dtype=int), 100.0
        )

    assert np.isnan(median_rate)
    assert breath_to_breath.size == 0
