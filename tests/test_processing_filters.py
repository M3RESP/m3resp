"""Tests for `m3resp.processing.filters.compute_power_loss`.

Every other shared filter primitive is pinned against upstream in
`tests/regression/test_processing_filter_equivalence.py`. `compute_power_loss`
cannot be, because the ReSurfEMG version it derives from is wrong in two ways:

1. `scipy.signal.welch` returns a `(frequencies, density)` pair. Upstream sums
   the whole pair rather than the density alone, so the frequency axis - which
   on a 1 kHz signal is around five orders of magnitude larger than the
   density - swamps the result.
2. The ratio is inverted: upstream computes `100 * (1 - original/processed)`
   where power loss is `100 * (1 - processed/original)`.

So these tests pin the *correct* answer instead of upstream's. They are written
against values that can be worked out on paper, so they stay meaningful without
a reference implementation to compare against.

If ReSurfEMG fixes this, an equivalence test belongs in the regression file
alongside the others and this module can be reduced to the edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.processing.filters import compute_power_loss

pytest.importorskip("scipy")

SAMPLE_FREQUENCY = 1000.0


def _noise(n_samples: int = 8000, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n_samples)


@pytest.mark.parametrize(
    ("scale", "expected_loss"),
    [
        (1.0, 0.0),  # nothing removed
        (0.5, 75.0),  # half the amplitude is a quarter of the power
        (0.25, 93.75),
        (0.0, 100.0),  # everything removed
    ],
)
def test_scaling_a_signal_loses_the_analytically_expected_power(
    scale: float, expected_loss: float
) -> None:
    """Power goes as amplitude squared, so scaling by `k` loses `1 - k**2`."""

    original = _noise()

    actual = compute_power_loss(
        original,
        original * scale,
        original_frequency=SAMPLE_FREQUENCY,
        processed_frequency=SAMPLE_FREQUENCY,
    )

    assert actual == pytest.approx(expected_loss)


def test_power_loss_ignores_the_welch_frequency_axis() -> None:
    """Guards defect (1) in the module docstring.

    Only the power density may enter the calculation. The frequency axis
    changes with the sampling rate while the power ratio does not, so a result
    that shifts when the rate does is a result that is summing the frequencies.
    """

    original = _noise()
    processed = original * 0.5

    at_1khz = compute_power_loss(
        original,
        processed,
        original_frequency=1000.0,
        processed_frequency=1000.0,
    )
    at_2khz = compute_power_loss(
        original,
        processed,
        original_frequency=2000.0,
        processed_frequency=2000.0,
    )

    assert at_1khz == pytest.approx(75.0)
    assert at_2khz == pytest.approx(75.0)


def test_power_loss_is_negative_when_processing_adds_power() -> None:
    """Guards defect (2): an inverted ratio changes the sign of the answer."""

    original = _noise()

    actual = compute_power_loss(
        original,
        original * 2.0,
        original_frequency=SAMPLE_FREQUENCY,
        processed_frequency=SAMPLE_FREQUENCY,
    )

    assert actual == pytest.approx(-300.0)


def test_power_loss_matches_a_hand_rolled_welch_density_ratio() -> None:
    """The whole calculation, spelled out, on a realistic filtered signal."""

    from scipy import signal

    fs = SAMPLE_FREQUENCY
    time = np.arange(int(fs * 3)) / fs
    original = np.sin(2 * np.pi * 4 * time) + 0.2 * np.sin(2 * np.pi * 110 * time)
    sos = signal.butter(4, 20.0, btype="lowpass", fs=fs, output="sos")
    processed = signal.sosfiltfilt(sos, original)

    n_segment = int(fs) // 2
    noverlap = int(25 / 100 * fs)
    _, original_density = signal.welch(
        original, fs, nperseg=n_segment, noverlap=noverlap
    )
    _, processed_density = signal.welch(
        processed, fs, nperseg=n_segment, noverlap=noverlap
    )
    expected = 100 * (1 - np.sum(processed_density) / np.sum(original_density))

    actual = compute_power_loss(
        original,
        processed,
        original_frequency=fs,
        processed_frequency=fs,
    )

    assert actual == pytest.approx(expected)
    # The 110 Hz component is most of what a 20 Hz lowpass throws away.
    assert 0.0 < actual < 100.0
