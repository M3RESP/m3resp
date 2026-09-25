"""Breaths found as dips in the airway pressure of a spontaneously breathing
subject (`detect_pressure_dip_breaths`, `ventilator.detect_pressure_breaths`)."""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.core.exceptions import MissingModalityDataError
from m3resp.processing.peaks import detect_pressure_dip_breaths
from m3resp.workflows.steps.ventilator.detection import detect_pressure_breaths

FS = 100.0


def _pressure(dip_times, dip_depths, duration_seconds=60.0, noise=0.03, seed=0):
    """Airway pressure resting at -0.3 cmH2O with one dip per breath.

    Each dip is a smooth 1.5 s trough of the given depth, centred on its time.
    """

    rng = np.random.default_rng(seed)
    time = np.arange(int(duration_seconds * FS)) / FS
    pressure = np.full(time.size, -0.3) + rng.normal(0.0, noise, time.size)
    for centre, depth in zip(dip_times, dip_depths, strict=True):
        inside = np.abs(time - centre) < 0.75
        pressure[inside] -= depth * np.cos(np.pi * (time[inside] - centre) / 1.5) ** 2
    return pressure


def test_every_breath_is_found_at_its_lowest_point():
    # 12 breaths/min, dips from 0.5 to 3 cmH2O deep.
    dip_times = np.arange(3.0, 60.0, 5.0)
    depths = np.linspace(0.5, 3.0, dip_times.size)

    indices = detect_pressure_dip_breaths(
        _pressure(dip_times, depths), sample_frequency=FS
    )

    assert indices.size == dip_times.size
    np.testing.assert_allclose(indices / FS, dip_times, atol=0.1)


def test_dips_shallower_than_min_depth_are_not_breaths():
    dip_times = [5.0, 15.0, 25.0, 35.0]
    depths = [1.0, 0.05, 1.0, 0.05]

    indices = detect_pressure_dip_breaths(
        _pressure(dip_times, depths, noise=0.005), sample_frequency=FS
    )

    np.testing.assert_allclose(indices / FS, [5.0, 25.0], atol=0.1)


def test_missing_samples_warn_and_never_hold_a_breath():
    dip_times = np.arange(3.0, 60.0, 5.0)
    pressure = _pressure(dip_times, np.ones(dip_times.size))
    # Remove the lowest part of the breath at 13 s.
    pressure[int(12.8 * FS) : int(13.2 * FS)] = np.nan

    with pytest.warns(UserWarning, match="missing"):
        indices = detect_pressure_dip_breaths(pressure, sample_frequency=FS)

    assert not np.isnan(pressure[indices]).any()
    assert indices.size >= dip_times.size - 1


def test_step_writes_ventilator_breath_indices():
    dip_times = np.arange(3.0, 60.0, 5.0)
    signals = {"pressure": _pressure(dip_times, np.ones(dip_times.size)), "fs": FS}

    result = detect_pressure_breaths(signals)

    assert result["ventilator_breath_indices"].size == dip_times.size


def test_step_needs_a_pressure_channel():
    with pytest.raises(MissingModalityDataError, match="pressure"):
        detect_pressure_breaths({"fs": FS})
