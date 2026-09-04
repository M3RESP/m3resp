"""PEEP estimated from end-expiration (Warnaar et al. 2024).

Mirrors ReSurfEMG's `VentilatorDataGroup.find_peep`.
"""

from __future__ import annotations

import pytest

from m3resp.core.exceptions import MissingModalityDataError
from m3resp.processing.ventilator import estimate_peep

np = pytest.importorskip("numpy")

FS = 100.0
PEEP = 5.0
PEAK_PRESSURE = 15.0


def _quiet_breathing(duration_seconds: float = 30.0, breath_hz: float = 0.25):
    """Volume and airway pressure rising together on inspiration, both
    returning to their end-expiratory floor between breaths."""

    time = np.arange(0, duration_seconds, 1 / FS)
    inspiration = np.maximum(np.sin(2 * np.pi * breath_hz * time), 0.0)
    volume = inspiration
    pressure = PEEP + (PEAK_PRESSURE - PEEP) * inspiration
    return pressure, volume


class TestEstimatePeep:
    def test_recovers_the_end_expiratory_pressure(self):
        pressure, volume = _quiet_breathing()

        assert estimate_peep(pressure, volume) == pytest.approx(PEEP)

    def test_is_not_the_whole_trace_median(self):
        # The whole-trace median includes inspiration and so sits above PEEP.
        # This is the error the estimate exists to avoid.
        pressure, volume = _quiet_breathing()

        assert float(np.nanmedian(pressure)) > PEEP
        assert estimate_peep(pressure, volume) < float(np.nanmedian(pressure))

    def test_ignores_pressure_excursions_away_from_end_expiration(self):
        # An occlusion manoeuvre drags pressure down mid-record; PEEP is
        # defined at end-expiration and should not follow it.
        pressure, volume = _quiet_breathing()
        pressure[1000:1100] -= 4.0

        assert estimate_peep(pressure, volume) == pytest.approx(PEEP)

    def test_rounds_to_the_nearest_whole_cmh2o(self):
        # Ventilators are set in whole cmH2O; a measured end-expiratory
        # pressure just either side of the set value reports that value.
        for offset in (-0.3, 0.4):
            pressure, volume = _quiet_breathing()
            pressure += offset

            assert estimate_peep(pressure, volume) == pytest.approx(PEEP)

    def test_round_to_integer_off_keeps_the_measured_value(self):
        pressure, volume = _quiet_breathing()
        pressure += 0.4

        assert estimate_peep(pressure, volume, round_to_integer=False) == pytest.approx(
            PEEP + 0.4
        )

    def test_rejects_a_flat_volume_signal(self):
        pressure, _ = _quiet_breathing()

        with pytest.raises(MissingModalityDataError, match="end-expiratory minima"):
            estimate_peep(pressure, np.zeros_like(pressure))

    def test_rejects_pressure_and_volume_on_different_time_bases(self):
        pressure, volume = _quiet_breathing()

        with pytest.raises(ValueError, match="same time base"):
            estimate_peep(pressure, volume[:-10])
