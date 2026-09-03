"""An unmeasured channel in an EIT recording is not a measurement.

A Draeger recording writes out every ventilator channel it can carry, whether
or not the matching sensor was recording. A channel nobody measured - a
pressure pod's esophageal pressure with no pod attached, a Medibus field the
ventilator never populated - is not omitted from the file; every sample is
filled with a large negative number standing in for NaN. Read literally those
samples are pressures of around -1e30 mbar, and they would be filtered,
plotted and averaged like any other reading.

The exact number differs between device software versions, so the values are
compared against a cutoff rather than one sentinel value. These tests cover
that conversion and what happens to a channel that holds nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.adapters.ventilator_adapter import (
    SENTINEL_CUTOFF,
    available_ventilator_channels,
    ventilator_payload_from_sequence,
)
from m3resp.core.exceptions import UnsupportedWorkflowError

FS = 100.0
N = 300


def _wave(scale: float = 1.0) -> np.ndarray:
    return scale * np.sin(2 * np.pi * 0.25 * np.arange(N) / FS)


class _ContinuousData:
    def __init__(self, label, values, unit):
        self.label = label
        self.name = label
        self.values = np.asarray(values, dtype=float)
        self.unit = unit
        self.sample_frequency = FS
        self.time = np.arange(len(self.values)) / FS


class _Sequence:
    def __init__(self, **channels):
        self.continuous_data = {
            label: _ContinuousData(label, values, unit)
            for label, (values, unit) in channels.items()
        }


def _sequence(pressure=None, esophageal=None):
    channels = {
        "airway pressure": (
            _wave(10.0) if pressure is None else pressure,
            "mbar",
        ),
        "flow": (_wave(5.0), "L/min"),
        "volume": (_wave(500.0), "mL"),
    }
    if esophageal is not None:
        channels["esophageal pressure (pod)"] = (esophageal, "mbar")
    return _Sequence(**channels)


def _unmeasured(value: float = -1.7e38) -> np.ndarray:
    return np.full(N, value, dtype=float)


class TestTheSentinelBecomesNaN:
    def test_a_stretch_the_device_did_not_measure_reads_as_nan(self):
        pressure = _wave(10.0)
        pressure[50:70] = -1.7e38
        payload = ventilator_payload_from_sequence(_sequence(pressure=pressure))

        row = payload["array"][0]
        assert np.all(np.isnan(row[50:70]))
        assert not np.any(np.isnan(row[:50]))
        assert not np.any(np.isnan(row[70:]))

    def test_those_samples_are_counted_as_missing(self):
        pressure = _wave(10.0)
        pressure[50:70] = -1.7e38
        payload = ventilator_payload_from_sequence(_sequence(pressure=pressure))

        assert payload["metadata"]["nan_samples"]["pressure"] == 20

    @pytest.mark.parametrize("value", [-1.7e38, -3.4e38, SENTINEL_CUTOFF * 10])
    def test_the_cutoff_catches_the_values_different_versions_write(self, value):
        pressure = _wave(10.0)
        pressure[10] = value
        payload = ventilator_payload_from_sequence(_sequence(pressure=pressure))

        assert np.isnan(payload["array"][0][10])

    def test_a_real_negative_reading_is_left_alone(self):
        # Expiratory flow is negative, and an esophageal pressure swing can sit
        # below zero throughout. Nothing physiological approaches the cutoff.
        payload = ventilator_payload_from_sequence(
            _sequence(pressure=_wave(10.0) - 40.0)
        )

        row = payload["array"][0]
        assert not np.any(np.isnan(row))
        assert row.min() < 0


class TestAChannelWithNoMeasurementIsRefused:
    def test_a_pod_channel_recorded_without_a_pod_is_rejected(self):
        sequence = _sequence(esophageal=_unmeasured())

        with pytest.raises(UnsupportedWorkflowError) as excinfo:
            ventilator_payload_from_sequence(
                sequence, channels=("pressure", "esophageal_pressure")
            )

        message = str(excinfo.value)
        assert "esophageal_pressure" in message
        assert "pressure pod" in message

    def test_the_other_channels_still_load(self):
        payload = ventilator_payload_from_sequence(
            _sequence(esophageal=_unmeasured()),
            channels=("pressure", "flow", "volume"),
        )

        assert payload["array"].shape == (3, N)
        assert not np.any(np.isnan(payload["array"]))


class TestListingChannelsDoesNotReadThem:
    def test_an_unmeasured_channel_is_still_listed(self):
        # Listing answers "what does this recording carry", which is a header
        # question; deciding it was never measured means reading every sample.
        available = available_ventilator_channels(_sequence(esophageal=_unmeasured()))

        assert "esophageal_pressure" in available
