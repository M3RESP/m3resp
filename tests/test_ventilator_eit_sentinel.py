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

from pathlib import Path

import numpy as np
import pytest

from m3resp.adapters.ventilator_adapter import (
    SENTINEL_CUTOFF,
    available_ventilator_channels,
    ventilator_payload_from_eit,
)
from m3resp.core.exceptions import UnsupportedWorkflowError
from m3resp.data import Signal
from m3resp.eit.io import EITFileData

FS = 100.0
N = 300


def _wave(scale: float = 1.0) -> np.ndarray:
    return scale * np.sin(2 * np.pi * 0.25 * np.arange(N) / FS)


def _recording(**channels) -> EITFileData:
    """The data of a Draeger file with these waveforms: label -> (values, unit)."""

    time = np.arange(N) / FS
    return EITFileData(
        vendor="draeger",
        paths=(Path("recording.bin"),),
        sample_frequency=FS,
        pixel_impedance=Signal(
            values=np.zeros((N, 2, 2)), time=time, sample_frequency=FS
        ),
        signals={
            label: Signal(
                values=np.asarray(values, dtype=float),
                time=time,
                sample_frequency=FS,
                unit=unit,
                name=label,
                modality="eit",
            )
            for label, (values, unit) in channels.items()
        },
    )


def _recording_with(pressure=None, esophageal=None):
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
    return _recording(**channels)


def _unmeasured(value: float = -1.7e38) -> np.ndarray:
    return np.full(N, value, dtype=float)


class TestTheSentinelBecomesNaN:
    def test_a_stretch_the_device_did_not_measure_reads_as_nan(self):
        pressure = _wave(10.0)
        pressure[50:70] = -1.7e38
        payload = ventilator_payload_from_eit(_recording_with(pressure=pressure))

        row = payload["array"][0]
        assert np.all(np.isnan(row[50:70]))
        assert not np.any(np.isnan(row[:50]))
        assert not np.any(np.isnan(row[70:]))

    def test_those_samples_are_counted_as_missing(self):
        pressure = _wave(10.0)
        pressure[50:70] = -1.7e38
        payload = ventilator_payload_from_eit(_recording_with(pressure=pressure))

        assert payload["metadata"]["nan_samples"]["airway_pressure"] == 20

    @pytest.mark.parametrize("value", [-1.7e38, -3.4e38, SENTINEL_CUTOFF * 10])
    def test_the_cutoff_catches_the_values_different_versions_write(self, value):
        pressure = _wave(10.0)
        pressure[10] = value
        payload = ventilator_payload_from_eit(_recording_with(pressure=pressure))

        assert np.isnan(payload["array"][0][10])

    def test_a_real_negative_reading_is_left_alone(self):
        # Expiratory flow is negative, and an esophageal pressure swing can sit
        # below zero throughout. Nothing physiological approaches the cutoff.
        payload = ventilator_payload_from_eit(
            _recording_with(pressure=_wave(10.0) - 40.0)
        )

        row = payload["array"][0]
        assert not np.any(np.isnan(row))
        assert row.min() < 0


class TestAChannelWithNoMeasurementIsRefused:
    def test_a_pod_channel_recorded_without_a_pod_is_rejected(self):
        recording = _recording_with(esophageal=_unmeasured())

        with pytest.raises(UnsupportedWorkflowError) as excinfo:
            ventilator_payload_from_eit(
                recording, channels=("airway_pressure", "esophageal_pressure")
            )

        message = str(excinfo.value)
        assert "esophageal_pressure" in message
        assert "pressure pod" in message

    def test_the_other_channels_still_load(self):
        payload = ventilator_payload_from_eit(
            _recording_with(esophageal=_unmeasured()),
            channels=("airway_pressure", "flow", "volume"),
        )

        assert payload["array"].shape == (3, N)
        assert not np.any(np.isnan(payload["array"]))


class TestListingChannelsDoesNotReadThem:
    def test_an_unmeasured_channel_is_still_listed(self):
        # Listing answers "what does this recording carry", which is a header
        # question; deciding it was never measured means reading every sample.
        available = available_ventilator_channels(
            _recording_with(esophageal=_unmeasured())
        )

        assert "esophageal_pressure" in available
