"""How the ventilator steps read the airway pressure and the recording length
from a channel bundle made by `ventilator.channels`."""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.adapters.ventilator_adapter import split_channels
from m3resp.core.exceptions import MissingModalityDataError
from m3resp.core.session import M3Session
from m3resp.workflows.steps.ventilator._shared import _airway_pressure
from m3resp.workflows.steps.ventilator.detection import find_occluded_breaths
from m3resp.workflows.steps.ventilator.normalization import normalize_breaths

FS = 100.0
N = 1000


def _bundle(labels, units=None, **kwargs):
    rows = np.vstack([np.full(N, float(i + 1)) for i in range(len(labels))])
    metadata = {"fs": FS, "labels": list(labels)}
    if units is not None:
        metadata["units"] = list(units)
    return split_channels({"array": rows, "metadata": metadata}, **kwargs)


class TestAirwayPressure:
    def test_the_unit_is_the_one_the_recording_reported(self):
        bundle = _bundle(["Paw", "Flow", "Volume"], units=["kPa", "L/min", "mL"])
        pressure, unit = _airway_pressure(bundle, "a step")
        assert unit == "kPa"
        assert np.all(pressure == 1.0)

    def test_a_second_recordings_pressure_is_found_under_its_own_name(self):
        bundle = _bundle(["Paw", "Flow", "Volume"], origin="monitor", qualify=True)
        assert "airway_pressure" not in bundle
        pressure, _ = _airway_pressure(bundle, "a step")
        assert np.all(pressure == 1.0)

    def test_a_missing_pressure_is_a_clear_error(self):
        bundle = _bundle(["Flow", "Volume"], channels=("flow", "volume"))
        with pytest.raises(MissingModalityDataError, match="airway_pressure"):
            find_occluded_breaths(bundle, peep=5.0)


class TestRecordingLength:
    def test_a_breath_is_cut_at_the_end_without_an_airway_pressure(self):
        # Volume only: the recording length still comes from the bundle, so a
        # breath at the very last sample does not run past the end.
        bundle = _bundle(["Volume"], channels=("volume",))
        session = M3Session()
        normalize_breaths(np.array([N - 1]), bundle, session)
        (breath,) = session.events["ventilator_breaths"]
        assert breath.end_time <= N / FS
