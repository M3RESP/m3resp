"""Names of the cutting steps, and cutting one m3resp `Signal`.

`*.slice_recording` cuts the loaded recording in the session;
`*.slice_signal` cuts one signal passed along in a workflow. The names used
before the rename still work.
"""

from __future__ import annotations

import numpy as np
import pytest

import m3resp.workflows.steps  # noqa: F401 - registers the built-in steps
from m3resp.data.signals import Signal
from m3resp.workflows.registry import available_steps, get_step
from m3resp.workflows.steps.eit.slicing import slice_signal

FS = 10.0


def _signal() -> Signal:
    """Five seconds of global impedance whose values equal the sample index."""

    time = 36528.5 + np.arange(50) / FS
    return Signal(
        values=np.arange(50, dtype=float),
        time=time,
        sample_frequency=FS,
        unit="AU",
        name="global impedance",
        modality="eit",
        channel="global_impedance",
        source="eitprocessing",
    )


class TestStepNames:
    @pytest.mark.parametrize(
        ("old_name", "new_name"),
        [("eit.slice", "eit.slice_signal"), ("emg.slice", "emg.slice_recording")],
    )
    def test_the_old_name_still_finds_the_step(self, old_name, new_name):
        assert get_step(old_name) is get_step(new_name)

    def test_only_the_new_names_are_listed(self):
        names = available_steps()
        assert {
            "eit.slice_signal",
            "eit.slice_recording",
            "emg.slice_recording",
        } <= set(names)
        assert "eit.slice" not in names
        assert "emg.slice" not in names


class TestSliceSignalOnAnM3respSignal:
    def test_index_mode_cuts_values_and_time_together(self):
        result = slice_signal(_signal(), start=10, end=20, mode="index")["result"]

        assert isinstance(result, Signal)
        assert result.values.tolist() == list(range(10, 20))
        assert result.time[0] == pytest.approx(36528.5 + 1.0)
        assert len(result.time) == 10

    def test_time_mode_uses_the_signals_own_time_values(self):
        # For EIT data that is the time of day stored in the file.
        result = slice_signal(
            _signal(), start=36528.5 + 1.0, end=36528.5 + 2.0, mode="time"
        )["result"]

        assert result.values.tolist() == list(range(10, 20))

    def test_everything_else_about_the_signal_is_kept(self):
        original = _signal()

        result = slice_signal(original, start=10, end=20, mode="index")["result"]

        assert (result.unit, result.modality, result.channel, result.source) == (
            original.unit,
            original.modality,
            original.channel,
            original.source,
        )
        assert result.sample_frequency == FS
        assert result.metadata["slice"] == {"start_index": 10, "end_index": 20}
        # The input signal itself is not changed.
        assert len(original.values) == 50
        assert "slice" not in original.metadata

    def test_a_window_with_no_samples_is_refused(self):
        with pytest.raises(ValueError, match="keeps no samples"):
            slice_signal(_signal(), start=0.0, end=1.0, mode="time")
