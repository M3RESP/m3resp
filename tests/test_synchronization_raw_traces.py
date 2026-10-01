"""Before/after traces for the raw synchronization plot
(`m3resp.synchronization.raw_traces`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from m3resp.core.session import M3Session
from m3resp.modalities.emg import EMGRecording
from m3resp.synchronization.raw_traces import raw_synchronization_traces


class TestRawSynchronizationTraces:
    def test_no_loaded_modality_returns_empty_traces(self):
        session = M3Session()
        assert raw_synchronization_traces(session, "emg") == {}

    def test_emg_trace_includes_time_and_values(self):
        session = M3Session()
        array = np.array([[0.0, 1.0, 2.0, 3.0]])
        session.emg = EMGRecording(
            data={
                "array": array,
                "metadata": {"fs": 2.0, "labels": ["ch0"], "units": ["mV"]},
            },
            path=Path("synthetic.npy"),
        )

        traces = raw_synchronization_traces(session, "emg")

        assert traces["emg"]["values"] == [0.0, 1.0, 2.0, 3.0]
        assert traces["emg"]["time"] == [0.0, 0.5, 1.0, 1.5]
