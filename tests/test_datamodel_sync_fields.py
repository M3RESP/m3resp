"""The data model's `SignalStream.sync_method` and `time_offset_ms` follow the
session's synchronization (`session.sync_methods`, `session.start_times`)."""

from __future__ import annotations

import numpy as np

from m3resp.adapters import EITProcessingAdapter, ReSurfEMGAdapter
from m3resp.core.session import M3Session
from m3resp.data.signals import Signal
from m3resp.datamodel import DataModelRecorder, DataModelStore

FS = 100.0


def _payload() -> dict:
    time = np.arange(int(10 * FS)) / FS
    return {
        "array": np.vstack([np.sin(time), np.cos(time), time]),
        "metadata": {"fs": FS, "labels": ["pressure", "flow", "volume"]},
    }


def _session() -> tuple[M3Session, DataModelStore]:
    session = M3Session(
        eit_adapter=EITProcessingAdapter(loader=lambda path, **kwargs: {"path": path}),
        emg_adapter=ReSurfEMGAdapter(loader=lambda path, **kwargs: _payload()),
    )
    store = DataModelStore()
    session.datamodel = DataModelRecorder(session, store)
    return session, store


def _stream(store: DataModelStore, signal_type: str):
    (stream,) = [
        s for s in store.signal_streams.values() if s.signal_type == signal_type
    ]
    return stream


def _pressure(channel: str, **metadata) -> Signal:
    return Signal(
        values=np.zeros(10),
        time=np.arange(10) / FS,
        sample_frequency=FS,
        modality="ventilator",
        category="airway_pressure",
        channel=channel,
        metadata=metadata,
    )


class TestEitAndEmgStreams:
    def test_a_stream_is_unsynchronized_until_the_recording_is(self):
        session, store = _session()
        session.load_emg("emg.txt")

        assert _stream(store, "emg_raw").sync_method is None
        assert _stream(store, "emg_raw").time_offset_ms is None

        session.synchronize_raw_modalities(
            offset_seconds={"emg": 1.5}, reference_modality="eit"
        )

        assert _stream(store, "emg_raw").sync_method == "manual"
        assert _stream(store, "emg_raw").time_offset_ms == 1500.0

    def test_skip_synchronization_is_recorded_as_none(self):
        session, store = _session()
        session.load_eit("recording.bin")
        session.load_emg("emg.txt")

        session.skip_synchronization()

        for signal_type in ("eit_waveform", "emg_raw"):
            assert _stream(store, signal_type).sync_method == "none"
            assert _stream(store, signal_type).time_offset_ms == 0.0

    def test_reloading_clears_the_stream_fields(self):
        session, store = _session()
        session.load_emg("emg.txt")
        session.synchronize_raw_modalities(
            offset_seconds={"emg": 1.5}, reference_modality="eit"
        )

        session.load_emg("emg.txt")

        assert _stream(store, "emg_raw").sync_method is None
        assert _stream(store, "emg_raw").time_offset_ms is None

    def test_slicing_moves_the_offset(self):
        session, store = _session()
        session.load_emg("emg.txt")
        session.synchronize_raw_modalities(
            offset_seconds={"emg": 1.5}, reference_modality="eit"
        )

        session.slice_emg(2.0)

        assert _stream(store, "emg_raw").time_offset_ms == 3500.0


class TestVentilatorStreams:
    def test_each_standalone_recording_gets_its_own_synchronization(self):
        session, _ = _session()
        session.load_ventilator("ventilator.txt", source="ventilator")
        session.load_ventilator("monitor.txt", source="ventilator", name="monitor")
        session.synchronize_raw_modalities(
            offset_seconds={"ventilator": 1.0, "ventilator:monitor": 12.5},
            reference_modality="eit",
        )

        primary = session.datamodel.record_signal(_pressure("pressure"))
        monitor = session.datamodel.record_signal(
            _pressure("pressure__monitor", recording="monitor")
        )

        assert (primary.sync_method, primary.time_offset_ms) == ("manual", 1000.0)
        assert (monitor.sync_method, monitor.time_offset_ms) == ("manual", 12500.0)

    def test_a_pod_channel_inside_one_recording_belongs_to_that_recording(self):
        # A `__pod` suffix here marks a second airway pressure inside the same
        # file, not a recording named "pod".
        session, _ = _session()
        session.load_ventilator("ventilator.txt", source="ventilator")
        session.synchronize_raw_modalities(
            offset_seconds={"ventilator": 1.0}, reference_modality="eit"
        )

        pod = session.datamodel.record_signal(_pressure("pressure__pod"))

        assert (pod.sync_method, pod.time_offset_ms) == ("manual", 1000.0)

    def test_ventilator_data_from_the_emg_file_takes_the_emg_synchronization(self):
        session, _ = _session()
        session.load_emg("biopac.txt")
        session.load_ventilator("biopac.txt")  # on the EMG clock
        session.synchronize_raw_modalities(
            offset_seconds={"emg": -2.0, "ventilator": 7.0}, reference_modality="eit"
        )

        stream = session.datamodel.record_signal(_pressure("pressure"))

        assert (stream.sync_method, stream.time_offset_ms) == ("manual", -2000.0)

    def test_preprocess_ventilator_names_the_recording_on_its_signals(self):
        session, _ = _session()
        session.load_ventilator("ventilator.txt", source="ventilator")
        session.load_ventilator("monitor.txt", source="ventilator", name="monitor")

        session.preprocess_ventilator()
        session.preprocess_ventilator(name="monitor")

        recordings = {
            signal.channel: signal.metadata.get("recording")
            for signal in session.signals
            if signal.modality == "ventilator"
        }
        assert recordings["pressure"] == "default"
        assert recordings["pressure__monitor"] == "monitor"
