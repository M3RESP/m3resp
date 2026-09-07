"""Several instruments recording ventilator data in one session.

A ventilator export and an EIT `*.bin` can both carry an airway pressure, and
they are different measurements: different transducer, different position,
different sampling. The session therefore files ventilator recordings by name
rather than holding one, and a non-primary recording's channels are named apart
so the two airway pressures stay distinct all the way to `session.signals`.

Also covers the two producers outside the ventilator adapter that used to pick
ventilator columns by position independently of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.adapters.resurfemg_adapter import ReSurfEMGAdapter, _ventilator_signals
from m3resp.core.exceptions import MissingModalityDataError
from m3resp.core.session import M3Session

FS = 100.0
N = 500


def _wave(scale: float = 1.0) -> np.ndarray:
    return scale * np.sin(2 * np.pi * 0.25 * np.arange(N) / FS)


def _payload(labels, units=None, rows=None) -> dict:
    rows = rows if rows is not None else [_wave(i + 1) for i in range(len(labels))]
    metadata: dict = {"fs": FS, "labels": list(labels)}
    if units is not None:
        metadata["units"] = list(units)
    return {"array": np.vstack(rows), "metadata": metadata}


def _session(
    payloads: dict[str, dict], eit_sequence: object | None = None
) -> M3Session:
    """A session whose EMG loader returns a different payload per path.

    `eit_sequence` injects an EIT loader too, for the `.bin` route where the
    ventilator waveforms come out of the EIT recording itself.
    """

    from m3resp.adapters.eitprocessing_adapter import EITProcessingAdapter

    return M3Session(
        emg_adapter=ReSurfEMGAdapter(loader=lambda path, **kwargs: payloads[str(path)]),
        eit_adapter=(
            EITProcessingAdapter(
                loader=lambda path, vendor=None, **kwargs: eit_sequence
            )
            if eit_sequence is not None
            else None
        ),
    )


class _ContinuousData:
    """The `eitprocessing.ContinuousData` surface the ventilator path reads."""

    def __init__(self, label, values, unit=None):
        self.label = label
        self.name = label
        self.values = np.asarray(values, dtype=float)
        self.unit = unit
        self.sample_frequency = FS
        self.time = np.arange(len(self.values)) / FS


class _Sequence:
    """A loaded EIT sequence carrying Medibus ventilator waveforms."""

    def __init__(self):
        self.continuous_data = {
            channel.label: channel
            for channel in (
                _ContinuousData("global_impedance_(raw)", _wave(100.0), "a.u."),
                _ContinuousData("airway pressure", _wave(10.0), "mbar"),
                _ContinuousData("flow", _wave(5.0), "L/min"),
                _ContinuousData("volume", _wave(500.0), "mL"),
            )
        }


class TestPositionalProducersNowResolveByName:
    def test_emg_postprocessing_finds_channels_by_label(self):
        # Columns deliberately out of the historical pressure/flow/volume
        # order: only name resolution gets these right.
        signals = _ventilator_signals(_payload(["Volume", "Paw", "Flow"]))
        assert signals is not None
        assert signals["channel_indices"] == {"pressure": 1, "flow": 2, "volume": 0}

    def test_an_unlabelled_recording_still_uses_the_old_positions(self):
        payload = {"array": np.vstack([_wave(i + 1) for i in range(3)])}
        signals = _ventilator_signals(payload, fs=FS)
        assert signals is not None
        assert signals["channel_indices"] == {"pressure": 0, "flow": 1, "volume": 2}

    def test_explicit_indices_still_win(self):
        signals = _ventilator_signals(
            _payload(["Paw", "Flow", "Volume"]), volume_channel=0
        )
        assert signals is not None
        assert signals["channel_indices"]["volume"] == 0

    def test_no_ventilator_is_still_no_signals(self):
        assert _ventilator_signals(None) is None

    def test_a_missing_array_is_still_an_error(self):
        with pytest.raises(TypeError, match="needs an array"):
            _ventilator_signals({"metadata": {"fs": FS}})

    def test_a_missing_sample_rate_is_still_an_error(self):
        with pytest.raises(TypeError, match="needs a sampling rate"):
            _ventilator_signals({"array": np.vstack([_wave()] * 3), "metadata": {}})

    def test_the_pipeline_step_can_select_channels(self):
        from m3resp.workflows.steps.emg import ventilator_channels

        result = ventilator_channels(
            _payload(["Paw", "Flow", "Volume", "esophageal pressure (pod)"]),
            channels=("pressure", "esophageal_pressure"),
        )
        assert set(result["ventilator_signals"]["channels"]) == {
            "pressure",
            "esophageal_pressure",
        }


class TestOneRecordingIsUnchanged:
    def test_the_primary_recording_is_still_reachable_the_old_way(self):
        session = _session({"vent.txt": _payload(["Paw", "Flow", "Volume"])})
        session.load_ventilator("vent.txt")

        assert session.ventilator is not None
        assert session.raw["ventilator"] is session.ventilator
        assert session.raw["vent"] is session.ventilator
        assert session.primary_ventilator_name() == "default"

    def test_channels_keep_their_bare_names(self):
        session = _session({"vent.txt": _payload(["Paw", "Flow", "Volume"])})
        session.load_ventilator("vent.txt")
        session.preprocess_ventilator()

        assert {signal.channel for signal in session.signals} == {
            "pressure",
            "flow",
            "volume",
        }

    def test_the_recording_still_gets_its_channels_back(self):
        session = _session({"vent.txt": _payload(["Paw", "Flow", "Volume"])})
        session.load_ventilator("vent.txt")
        session.preprocess_ventilator()

        assert session.ventilator is not None
        assert session.ventilator.pressure is not None
        assert session.ventilator.fs == FS


class TestTwoRecordings:
    def _session(self) -> M3Session:
        session = _session(
            {
                "ventilator.txt": _payload(
                    ["Paw", "Flow", "Volume"], units=["cmH2O", "L/min", "L"]
                ),
                "eit_medibus.txt": _payload(
                    ["airway pressure", "flow", "volume"],
                    units=["mbar", "L/min", "mL"],
                ),
            }
        )
        session.load_ventilator("ventilator.txt")
        session.load_ventilator("eit_medibus.txt", name="eit")
        return session

    def test_both_are_filed_under_their_names(self):
        session = self._session()
        assert set(session.ventilators) == {"default", "eit"}

    def test_the_first_stays_primary(self):
        session = self._session()
        assert session.primary_ventilator_name() == "default"
        assert session.raw["ventilator"] is session.ventilators["default"]

    def test_a_named_recording_is_retrievable(self):
        session = self._session()
        assert session.get_ventilator("eit") is session.ventilators["eit"]

    def test_an_unknown_name_names_what_is_loaded(self):
        session = self._session()
        with pytest.raises(MissingModalityDataError, match="'eit'"):
            session.get_ventilator("nope")

    def test_asking_before_loading_is_an_error(self):
        with pytest.raises(MissingModalityDataError, match="load_ventilator"):
            M3Session().get_ventilator()

    def test_the_second_recordings_channels_are_named_apart(self):
        session = self._session()
        session.preprocess_ventilator()
        session.preprocess_ventilator(name="eit")

        channels = {signal.channel for signal in session.signals}
        assert {"pressure", "flow", "volume"} <= channels
        assert {"pressure__eit", "flow__eit", "volume__eit"} <= channels

    def test_both_airway_pressures_survive_as_the_same_quantity(self):
        session = self._session()
        session.preprocess_ventilator()
        session.preprocess_ventilator(name="eit")

        airway = session.signals.for_category("airway_pressure")
        assert {signal.channel for signal in airway} == {"pressure", "pressure__eit"}

    def test_each_pressure_keeps_the_unit_its_device_reported(self):
        session = self._session()
        session.preprocess_ventilator()
        session.preprocess_ventilator(name="eit")

        units = {
            signal.channel: signal.unit
            for signal in session.signals.for_category("volume")
        }
        # Not converted: the ventilator export reports litres, the Medibus
        # channels millilitres.
        assert units == {"volume": "L", "volume__eit": "mL"}

    def test_each_recording_lands_in_its_own_variant(self):
        session = self._session()
        session.preprocess_ventilator()
        session.preprocess_ventilator(name="eit")

        assert set(session.processed_variants["ventilator"]) == {"default", "eit"}

    def test_the_named_recording_gets_its_own_channels_back(self):
        session = self._session()
        session.preprocess_ventilator(name="eit")

        recording = session.get_ventilator("eit")
        assert recording.pressure is not None
        assert recording.volume is not None
        # The primary recording was not touched by preprocessing the other one.
        assert session.ventilators["default"].pressure is None

    def test_the_two_pressures_hold_different_data(self):
        session = _session(
            {
                "a.txt": _payload(["Paw", "Flow", "Volume"], rows=[_wave(10.0)] * 3),
                "b.txt": _payload(["Paw", "Flow", "Volume"], rows=[_wave(3.0)] * 3),
            }
        )
        session.load_ventilator("a.txt")
        session.load_ventilator("b.txt", name="second")
        session.preprocess_ventilator()
        session.preprocess_ventilator(name="second")

        by_channel = {
            signal.channel: signal
            for signal in session.signals
            if signal.processing_state == "raw"
        }
        assert not np.allclose(
            by_channel["pressure"].values, by_channel["pressure__second"].values
        )


class TestVentilatorInheritsItsHostsClock:
    """Ventilator channels carried inside another modality's file move with it.

    Waveforms read out of the EIT `*.bin` or the sEMG export are samples of
    that recording's own time base. They are copied into a separate array
    rather than kept as a view, so aligning the host has to crop them too - and
    must not also shift them by a ventilator offset, which would move them
    twice.
    """

    def test_a_bin_is_recognised_as_carrying_the_eit_clock(self):
        from m3resp.adapters.ventilator_adapter import resolve_ventilator_source

        assert resolve_ventilator_source("study.bin") == "eit"
        assert resolve_ventilator_source("study.txt") == "emg"
        assert resolve_ventilator_source("study.txt", "ventilator") == "ventilator"

    def test_a_bin_sourced_recording_reports_the_eit_clock(self):
        from m3resp.synchronization.cropping import ventilator_clock

        session = _session({}, eit_sequence=_Sequence())
        session.load_ventilator("study.bin")
        assert ventilator_clock(session.ventilator) == "eit"

    def test_cropping_eit_also_crops_a_bin_sourced_ventilator(self):
        from m3resp.synchronization.cropping import crop_loaded_modality

        session = _session({}, eit_sequence=_Sequence())
        session.load_ventilator("study.bin")
        before = session.ventilator.data["array"].shape[1]

        crop_loaded_modality(session, "eit", 1.0)
        assert session.ventilator.data["array"].shape[1] == before - int(1.0 * FS)

    def test_the_ventilator_offset_does_not_move_a_bin_sourced_recording(self):
        from m3resp.synchronization.cropping import crop_loaded_modality

        session = _session({}, eit_sequence=_Sequence())
        session.load_ventilator("study.bin")
        before = session.ventilator.data["array"].shape[1]

        assert crop_loaded_modality(session, "ventilator", 1.0) == 0
        assert session.ventilator.data["array"].shape[1] == before

    def test_an_semg_sourced_recording_reports_the_emg_clock(self):
        from m3resp.synchronization.cropping import ventilator_clock

        session = _session({"study.txt": _payload(["Paw", "Flow", "Volume"])})
        session.load_ventilator("study.txt")
        assert ventilator_clock(session.ventilator) == "emg"

    def test_a_standalone_export_keeps_its_own_clock(self):
        from m3resp.synchronization.cropping import ventilator_clock

        session = _session({"monitor.txt": _payload(["Paw", "Flow", "Volume"])})
        session.load_ventilator("monitor.txt", source="ventilator")
        assert ventilator_clock(session.ventilator) == "ventilator"

    def test_cropping_emg_also_crops_the_ventilator_it_carried(self):
        from m3resp.synchronization.cropping import crop_loaded_modality

        session = _session({"study.txt": _payload(["Paw", "Flow", "Volume"])})
        session.load_ventilator("study.txt")
        before = session.ventilator.data["array"].shape[1]

        crop_loaded_modality(session, "emg", 1.0)
        after = session.ventilator.data["array"].shape[1]
        assert after == before - int(1.0 * FS)

    def test_the_ventilator_offset_does_not_move_a_hosted_recording_again(self):
        from m3resp.synchronization.cropping import crop_loaded_modality

        session = _session({"study.txt": _payload(["Paw", "Flow", "Volume"])})
        session.load_ventilator("study.txt")
        before = session.ventilator.data["array"].shape[1]

        # Already aligned with the sEMG it came from: a ventilator offset must
        # be a no-op for it, or it would be shifted twice.
        assert crop_loaded_modality(session, "ventilator", 1.0) == 0
        assert session.ventilator.data["array"].shape[1] == before

    def test_a_standalone_export_is_cropped_by_the_ventilator_offset(self):
        from m3resp.synchronization.cropping import crop_loaded_modality

        session = _session({"monitor.txt": _payload(["Paw", "Flow", "Volume"])})
        session.load_ventilator("monitor.txt", source="ventilator")
        before = session.ventilator.data["array"].shape[1]

        crop_loaded_modality(session, "ventilator", 1.0)
        assert session.ventilator.data["array"].shape[1] == before - int(1.0 * FS)

    def test_a_standalone_export_is_untouched_by_the_emg_offset(self):
        from m3resp.synchronization.cropping import crop_loaded_modality

        session = _session({"monitor.txt": _payload(["Paw", "Flow", "Volume"])})
        session.load_ventilator("monitor.txt", source="ventilator")
        before = session.ventilator.data["array"].shape[1]

        crop_loaded_modality(session, "emg", 1.0)
        assert session.ventilator.data["array"].shape[1] == before

    def test_each_recording_follows_its_own_host(self):
        from m3resp.synchronization.cropping import crop_loaded_modality

        session = _session(
            {
                "study.txt": _payload(["Paw", "Flow", "Volume"]),
                "monitor.txt": _payload(["Paw", "Flow", "Volume"]),
            }
        )
        session.load_ventilator("study.txt")
        session.load_ventilator("monitor.txt", name="monitor", source="ventilator")
        hosted = session.ventilators["default"].data["array"].shape[1]
        standalone = session.ventilators["monitor"].data["array"].shape[1]

        crop_loaded_modality(session, "emg", 1.0)
        assert session.ventilators["default"].data["array"].shape[1] < hosted
        assert session.ventilators["monitor"].data["array"].shape[1] == standalone
