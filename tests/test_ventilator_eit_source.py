"""Ventilator waveforms read out of an EIT recording.

Draeger and Timpel write ventilator waveforms into the EIT file, next to the
images; the m3resp EIT readers give them as Signals in
``EITFileData.signals``. These tests cover the vendor channel names that map
onto the m3resp channel names, and the payload, which has the form of the
payload of the sEMG-file path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from m3resp.adapters.ventilator_adapter import (
    SUGGESTED_LOWPASS_HZ,
    VentilatorAdapter,
    available_ventilator_channels,
    split_channels,
    ventilator_payload_from_eit,
)
from m3resp.core.exceptions import UnsupportedWorkflowError
from m3resp.data import Signal
from m3resp.eit.io import EITFileData
from m3resp.modalities.eit import EITRecording

FS = 20.0
N = 400


def _waveform(label, values, unit=None, sample_frequency=FS, time=None) -> Signal:
    """A waveform of an EIT file, as the m3resp EIT readers give it."""

    values = np.asarray(values, dtype=float)
    return Signal(
        values=values,
        time=np.arange(len(values)) / FS if time is None else time,
        sample_frequency=sample_frequency,
        unit=unit,
        name=label,
        modality="eit",
        source="draeger",
    )


def _recording(*waveforms: Signal) -> EITFileData:
    """The data of an EIT file with these waveforms, keyed by their names."""

    return EITFileData(
        vendor="draeger",
        paths=(Path("recording.bin"),),
        sample_frequency=FS,
        pixel_impedance=Signal(
            values=np.zeros((N, 2, 2)), time=np.arange(N) / FS, sample_frequency=FS
        ),
        signals={waveform.name: waveform for waveform in waveforms},
    )


def _wave(scale: float = 1.0) -> np.ndarray:
    return scale * np.sin(2 * np.pi * 0.25 * np.arange(N) / FS)


def _draeger(**overrides) -> EITFileData:
    """A Draeger recording: global impedance and the three Medibus waveforms."""

    channels = {
        "global_impedance_(raw)": _waveform(
            "global_impedance_(raw)", _wave(100.0), unit="a.u."
        ),
        "airway pressure": _waveform("airway pressure", _wave(10.0), unit="mbar"),
        "flow": _waveform("flow", _wave(5.0), unit="L/min"),
        "volume": _waveform("volume", _wave(500.0), unit="mL"),
    }
    channels.update(overrides)
    return _recording(*channels.values())


def _timpel() -> EITFileData:
    return _recording(
        _waveform("global_impedance_(raw)", _wave(100.0), unit="a.u."),
        _waveform("airway_pressure_(timpel)", _wave(10.0), unit="cmH2O"),
        _waveform("flow_(timpel)", _wave(5.0), unit="L/s"),
        _waveform("volume_(timpel)", _wave(0.5), unit="L"),
    )


class TestChannelResolution:
    def test_resolves_draeger_medibus_names(self):
        assert available_ventilator_channels(_draeger()) == {
            "airway_pressure": "airway pressure",
            "flow": "flow",
            "volume": "volume",
        }

    def test_resolves_timpel_names_through_the_vendor_tag(self):
        # Timpel suffixes every label with `_(timpel)`; the same three
        # quantities must land on the same canonical names as Draeger's.
        assert available_ventilator_channels(_timpel()) == {
            "airway_pressure": "airway_pressure_(timpel)",
            "flow": "flow_(timpel)",
            "volume": "volume_(timpel)",
        }

    def test_ignores_non_ventilator_channels(self):
        assert (
            "global_impedance_(raw)"
            not in available_ventilator_channels(_draeger()).values()
        )

    def test_resolves_the_pressure_pod_channels(self):
        # A Draeger pressure pod records five distinct pressures. Each must
        # stay separable rather than collapsing onto "airway_pressure".
        recording = _draeger(
            **{
                "esophageal pressure (pod)": _waveform(
                    "esophageal pressure (pod)", _wave(8.0), unit="mbar"
                ),
                "transpulmonary pressure (pod)": _waveform(
                    "transpulmonary pressure (pod)", _wave(2.0), unit="mbar"
                ),
                "gastric pressure/auxiliary pressure (pod)": _waveform(
                    "gastric pressure/auxiliary pressure (pod)", _wave(6.0), unit="mbar"
                ),
            }
        )
        available = available_ventilator_channels(recording)
        assert available["esophageal_pressure"] == "esophageal pressure (pod)"
        assert available["transpulmonary_pressure"] == "transpulmonary pressure (pod)"
        assert (
            available["gastric_pressure"] == "gastric pressure/auxiliary pressure (pod)"
        )
        # `(pod)` marks a separate transducer, so it is not stripped the way a
        # vendor tag is: airway pressure stays the un-podded channel.
        assert available["airway_pressure"] == "airway pressure"

    def test_a_recording_without_ventilator_data_resolves_nothing(self):
        recording = _recording(
            _waveform("global_impedance_(raw)", _wave(100.0), unit="a.u.")
        )
        assert available_ventilator_channels(recording) == {}


class TestPayload:
    def test_rows_are_in_the_requested_channel_order(self):
        payload = ventilator_payload_from_eit(_draeger())
        assert payload["array"].shape == (3, N)
        assert np.allclose(payload["array"][0], _wave(10.0))
        assert np.allclose(payload["array"][1], _wave(5.0))
        assert np.allclose(payload["array"][2], _wave(500.0))

    def test_carries_the_sample_rate(self):
        assert ventilator_payload_from_eit(_draeger())["metadata"]["fs"] == FS

    def test_carries_vendor_labels_and_units(self):
        metadata = ventilator_payload_from_eit(_timpel())["metadata"]
        assert metadata["labels"] == [
            "airway_pressure_(timpel)",
            "flow_(timpel)",
            "volume_(timpel)",
        ]
        assert metadata["units"] == ["cmH2O", "L/s", "L"]

    def test_records_the_source(self):
        assert ventilator_payload_from_eit(_draeger())["metadata"]["source"] == "eit"

    def test_reports_channels_it_did_not_load(self):
        # The default selection is three channels; a caller must be able to
        # discover the pod pressures without loading the file again.
        recording = _draeger(
            **{
                "esophageal pressure (pod)": _waveform(
                    "esophageal pressure (pod)", _wave(8.0), unit="mbar"
                )
            }
        )
        metadata = ventilator_payload_from_eit(recording)["metadata"]
        assert "esophageal_pressure" in metadata["available_channels"]
        assert metadata["channels"] == ["airway_pressure", "flow", "volume"]

    def test_extra_channels_can_be_requested(self):
        recording = _draeger(
            **{
                "esophageal pressure (pod)": _waveform(
                    "esophageal pressure (pod)", _wave(8.0), unit="mbar"
                )
            }
        )
        payload = ventilator_payload_from_eit(
            recording,
            channels=("airway_pressure", "flow", "volume", "esophageal_pressure"),
        )
        assert payload["array"].shape == (4, N)
        assert np.allclose(payload["array"][3], _wave(8.0))

    def test_falls_back_to_the_time_axis_for_the_sample_rate(self):
        recording = _draeger()
        for channel in recording.signals.values():
            channel.sample_frequency = None
        fs = ventilator_payload_from_eit(recording)["metadata"]["fs"]
        assert fs == pytest.approx(FS)

    def test_an_explicit_sample_rate_wins(self):
        payload = ventilator_payload_from_eit(_draeger(), fs=50.0)
        assert payload["metadata"]["fs"] == 50.0

    def test_counts_nan_samples_per_channel(self):
        # Draeger writes NaN for a Medibus field that went unreported.
        recording = _draeger()
        recording.signals["flow"].values[:10] = np.nan
        metadata = ventilator_payload_from_eit(recording)["metadata"]
        assert metadata["nan_samples"] == {
            "airway_pressure": 0,
            "flow": 10,
            "volume": 0,
        }


def test_a_recording_gives_the_payload_of_its_file_data():
    file_data = _draeger()
    recording = EITRecording.from_file_data(file_data)

    actual = ventilator_payload_from_eit(recording)
    expected = ventilator_payload_from_eit(file_data)

    np.testing.assert_array_equal(actual["array"], expected["array"])
    np.testing.assert_array_equal(
        actual["metadata"].pop("time"), expected["metadata"].pop("time")
    )
    assert actual["metadata"] == expected["metadata"]


class TestPayloadErrors:
    def test_channels_on_different_time_axes_are_refused(self):
        recording = _draeger()
        recording.signals["flow"].time = np.arange(N) / FS + 0.01
        with pytest.raises(UnsupportedWorkflowError, match="different time axis"):
            ventilator_payload_from_eit(recording)

    def test_a_missing_channel_names_what_is_available(self):
        recording = _recording(_waveform("airway pressure", _wave(10.0), unit="mbar"))
        with pytest.raises(UnsupportedWorkflowError, match="flow"):
            ventilator_payload_from_eit(recording)

    def test_an_all_nan_channel_is_rejected_not_filtered(self):
        # An unconnected ventilator yields all-NaN. Failing here beats a
        # silently all-NaN filter result three steps downstream.
        recording = _draeger()
        recording.signals["volume"].values[:] = np.nan
        with pytest.raises(UnsupportedWorkflowError, match="not connected"):
            ventilator_payload_from_eit(recording)

    def test_channels_of_differing_lengths_are_rejected(self):
        recording = _draeger()
        recording.signals["flow"].values = _wave(5.0)[:-5]
        with pytest.raises(UnsupportedWorkflowError, match="lengths"):
            ventilator_payload_from_eit(recording)

    def test_an_unknown_channel_name_is_an_error(self):
        with pytest.raises(ValueError, match="Unknown ventilator channel"):
            ventilator_payload_from_eit(_draeger(), channels=("tidal_impedance",))

    def test_an_object_that_is_not_an_eit_recording_is_an_error(self):
        with pytest.raises(TypeError, match="EITRecording or EITFileData"):
            ventilator_payload_from_eit(object())


class TestPayloadFeedsTheExistingVentilatorPath:
    def test_split_channels_accepts_it_unchanged(self):
        bundle = split_channels(ventilator_payload_from_eit(_draeger()))
        assert bundle["fs"] == FS
        assert bundle["units"] == {
            "airway_pressure": "mbar",
            "flow": "L/min",
            "volume": "mL",
        }

    def test_preprocessing_runs_end_to_end(self):
        adapter = VentilatorAdapter()
        processed = adapter.preprocess(
            ventilator_payload_from_eit(_draeger()),
            lowpass_hz=SUGGESTED_LOWPASS_HZ,
        )
        assert processed["airway_pressure"].shape == (N,)
        # The requested cutoff is clamped below Nyquist for this 20 Hz
        # recording, so check that a filter ran rather than the exact value.
        assert processed["filter"]["lowpass_hz"] is not None

    def test_signals_carry_the_ventilator_categories(self):
        adapter = VentilatorAdapter()
        signals = adapter.to_signals(
            adapter.preprocess(ventilator_payload_from_eit(_draeger()))
        )
        assert {signal.category for signal in signals} == {
            "airway_pressure",
            "airflow",
            "volume",
        }


class TestLoadDispatch:
    def test_a_bin_file_goes_through_the_eit_loader(self):
        recording = _draeger()
        adapter = VentilatorAdapter(
            loader=lambda path, **kwargs: pytest.fail("used the sEMG loader"),
            eit_loader=lambda path, **kwargs: recording,
        )
        payload = adapter.load("recording.bin")
        assert payload["metadata"]["source"] == "eit"

    def test_a_non_bin_file_still_goes_through_the_semg_loader(self):
        payload = {"array": np.vstack([_wave()] * 3), "metadata": {"fs": FS}}
        adapter = VentilatorAdapter(
            loader=lambda path, **kwargs: payload,
            eit_loader=lambda path, **kwargs: pytest.fail("used the EIT loader"),
        )
        assert adapter.load("recording.txt") is payload

    def test_the_suffix_check_is_case_insensitive(self):
        adapter = VentilatorAdapter(eit_loader=lambda path, **kwargs: _draeger())
        assert adapter.load("RECORDING.BIN")["metadata"]["source"] == "eit"

    def test_source_forces_the_eit_path_for_an_odd_extension(self):
        adapter = VentilatorAdapter(
            loader=lambda path, **kwargs: pytest.fail("used the sEMG loader"),
            eit_loader=lambda path, **kwargs: _draeger(),
        )
        assert (
            adapter.load("recording.dat", source="eit")["metadata"]["source"] == "eit"
        )

    def test_source_forces_the_semg_path_for_a_bin(self):
        payload = {"array": np.vstack([_wave()] * 3), "metadata": {"fs": FS}}
        adapter = VentilatorAdapter(loader=lambda path, **kwargs: payload)
        assert adapter.load("recording.bin", source="emg") is payload

    def test_an_unknown_source_is_an_error(self):
        with pytest.raises(ValueError, match="must be 'eit', 'emg' or 'ventilator'"):
            VentilatorAdapter().load("recording.bin", source="impedance")

    def test_a_standalone_export_can_be_marked_as_its_own_source(self):
        # A ventilator/monitor export readable by the sEMG loader but carrying
        # its own clock, so synchronization aligns it separately.
        payload = {"array": np.vstack([_wave()] * 3), "metadata": {"fs": FS}}
        adapter = VentilatorAdapter(loader=lambda path, **kwargs: payload)
        assert adapter.load("monitor.txt", source="ventilator") is payload

    def test_channel_selection_reaches_the_payload(self):
        recording = _draeger(
            **{
                "esophageal pressure (pod)": _waveform(
                    "esophageal pressure (pod)", _wave(8.0), unit="mbar"
                )
            }
        )
        adapter = VentilatorAdapter(eit_loader=lambda path, **kwargs: recording)
        payload = adapter.load(
            "recording.bin",
            ventilator_channels=("airway_pressure", "flow", "esophageal_pressure"),
        )
        assert payload["metadata"]["channels"] == [
            "airway_pressure",
            "flow",
            "esophageal_pressure",
        ]


class TestSessionLoadsVentilatorFromBin:
    def test_the_sessions_eit_loader_covers_the_ventilator_path(self):
        from m3resp.core.session import M3Session

        file_data = _draeger()
        session = M3Session(eit_loader=lambda path, vendor=None, **kwargs: file_data)
        payload = session.load_ventilator("recording.bin")

        assert payload["metadata"]["source"] == "eit"
        assert session.raw["ventilator"].fs == FS
        # The legacy alias still points at the same recording.
        assert session.raw["vent"] is session.raw["ventilator"]

    def test_the_loaded_bin_preprocesses_like_any_other_recording(self):
        from m3resp.core.session import M3Session

        session = M3Session(eit_loader=lambda path, vendor=None, **kwargs: _draeger())
        session.load_ventilator("recording.bin")
        processed = session.preprocess_ventilator()

        assert processed["volume"].shape == (N,)
        assert {signal.channel for signal in session.signals} == {
            "airway_pressure",
            "flow",
            "volume",
        }
