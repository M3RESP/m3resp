"""Ventilator waveforms read out of an EIT recording rather than the sEMG file.

Ventilator data has two sources, and before this the ventilator path only knew
about one: the multi-channel file it shares with the sEMG. Draeger and Timpel
both write ventilator waveforms into the EIT ``*.bin`` itself, which
`eitprocessing` already parses into `ContinuousData`. These tests cover
resolving the vendors' channel names onto m3resp's canonical ones and packing
them into the same payload the sEMG-file path produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.adapters.ventilator_adapter import (
    VentilatorAdapter,
    available_ventilator_channels,
    split_channels,
    ventilator_payload_from_sequence,
)
from m3resp.core.exceptions import UnsupportedWorkflowError

FS = 20.0
N = 400


class _ContinuousData:
    """The `eitprocessing.ContinuousData` surface `_eit_source` duck-types on."""

    def __init__(self, label, values, unit=None, sample_frequency=FS, time=None):
        self.label = label
        self.name = label
        self.values = np.asarray(values, dtype=float)
        self.unit = unit
        self.sample_frequency = sample_frequency
        self.time = np.arange(len(self.values)) / FS if time is None else time


class _Sequence:
    """A loaded sequence, reduced to the collection `_eit_source` reads."""

    def __init__(self, *channels):
        self.continuous_data = {channel.label: channel for channel in channels}


def _wave(scale: float = 1.0) -> np.ndarray:
    return scale * np.sin(2 * np.pi * 0.25 * np.arange(N) / FS)


def _draeger(**overrides) -> _Sequence:
    """A Draeger sequence: global impedance plus the three Medibus waveforms."""

    channels = {
        "global_impedance_(raw)": _ContinuousData(
            "global_impedance_(raw)", _wave(100.0), unit="a.u."
        ),
        "airway pressure": _ContinuousData("airway pressure", _wave(10.0), unit="mbar"),
        "flow": _ContinuousData("flow", _wave(5.0), unit="L/min"),
        "volume": _ContinuousData("volume", _wave(500.0), unit="mL"),
    }
    channels.update(overrides)
    return _Sequence(*channels.values())


def _timpel() -> _Sequence:
    return _Sequence(
        _ContinuousData("global_impedance_(raw)", _wave(100.0), unit="a.u."),
        _ContinuousData("airway_pressure_(timpel)", _wave(10.0), unit="cmH2O"),
        _ContinuousData("flow_(timpel)", _wave(5.0), unit="L/s"),
        _ContinuousData("volume_(timpel)", _wave(0.5), unit="L"),
    )


class TestChannelResolution:
    def test_resolves_draeger_medibus_names(self):
        assert available_ventilator_channels(_draeger()) == {
            "pressure": "airway pressure",
            "flow": "flow",
            "volume": "volume",
        }

    def test_resolves_timpel_names_through_the_vendor_tag(self):
        # Timpel suffixes every label with `_(timpel)`; the same three
        # quantities must land on the same canonical names as Draeger's.
        assert available_ventilator_channels(_timpel()) == {
            "pressure": "airway_pressure_(timpel)",
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
        # stay separable rather than collapsing onto "pressure".
        sequence = _draeger(
            **{
                "esophageal pressure (pod)": _ContinuousData(
                    "esophageal pressure (pod)", _wave(8.0), unit="mbar"
                ),
                "transpulmonary pressure (pod)": _ContinuousData(
                    "transpulmonary pressure (pod)", _wave(2.0), unit="mbar"
                ),
                "gastric pressure/auxiliary pressure (pod)": _ContinuousData(
                    "gastric pressure/auxiliary pressure (pod)", _wave(6.0), unit="mbar"
                ),
            }
        )
        available = available_ventilator_channels(sequence)
        assert available["esophageal_pressure"] == "esophageal pressure (pod)"
        assert available["transpulmonary_pressure"] == "transpulmonary pressure (pod)"
        assert (
            available["gastric_pressure"] == "gastric pressure/auxiliary pressure (pod)"
        )
        # `(pod)` marks a separate transducer, so it is not stripped the way a
        # vendor tag is: airway pressure stays the un-podded channel.
        assert available["pressure"] == "airway pressure"

    def test_a_recording_without_ventilator_data_resolves_nothing(self):
        sequence = _Sequence(
            _ContinuousData("global_impedance_(raw)", _wave(100.0), unit="a.u.")
        )
        assert available_ventilator_channels(sequence) == {}


class TestPayload:
    def test_rows_are_in_the_requested_channel_order(self):
        payload = ventilator_payload_from_sequence(_draeger())
        assert payload["array"].shape == (3, N)
        assert np.allclose(payload["array"][0], _wave(10.0))
        assert np.allclose(payload["array"][1], _wave(5.0))
        assert np.allclose(payload["array"][2], _wave(500.0))

    def test_carries_the_sample_rate(self):
        assert ventilator_payload_from_sequence(_draeger())["metadata"]["fs"] == FS

    def test_carries_vendor_labels_and_units(self):
        metadata = ventilator_payload_from_sequence(_timpel())["metadata"]
        assert metadata["labels"] == [
            "airway_pressure_(timpel)",
            "flow_(timpel)",
            "volume_(timpel)",
        ]
        assert metadata["units"] == ["cmH2O", "L/s", "L"]

    def test_records_the_source(self):
        assert (
            ventilator_payload_from_sequence(_draeger())["metadata"]["source"] == "eit"
        )

    def test_reports_channels_it_did_not_load(self):
        # The default selection is three channels; a caller must be able to
        # discover the pod pressures without loading the file again.
        sequence = _draeger(
            **{
                "esophageal pressure (pod)": _ContinuousData(
                    "esophageal pressure (pod)", _wave(8.0), unit="mbar"
                )
            }
        )
        metadata = ventilator_payload_from_sequence(sequence)["metadata"]
        assert "esophageal_pressure" in metadata["available_channels"]
        assert metadata["channels"] == ["pressure", "flow", "volume"]

    def test_extra_channels_can_be_requested(self):
        sequence = _draeger(
            **{
                "esophageal pressure (pod)": _ContinuousData(
                    "esophageal pressure (pod)", _wave(8.0), unit="mbar"
                )
            }
        )
        payload = ventilator_payload_from_sequence(
            sequence, channels=("pressure", "flow", "volume", "esophageal_pressure")
        )
        assert payload["array"].shape == (4, N)
        assert np.allclose(payload["array"][3], _wave(8.0))

    def test_falls_back_to_the_time_axis_for_the_sample_rate(self):
        sequence = _draeger()
        for channel in sequence.continuous_data.values():
            channel.sample_frequency = None
        fs = ventilator_payload_from_sequence(sequence)["metadata"]["fs"]
        assert fs == pytest.approx(FS)

    def test_an_explicit_sample_rate_wins(self):
        payload = ventilator_payload_from_sequence(_draeger(), fs=50.0)
        assert payload["metadata"]["fs"] == 50.0

    def test_counts_nan_samples_per_channel(self):
        # Draeger writes NaN for a Medibus field that went unreported.
        sequence = _draeger()
        sequence.continuous_data["flow"].values[:10] = np.nan
        metadata = ventilator_payload_from_sequence(sequence)["metadata"]
        assert metadata["nan_samples"] == {"pressure": 0, "flow": 10, "volume": 0}


class TestPayloadErrors:
    def test_a_missing_channel_names_what_is_available(self):
        sequence = _Sequence(
            _ContinuousData("airway pressure", _wave(10.0), unit="mbar")
        )
        with pytest.raises(UnsupportedWorkflowError, match="flow"):
            ventilator_payload_from_sequence(sequence)

    def test_an_all_nan_channel_is_rejected_not_filtered(self):
        # An unconnected ventilator yields all-NaN. Failing here beats a
        # silently all-NaN filter result three steps downstream.
        sequence = _draeger()
        sequence.continuous_data["volume"].values[:] = np.nan
        with pytest.raises(UnsupportedWorkflowError, match="not connected"):
            ventilator_payload_from_sequence(sequence)

    def test_channels_of_differing_lengths_are_rejected(self):
        sequence = _draeger()
        sequence.continuous_data["flow"].values = _wave(5.0)[:-5]
        with pytest.raises(UnsupportedWorkflowError, match="lengths"):
            ventilator_payload_from_sequence(sequence)

    def test_an_unknown_channel_name_is_an_error(self):
        with pytest.raises(ValueError, match="Unknown ventilator channel"):
            ventilator_payload_from_sequence(_draeger(), channels=("tidal_impedance",))

    def test_a_non_sequence_is_an_error(self):
        with pytest.raises(TypeError, match="eitprocessing Sequence"):
            ventilator_payload_from_sequence(object())


class TestPayloadFeedsTheExistingVentilatorPath:
    def test_split_channels_accepts_it_unchanged(self):
        bundle = split_channels(ventilator_payload_from_sequence(_draeger()))
        assert bundle["fs"] == FS
        assert bundle["units"] == {
            "pressure": "mbar",
            "flow": "L/min",
            "volume": "mL",
        }

    def test_preprocessing_runs_end_to_end(self):
        adapter = VentilatorAdapter()
        processed = adapter.preprocess(ventilator_payload_from_sequence(_draeger()))
        assert processed["pressure"].shape == (N,)
        assert processed["filter"]["lowpass_hz"] is not None

    def test_signals_carry_the_ventilator_categories(self):
        adapter = VentilatorAdapter()
        signals = adapter.to_signals(
            adapter.preprocess(ventilator_payload_from_sequence(_draeger()))
        )
        assert {signal.category for signal in signals} == {
            "airway_pressure",
            "airflow",
            "volume",
        }


class TestLoadDispatch:
    def test_a_bin_file_goes_through_the_eit_loader(self):
        sequence = _draeger()
        adapter = VentilatorAdapter(
            loader=lambda path, **kwargs: pytest.fail("used the sEMG loader"),
            eit_loader=lambda path, **kwargs: sequence,
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
        with pytest.raises(ValueError, match="'eit' or 'emg'"):
            VentilatorAdapter().load("recording.bin", source="ventilator")

    def test_channel_selection_reaches_the_payload(self):
        sequence = _draeger(
            **{
                "esophageal pressure (pod)": _ContinuousData(
                    "esophageal pressure (pod)", _wave(8.0), unit="mbar"
                )
            }
        )
        adapter = VentilatorAdapter(eit_loader=lambda path, **kwargs: sequence)
        payload = adapter.load(
            "recording.bin",
            ventilator_channels=("pressure", "flow", "esophageal_pressure"),
        )
        assert payload["metadata"]["channels"] == [
            "pressure",
            "flow",
            "esophageal_pressure",
        ]


class TestSessionLoadsVentilatorFromBin:
    def test_the_sessions_eit_loader_covers_the_ventilator_path(self):
        from m3resp.adapters.eitprocessing_adapter import EITProcessingAdapter
        from m3resp.core.session import M3Session

        sequence = _draeger()
        session = M3Session(
            eit_adapter=EITProcessingAdapter(
                loader=lambda path, vendor=None, **kwargs: sequence
            )
        )
        payload = session.load_ventilator("recording.bin")

        assert payload["metadata"]["source"] == "eit"
        assert session.raw["ventilator"].fs == FS
        # The legacy alias still points at the same recording.
        assert session.raw["vent"] is session.raw["ventilator"]

    def test_the_loaded_bin_preprocesses_like_any_other_recording(self):
        from m3resp.adapters.eitprocessing_adapter import EITProcessingAdapter
        from m3resp.core.session import M3Session

        session = M3Session(
            eit_adapter=EITProcessingAdapter(
                loader=lambda path, vendor=None, **kwargs: _draeger()
            )
        )
        session.load_ventilator("recording.bin")
        processed = session.preprocess_ventilator()

        assert processed["volume"].shape == (N,)
        assert {signal.channel for signal in session.signals} == {
            "pressure",
            "flow",
            "volume",
        }
