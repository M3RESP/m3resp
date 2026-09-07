"""`VentilatorAdapter`: the first adapter with no upstream library behind it.

Neither `eitprocessing` nor `resurfemg` implements ventilator preprocessing, so
these defaults are native (`m3resp.processing.filters`/`.peaks`) rather than a
wrapper. Ventilator channels previously reached `session.signals` not at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.adapters.ventilator_adapter import (
    SUGGESTED_LOWPASS_HZ,
    VentilatorAdapter,
    split_channels,
)
from m3resp.core.exceptions import UnsupportedWorkflowError
from m3resp.modalities.ventilator import VentilatorRecording

FS = 100.0
N = 1000


def _breathing(frequency: float = 0.25, noise_hz: float = 40.0, amplitude=0.2):
    time = np.arange(N) / FS
    return np.sin(2 * np.pi * frequency * time) + amplitude * np.sin(
        2 * np.pi * noise_hz * time
    )


def _payload(fs: float = FS) -> dict:
    wave = _breathing()
    return {
        "array": np.vstack([wave * 10, wave * 5, wave]),
        "metadata": {
            "fs": fs,
            "labels": ["Paw", "Flow", "Volume"],
            "units": ["cmH2O", "L/min", "L"],
        },
    }


def _adapter(payload: dict | None = None) -> VentilatorAdapter:
    data = payload if payload is not None else _payload()
    return VentilatorAdapter(loader=lambda path, **kwargs: data)


class TestSplitChannels:
    def test_names_the_three_quantities(self):
        bundle = split_channels(_payload())
        assert bundle["pressure"].shape == (N,)
        assert bundle["flow"].shape == (N,)
        assert bundle["volume"].shape == (N,)

    def test_reads_the_sample_rate_from_metadata(self):
        assert split_channels(_payload())["fs"] == FS

    def test_an_explicit_sample_rate_overrides_metadata(self):
        assert split_channels(_payload(), fs=50.0)["fs"] == 50.0

    def test_resolves_units_from_metadata(self):
        units = split_channels(_payload())["units"]
        assert units == {"pressure": "cmH2O", "flow": "L/min", "volume": "L"}

    def test_falls_back_to_default_units(self):
        payload = _payload()
        payload["metadata"].pop("units")
        assert split_channels(payload)["units"]["pressure"] == "cmH2O"

    def test_channel_indices_are_configurable(self):
        bundle = split_channels(_payload(), pressure_channel=2, volume_channel=0)
        # Channel 2 is the unscaled wave, channel 0 is scaled by 10.
        assert np.allclose(bundle["volume"] / 10.0, bundle["pressure"], atol=1e-9)

    def test_accepts_a_recording_object(self):
        recording = VentilatorRecording(data=_payload(), path="v.txt")
        assert split_channels(recording)["fs"] == FS

    def test_missing_sample_rate_is_an_error(self):
        payload = _payload()
        payload["metadata"].pop("fs")
        with pytest.raises(TypeError, match="sampling rate"):
            split_channels(payload)

    def test_out_of_range_channel_is_an_error(self):
        with pytest.raises(IndexError, match="out of range"):
            split_channels(_payload(), volume_channel=9)


class TestPreprocess:
    def test_does_not_filter_unless_asked(self):
        # Low-passing ventilator waveforms is not standard practice, so
        # loading a recording must return what the ventilator recorded.
        processed = _adapter().preprocess(_payload())
        assert processed["filter"]["lowpass_hz"] is None
        assert processed["filtered"] == {}
        assert np.allclose(processed["volume"], processed["raw"]["volume"])

    def test_attenuates_out_of_band_noise_when_a_cutoff_is_given(self):
        processed = _adapter().preprocess(_payload(), lowpass_hz=SUGGESTED_LOWPASS_HZ)
        clean = np.sin(2 * np.pi * 0.25 * np.arange(N) / FS)
        raw_error = np.abs(processed["raw"]["volume"] - clean).mean()
        filtered_error = np.abs(processed["volume"] - clean).mean()
        assert filtered_error < raw_error / 2

    def test_keeps_the_unfiltered_channels_available(self):
        processed = _adapter().preprocess(_payload(), lowpass_hz=SUGGESTED_LOWPASS_HZ)
        assert set(processed["raw"]) == {"pressure", "flow", "volume"}
        assert not np.allclose(processed["raw"]["volume"], processed["volume"])

    def test_plain_keys_expose_the_processed_signal(self):
        processed = _adapter().preprocess(_payload(), lowpass_hz=SUGGESTED_LOWPASS_HZ)
        assert np.allclose(processed["volume"], processed["filtered"]["volume"])

    def test_records_the_filter_it_applied(self):
        processed = _adapter().preprocess(_payload(), lowpass_hz=SUGGESTED_LOWPASS_HZ)
        assert processed["filter"]["lowpass_hz"] == SUGGESTED_LOWPASS_HZ
        assert processed["filter"]["filter_order"] == 4

    def test_cutoff_is_clamped_below_nyquist(self):
        # A low-rate export must stay usable rather than raise.
        processed = _adapter().preprocess(_payload(fs=20.0), lowpass_hz=50.0)
        assert processed["filter"]["lowpass_hz"] == pytest.approx(9.5)

    def test_a_custom_callable_bypasses_the_default(self):
        sentinel = {"custom": True}
        result = _adapter().preprocess(
            _payload(), preprocess=lambda rec, **kw: sentinel
        )
        assert result is sentinel


class TestToSignals:
    def _signals(self, **kwargs):
        adapter = _adapter()
        return adapter.to_signals(adapter.preprocess(_payload(), **kwargs))

    def test_emits_one_raw_signal_per_channel_by_default(self):
        signals = self._signals()
        assert len(signals) == 3
        assert {s.processing_state for s in signals} == {"raw"}

    def test_emits_raw_and_processed_per_channel_when_filtered(self):
        assert len(self._signals(lowpass_hz=SUGGESTED_LOWPASS_HZ)) == 6

    def test_every_signal_is_the_ventilator_modality(self):
        assert {signal.modality for signal in self._signals()} == {"ventilator"}

    def test_channels_are_distinguished_by_category(self):
        # The whole point of the modality/category split: one device, three
        # quantities, all still tellable apart.
        categories = {signal.category for signal in self._signals()}
        assert categories == {"airway_pressure", "airflow", "volume"}

    def test_carries_units_and_sample_rate(self):
        pressure = next(s for s in self._signals() if s.category == "airway_pressure")
        assert pressure.unit == "cmH2O"
        assert pressure.sample_frequency == FS

    def test_processed_signals_record_the_method(self):
        signals = self._signals(lowpass_hz=SUGGESTED_LOWPASS_HZ)
        processed = [s for s in signals if s.processing_state == "processed"]
        assert len(processed) == 3
        assert all("lowpass_filter" in (s.method or "") for s in processed)

    def test_raw_signals_have_no_method(self):
        raw = [s for s in self._signals() if s.processing_state == "raw"]
        assert len(raw) == 3
        assert all(s.method is None for s in raw)

    def test_time_axis_matches_the_values(self):
        for signal in self._signals():
            assert signal.time.shape == signal.values.shape

    def test_rejects_a_bundle_it_did_not_produce(self):
        with pytest.raises(UnsupportedWorkflowError, match="preprocess_ventilator"):
            _adapter().to_signals({"not": "a bundle"})


class TestDetectBreaths:
    def test_detects_breaths_on_the_volume_channel(self):
        adapter = _adapter()
        breaths = adapter.detect_breaths(adapter.preprocess(_payload()))
        # 1000 samples at 100 Hz = 10 s of 0.25 Hz breathing.
        assert 2 <= len(breaths) <= 3

    def test_breaths_carry_the_canonical_modality(self):
        adapter = _adapter()
        breaths = adapter.detect_breaths(adapter.preprocess(_payload()))
        assert {breath.modality for breath in breaths} == {"ventilator"}

    def test_breaths_have_real_times_not_just_indices(self):
        adapter = _adapter()
        breaths = adapter.detect_breaths(adapter.preprocess(_payload()))
        assert all(breath.peak_time is not None for breath in breaths)
        assert all(breath.end_time > breath.start_time for breath in breaths)

    def test_a_custom_detector_is_used_when_given(self):
        adapter = _adapter()
        processed = adapter.preprocess(_payload())
        breaths = adapter.detect_breaths(processed, detector=lambda bundle, **kw: [5])
        assert len(breaths) == 1
        assert breaths[0].peak_time == pytest.approx(5 / FS)

    def test_rejects_a_bundle_it_did_not_produce(self):
        with pytest.raises(UnsupportedWorkflowError, match="detector"):
            _adapter().detect_breaths({"not": "a bundle"})


class TestConversionSurfaceMatchesOtherAdapters:
    """Present so `M3Session` can call the same conversions on every adapter."""

    def test_no_parameters_are_produced_by_preprocessing(self):
        adapter = _adapter()
        assert adapter.to_parameters(adapter.preprocess(_payload())) == []

    def test_no_quality_flags_are_produced_by_preprocessing(self):
        adapter = _adapter()
        assert adapter.to_quality_flags(adapter.preprocess(_payload())) == []


class TestLoad:
    def test_uses_an_injected_loader(self):
        payload = _payload()
        assert _adapter(payload).load("v.txt") is payload

    def test_falls_back_to_the_resurfemg_loader(self, monkeypatch):
        # No injected loader: ventilator files share the sEMG's formats, so
        # loading delegates rather than reimplementing readers.
        from m3resp.adapters import resurfemg_adapter

        payload = _payload()
        monkeypatch.setattr(
            resurfemg_adapter.ReSurfEMGAdapter,
            "load",
            lambda self, path, **kwargs: payload,
        )
        assert VentilatorAdapter().load("v.txt") is payload
