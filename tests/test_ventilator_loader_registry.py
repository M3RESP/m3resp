"""Registering a reader for a third-party ventilator/EIT file format.

Loading previously only knew two sources, dispatched by file suffix: the EIT
`*.bin` and the sEMG multi-channel export. A file in some other vendor's
format - one this codebase has never heard of - needed a caller to hand-write
`loader=`/`eit_loader=` on every construction that reads it. This registry
lets a reader be declared once, by extension, so a new format becomes ordinary
dispatch rather than something every caller has to remember to inject.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.adapters.ventilator_adapter import (
    VentilatorAdapter,
    register_ventilator_loader,
    reset_ventilator_loaders,
    resolve_ventilator_source,
    unregister_ventilator_loader,
    ventilator_loaders,
)

FS = 100.0
N = 300


@pytest.fixture(autouse=True)
def _restore_registry():
    yield
    reset_ventilator_loaders()


def _wave(scale: float = 1.0) -> np.ndarray:
    return scale * np.sin(2 * np.pi * 0.25 * np.arange(N) / FS)


def _payload() -> dict:
    return {
        "array": np.vstack([_wave(10.0), _wave(5.0), _wave(1.0)]),
        "metadata": {"fs": FS, "labels": ["Paw", "Flow", "Volume"]},
    }


class TestRegistration:
    def test_an_extension_can_be_registered_with_or_without_a_leading_dot(self):
        register_ventilator_loader("mfx", lambda path, **kwargs: _payload())
        register_ventilator_loader(".mfy", lambda path, **kwargs: _payload())
        assert ventilator_loaders() == {".mfx": "ventilator", ".mfy": "ventilator"}

    def test_matching_is_case_insensitive(self):
        register_ventilator_loader(".MFX", lambda path, **kwargs: _payload())
        assert resolve_ventilator_source("recording.MfX") == "ventilator"

    def test_defaults_to_the_standalone_ventilator_clock(self):
        register_ventilator_loader("mfx", lambda path, **kwargs: _payload())
        assert resolve_ventilator_source("recording.mfx") == "ventilator"

    def test_an_invalid_source_is_rejected(self):
        with pytest.raises(ValueError, match="'eit'"):
            register_ventilator_loader(
                "mfx", lambda path, **kwargs: _payload(), source="impedance"
            )

    def test_unregistering_removes_it(self):
        register_ventilator_loader("mfx", lambda path, **kwargs: _payload())
        unregister_ventilator_loader("mfx")
        assert ventilator_loaders() == {}

    def test_reset_clears_every_registration(self):
        register_ventilator_loader("mfx", lambda path, **kwargs: _payload())
        register_ventilator_loader("mfy", lambda path, **kwargs: _payload())
        reset_ventilator_loaders()
        assert ventilator_loaders() == {}


class TestDispatch:
    def test_a_registered_extension_is_used_automatically(self):
        payload = _payload()
        register_ventilator_loader("mfx", lambda path, **kwargs: payload)
        adapter = VentilatorAdapter()
        assert adapter.load("recording.mfx") is payload

    def test_the_loader_receives_the_path(self):
        seen = {}

        def reader(path, **kwargs):
            seen["path"] = path
            return _payload()

        register_ventilator_loader("mfx", reader)
        VentilatorAdapter().load("study.mfx")
        assert seen["path"] == "study.mfx"

    def test_an_unregistered_extension_still_falls_back_to_the_semg_path(self):
        register_ventilator_loader("mfx", lambda path, **kwargs: _payload())
        called = {}
        adapter = VentilatorAdapter(
            loader=lambda path, **kwargs: called.setdefault("used", True) or _payload()
        )
        adapter.load("recording.txt")
        assert called == {"used": True}

    def test_an_explicit_source_bypasses_the_registry(self):
        # A registered loader for `.mfx`, but the caller insists this
        # particular file is really the sEMG-style path - the injected
        # `loader=` should win, not the registration.
        register_ventilator_loader("mfx", lambda path, **kwargs: _payload())
        semg_payload = _payload()
        adapter = VentilatorAdapter(loader=lambda path, **kwargs: semg_payload)
        assert adapter.load("recording.mfx", source="emg") is semg_payload

    def test_the_loaded_payload_preprocesses_like_any_other_recording(self):
        register_ventilator_loader("mfx", lambda path, **kwargs: _payload())
        adapter = VentilatorAdapter()
        recording = adapter.load("study.mfx")
        processed = adapter.preprocess(recording)
        assert processed["pressure"].shape == (N,)

    def test_a_bin_suffix_can_be_overridden_by_a_registered_extension(self):
        # A registered extension takes priority over the built-in suffix
        # dispatch, even one that would otherwise mean "read as EIT".
        payload = _payload()
        register_ventilator_loader("bin", lambda path, **kwargs: payload, source="emg")
        adapter = VentilatorAdapter()
        assert adapter.load("recording.bin") is payload


class TestEitSourcedRegistration:
    def _sequence(self):
        class _ContinuousData:
            def __init__(self, label, values, unit):
                self.label = label
                self.name = label
                self.values = np.asarray(values, dtype=float)
                self.unit = unit
                self.sample_frequency = FS
                self.time = np.arange(len(self.values)) / FS

        class _Sequence:
            def __init__(self):
                self.continuous_data = {
                    "airway pressure": _ContinuousData(
                        "airway pressure", _wave(10.0), "mbar"
                    ),
                    "flow": _ContinuousData("flow", _wave(5.0), "L/min"),
                    "volume": _ContinuousData("volume", _wave(500.0), "mL"),
                }

        return _Sequence()

    def test_an_eit_sourced_registration_is_unpacked_like_the_builtin_eit_path(self):
        sequence = self._sequence()
        register_ventilator_loader("mfz", lambda path, **kwargs: sequence, source="eit")
        payload = VentilatorAdapter().load("study.mfz")
        assert payload["metadata"]["source"] == "eit"
        assert payload["array"].shape == (3, N)

    def test_the_reader_does_not_receive_channel_or_fs_kwargs(self):
        sequence = self._sequence()
        seen_kwargs = {}

        def reader(path, **kwargs):
            seen_kwargs.update(kwargs)
            return sequence

        register_ventilator_loader("mfz", reader, source="eit")
        VentilatorAdapter().load(
            "study.mfz", ventilator_channels=("pressure",), fs=50.0
        )
        assert "ventilator_channels" not in seen_kwargs
        assert "fs" not in seen_kwargs

    def test_ventilator_channels_still_selects_which_channels_are_read(self):
        sequence = self._sequence()
        register_ventilator_loader("mfz", lambda path, **kwargs: sequence, source="eit")
        payload = VentilatorAdapter().load(
            "study.mfz", ventilator_channels=("pressure", "flow")
        )
        assert payload["array"].shape == (2, N)

    def test_the_reported_clock_is_eit(self):
        sequence = self._sequence()
        register_ventilator_loader("mfz", lambda path, **kwargs: sequence, source="eit")
        assert resolve_ventilator_source("study.mfz") == "eit"
