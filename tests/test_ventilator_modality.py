"""`VentilatorRecording` and `M3Session.load_ventilator`, the ventilator
counterparts of the EIT/EMG loading path.

Ventilator data used to reach a session only as a keyword argument to
`postprocess_emg` or via the `emg.load_ventilator` pipeline step, which stashed
a bare payload dict in `session.raw["vent"]`. It is now loaded like any other
modality, and `session.raw["ventilator"]` holds a recording object exactly as
`raw["eit"]`/`raw["emg"]` do.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from m3resp.adapters import ReSurfEMGAdapter
from m3resp.core.session import M3Session
from m3resp.modalities.ventilator import VentilatorRecording
from m3resp.synchronization.cropping import ventilator_payload, ventilator_raw


def _payload(n_samples: int = 100, fs: float = 10.0) -> dict:
    return {
        "array": np.arange(3 * n_samples, dtype=float).reshape(3, n_samples),
        "metadata": {"fs": fs, "labels": ["pressure", "flow", "volume"]},
    }


def _session(payload: dict | None = None) -> M3Session:
    data = payload if payload is not None else _payload()
    return M3Session(
        emg_adapter=ReSurfEMGAdapter(loader=lambda path, **kwargs: data),
    )


class TestLoadVentilator:
    def test_returns_the_loader_payload(self):
        payload = _payload()
        assert _session(payload).load_ventilator("vent.txt") is payload

    def test_stores_a_recording_on_the_session(self):
        session = _session()
        session.load_ventilator("subject.txt")

        assert isinstance(session.ventilator, VentilatorRecording)
        assert session.ventilator.path == Path("subject.txt")

    def test_unpacks_metadata_sample_rate_and_array(self):
        session = _session()
        session.load_ventilator("subject.txt")

        assert session.ventilator.fs == 10.0
        assert session.ventilator.metadata["labels"] == ["pressure", "flow", "volume"]
        assert session.ventilator.raw.shape == (3, 100)

    def test_raw_holds_the_recording_under_both_keys(self):
        session = _session()
        session.load_ventilator("subject.txt")

        assert session.raw["ventilator"] is session.ventilator
        assert session.raw["vent"] is session.ventilator

    def test_records_provenance_like_the_other_loaders(self):
        session = _session()
        session.load_ventilator("subject.txt")

        entry = session.provenance[-1]
        assert entry.action == "load_ventilator"
        assert entry.modality == "ventilator"
        assert entry.parameters["path"] == "subject.txt"

    def test_sits_alongside_eit_and_emg_in_provenance(self):
        from m3resp.adapters import EITProcessingAdapter

        session = M3Session(
            eit_adapter=EITProcessingAdapter(
                loader=lambda path, vendor=None, **kwargs: {"path": path}
            ),
            emg_adapter=ReSurfEMGAdapter(loader=lambda path, **kwargs: _payload()),
        )
        session.load_eit("subject.eit")
        session.load_emg("subject.edf")
        session.load_ventilator("subject.txt")

        assert [entry.action for entry in session.provenance] == [
            "load_eit",
            "load_emg",
            "load_ventilator",
        ]


class TestAdapterInjection:
    def test_defaults_to_a_dedicated_ventilator_adapter(self):
        from m3resp.adapters import VentilatorAdapter

        assert isinstance(_session().ventilator_adapter, VentilatorAdapter)

    def test_loading_still_flows_through_the_emg_adapter(self):
        # Ventilator channels usually arrive in the same multi-channel file as
        # the sEMG, so injecting one EMG loader must cover both without a
        # second injection - even though processing is now native.
        payload = _payload()
        session = _session(payload)
        assert session.load_ventilator("shared.txt") is payload

    def test_a_dedicated_adapter_can_be_injected(self):
        ventilator_payload_dict = _payload()
        session = M3Session(
            emg_adapter=ReSurfEMGAdapter(loader=lambda path, **kwargs: {"emg": True}),
            ventilator_adapter=ReSurfEMGAdapter(
                loader=lambda path, **kwargs: ventilator_payload_dict
            ),
        )
        session.load_ventilator("vent.txt")

        assert session.ventilator_adapter is not session.emg_adapter
        assert session.ventilator.data is ventilator_payload_dict


class TestPayloadUnwrapping:
    def test_unwraps_a_recording(self):
        session = _session()
        session.load_ventilator("subject.txt")
        assert ventilator_payload(ventilator_raw(session)) is session.ventilator.data

    def test_accepts_a_bare_legacy_dict(self):
        # Stage 1 stored the payload directly under `raw["vent"]`.
        session = _session()
        payload = _payload()
        session.raw["vent"] = payload
        assert ventilator_payload(ventilator_raw(session)) is payload

    @pytest.mark.parametrize("value", [None, {}, {"no_array": 1}, object()])
    def test_returns_none_for_anything_without_a_payload(self, value):
        assert ventilator_payload(value) is None


class TestCroppingALoadedRecording:
    def test_crops_the_payload_in_place(self):
        session = _session()
        session.load_ventilator("subject.txt")

        session.synchronize_raw_modalities(
            offset_seconds={"ventilator": 1.0}, reference_modality="eit"
        )

        assert session.ventilator.data["array"].shape[1] == 90

    def test_refreshes_the_recordings_convenience_fields(self):
        # Mirrors `_crop_emg_recording`: `.raw` must not keep pointing at the
        # pre-crop array after the payload is cropped.
        session = _session()
        session.load_ventilator("subject.txt")

        session.synchronize_raw_modalities(
            offset_seconds={"ventilator": 1.0}, reference_modality="eit"
        )

        assert session.ventilator.raw is session.ventilator.data["array"]
        assert session.ventilator.raw.shape[1] == 90

    def test_both_raw_keys_observe_the_crop(self):
        session = _session()
        session.load_ventilator("subject.txt")

        session.synchronize_raw_modalities(
            offset_seconds={"vent": 1.0}, reference_modality="eit"
        )

        assert session.raw["vent"].data["array"].shape[1] == 90
        assert session.raw["ventilator"] is session.raw["vent"]

    def test_a_legacy_bare_dict_is_still_cropped(self):
        session = _session()
        payload = _payload()
        session.raw["vent"] = payload

        session.synchronize_raw_modalities(
            offset_seconds={"ventilator": 1.0}, reference_modality="eit"
        )

        assert payload["array"].shape[1] == 90


class TestPipelineStepDelegates:
    def test_load_ventilator_step_populates_the_session_recording(self):
        import m3resp.workflows.steps  # noqa: F401 - registers built-in steps
        from m3resp.workflows.registry import get_step

        session = _session()
        result = get_step("ventilator.load").func(session=session, file="vent.txt")

        # The step still emits the raw payload dict its downstream consumer
        # (`emg.ventilator_channels`) expects...
        assert result["ventilator_raw"] is session.ventilator.data
        # ...while the session now also carries the typed recording.
        assert isinstance(session.ventilator, VentilatorRecording)
        assert session.provenance[-1].action == "load_ventilator"
