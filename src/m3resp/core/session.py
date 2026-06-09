"""The central Stage 1 M3Resp session object."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from m3resp.adapters.eitprocessing_adapter import EITProcessingAdapter
from m3resp.adapters.resurfemg_adapter import ReSurfEMGAdapter
from m3resp.core.exceptions import MissingModalityDataError
from m3resp.core.metadata import SessionMetadata
from m3resp.core.provenance import ProvenanceRecord, record
from m3resp.export.session_export import export_session_summary
from m3resp.modalities.eit import EITRecording, load as load_eit_recording
from m3resp.modalities.emg import EMGRecording, load as load_emg_recording
from m3resp.synchronization.alignment import align_events_manual_offset


class M3Session:
    """Small, explicit session object for Stage 1 multimodal workflows."""

    def __init__(
        self,
        eit_adapter: EITProcessingAdapter | None = None,
        emg_adapter: ReSurfEMGAdapter | None = None,
        metadata: SessionMetadata | dict[str, Any] | None = None,
    ):
        self.eit_adapter = eit_adapter or EITProcessingAdapter()
        self.emg_adapter = emg_adapter or ReSurfEMGAdapter()

        self.eit: EITRecording | None = None
        self.emg: EMGRecording | None = None
        self.raw: dict[str, Any] = {}
        self.processed: dict[str, Any] = {}
        self.events: dict[str, Any] = {}
        self.parameters: dict[str, Any] = {}
        self.quality: dict[str, Any] = {}
        self.metadata = _coerce_metadata(metadata)
        self.provenance: list[ProvenanceRecord] = []

    def load_eit(
        self, path: str | Path, vendor: str | None = None, **kwargs: Any
    ) -> Any:
        """Load EIT data and store it under `raw["eit"]`."""

        recording = load_eit_recording(
            path,
            vendor=vendor,
            adapter=self.eit_adapter,
            **kwargs,
        )
        self.eit = recording
        self.raw["eit"] = recording
        self._record("load_eit", "eit", path=str(path), vendor=vendor)
        return recording.data

    def load_emg(self, path: str | Path, **kwargs: Any) -> Any:
        """Load EMG data and store it under `raw["emg"]`."""

        recording = load_emg_recording(path, adapter=self.emg_adapter, **kwargs)
        self.emg = recording
        self.raw["emg"] = recording
        self._record("load_emg", "emg", path=str(path))
        return recording.data

    def preprocess_eit(self, **kwargs: Any) -> Any:
        """Run a provided or upstream EIT preprocessing function."""

        recording = self._require_raw("eit")
        preprocess = kwargs.pop("preprocess", None)
        if preprocess is None:
            self.processed["eit"] = self.eit_adapter.preprocess(
                recording.data, **kwargs
            )
        else:
            self.processed["eit"] = preprocess(recording.data, **kwargs)
        self._record("preprocess_eit", "eit", **kwargs)
        return self.processed["eit"]

    def preprocess_emg(self, **kwargs: Any) -> Any:
        """Run EMG preprocessing through the adapter."""

        recording = self._require_raw("emg")
        self.processed["emg"] = self.emg_adapter.preprocess(recording.data, **kwargs)
        if self.emg is not None and isinstance(self.processed["emg"], dict):
            self.emg.filtered = self.processed["emg"].get("filtered")
            self.emg.envelope = self.processed["emg"].get("envelope")
            self.emg.channel = self.processed["emg"].get("channel")
            self.emg.fs = self.processed["emg"].get("fs")
        self._record("preprocess_emg", "emg", **kwargs)
        return self.processed["emg"]

    def detect_eit_breaths(self, **kwargs: Any) -> Any:
        """Detect EIT breaths and store normalized events."""

        data = self.processed.get("eit") or self._require_raw("eit").data
        events = self.eit_adapter.detect_breaths(data, **kwargs)
        self.add_events("eit_breaths", events)
        self._record("detect_eit_breaths", "eit", **kwargs)
        return self.events["eit_breaths"]

    def detect_emg_breaths(self, **kwargs: Any) -> Any:
        """Detect EMG breaths and store normalized events."""

        data = self.processed.get("emg") or self._require_raw("emg").data
        events = self.emg_adapter.detect_breaths(data, **kwargs)
        self.add_events("emg_breaths", events)
        self._record("detect_emg_breaths", "emg", **kwargs)
        return self.events["emg_breaths"]

    def add_events(self, name: str, events: Any) -> list[Any]:
        """Store a named event list while keeping `session.events` as backing data."""

        self.events[name] = list(events)
        return self.events[name]

    def get_events(self, name: str, default: Any = None) -> Any:
        """Return a named event list from `session.events`."""

        return self.events.get(name, default)

    def postprocess_emg(self, **kwargs: Any) -> Any:
        """Run EMG postprocessing through the adapter."""

        data = self.processed.get("emg") or self._require_raw("emg").data
        events = self.events.get("emg_breaths")
        self.parameters["emg_postprocessing"] = self.emg_adapter.postprocess(
            data,
            events=events,
            **kwargs,
        )
        self._record("postprocess_emg", "emg", **kwargs)
        return self.parameters["emg_postprocessing"]

    def align_modalities(
        self, method: str = "manual_offset", offset_seconds: float = 0.0
    ) -> dict[str, Any]:
        """Apply basic Stage 1 alignment to stored event lists."""

        if method != "manual_offset":
            raise ValueError("Stage 1 supports only method='manual_offset'")

        synchronized: dict[str, Any] = {}
        for name, events in self.events.items():
            if not isinstance(events, list):
                continue
            synchronized[name] = align_events_manual_offset(events, offset_seconds)

        self.processed["synchronized"] = synchronized
        self._record(
            "align_modalities",
            parameters={"method": method, "offset_seconds": offset_seconds},
        )
        return synchronized

    def export_summary(self, output_dir: str | Path) -> Path:
        """Export the session summary to disk."""

        output_path = export_session_summary(self, output_dir)
        self._record("export_summary", parameters={"output_dir": str(output_path)})
        return output_path

    def _require_raw(self, modality: str) -> Any:
        if modality not in self.raw:
            raise MissingModalityDataError(
                f"No raw {modality.upper()} data loaded. Call load_{modality} first."
            )
        return self.raw[modality]

    def _record(
        self,
        action: str,
        modality: str | None = None,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        record_parameters = parameters or kwargs
        self.provenance.append(record(action, modality, **record_parameters))


def _coerce_metadata(
    metadata: SessionMetadata | dict[str, Any] | None,
) -> SessionMetadata:
    if metadata is None:
        return SessionMetadata()
    if isinstance(metadata, SessionMetadata):
        return metadata
    return SessionMetadata(attributes=dict(metadata))
