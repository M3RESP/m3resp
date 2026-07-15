"""The central Stage 1 M3Resp session object."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from m3resp.adapters.eitprocessing_adapter import EITProcessingAdapter
from m3resp.adapters.resurfemg_adapter import ReSurfEMGAdapter
from m3resp.core.events import BreathEvent, Event, coerce_breath_event
from m3resp.core.exceptions import MissingModalityDataError
from m3resp.core.metadata import SessionMetadata
from m3resp.core.provenance import ProvenanceRecord, record
from m3resp.data.collections import (
    ParameterResultCollection,
    QualityReport,
    SignalCollection,
)
from m3resp.data.linked_breath import LinkedBreath
from m3resp.export.session_export import export_session_summary
from m3resp.modalities.eit import EITRecording, load as load_eit_recording
from m3resp.modalities.emg import EMGRecording, load as load_emg_recording
from m3resp.synchronization.alignment import align_events_by_modality_offset
from m3resp.synchronization.linking import link_breaths_by_time

if TYPE_CHECKING:
    from m3resp.datamodel.recorder import DataModelRecorder

ALIGNMENT_EVENT_LISTS = {
    "eit": "eit_breaths",
    "emg": "emg_breaths",
    "vent": "ventilator_breaths",
}


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
        # Named alternate preprocessing results, e.g. for algorithms that
        # need the same raw recording preprocessed differently (see
        # preprocess_eit`/`preprocess_emg`'s `variant` parameter).
        self.processed_variants: dict[str, dict[str, Any]] = {"eit": {}, "emg": {}}
        self.events: dict[str, Any] = {}
        self.parameters: dict[str, Any] = {}
        # Milestone 2.2 (plan/plan_stage2.md Sec 14): typed collections that
        # let EIT and EMG data live in the same structure, populated from the
        # default preprocess/postprocess paths via each adapter's
        # to_signals/to_parameters/to_quality_flags. These are additive: the
        # `raw`/`processed`/`parameters` dicts above keep their Stage 1 shape
        # and behavior unchanged.
        self.signals = SignalCollection()
        self.parameter_results = ParameterResultCollection()
        self.quality = QualityReport()
        # Milestone 2.5 (plan/plan_stage2.md Sec 20): breaths matched across
        # modalities by `link_breaths`, once per-modality breath events exist.
        self.linked_breaths: list[LinkedBreath] = []
        self.metadata = _coerce_metadata(metadata)
        self.provenance: list[ProvenanceRecord] = []
        # Stage 2 data model wrapper (opt-in, see m3resp.datamodel). ``None``
        # leaves Stage 1 behavior completely unchanged.
        self.datamodel: DataModelRecorder | None = None

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

    def preprocess_eit(self, *, variant: str | None = None, **kwargs: Any) -> Any:
        """Run a provided or upstream EIT preprocessing function.

        By default this overwrites `session.processed["eit"]`, so a session
        only ever keeps one preprocessed EIT result at a time. Pass
        `variant=<name>` to also persist this result under
        `session.processed_variants["eit"][name]`, so different downstream
        algorithms can each get their own differently preprocessed copy of
        the same raw recording (e.g. `preprocess_eit(filter_mode="mdn",
        variant="mdn")` and `preprocess_eit(filter_mode="lowpass",
        variant="lowpass")` can both coexist). See
        `detect_eit_breaths(variant=...)` to detect breaths against a
        specific variant.
        """

        recording = self._require_raw("eit")
        preprocess = kwargs.pop("preprocess", None)
        if preprocess is None:
            self.processed["eit"] = self.eit_adapter.preprocess(
                recording.data, **kwargs
            )
            self._extend_typed_collections_from_eit(self.processed["eit"])
        else:
            # A custom `preprocess` callable's output shape is not guaranteed
            # to match what `EITProcessingAdapter.to_signals/to_parameters/
            # to_quality_flags` expect, so the typed collections are only
            # populated on the default adapter path.
            self.processed["eit"] = preprocess(recording.data, **kwargs)
        if variant is not None:
            self.processed_variants["eit"][variant] = self.processed["eit"]
        self._record("preprocess_eit", "eit", variant=variant, **kwargs)
        return self.processed["eit"]

    def preprocess_emg(self, *, variant: str | None = None, **kwargs: Any) -> Any:
        """Run EMG preprocessing through the adapter.

        See `preprocess_eit` for what `variant` does - it persists this
        result under `session.processed_variants["emg"][variant]` in
        addition to the default `session.processed["emg"]` slot.
        """

        recording = self._require_raw("emg")
        self.processed["emg"] = self.emg_adapter.preprocess(recording.data, **kwargs)
        if self.emg is not None and isinstance(self.processed["emg"], dict):
            self.emg.filtered = self.processed["emg"].get("filtered")
            self.emg.envelope = self.processed["emg"].get("envelope")
            self.emg.channel = self.processed["emg"].get("channel")
            self.emg.fs = self.processed["emg"].get("fs")
        for signal in self.emg_adapter.to_signals(self.processed["emg"]):
            self.signals.add(signal)
        if variant is not None:
            self.processed_variants["emg"][variant] = self.processed["emg"]
        self._record("preprocess_emg", "emg", variant=variant, **kwargs)
        return self.processed["emg"]

    def synchronize_raw_modalities(
        self,
        method: str = "manual_offset",
        offset_seconds: float | Mapping[str, float] = 0.0,
        reference_modality: str | None = None,
    ) -> dict[str, Any]:
        """Crop loaded raw modality signals before downstream processing."""

        if method != "manual_offset":
            raise ValueError("Stage 1 supports only method='manual_offset'")

        configured_offsets = _resolve_alignment_offsets(offset_seconds)
        resolved_reference = self._resolve_raw_alignment_reference(reference_modality)
        offsets = _offsets_relative_to_reference(configured_offsets, resolved_reference)
        synchronized: dict[str, Any] = {}
        traces: dict[str, Any] = {}
        for modality, offset in offsets.items():
            before_traces = _raw_synchronization_traces(self, modality)
            n_samples = _crop_loaded_modality(self, modality, float(offset))
            after_traces = _raw_synchronization_traces(self, modality)
            for trace_name, before_trace in before_traces.items():
                after_trace = after_traces.get(trace_name)
                if after_trace is not None:
                    traces[trace_name] = {
                        "before": before_trace,
                        "after": after_trace,
                        "offset_seconds": float(offset),
                    }
            if n_samples:
                synchronized[modality] = {
                    "offset_seconds": float(offset),
                    "cropped_samples": n_samples,
                }

        self.processed["raw_synchronization"] = traces
        self.parameters["raw_alignment"] = {
            "method": method,
            "reference_modality": resolved_reference,
            "requested_reference_modality": reference_modality,
            "offset_seconds": offsets,
            "configured_offset_seconds": configured_offsets,
            "synchronized_modalities": sorted(synchronized),
            "cropped_samples": synchronized,
        }
        self._record(
            "synchronize_raw_modalities",
            parameters={
                "method": method,
                "reference_modality": resolved_reference,
                "offset_seconds": offsets,
                "configured_offset_seconds": configured_offsets,
                "cropped_samples": synchronized,
            },
        )
        return synchronized

    def detect_eit_breaths(self, *, variant: str | None = None, **kwargs: Any) -> Any:
        """Detect EIT breaths and store normalized events.

        Pass `variant=<name>` to detect breaths against a
        `preprocess_eit(..., variant=<name>)` result instead of the default
        `processed["eit"]`; the events are then stored under
        `session.events["eit_breaths:<name>"]` instead of
        `session.events["eit_breaths"]`, so multiple variants' detections
        can coexist.
        """

        if variant is not None:
            data = self.processed_variants["eit"].get(variant)
            if data is None:
                raise MissingModalityDataError(
                    f"No EIT preprocessing variant {variant!r}; call "
                    f"preprocess_eit(variant={variant!r}, ...) first."
                )
            event_key = f"eit_breaths:{variant}"
        else:
            data = self.processed.get("eit") or self._require_raw("eit").data
            event_key = "eit_breaths"
        events = self.eit_adapter.detect_breaths(data, **kwargs)
        self.add_events(event_key, events)
        self._record("detect_eit_breaths", "eit", variant=variant, **kwargs)
        return self.events[event_key]

    def detect_emg_breaths(self, *, variant: str | None = None, **kwargs: Any) -> Any:
        """Detect EMG breaths and store normalized events.

        See `detect_eit_breaths` for what `variant` does.
        """

        if variant is not None:
            data = self.processed_variants["emg"].get(variant)
            if data is None:
                raise MissingModalityDataError(
                    f"No EMG preprocessing variant {variant!r}; call "
                    f"preprocess_emg(variant={variant!r}, ...) first."
                )
            event_key = f"emg_breaths:{variant}"
        else:
            data = self.processed.get("emg") or self._require_raw("emg").data
            event_key = "emg_breaths"
        events = self.emg_adapter.detect_breaths(data, **kwargs)
        self.add_events(event_key, events)
        self._record("detect_emg_breaths", "emg", variant=variant, **kwargs)
        return self.events[event_key]

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
        for parameter in self.emg_adapter.to_parameters(
            self.parameters["emg_postprocessing"]
        ):
            self.parameter_results.add(parameter)
        for flag in self.emg_adapter.to_quality_flags(
            self.parameters["emg_postprocessing"]
        ):
            self.quality.add(flag)
        ventilator_breaths = self._normalize_ventilator_breaths(
            self.parameters["emg_postprocessing"],
            ventilator=kwargs.get("ventilator"),
            ventilator_fs=kwargs.get("ventilator_fs"),
            ventilator_breath_width_seconds=kwargs.get(
                "ventilator_breath_width_seconds",
            ),
        )
        if ventilator_breaths:
            self.add_events("ventilator_breaths", ventilator_breaths)
        self._record("postprocess_emg", "emg", **kwargs)
        return self.parameters["emg_postprocessing"]

    def align_modalities(
        self,
        method: str = "manual_offset",
        offset_seconds: float | Mapping[str, float] = 0.0,
        reference_modality: str | None = None,
    ) -> dict[str, Any]:
        """Apply basic Stage 1 alignment to stored event lists."""

        if method != "manual_offset":
            raise ValueError("Stage 1 supports only method='manual_offset'")

        offsets = _resolve_alignment_offsets(offset_seconds)
        requested_reference = reference_modality
        resolved_reference, fallback_reference = self._resolve_alignment_reference(
            reference_modality
        )
        synchronized: dict[str, Any] = {}
        aligned_event_lists: list[str] = []
        missing_event_lists: list[str] = []
        for name in ALIGNMENT_EVENT_LISTS.values():
            events = self.events.get(name)
            if events is None:
                missing_event_lists.append(name)
                continue
            if not isinstance(events, list):
                continue
            synchronized[name] = align_events_by_modality_offset(events, offsets)
            aligned_event_lists.append(name)

        self.processed["synchronized"] = synchronized
        self.parameters["alignment"] = {
            "method": method,
            "reference_modality": resolved_reference,
            "requested_reference_modality": requested_reference,
            "fallback_reference_modality": fallback_reference,
            "offset_seconds": offsets,
            "aligned_event_lists": aligned_event_lists,
            "missing_event_lists": missing_event_lists,
        }
        self._record(
            "align_modalities",
            parameters={
                "method": method,
                "reference_modality": resolved_reference,
                "offset_seconds": offsets,
            },
        )
        return synchronized

    def link_breaths(self, *, time_tolerance: float = 0.5) -> list[LinkedBreath]:
        """Link breaths across modalities into `LinkedBreath` objects (Milestone 2.5).

        Prefers the aligned breath lists produced by `align_modalities`
        (``self.processed["synchronized"]``) over the raw per-modality event
        lists in ``self.events``, so breaths are matched on a common time
        axis whenever alignment has already been run.
        """

        synchronized = self.processed.get("synchronized")
        if not isinstance(synchronized, dict):
            synchronized = {}

        def _breaths(name: str) -> list[BreathEvent] | None:
            events = synchronized.get(name)
            if events is None:
                events = self.events.get(name)
            return events

        self.linked_breaths = link_breaths_by_time(
            {
                "eit": _breaths("eit_breaths"),
                "emg": _breaths("emg_breaths"),
                "ventilator": _breaths("ventilator_breaths"),
            },
            time_tolerance=time_tolerance,
        )
        self._record("link_breaths", parameters={"time_tolerance": time_tolerance})
        return self.linked_breaths

    def run_pipeline(
        self, name: str, *, config: Mapping[str, Mapping[str, Any]] | None = None
    ) -> M3Session:
        """Run a named, built-in `Pipeline` preset against this session.

        This is a different mechanism from the module-level
        ``m3resp.run_pipeline(spec, session=...)``, which executes a fully
        custom declarative step-list spec (the Stage 1 pipeline engine in
        ``m3resp.workflows``). ``session.run_pipeline(name)`` instead runs one
        of the small, built-in presets registered in ``m3resp.presets``
        (``"eit"``, ``"emg"``, ``"multimodal"``), which simply call this
        session's own already-instrumented methods in sequence - see
        ``m3resp.presets.base`` for the rationale.
        """

        from m3resp.presets import get_pipeline

        pipeline_cls = get_pipeline(name)
        return pipeline_cls().run(self, config=config)

    def export_summary(
        self, output_dir: str | Path, *, processing_run_id: str | None = None
    ) -> Path:
        """Export the session summary to disk.

        ``processing_run_id`` (typically `PipelineResult.processing_run_id`)
        links a written parameter-array archive to the `ProcessingRun` that
        produced it when a `DataModelRecorder` is attached; omit it for a
        manual export with no associated pipeline run.
        """

        output_path = export_session_summary(
            self, output_dir, processing_run_id=processing_run_id
        )
        self._record("export_summary", parameters={"output_dir": str(output_path)})
        return output_path

    def _require_raw(self, modality: str) -> Any:
        if modality not in self.raw:
            raise MissingModalityDataError(
                f"No raw {modality.upper()} data loaded. Call load_{modality} first."
            )
        return self.raw[modality]

    def _extend_typed_collections_from_eit(self, preprocessed: dict[str, Any]) -> None:
        for signal in self.eit_adapter.to_signals(preprocessed):
            self.signals.add(signal)
        for parameter in self.eit_adapter.to_parameters(preprocessed):
            self.parameter_results.add(parameter)
        for flag in self.eit_adapter.to_quality_flags(preprocessed):
            self.quality.add(flag)

    def _record(
        self,
        action: str,
        modality: str | None = None,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        record_parameters = parameters or kwargs
        provenance_record = record(action, modality, **record_parameters)
        self.provenance.append(provenance_record)
        if self.datamodel is not None:
            self.datamodel.record_provenance(provenance_record)

    def _resolve_alignment_reference(
        self,
        reference_modality: str | None,
    ) -> tuple[str, str | None]:
        if reference_modality is not None:
            return _normalize_modality(reference_modality), None
        if self.events.get("ventilator_breaths"):
            return "vent", None
        return "eit", "eit"

    def _resolve_raw_alignment_reference(self, reference_modality: str | None) -> str:
        if reference_modality is not None:
            return _normalize_modality(reference_modality)
        if "vent" in self.raw:
            return "vent"
        if "eit" in self.raw:
            return "eit"
        if "emg" in self.raw:
            return "emg"
        return "eit"

    def _normalize_ventilator_breaths(
        self,
        postprocessing: Any,
        *,
        ventilator: Any | None,
        ventilator_fs: float | None,
        ventilator_breath_width_seconds: float | None,
    ) -> list[BreathEvent]:
        detections = (
            postprocessing.get("computed", {})
            .get("event_detection", {})
            .get("detect_ventilator_breath", [])
            if isinstance(postprocessing, dict)
            else []
        )
        if detections is None:
            return []

        fs = _infer_ventilator_fs(ventilator, ventilator_fs)
        width_seconds = (
            0.0
            if ventilator_breath_width_seconds is None
            else float(ventilator_breath_width_seconds)
        )
        return [
            _normalize_ventilator_breath(
                detection,
                fs=fs,
                width_seconds=width_seconds,
            )
            for detection in _iter_ventilator_detections(detections)
        ]


def _coerce_metadata(
    metadata: SessionMetadata | dict[str, Any] | None,
) -> SessionMetadata:
    if metadata is None:
        return SessionMetadata()
    if isinstance(metadata, SessionMetadata):
        return metadata
    return SessionMetadata(attributes=dict(metadata))


def _resolve_alignment_offsets(
    offset_seconds: float | Mapping[str, float],
) -> dict[str, float]:
    if isinstance(offset_seconds, Mapping):
        offsets = {"eit": 0.0, "emg": 0.0, "vent": 0.0}
        for modality, offset in offset_seconds.items():
            offsets[_normalize_modality(modality)] = float(offset)
        return offsets
    return {"eit": 0.0, "emg": float(offset_seconds), "vent": 0.0}


def _offsets_relative_to_reference(
    offsets: Mapping[str, float],
    reference_modality: str,
) -> dict[str, float]:
    reference = _normalize_modality(reference_modality)
    reference_offset = float(offsets.get(reference, 0.0))
    return {
        _normalize_modality(modality): float(offset) - reference_offset
        for modality, offset in offsets.items()
    }


def _normalize_modality(modality: str) -> str:
    normalized = str(modality).lower()
    if normalized in {"ventilator", "ventilation"}:
        return "vent"
    return normalized


def _crop_loaded_modality(session: M3Session, modality: str, offset: float) -> int:
    if offset == 0.0:
        return 0
    if modality == "emg" and session.emg is not None:
        return _crop_emg_recording(session.emg, offset)
    if modality == "vent":
        ventilator = session.raw.get("vent")
        return _crop_recording_dict(ventilator, offset)
    if modality == "eit" and session.eit is not None:
        return _crop_eit_recording(session.eit, offset)
    return 0


def _raw_synchronization_traces(session: M3Session, modality: str) -> dict[str, Any]:
    if modality == "emg" and session.emg is not None:
        trace = _emg_raw_trace(session.emg)
        return {"emg": trace} if trace is not None else {}
    if modality == "eit" and session.eit is not None:
        trace = _eit_raw_trace(session.eit)
        return {"eit": trace} if trace is not None else {}
    if modality == "vent":
        return _ventilator_raw_traces(session.raw.get("vent"))
    return {}


def _emg_raw_trace(recording: EMGRecording) -> dict[str, Any] | None:
    data = recording.data if isinstance(recording.data, dict) else None
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    labels = metadata.get("labels") or []
    units = metadata.get("units") or []
    label = labels[0] if labels else "emg_0"
    unit = units[0] if units else "a.u."
    return _recording_dict_trace(
        data,
        title=f"EMG raw ({label})",
        ylabel=f"EMG amplitude ({unit})" if unit else "EMG amplitude",
    )


def _recording_dict_trace(
    recording: Any,
    *,
    title: str,
    ylabel: str,
    channel: int = 0,
) -> dict[str, Any] | None:
    if not isinstance(recording, dict) or "array" not in recording:
        return None
    metadata = recording.get("metadata") or {}
    fs = metadata.get("fs")
    if fs is None:
        return None
    array = np.asarray(recording["array"], dtype=float)
    if array.ndim == 0 or array.size == 0:
        return None
    if array.ndim > 1 and channel >= array.shape[0]:
        return None
    values = array[channel] if array.ndim > 1 else array
    time = np.arange(len(values), dtype=float) / float(fs)
    return {
        "title": title,
        "time": time.tolist(),
        "values": np.asarray(values, dtype=float).tolist(),
        "ylabel": ylabel,
    }


def _ventilator_raw_traces(recording: Any) -> dict[str, Any]:
    pressure = _recording_dict_trace(
        recording,
        title="Ventilator pressure",
        ylabel="Pressure",
        channel=0,
    )
    flow = _recording_dict_trace(
        recording,
        title="Ventilator flow",
        ylabel="Flow",
        channel=1,
    )
    volume = _recording_dict_trace(
        recording,
        title="Ventilator volume",
        ylabel="Volume",
        channel=2,
    )
    traces: dict[str, Any] = {}
    if pressure is not None:
        traces["vent_pressure"] = pressure
    if flow is not None:
        traces["vent_flow"] = flow
    if volume is not None:
        traces["vent_volume"] = volume
    return traces


def _eit_raw_trace(recording: EITRecording) -> dict[str, Any] | None:
    signal = recording.global_impedance
    if signal is None:
        return None
    time = getattr(signal, "time", None)
    values = getattr(signal, "values", None)
    if time is None or values is None:
        return None
    return {
        "title": "EIT raw global impedance",
        "time": np.asarray(time, dtype=float).tolist(),
        "values": np.asarray(values, dtype=float).tolist(),
        "ylabel": getattr(signal, "label", "global_impedance_(raw)"),
    }


def _crop_emg_recording(recording: EMGRecording, offset: float) -> int:
    cropped = _crop_recording_dict(recording.data, offset)
    if not cropped:
        return 0
    if isinstance(recording.data, dict):
        recording.raw = recording.data.get("array")
        recording.metadata = recording.data.get("metadata")
    return cropped


def _crop_recording_dict(recording: Any, offset: float) -> int:
    if not isinstance(recording, dict) or "array" not in recording:
        return 0
    metadata = recording.get("metadata") or {}
    fs = metadata.get("fs")
    if fs is None:
        return 0
    array, n_samples = _crop_sample_array(recording["array"], float(fs), offset)
    if not n_samples:
        return 0
    recording["array"] = array
    return n_samples


def _crop_eit_recording(recording: EITRecording, offset: float) -> int:
    sequence = recording.data
    fs = _infer_eit_sample_frequency(recording)
    if fs is None:
        return 0

    n_samples = 0
    if recording.raw is not None:
        n_samples = max(n_samples, _crop_eit_like_object(recording.raw, fs, offset))
    if recording.global_impedance is not None:
        n_samples = max(
            n_samples,
            _crop_eit_like_object(recording.global_impedance, fs, offset),
        )
    for collection_name in ("eit_data", "continuous_data"):
        collection = getattr(sequence, collection_name, None)
        if isinstance(collection, Mapping):
            for item in collection.values():
                n_samples = max(n_samples, _crop_eit_like_object(item, fs, offset))
    return n_samples


def _crop_eit_like_object(obj: Any, fs: float, offset: float) -> int:
    n_samples = 0
    for attr in ("pixel_impedance", "values", "time"):
        if not hasattr(obj, attr):
            continue
        values = getattr(obj, attr)
        cropped, cropped_samples = _crop_sample_array(
            values,
            fs,
            offset,
            sample_axis=0,
        )
        if cropped_samples:
            setattr(obj, attr, cropped)
            n_samples = max(n_samples, cropped_samples)
    return n_samples


def _crop_sample_array(
    values: Any,
    fs: float,
    offset: float,
    sample_axis: int | None = None,
) -> tuple[Any, int]:
    array = np.asarray(values)
    if array.size == 0:
        return values, 0

    if sample_axis is None:
        sample_axis = 1 if array.ndim > 1 and array.shape[1] >= array.shape[0] else 0
    n_samples = int(round(abs(offset) * float(fs)))
    axis_len = array.shape[sample_axis]
    if n_samples <= 0 or n_samples >= axis_len:
        return values, 0

    if offset < 0:
        sample_slice = slice(n_samples, None)
    else:
        sample_slice = slice(None, axis_len - n_samples)
    slices = [slice(None)] * array.ndim
    slices[sample_axis] = sample_slice
    cropped = array[tuple(slices)]
    if isinstance(values, list):
        return cropped.tolist(), n_samples
    return cropped, n_samples


def _infer_eit_sample_frequency(recording: EITRecording) -> float | None:
    for obj in (recording.raw, recording.global_impedance, recording.data):
        for attr in ("sample_frequency", "sample_frequency_hz", "fs"):
            value = getattr(obj, attr, None)
            if value is not None:
                return float(value)
        metadata = getattr(obj, "metadata", None)
        if isinstance(metadata, Mapping):
            for key in ("sample_frequency", "sample_frequency_hz", "fs"):
                value = metadata.get(key)
                if value is not None:
                    return float(value)
    return None


def _iter_ventilator_detections(detections: Any) -> list[Any]:
    if isinstance(detections, (BreathEvent, Event, Mapping)):
        return [detections]
    if hasattr(detections, "tolist"):
        detections = detections.tolist()
    return list(detections)


def _normalize_ventilator_breath(
    detection: Any,
    *,
    fs: float | None,
    width_seconds: float,
) -> BreathEvent:
    if isinstance(detection, BreathEvent):
        return replace(detection, modality="vent")
    if isinstance(detection, Mapping):
        breath = coerce_breath_event(detection, modality="vent", source="ventilator")
        return replace(breath, modality="vent")

    if hasattr(detection, "start_time") and hasattr(detection, "end_time"):
        breath = coerce_breath_event(detection, modality="vent", source="ventilator")
        return replace(breath, modality="vent")

    if fs is None:
        raise ValueError(
            "Ventilator breath indices require a ventilator sampling rate. "
            "Pass ventilator_fs or include metadata['fs'] in the ventilator input."
        )

    sample_index = int(detection)
    peak_time = sample_index / float(fs)
    half_width = width_seconds / 2
    return BreathEvent(
        modality="vent",
        start_time=max(0.0, peak_time - half_width),
        end_time=peak_time + half_width,
        peak_time=peak_time,
        source="resurfemg.detect_ventilator_breath",
        metadata={
            "sample_index": sample_index,
            "fs": float(fs),
            "width_seconds": width_seconds,
        },
    )


def _infer_ventilator_fs(
    ventilator: Any | None,
    ventilator_fs: float | None,
) -> float | None:
    if ventilator_fs is not None:
        return float(ventilator_fs)
    if isinstance(ventilator, Mapping):
        metadata = ventilator.get("metadata", {})
        if isinstance(metadata, Mapping) and metadata.get("fs") is not None:
            return float(metadata["fs"])
    return None
