"""The central Stage 1 M3Resp session object."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from m3resp.adapters.eitprocessing_adapter import EITProcessingAdapter
from m3resp.adapters.resurfemg_adapter import ReSurfEMGAdapter
from m3resp.core.events import BreathEvent
from m3resp.core.exceptions import MissingModalityDataError, VariantAlreadyExistsError
from m3resp.core.metadata import SessionMetadata
from m3resp.core.provenance import ProvenanceRecord, record
from m3resp.data.collections import (
    ParameterResultCollection,
    QualityReport,
    SignalCollection,
)
from m3resp.data.linked_breath import LinkedBreath
from m3resp.data.parameters import ParameterResult
from m3resp.data.processing import ProcessingHistory
from m3resp.export.session_export import export_session_summary
from m3resp.modalities.eit import EITRecording, load as load_eit_recording
from m3resp.modalities.emg import EMGRecording, load as load_emg_recording
from m3resp.synchronization.alignment import align_events_by_modality_offset
from m3resp.synchronization.cropping import (
    crop_loaded_modality,
    normalize_modality,
    offsets_relative_to_reference,
    raw_synchronization_traces,
    resolve_alignment_offsets,
)
from m3resp.synchronization.linking import link_breaths_by_time
from m3resp.synchronization.multimodal_parameters import compute_multimodal_parameters
from m3resp.synchronization.ventilator import (
    infer_ventilator_fs,
    iter_ventilator_detections,
    normalize_ventilator_breath,
)

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
        allow_overwrite: bool = False,
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
        # Session-wide default for preprocess_eit/preprocess_emg's `overwrite`
        # kwarg, so notebook/exploratory code can set this once instead of
        # passing `overwrite=True` on every call. Left off by default so code
        # copied into reusable/production paths is safe unless it opts in
        # explicitly, per-call, or here.
        self.allow_overwrite = allow_overwrite
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
        # Stage 2 pipeline-structure Phase 5.1: a universal, engine-populated
        # log of every executed workflow step (name/bindings/parameters/
        # timing), independent of whether any step function calls
        # `self._record()` itself. Distinct from `provenance` (the older,
        # session-method-level "action + modality" log) and from the
        # datamodel's per-pipeline `ProcessingRun` (see
        # `m3resp.workflows.engine.run_pipeline` and
        # `DataModelRecorder.record_pipeline_result`).
        self.processing_history = ProcessingHistory()
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

    def preprocess_eit(
        self,
        *,
        variant: str | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run a provided or upstream EIT preprocessing function.

        Every result is stored under `session.processed_variants["eit"][name]`,
        `name` being `variant` if given, otherwise `"default"` - there is no
        implicit, ambiguously-overwritten "current" result. Writing to a name
        that's already populated raises `VariantAlreadyExistsError` unless
        `overwrite=True` is passed (or `session.allow_overwrite = True` is
        set, so notebook/exploratory code can opt in once instead of passing
        `overwrite=True` on every call), so a reference like
        `processed_variants["eit"]["mdn"]` can't silently change meaning
        underneath a caller that stashed it earlier. `session.processed["eit"]`
        mirrors the `"default"` variant only, for convenience/backwards
        compatibility with code that just wants "the" EIT result.

        `preprocess_eit(filter_mode="mdn", variant="mdn")` and
        `preprocess_eit(filter_mode="lowpass", variant="lowpass")` can both
        coexist. See `detect_eit_breaths(variant=...)` to detect breaths
        against a specific variant.
        """

        recording = self._require_raw("eit")
        name = variant if variant is not None else "default"
        if (
            not (overwrite or self.allow_overwrite)
            and name in self.processed_variants["eit"]
        ):
            raise VariantAlreadyExistsError(
                f"EIT preprocessing variant {name!r} already exists; pass "
                "a different `variant=`, or `overwrite=True` to replace it."
            )
        preprocess = kwargs.pop("preprocess", None)
        if preprocess is None:
            result = self.eit_adapter.preprocess(recording.data, **kwargs)
            self._extend_typed_collections_from_eit(result)
        else:
            # A custom `preprocess` callable's output shape is not guaranteed
            # to match what `EITProcessingAdapter.to_signals/to_parameters/
            # to_quality_flags` expect, so the typed collections are only
            # populated on the default adapter path.
            result = preprocess(recording.data, **kwargs)
        self.processed_variants["eit"][name] = result
        if name == "default":
            self.processed["eit"] = result
        self._record("preprocess_eit", "eit", variant=variant, **kwargs)
        return result

    def preprocess_emg(
        self,
        *,
        variant: str | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run EMG preprocessing through the adapter.

        See `preprocess_eit` for what `variant`/`overwrite`/`allow_overwrite`
        do - it persists this result under
        `session.processed_variants["emg"][name]`, raising
        `VariantAlreadyExistsError` if `name` is already populated, and
        mirrors it onto `session.processed["emg"]` only when `name` is
        `"default"`.
        """

        recording = self._require_raw("emg")
        name = variant if variant is not None else "default"
        if (
            not (overwrite or self.allow_overwrite)
            and name in self.processed_variants["emg"]
        ):
            raise VariantAlreadyExistsError(
                f"EMG preprocessing variant {name!r} already exists; pass "
                "a different `variant=`, or `overwrite=True` to replace it."
            )
        result = self.emg_adapter.preprocess(recording.data, **kwargs)
        if self.emg is not None and isinstance(result, dict):
            self.emg.filtered = result.get("filtered")
            self.emg.envelope = result.get("envelope")
            self.emg.channel = result.get("channel")
            self.emg.fs = result.get("fs")
        for signal in self.emg_adapter.to_signals(result):
            self.signals.add(signal)
        self.processed_variants["emg"][name] = result
        if name == "default":
            self.processed["emg"] = result
        self._record("preprocess_emg", "emg", variant=variant, **kwargs)
        return result

    def synchronize_raw_modalities(
        self,
        method: str = "manual_offset",
        offset_seconds: float | Mapping[str, float] = 0.0,
        reference_modality: str | None = None,
    ) -> dict[str, Any]:
        """Crop loaded raw modality signals before downstream processing."""

        if method != "manual_offset":
            raise ValueError("Stage 1 supports only method='manual_offset'")

        configured_offsets = resolve_alignment_offsets(offset_seconds)
        resolved_reference = self._resolve_raw_alignment_reference(reference_modality)
        offsets = offsets_relative_to_reference(configured_offsets, resolved_reference)
        synchronized: dict[str, Any] = {}
        traces: dict[str, Any] = {}
        for modality, offset in offsets.items():
            before_traces = raw_synchronization_traces(self, modality)
            n_samples = crop_loaded_modality(self, modality, float(offset))
            after_traces = raw_synchronization_traces(self, modality)
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

        offsets = resolve_alignment_offsets(offset_seconds)
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

    def compute_multimodal_parameters(
        self,
        *,
        delay_pairs: Sequence[tuple[str, str]] | None = None,
        duration_pairs: Sequence[tuple[str, str]] | None = None,
        anchor: str = "start",
    ) -> list[ParameterResult]:
        """Compute timing-delay/duration-difference/event-agreement
        `ParameterResult`s from `self.linked_breaths` (plan_stage2.md Sec 21).

        Call `link_breaths` first; an empty `self.linked_breaths` yields an
        empty result rather than raising. Results are added to
        `self.parameter_results` and also returned.
        """

        results = compute_multimodal_parameters(
            self.linked_breaths,
            delay_pairs=delay_pairs,
            duration_pairs=duration_pairs,
            anchor=anchor,
        )
        for parameter in results:
            self.parameter_results.add(parameter)
        self._record(
            "compute_multimodal_parameters",
            parameters={"anchor": anchor, "n_linked_breaths": len(self.linked_breaths)},
        )
        return results

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
            return normalize_modality(reference_modality), None
        if self.events.get("ventilator_breaths"):
            return "vent", None
        return "eit", "eit"

    def _resolve_raw_alignment_reference(self, reference_modality: str | None) -> str:
        if reference_modality is not None:
            return normalize_modality(reference_modality)
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

        fs = infer_ventilator_fs(ventilator, ventilator_fs)
        width_seconds = (
            0.0
            if ventilator_breath_width_seconds is None
            else float(ventilator_breath_width_seconds)
        )
        return [
            normalize_ventilator_breath(
                detection,
                fs=fs,
                width_seconds=width_seconds,
            )
            for detection in iter_ventilator_detections(detections)
        ]


def _coerce_metadata(
    metadata: SessionMetadata | dict[str, Any] | None,
) -> SessionMetadata:
    if metadata is None:
        return SessionMetadata()
    if isinstance(metadata, SessionMetadata):
        return metadata
    return SessionMetadata(attributes=dict(metadata))
