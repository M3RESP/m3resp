"""Wraps an ``M3Session`` / pipeline run, turning Stage 1 activity into
data model entities in a ``DataModelStore``.

This is the "merge" point between Stage 1 (``M3Session``, adapters, the
pipeline engine) and the Stage 2 data model: it does not change what Stage 1
does, it only observes the two seams that already exist and existed before
this module:

- ``M3Session._record`` (``core/session.py``) is the single provenance choke
  point every session method already calls through. Attaching a recorder adds
  one call from ``_record`` into ``record_provenance`` here; no other method
  on ``M3Session`` changes.
- ``run_pipeline`` (``pipeline/engine.py``) calls ``record_pipeline_result``
  once, after a run finishes, turning named context artifacts into store
  entities.

Attaching a recorder is opt-in (``session.datamodel = DataModelRecorder(...)``)
so Stage 1 behavior is unchanged for sessions that never attach one.

Per ``plan/stage2_consolidation.md``, this recorder prefers Layer 1 runtime
objects (``m3resp.data.Signal``/``ParameterResult``/``QualityFlag``/
``ProcessingStep``) when adapters or pipeline steps already produce them
(Milestone 2.3), and falls back to inferring from ``ProvenanceRecord``/raw
dicts for anything not yet migrated (Milestone 2.1/1). Both paths converge on
the same store rows, keyed by modality, so it does not matter which path ran
first.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from m3resp.data.parameters import ParameterResult
from m3resp.data.processing import ProcessingStep
from m3resp.data.quality import QualityFlag
from m3resp.data.signals import Signal
from m3resp.datamodel.entities import (
    Case,
    DataFile,
    Device,
    DeviceType,
    DerivedFeature,
    FileFormat,
    FileRole,
    ProcessingRun,
    QualityAnnotation,
    RecordingSession,
    SignalStream,
    SignalType,
    TargetType,
)
from m3resp.datamodel.store import DataModelStore

if TYPE_CHECKING:
    from m3resp.core.provenance import ProvenanceRecord
    from m3resp.core.session import M3Session
    from m3resp.workflows.engine import PipelineResult

#: Provenance actions that correspond to loading a modality's raw data.
_LOAD_ACTIONS = {"load_eit": "eit", "load_emg": "emg"}

#: Maps both Stage 1 session modality keys ("vent") and Layer 1
#: ``Signal.modality`` values ("ventilator", "pressure", "flow") onto the
#: persisted ``Device.device_type`` vocabulary.
_MODALITY_DEVICE_TYPE: dict[str, DeviceType] = {
    "eit": "eit",
    "emg": "emg",
    "vent": "ventilator",
    "ventilator": "ventilator",
    "pressure": "ventilator",
    "flow": "ventilator",
}

#: Fallback signal type for the provenance-inference path (Milestone 1),
#: keyed by Stage 1 session modality.
_MODALITY_SIGNAL_TYPE: dict[str, SignalType] = {
    "eit": "eit_waveform",
    "emg": "emg_raw",
}

_FILE_FORMAT_BY_SUFFIX: dict[str, FileFormat] = {
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".edf": "edf",
    ".csv": "csv",
    ".json": "json",
    ".parquet": "parquet",
    ".zarr": "zarr",
}


class DataModelRecorder:
    """Mirrors ``M3Session``/pipeline activity into a ``DataModelStore``."""

    def __init__(
        self,
        session: M3Session,
        store: DataModelStore | None = None,
        *,
        case: Case | None = None,
    ) -> None:
        self.session = session
        self.store = store or DataModelStore()

        self.case = case or self.store.add_case(
            Case(external_case_ref=session.metadata.subject_id)
        )
        if case is not None and case.case_id not in self.store.cases:
            self.store.add_case(case)

        self.recording_session = self.store.add_session(
            RecordingSession(
                case_id=self.case.case_id,
                notes=session.metadata.notes,
            )
        )

        self._devices: dict[str, str] = {}
        self._signals: dict[str, str] = {}
        self._files: dict[str, str] = {}

    # -- Layer 1 objects -> persisted entities (Milestone 2.3) ---------------

    def record_signal(
        self,
        signal: Signal,
        *,
        file_path: str | Path | None = None,
        file_role: FileRole = "raw",
    ) -> SignalStream | None:
        """Record a ``Signal`` as a ``SignalStream`` (+ ``DataFile``).

        A signal whose modality is not one the data model has a stream type for
        (for example the default ``"unknown"`` modality, before the signal has
        been tagged as EIT/EMG/ventilator) cannot be stored, so it is skipped
        with a warning rather than stopping the whole recording. Returns the
        new stream, or ``None`` when the signal was skipped.
        """

        signal_type = _signal_type_for(signal)
        if signal_type is None:
            logger.warning(
                "Skipping signal with modality {!r}: no data model stream type "
                "for it. Tag the signal's modality (eit/emg/ventilator/...) to "
                "record it.",
                signal.modality,
            )
            return None

        device_id = self._ensure_device(signal.modality)
        stream = self.store.add_signal_stream(
            SignalStream(
                session_id=self.recording_session.session_id,
                device_id=device_id,
                signal_type=signal_type,
                unit=signal.unit,
                sampling_frequency_hz=signal.sample_frequency,
                sample_count=signal.n_samples,
            )
        )
        self._signals[signal.modality] = stream.signal_id

        if file_path is not None:
            data_file = self.store.add_data_file(
                DataFile(
                    session_id=self.recording_session.session_id,
                    signal_id=stream.signal_id,
                    file_path=str(file_path),
                    file_format=_infer_file_format(file_path),
                    file_role=file_role,
                )
            )
            self._files[signal.modality] = data_file.file_id
        return stream

    def record_parameter(
        self, parameter: ParameterResult, *, processing_run_id: str
    ) -> DerivedFeature:
        """Materialize a ``ParameterResult`` as a ``DerivedFeature``."""

        return self.store.add_derived_feature(
            DerivedFeature(
                source_signal_id=self._signals.get(parameter.modality),
                processing_run_id=processing_run_id,
                feature_name=parameter.name,
                value=_scalar_value(parameter.value),
                unit=parameter.unit,
            )
        )

    def record_quality_flag(
        self,
        flag: QualityFlag,
        *,
        target_type: TargetType | None = None,
        target_id: str | None = None,
    ) -> QualityAnnotation:
        """Materialize a ``QualityFlag`` as a ``QualityAnnotation``.

        Without an explicit target, this annotates the ``SignalStream``
        already recorded for ``flag.modality``, falling back to the session
        itself if no such stream exists yet (e.g. a session-level check).
        """

        resolved_type: TargetType = target_type or "signal"
        resolved_id = target_id
        if resolved_id is None and flag.modality is not None:
            resolved_id = self._signals.get(flag.modality)
        if resolved_id is None:
            resolved_type = "session"
            resolved_id = self.recording_session.session_id

        return self.store.add_quality_annotation(
            QualityAnnotation(
                target_type=resolved_type,
                target_id=resolved_id,
                quality_label="valid" if flag.passed else "suspect",
                annotation_source="automated",
                annotator_ref=flag.name,
            )
        )

    def record_processing_step(self, step: ProcessingStep) -> ProcessingRun:
        """Materialize a ``ProcessingStep`` as a ``ProcessingRun``."""

        input_file_ids = [
            self._files[key] for key in step.input_keys if key in self._files
        ]
        run = ProcessingRun(
            pipeline_name=step.name,
            run_time=_parse_timestamp(step.timestamp),
            input_file_ids=input_file_ids,
            parameters=_json_safe_parameters(step.parameters),
        )
        return self.store.add_processing_run(run)

    # -- provenance -> ProcessingRun (Milestone 1 fallback) -------------------

    def record_provenance(self, provenance: ProvenanceRecord) -> ProcessingRun:
        """Mirror one ``ProvenanceRecord`` into the store as a ``ProcessingRun``.

        Used for session actions that have not been migrated to emit a
        ``ProcessingStep`` yet.
        """

        input_file_ids: list[str] = []
        modality = _LOAD_ACTIONS.get(provenance.action)
        if modality is not None:
            self._record_load(modality)
        if provenance.modality is not None:
            file_id = self._files.get(provenance.modality)
            if file_id is not None:
                input_file_ids = [file_id]

        run = ProcessingRun(
            pipeline_name=provenance.action,
            run_time=_parse_timestamp(provenance.timestamp),
            input_file_ids=input_file_ids,
            parameters=_json_safe_parameters(provenance.parameters),
        )
        return self.store.add_processing_run(run)

    def _ensure_device(self, modality: str) -> str:
        if modality in self._devices:
            return self._devices[modality]
        device_type = _MODALITY_DEVICE_TYPE.get(modality, "monitor")
        device = self.store.add_device(Device(device_type=device_type))
        self._devices[modality] = device.device_id
        return device.device_id

    def _record_load(self, modality: str) -> None:
        if modality in self._signals:
            return

        recording = self.session.raw.get(modality)
        if recording is None:
            return

        device_id = self._ensure_device(modality)
        signal_type = _MODALITY_SIGNAL_TYPE.get(modality)
        if signal_type is not None:
            stream = self.store.add_signal_stream(
                SignalStream(
                    session_id=self.recording_session.session_id,
                    device_id=device_id,
                    signal_type=signal_type,
                )
            )
            self._signals[modality] = stream.signal_id

            path = getattr(recording, "path", None)
            if path is not None:
                data_file = self.store.add_data_file(
                    DataFile(
                        session_id=self.recording_session.session_id,
                        signal_id=stream.signal_id,
                        file_path=str(path),
                        file_format=_infer_file_format(path),
                        file_role="raw",
                    )
                )
                self._files[modality] = data_file.file_id

    # -- pipeline outputs -> DerivedFeature/QualityAnnotation/SignalStream ---

    def record_pipeline_result(self, result: PipelineResult) -> ProcessingRun:
        """Turn a finished pipeline run's outputs into store entities.

        Named context artifacts that are already ``ParameterResult``/
        ``QualityFlag``/``Signal`` objects are materialized directly; bare
        numeric outputs (Milestone 1 pipelines that have not adopted Layer 1
        objects yet) still become a ``DerivedFeature`` with just a value. The
        run's ``parameters["outputs"]`` also records a JSON-safe provenance
        summary (method/unit/metadata) for every native result, keyed by its
        context name, so the run's full output provenance survives even for
        array-valued results whose ``DerivedFeature.value`` stays ``None``.
        """

        run = self.store.add_processing_run(
            ProcessingRun(pipeline_name=result.name, parameters={})
        )
        output_provenance: dict[str, Any] = {}
        for name, value in result.outputs.items():
            if isinstance(value, ParameterResult):
                self.record_parameter(value, processing_run_id=run.processing_run_id)
                output_provenance[name] = _output_provenance_entry(value)
            elif isinstance(value, QualityFlag):
                self.record_quality_flag(value)
            elif isinstance(value, Signal):
                self.record_signal(value)
                output_provenance[name] = _output_provenance_entry(value)
            elif isinstance(value, bool):
                continue
            elif isinstance(value, (int, float)):
                self.store.add_derived_feature(
                    DerivedFeature(
                        processing_run_id=run.processing_run_id,
                        feature_name=name,
                        value=float(value),
                    )
                )
        run.parameters["outputs"] = output_provenance
        return run

    def record_parameter_file(
        self, path: str | Path, *, processing_run_id: str
    ) -> DataFile:
        """Record a structured-export parameter artifact (e.g. the
        ``parameter_result_arrays.npz`` archive) as a ``DataFile`` with role
        ``"parameter"``, and link it onto the ``ProcessingRun`` that produced
        it via ``ProcessingRun.parameter_file_id``.

        Does not create a new entity type for array-valued results (per
        ``plan/stage2/1_eit_gap_migration_implementation_plan.md`` Phase 5.3):
        the existing ``DataFile``/``ProcessingRun`` link is reused.
        """

        data_file = self.store.add_data_file(
            DataFile(
                session_id=self.recording_session.session_id,
                file_path=str(path),
                file_format="other",
                file_role="parameter",
                checksum_sha256=_sha256_file(path),
                file_size_bytes=os.path.getsize(path),
            )
        )
        run = self.store.processing_runs[processing_run_id]
        run.parameter_file_id = data_file.file_id
        return data_file


def _signal_type_for(signal: Signal) -> SignalType | None:
    """Map a runtime signal's modality to a stored stream type.

    Returns ``None`` for a modality the data model has no stream type for (for
    example the default ``"unknown"``), so the caller can skip it instead of
    failing.
    """

    if signal.modality == "eit":
        return "eit_waveform"
    if signal.modality == "emg":
        return "emg_envelope" if signal.processing_state == "processed" else "emg_raw"
    if signal.modality in ("ventilator", "pressure"):
        return "ventilator_pressure"
    if signal.modality == "flow":
        return "ventilator_flow"
    return None


def _output_provenance_entry(value: ParameterResult | Signal) -> dict[str, Any]:
    """Build a JSON-safe provenance summary for one pipeline-output result,
    for ``ProcessingRun.parameters["outputs"]``."""

    entry: dict[str, Any] = {
        "type": type(value).__name__,
        "method": getattr(value, "method", None),
        "modality": value.modality,
        "unit": value.unit,
        "metadata": _json_safe_parameters(value.metadata),
    }
    if isinstance(value, ParameterResult):
        entry["is_scalar"] = value.is_scalar
    else:
        entry["channel"] = value.channel
        entry["processing_state"] = value.processing_state
    return entry


def _json_safe_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Make a step's recorded parameters safe to write to a JSON file."""

    return {str(key): _json_safe(value) for key, value in parameters.items()}


def _json_safe(value: Any) -> Any:
    """Return a version of ``value`` that can be written to a JSON file.

    Recorded parameters are the settings a step was run with, not the
    scientific results themselves (those are kept in data files). So a bulky
    or non-text value passed as a setting - most often a numpy array handed in
    as an argument - is recorded as a short note of its shape and type instead
    of being copied in full. That keeps the audit file small and always
    writable, while still recording that the argument was given.
    """

    if isinstance(value, np.ndarray):
        return f"<array shape={value.shape} dtype={value.dtype}>"
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _scalar_value(value: float | np.ndarray) -> float | None:
    if np.ndim(value) == 0:
        return float(value)
    return None


def _infer_file_format(path: Path | str) -> FileFormat | None:
    suffix = Path(path).suffix.lower()
    return _FILE_FORMAT_BY_SUFFIX.get(suffix)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)
