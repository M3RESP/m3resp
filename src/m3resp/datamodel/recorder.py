"""Record session activity and workflow results in a DataModelStore.

An attached recorder stores session provenance, signal and result records,
and completed workflow outputs. Native result objects are recorded through
their corresponding methods; numeric workflow outputs become derived features.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from m3resp.data.event_data import EventData, IntervalData
from m3resp.data.parameters import ParameterResult
from m3resp.data.pixel_maps import PixelMap, PixelMask
from m3resp.data.processing import ProcessingStep
from m3resp.data.quality import QualityFlag
from m3resp.data.signals import Signal
from m3resp.datamodel.entities import (
    Case,
    DataFile,
    DerivedFeature,
    Device,
    DeviceType,
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
from m3resp.modalities.names import VENTILATOR, normalize_modality
from m3resp.synchronization.start_times import recording_start_time
from m3resp.synchronization.sync_methods import clock_key

if TYPE_CHECKING:
    from m3resp.core.provenance import ProvenanceRecord
    from m3resp.core.session import M3Session
    from m3resp.workflows.engine import WorkflowResult

#: Provenance actions that correspond to loading a modality's raw data.
_LOAD_ACTIONS = {"load_eit": "eit", "load_emg": "emg"}

#: Maps Stage 1 session modality keys ("vent") and Layer 1 ``Signal.modality``
#: values onto the persisted ``Device.device_type`` vocabulary. Since
#: ``modality`` is now purely the device axis (physical quantities moved to
#: ``Signal.category``), this is close to an identity map.
#:
#: ``"pressure"``/``"flow"`` are retained only as legacy aliases, for signals
#: produced before the split or by third-party code still using the old
#: combined vocabulary. They assume a ventilator, which is exactly the wrong
#: guess for esophageal or gastric pressure - tag those ``modality="monitor"``
#: (or the actual device) with ``category="esophageal_pressure"`` instead.
_MODALITY_DEVICE_TYPE: dict[str, DeviceType] = {
    "eit": "eit",
    "emg": "emg",
    "vent": "ventilator",
    "ventilator": "ventilator",
    "monitor": "monitor",
    "pressure": "ventilator",
    "flow": "ventilator",
}


def _stream_key(
    modality: str | None, category: str | None, channel: str | None = None
) -> str:
    """Cache key identifying one recorded ``SignalStream``.

    All three axes are needed. One device emits several streams (a ventilator
    produces pressure, flow and volume), and one device can emit several
    streams of the *same* quantity, which only the channel tells apart. Both
    cases are real: EIT records the global impedance and the pixel impedance
    as ``eit``/``impedance``, and a study can capture an airway pressure both
    from a ventilator export and from the EIT file's own Medibus channels.
    Keying on modality and category alone made each of those pairs
    indistinguishable, so the second overwrote the first and every derived
    feature and quality flag was attributed to whichever was recorded last.

    Falls back to the bare modality when no category is set, so signals
    recorded before categories existed keep the keys they had, and omits the
    channel when there is none.
    """

    resolved = modality or "unknown"
    if not category:
        return resolved
    if not channel:
        return f"{resolved}:{category}"
    return f"{resolved}:{category}:{channel}"


def _instrument_of(signal: Signal) -> str | None:
    """Return the instrument qualifier after __ in a signal's channel key.

    For example, airway_pressure__pod yields "pod" for the Device record. A
    channel key without a qualifier returns None.
    """

    channel = signal.channel
    if channel and "__" in channel:
        return channel.split("__", 1)[1]
    return None


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
    ".mat": "mat",
    ".parquet": "parquet",
    ".zarr": "zarr",
}


class DataModelRecorder:
    """Store a session's recordings, results and processing provenance.

    Construction creates or registers a case and a recording session in the
    store. Assign the recorder to session.datamodel to record session actions
    and workflow results automatically.

    Attributes:
        session: Runtime session being recorded.
        store: DataModelStore receiving records.
        case: Case associated with the session.
        recording_session: Stored session record.
    """

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
        # Keyed by `_stream_key(modality, category, channel)`. Every stream is
        # filed under its own fully qualified key; the first channel to claim
        # a quantity is additionally filed under the unqualified
        # `modality:category` key, so a result that names no channel still
        # resolves to it.
        self._signals: dict[str, str] = {}
        self._files: dict[str, str] = {}
        # Which channel claimed each unqualified key. A later signal on that
        # same channel (its processed version, say) may replace it; one on a
        # different channel may not, or the global and pixel impedances - or
        # two instruments' airway pressures - would overwrite each other.
        self._stream_owners: dict[str, str | None] = {}
        # Which recording each stream came from, as (modality, ventilator
        # recording name or None for the primary one), so its synchronization
        # can be copied from the session. See `record_synchronization`.
        self._stream_recordings: dict[str, tuple[str, str | None]] = {}

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
                "Skipping signal with modality={!r} category={!r}: no data "
                "model stream type for that pair. Tag the signal's modality "
                "(eit/emg/ventilator/...) and, for a device that emits several "
                "quantities, its category (airway_pressure/airflow/volume).",
                signal.modality,
                signal.category,
            )
            return None

        instrument = _instrument_of(signal)
        device_id = self._ensure_device(signal.modality, instrument)
        recording_name = self._ventilator_recording_name(signal, instrument)
        sync_method, time_offset_ms = self._sync_fields(signal.modality, recording_name)
        stream = self.store.add_signal_stream(
            SignalStream(
                session_id=self.recording_session.session_id,
                device_id=device_id,
                signal_type=signal_type,
                unit=signal.unit,
                sampling_frequency_hz=signal.sample_frequency,
                sample_count=signal.n_samples,
                sync_method=sync_method,
                time_offset_ms=time_offset_ms,
            )
        )
        self._stream_recordings[stream.signal_id] = (signal.modality, recording_name)
        # The channel-qualified key always points at this exact stream. The
        # unqualified key is owned by the first channel that claimed it, so a
        # later stream of the same modality and quantity on a *different*
        # channel (the pixel impedance arriving after the global one, or a
        # second instrument's airway pressure) cannot take over the
        # attribution of results that name only their modality and quantity.
        # Re-recording the *same* channel still replaces its own entry.
        stream_key = _stream_key(signal.modality, signal.category, signal.channel)
        self._signals[stream_key] = stream.signal_id
        shared_key = _stream_key(signal.modality, signal.category)
        owns_shared = (
            self._stream_owners.setdefault(shared_key, signal.channel) == signal.channel
        )
        if owns_shared:
            self._signals[shared_key] = stream.signal_id
        self._signals.setdefault(_stream_key(signal.modality, None), stream.signal_id)

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
            self._files[stream_key] = data_file.file_id
            if owns_shared:
                self._files[shared_key] = data_file.file_id
        return stream

    def _lookup_signal_id(
        self, modality: str | None, category: str | None, channel: str | None = None
    ) -> str | None:
        """Find the recorded stream a parameter/flag belongs to.

        Prefers an exact ``(modality, category, channel)`` match, then the
        channel that owns the bare ``(modality, category)`` pair. A result
        that names only its modality falls back to that modality's first
        recorded stream, which is what everything did before categories
        existed.
        """

        if modality is None:
            return None
        for key in (
            _stream_key(modality, category, channel),
            _stream_key(modality, category),
        ):
            signal_id = self._signals.get(key)
            if signal_id is not None:
                return signal_id
        signal_id = self._signals.get(modality)
        if signal_id is not None:
            return signal_id
        prefix = f"{modality}:"
        for stored_key, stored_id in self._signals.items():
            if stored_key.startswith(prefix):
                return stored_id
        return None

    def record_parameter(
        self, parameter: ParameterResult, *, processing_run_id: str
    ) -> DerivedFeature:
        """Materialize a ``ParameterResult`` as a ``DerivedFeature``."""

        signal_id = self._lookup_signal_id(
            parameter.modality, parameter.category, parameter.channel
        )
        return self.store.add_derived_feature(
            DerivedFeature(
                source_signal_ids=[signal_id] if signal_id is not None else [],
                processing_run_id=processing_run_id,
                feature_name=parameter.name,
                value=_scalar_value(parameter.value),
                unit=parameter.unit,
            )
        )

    def record_interval_data(
        self, interval_data: IntervalData | EventData, *, processing_run_id: str
    ) -> list[DerivedFeature]:
        """Add values per interval or event to the store as derived features.

        Scalar values create one feature per item. Times are copied in
        seconds on the input's clock; an event uses the same time for both
        ends of its window. Missing ``None`` or NaN values are stored as
        ``None``. Values that fail numeric conversion are stored as ``None``
        with a logged warning.

        A result containing any array values creates one feature for the
        whole result, with ``value=None``. Array data is kept in the separate
        export archive.

        Args:
            interval_data: ``IntervalData`` or ``EventData`` with values in
                item order and their physical unit.
            processing_run_id: Identifier of the run already in the store.

        Returns:
            list[DerivedFeature]: Stored features in item order for scalar
                values, or one feature for a result containing arrays.

        Raises:
            DataModelStoreError: If the processing run is absent from the store.
        """

        signal_id = self._lookup_signal_id(
            interval_data.modality, interval_data.category, None
        )
        source_signal_ids = [signal_id] if signal_id is not None else []
        values = interval_data.values
        items = (
            interval_data.intervals
            if isinstance(interval_data, IntervalData)
            else interval_data.events
        )
        if values is not None and any(np.ndim(value) > 0 for value in values):
            return [
                self.store.add_derived_feature(
                    DerivedFeature(
                        source_signal_ids=source_signal_ids,
                        processing_run_id=processing_run_id,
                        feature_name=interval_data.name,
                        value=None,
                        unit=interval_data.unit,
                    )
                )
            ]

        features = []
        for position, item in enumerate(items):
            value = None if values is None else values[position]
            start = getattr(item, "start_time", getattr(item, "time", None))
            end = getattr(item, "end_time", start)
            features.append(
                self.store.add_derived_feature(
                    DerivedFeature(
                        source_signal_ids=source_signal_ids,
                        processing_run_id=processing_run_id,
                        feature_name=interval_data.name,
                        time_window_start=start,
                        time_window_end=end,
                        value=_feature_value(value, interval_data.name),
                        unit=interval_data.unit,
                    )
                )
            )
        return features

    def record_pixel_grid(
        self, grid: PixelMap | PixelMask, *, processing_run_id: str
    ) -> DerivedFeature:
        """Add a pixel map or mask to the store as a derived-feature record.

        Args:
            grid: ``PixelMap`` or ``PixelMask`` with its name and modality.
                A pixel map also supplies the feature's unit.
            processing_run_id: Identifier of the run already in the store.

        Returns:
            DerivedFeature: Stored record with ``value=None``. The numeric
                grid is kept in a separate export archive.

        Raises:
            DataModelStoreError: If the processing run is absent from the store.
        """

        signal_id = self._lookup_signal_id(grid.modality, None, None)
        return self.store.add_derived_feature(
            DerivedFeature(
                source_signal_ids=[signal_id] if signal_id is not None else [],
                processing_run_id=processing_run_id,
                feature_name=grid.name,
                value=None,
                unit=getattr(grid, "unit", None),
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
            resolved_id = self._lookup_signal_id(flag.modality, flag.category)
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
        """Store one processing step as a ProcessingRun with kind=step.

        Args:
            step: ProcessingStep with a name, timestamp, input keys and settings.

        Returns:
            ProcessingRun: Record added to the store with the step's name, parsed
                timestamp and JSON-compatible settings. Known input-file keys are
                linked. Other run fields use ProcessingRun defaults.
        """

        input_file_ids = [
            self._files[key] for key in step.input_keys if key in self._files
        ]
        run = ProcessingRun(
            name=step.name,
            kind="step",
            run_time=_parse_timestamp(step.timestamp),
            input_file_ids=input_file_ids,
            parameters=_json_safe_parameters(step.parameters),
        )
        return self.store.add_processing_run(run)

    # -- Session actions -> ProcessingRun ------------------------------------

    def record_provenance(self, provenance: ProvenanceRecord) -> ProcessingRun:
        """Store one session action as a ProcessingRun with kind=session_action.

        Loading actions first record the raw signal and source file when available.
        The session's current synchronization settings are also copied to recorded
        streams.

        Args:
            provenance: Session action with a timestamp, modality and settings.

        Returns:
            ProcessingRun: Stored action record with converted settings and any
                known source-file link for its modality.
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
            name=provenance.action,
            kind="session_action",
            run_time=_parse_timestamp(provenance.timestamp),
            input_file_ids=input_file_ids,
            parameters=_json_safe_parameters(provenance.parameters),
        )
        # Every session action reaches here after it has run, so this is where
        # a synchronization, skip, slice or reload is copied onto the streams.
        self.record_synchronization()
        return self.store.add_processing_run(run)

    def record_synchronization(self) -> None:
        """Copy each recording's synchronization onto its recorded streams.

        Sets `SignalStream.sync_method` from `session.sync_methods`
        (``"manual"``, ``"none"``) and `SignalStream.time_offset_ms` from the
        recording's start time on the shared clock. A stream whose recording
        was never synchronized gets None for both. Ventilator data from the
        EIT or EMG file takes that file's synchronization.
        """

        for signal_id, (modality, instrument) in self._stream_recordings.items():
            stream = self.store.signal_streams.get(signal_id)
            if stream is not None:
                stream.sync_method, stream.time_offset_ms = self._sync_fields(
                    modality, instrument
                )

    def _sync_fields(
        self, modality: str | None, recording_name: str | None
    ) -> tuple[str | None, float | None]:
        """``(sync_method, time_offset_ms)`` for one recording's stream, or
        ``(None, None)`` when it was never synchronized or its modality is
        not one a start time is kept for. `recording_name` names the
        ventilator recording; None means the primary one."""

        normalized = normalize_modality(modality or "")
        if normalized not in ("eit", "emg", VENTILATOR):
            return None, None
        key = clock_key(self.session, normalized, recording_name)
        method = getattr(self.session, "sync_methods", {}).get(key)
        if method is None:
            return None, None
        start_seconds = recording_start_time(self.session, normalized, recording_name)
        return method, start_seconds * 1000.0

    def _ventilator_recording_name(
        self, signal: Signal, instrument: str | None
    ) -> str | None:
        """The loaded ventilator recording a signal came from, or None for
        the primary one.

        Uses the name `M3Session.preprocess_ventilator` stores in the
        signal's metadata. Without it, the channel suffix (`instrument`)
        counts only when it is the name of a loaded recording: a suffix can
        also mark a second channel of the same quantity inside one recording
        (a pressure pod's airway pressure), which belongs to that recording.
        """

        loaded = getattr(self.session, "ventilators", {}) or {}
        for candidate in (signal.metadata.get("recording"), instrument):
            if candidate is not None and candidate in loaded:
                return candidate
        return None

    def _ensure_device(self, modality: str, instrument: str | None = None) -> str:
        """The ``Device`` record for one instrument of one modality.

        Two instruments recording the same modality are two devices, each with
        its own manufacturer, model and serial number, so they cannot share a
        record. The primary recording keeps the bare modality as its key.
        """

        key = f"{modality}@{instrument}" if instrument else modality
        if key in self._devices:
            return self._devices[key]
        device_type = _MODALITY_DEVICE_TYPE.get(modality, "monitor")
        device = self.store.add_device(Device(device_type=device_type))
        self._devices[key] = device.device_id
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
            sync_method, time_offset_ms = self._sync_fields(modality, None)
            stream = self.store.add_signal_stream(
                SignalStream(
                    session_id=self.recording_session.session_id,
                    device_id=device_id,
                    signal_type=signal_type,
                    sync_method=sync_method,
                    time_offset_ms=time_offset_ms,
                )
            )
            self._signals[modality] = stream.signal_id
            self._stream_recordings[stream.signal_id] = (modality, None)

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

    # -- workflow outputs -> DerivedFeature/QualityAnnotation/SignalStream ---

    def record_workflow_result(self, result: WorkflowResult) -> ProcessingRun:
        """Store a workflow run and its recognized output values.

        Nested lists, tuples and dictionaries are visited recursively. Signals,
        measurements, per-event/per-interval results, pixel grids and quality flags
        use their corresponding recorder methods. Bare numeric outputs become
        derived features named by context key; booleans and unsupported objects
        are skipped.

        Args:
            result: Completed WorkflowResult containing named outputs.

        Returns:
            ProcessingRun: Stored record with kind=workflow and the workflow name.
                Native measurement, signal and grouped-result provenance is kept in
                parameters["outputs"]. Array values remain in exported archives.
                All input files known to this recorder are linked, including files
                from earlier runs on the same session. Status and run_time use
                ProcessingRun defaults.
        """

        run = self.store.add_processing_run(
            ProcessingRun(name=result.name, kind="workflow", parameters={})
        )
        output_provenance: dict[str, Any] = {}
        for name, value in result.outputs.items():
            entry = self._record_output_value(name, value, run)
            if entry is not None:
                output_provenance[name] = entry
        run.parameters["outputs"] = output_provenance
        run.input_file_ids = sorted(set(self._files.values()))
        return run

    def _record_output_value(self, name: str, value: Any, run: ProcessingRun) -> Any:
        """Record recognized leaves within nested lists, tuples and dictionaries.

        Return collected provenance entries, or None when none are available.
        Lists and tuples become lists; entries without provenance are omitted.
        """

        if isinstance(value, list | tuple):
            entries = [self._record_output_value(name, item, run) for item in value]
            entries = [entry for entry in entries if entry is not None]
            return entries or None
        if isinstance(value, dict):
            mapped = {
                str(key): self._record_output_value(f"{name}.{key}", item, run)
                for key, item in value.items()
            }
            mapped = {key: entry for key, entry in mapped.items() if entry is not None}
            return mapped or None
        return self._record_output_item(name, value, run)

    def _record_output_item(
        self, name: str, value: Any, run: ProcessingRun
    ) -> dict[str, Any] | None:
        """Record one workflow result and return its provenance description.

        ``name`` supplies the feature name for a numeric result. Signals,
        parameters, timed values and pixel grids return descriptive entries;
        quality flags and numbers are stored and return ``None``. Unsupported
        types and booleans return ``None``.
        """

        if isinstance(value, ParameterResult):
            self.record_parameter(value, processing_run_id=run.processing_run_id)
            return _output_provenance_entry(value)
        if isinstance(value, QualityFlag):
            self.record_quality_flag(value)
            return None
        if isinstance(value, Signal):
            self.record_signal(value)
            return _output_provenance_entry(value)
        if isinstance(value, (IntervalData, EventData)):
            self.record_interval_data(value, processing_run_id=run.processing_run_id)
            return _grouped_output_provenance_entry(value)
        if isinstance(value, (PixelMap, PixelMask)):
            self.record_pixel_grid(value, processing_run_id=run.processing_run_id)
            return _grouped_output_provenance_entry(value)
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            self.store.add_derived_feature(
                DerivedFeature(
                    processing_run_id=run.processing_run_id,
                    feature_name=name,
                    value=float(value),
                )
            )
            return None
        return None

    def record_parameter_file(
        self, path: str | Path, *, processing_run_id: str
    ) -> DataFile:
        """Record an exported array file and link it to its processing run.

        Exporting again to the same path replaces the earlier link to that
        path in the run's ``parameter_file_ids``. The earlier ``DataFile``
        remains in the store as a record of the earlier export.

        Args:
            path: Existing array-results file, such as
                ``interval_data_arrays.npz`` or ``pixel_masks.npz``.
            processing_run_id: Identifier of the run already in the store.

        Returns:
            DataFile: Stored file record with role ``'parameter'``, a SHA-256
                checksum and file size in bytes.

        Raises:
            OSError: If the file cannot be read or its size determined.
            KeyError: If the processing run is absent from the store.
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
        run.parameter_file_ids = [
            file_id
            for file_id in run.parameter_file_ids
            if self.store.data_files[file_id].file_path != str(path)
        ] + [data_file.file_id]
        return data_file


#: ``(modality, category) -> SignalType``. Layer 2 keys streams by a name that
#: fuses device and quantity, so both Layer 1 axes are needed to resolve one.
#: Before ``Signal.category`` existed this had to be guessed from ``modality``
#: alone, which meant every ventilator signal was recorded as
#: ``ventilator_pressure`` and ``ventilator_volume`` was unreachable.
_SIGNAL_TYPE_BY_MODALITY_CATEGORY: dict[tuple[str, str], SignalType] = {
    ("eit", "impedance"): "eit_waveform",
    ("ventilator", "airway_pressure"): "ventilator_pressure",
    ("ventilator", "airflow"): "ventilator_flow",
    ("ventilator", "volume"): "ventilator_volume",
    ("ventilator", "tidal_volume"): "ventilator_volume",
}

#: Fallback ``modality -> SignalType`` for signals with no ``category`` set,
#: used only where the modality alone is unambiguous. ``"ventilator"`` has no
#: entry: without a category, pressure, flow and volume cannot be told apart.
_SIGNAL_TYPE_BY_MODALITY: dict[str, SignalType] = {
    "eit": "eit_waveform",
}


def _signal_type_for(signal: Signal) -> SignalType | None:
    """Map a runtime signal's ``(modality, category)`` to a stored stream type.

    Returns ``None`` when the pair does not identify a stream type - either an
    untagged modality (the default ``"unknown"``), or a modality that needs a
    category to be resolvable (``"ventilator"``) and doesn't have one. The
    caller skips those with a warning instead of failing.
    """

    modality = signal.modality
    category = signal.category

    # EMG's split is by processing stage, not by physical quantity: raw trace
    # and envelope are both electrical potentials.
    if modality == "emg":
        return "emg_envelope" if signal.processing_state == "processed" else "emg_raw"

    if category is not None:
        signal_type = _SIGNAL_TYPE_BY_MODALITY_CATEGORY.get((modality, category))
        if signal_type is not None:
            return signal_type

    return _SIGNAL_TYPE_BY_MODALITY.get(modality)


def _output_provenance_entry(value: ParameterResult | Signal) -> dict[str, Any]:
    """Summarize a signal or parameter's method, units, channel and metadata.

    Parameter summaries include breath identity and scalar status; signal
    summaries include processing state. Metadata is converted for JSON storage.
    """

    entry: dict[str, Any] = {
        "type": type(value).__name__,
        "method": getattr(value, "method", None),
        "modality": value.modality,
        "unit": value.unit,
        "metadata": _json_safe_parameters(value.metadata),
    }
    if isinstance(value, ParameterResult):
        entry["is_scalar"] = value.is_scalar
        entry["breath_id"] = value.breath_id
        entry["channel"] = value.channel
    else:
        entry["channel"] = value.channel
        entry["processing_state"] = value.processing_state
    return entry


def _feature_value(value: Any, name: str) -> float | None:
    """Convert a feature value to a float, using ``None`` for missing values.

    ``None`` and NaN become ``None``. Failed numeric conversion also returns
    ``None`` and logs a warning identifying the result by ``name``.
    """

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        logger.warning(
            f"Values of {name!r} are not numbers (got {value!r}); stored "
            "without a value in the data model."
        )
        return None
    return None if math.isnan(number) else number


def _grouped_output_provenance_entry(
    value: IntervalData | EventData | PixelMap | PixelMask,
) -> dict[str, Any]:
    """Describe a timed result or pixel grid for the saved processing history.

    The description includes names, units and metadata, plus the item count
    for timed values or the row-column shape for a grid.
    """

    entry: dict[str, Any] = {
        "type": type(value).__name__,
        "name": value.name,
        "method": value.method,
        "modality": value.modality,
        "unit": getattr(value, "unit", None),
        "metadata": _json_safe_parameters(value.metadata),
    }
    if isinstance(value, (IntervalData, EventData)):
        entry["count"] = len(value)
    else:
        entry["shape"] = list(value.shape)
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


def _parse_timestamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()
