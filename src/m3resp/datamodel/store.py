"""In-memory relational store for data model entities.

Deliberately not a database: the data model doc's own recommended process
(Sec 4) is to build a small prototype against a handful of real sessions, run
real workflows, and revise the schema *before* freezing and persisting it.
This store gives us exactly that prototype - typed tables with cheap
foreign-key checks, no migrations - and can be swapped for SQLite/Postgres
later without changing the entities themselves.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from m3resp.datamodel.entities import (
    Breath,
    Case,
    ClinicalEvent,
    DataFile,
    DerivedFeature,
    Device,
    EITConfiguration,
    EMGChannel,
    ProcessingRun,
    QualityAnnotation,
    RecordingSession,
    SignalStream,
    VentilatorSetting,
)

EntityT = TypeVar("EntityT", bound=BaseModel)


class DataModelStoreError(Exception):
    """Raised on foreign-key or lookup failures in a ``DataModelStore``."""


class DataModelStore:
    """A dict-of-id-to-entity table per entity type, plus relational lookups."""

    def __init__(self) -> None:
        self.cases: dict[str, Case] = {}
        self.sessions: dict[str, RecordingSession] = {}
        self.devices: dict[str, Device] = {}
        self.signal_streams: dict[str, SignalStream] = {}
        self.eit_configurations: dict[str, EITConfiguration] = {}
        self.emg_channels: dict[str, EMGChannel] = {}
        self.ventilator_settings: dict[str, VentilatorSetting] = {}
        self.breaths: dict[str, Breath] = {}
        self.clinical_events: dict[str, ClinicalEvent] = {}
        self.data_files: dict[str, DataFile] = {}
        self.processing_runs: dict[str, ProcessingRun] = {}
        self.derived_features: dict[str, DerivedFeature] = {}
        self.quality_annotations: dict[str, QualityAnnotation] = {}

    # -- generic add/get -----------------------------------------------------

    def add_case(self, case: Case) -> Case:
        self.cases[case.case_id] = case
        return case

    def add_session(self, session: RecordingSession) -> RecordingSession:
        self._require(self.cases, session.case_id, "Case")
        self.sessions[session.session_id] = session
        return session

    def add_device(self, device: Device) -> Device:
        self.devices[device.device_id] = device
        return device

    def add_signal_stream(self, stream: SignalStream) -> SignalStream:
        self._require(self.sessions, stream.session_id, "RecordingSession")
        self._require(self.devices, stream.device_id, "Device")
        self.signal_streams[stream.signal_id] = stream
        return stream

    def add_eit_configuration(self, config: EITConfiguration) -> EITConfiguration:
        self._require(self.signal_streams, config.signal_id, "SignalStream")
        self.eit_configurations[config.eit_config_id] = config
        return config

    def add_emg_channel(self, channel: EMGChannel) -> EMGChannel:
        self._require(self.signal_streams, channel.signal_id, "SignalStream")
        self.emg_channels[channel.emg_channel_id] = channel
        return channel

    def add_ventilator_setting(self, setting: VentilatorSetting) -> VentilatorSetting:
        self._require(self.sessions, setting.session_id, "RecordingSession")
        self._require(self.devices, setting.device_id, "Device")
        self.ventilator_settings[setting.ventilator_record_id] = setting
        return setting

    def add_breath(self, breath: Breath) -> Breath:
        self._require(self.sessions, breath.session_id, "RecordingSession")
        self._require(self.signal_streams, breath.source_signal_id, "SignalStream")
        self.breaths[breath.breath_id] = breath
        return breath

    def add_clinical_event(self, event: ClinicalEvent) -> ClinicalEvent:
        self._require(self.sessions, event.session_id, "RecordingSession")
        self.clinical_events[event.event_id] = event
        return event

    def add_data_file(self, data_file: DataFile) -> DataFile:
        if data_file.session_id is not None:
            self._require(self.sessions, data_file.session_id, "RecordingSession")
        if data_file.signal_id is not None:
            self._require(self.signal_streams, data_file.signal_id, "SignalStream")
        self.data_files[data_file.file_id] = data_file
        return data_file

    def add_processing_run(self, run: ProcessingRun) -> ProcessingRun:
        for file_id in run.input_file_ids:
            self._require(self.data_files, file_id, "DataFile")
        self.processing_runs[run.processing_run_id] = run
        return run

    def add_derived_feature(self, feature: DerivedFeature) -> DerivedFeature:
        self._require(self.processing_runs, feature.processing_run_id, "ProcessingRun")
        for source_signal_id in feature.source_signal_ids:
            self._require(self.signal_streams, source_signal_id, "SignalStream")
        self.derived_features[feature.feature_id] = feature
        return feature

    def add_quality_annotation(
        self, annotation: QualityAnnotation
    ) -> QualityAnnotation:
        if not self.has_quality_annotation_target(
            annotation.target_type, annotation.target_id
        ):
            raise DataModelStoreError(
                f"Unknown {annotation.target_type} target id referenced: "
                f"{annotation.target_id!r}"
            )
        self.quality_annotations[annotation.quality_annotation_id] = annotation
        return annotation

    # -- relational lookups ---------------------------------------------------

    def sessions_for_case(self, case_id: str) -> list[RecordingSession]:
        return [s for s in self.sessions.values() if s.case_id == case_id]

    def streams_for_session(self, session_id: str) -> list[SignalStream]:
        return [s for s in self.signal_streams.values() if s.session_id == session_id]

    def files_for_signal(self, signal_id: str) -> list[DataFile]:
        return [f for f in self.data_files.values() if f.signal_id == signal_id]

    def runs_for_file(self, file_id: str) -> list[ProcessingRun]:
        return [r for r in self.processing_runs.values() if file_id in r.input_file_ids]

    def features_for_run(self, processing_run_id: str) -> list[DerivedFeature]:
        return [
            f
            for f in self.derived_features.values()
            if f.processing_run_id == processing_run_id
        ]

    def annotations_for_target(
        self, target_type: str, target_id: str
    ) -> list[QualityAnnotation]:
        return [
            a
            for a in self.quality_annotations.values()
            if a.target_type == target_type and a.target_id == target_id
        ]

    def has_quality_annotation_target(self, target_type: str, target_id: str) -> bool:
        """Return whether a quality annotation target exists in this store."""

        if target_type == "signal":
            return target_id in self.signal_streams
        if target_type == "file":
            return target_id in self.data_files
        if target_type == "session":
            return target_id in self.sessions
        if target_type == "event":
            return target_id in self.clinical_events
        if target_type == "feature":
            return target_id in self.derived_features
        return False

    # -- internals --------------------------------------------------------

    @staticmethod
    def _require(table: dict[str, EntityT], key: str, kind: str) -> None:
        if key not in table:
            raise DataModelStoreError(f"Unknown {kind} id referenced: {key!r}")
