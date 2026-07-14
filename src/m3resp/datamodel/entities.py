"""Pydantic entities for the M3Resp conceptual/logical data model.

Fields mirror the logical tables in ``main_v0.3.tex`` (data model design doc):
Sections 7.1-7.11 for the MVP entities (doc Sec 9) and Section 11 for
``QualityAnnotation``. Coded fields use ``Literal`` unions built from the
vocabularies the doc already spells out (Sec 2.5, 7.4, 7.9-7.11, 11).

Two fields go beyond the doc, both flagged here rather than hidden: they exist
so the future Controller/Service layer (Session Manager, State Manager,
Pipeline Manager, Error Handler, Task Runner) has somewhere to record
execution/async state without a schema break.

- ``ProcessingRun.status`` / ``ProcessingRun.error``: Task Runner needs to
  track jobs across pending/running/succeeded/failed, and the Error Handler
  needs a place to attach a technical error message to the run that failed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from m3resp.datamodel.ids import new_id


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Entity(BaseModel):
    """Base class for all data model entities."""

    model_config = {"extra": "forbid"}


# --- Study context - the spine (doc Sec 7.1-7.3) ----------------------------


class Case(Entity):
    """Opaque grouping of sessions belonging to one external study subject."""

    case_id: str = Field(default_factory=lambda: new_id("case"))
    external_case_ref: str | None = None
    study_ref: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)


BodyPosition = Literal[
    "supine", "prone", "semi_recumbent", "lateral", "dynamic", "other"
]
AcquisitionState = Literal["baseline", "intervention", "recovery", "other"]


class RecordingSession(Entity):
    """One continuous acquisition period (doc Sec 7.2)."""

    session_id: str = Field(default_factory=lambda: new_id("session"))
    case_id: str
    session_start_time: datetime | None = None
    session_end_time: datetime | None = None
    #: The session's body position, if fixed for its whole duration. Set to
    #: ``"dynamic"`` when position changes during the session rather than
    #: guessing a single value - record the actual changes as
    #: ``ClinicalEvent(event_type="position_change", ...)`` rows instead.
    body_position: BodyPosition | None = None
    acquisition_state: AcquisitionState | None = None
    operator_ref: str | None = None
    notes: str | None = None


#: Common device types (doc Sec 7.3). Open vocabulary, not an enum: a new
#: loading function can use any string here (e.g. a new device model) without
#: a schema change - these are listed for documentation/IDE-completion
#: convenience, not enforced at validation time.
DeviceType = Literal["eit", "emg", "ventilator", "monitor", "acquisition_pc"] | str
CalibrationStatus = Literal["valid", "expired", "unknown", "not_applicable"]


class Device(Entity):
    """Physical or software system that produces data (doc Sec 7.3)."""

    device_id: str = Field(default_factory=lambda: new_id("device"))
    device_type: DeviceType
    manufacturer: str | None = None
    model: str | None = None
    serial_number_hash: str | None = None
    software_version: str | None = None
    calibration_status: CalibrationStatus | None = None


# --- Signal layer (doc Sec 7.4-7.8) ------------------------------------------

#: Common signal types (doc Sec 7.4). Open vocabulary, not an enum: real
#: recordings can have several distinct pressure signals (e.g. airway,
#: esophageal, differential) from multiple devices at once, and new
#: modalities/quantities keep showing up - these are listed for
#: documentation/IDE-completion convenience, not enforced at validation time.
#:
#: - ``eit_waveform``: continuous data with a value per timepoint - 1D (e.g.
#:   global impedance), 2D, or 3D (per-pixel impedance over time).
#: - ``eit_frame``: a single 2D map, not a time series - either a slice of an
#:   ``eit_waveform`` at one timepoint, or a derived 2D map (e.g. a
#:   difference between two timepoints, or a breath-derived map like TIV).
SignalType = (
    Literal[
        "eit_waveform",
        "eit_frame",
        "emg_raw",
        "emg_envelope",
        "ventilator_pressure",
        "ventilator_flow",
        "ventilator_volume",
    ]
    | str
)
#: - ``absolute``: a calibrated real-world clock time (e.g. NTP/atomic).
#: - ``approximate``: a human-recorded or estimated real-world time (e.g.
#:   noted on paper during the session) - a real timestamp, but not
#:   calibrated, so it shouldn't be trusted to the same precision as
#:   ``absolute``.
#: - ``relative``: elapsed time from some reference point, not tied to a
#:   real-world clock.
#: - ``device_local``: the recording device's own internal clock.
#: - ``synchronized``: has been aligned onto a common time axis with other
#:   signals (see `m3resp.synchronization`).
TimeBase = Literal[
    "absolute", "approximate", "relative", "device_local", "synchronized"
]
SyncMethod = Literal["common_clock", "ntp", "trigger", "manual", "none"]
#: A coarse, human-readable summary of sync quality - not a substitute for
#: `SignalStream.sync_uncertainty_ms`. What counts as "good" vs "acceptable"
#: synchronization is application-dependent (e.g. millisecond precision
#: needed for some analyses, breath-level granularity sufficient for
#: others), so code that needs a concrete threshold should compare against
#: `sync_uncertainty_ms` directly rather than branch on this label.
SyncQualityLabel = Literal["good", "acceptable", "uncertain", "failed"]


class SignalStream(Entity):
    """Base type for every continuous signal (doc Sec 7.4)."""

    signal_id: str = Field(default_factory=lambda: new_id("signal"))
    session_id: str
    device_id: str
    signal_type: SignalType
    unit: str | None = None
    sampling_frequency_hz: float | None = None
    sample_count: int | None = None
    start_time: datetime | None = None
    device_local_start_time: datetime | None = None
    time_zone: str | None = None
    time_base: TimeBase | None = None
    sync_method: SyncMethod | None = None
    sync_reference_stream: str | None = None
    clock_drift_ppm: float | None = None
    time_offset_ms: float | None = None
    sync_uncertainty_ms: float | None = None
    sync_quality_label: SyncQualityLabel | None = None


class EITConfiguration(Entity):
    """Reconstruction/belt configuration for an EIT stream (doc Sec 7.5)."""

    eit_config_id: str = Field(default_factory=lambda: new_id("eitcfg"))
    signal_id: str
    electrode_count: int | None = None
    belt_position: str | None = None
    current_amplitude: float | None = None
    injection_pattern: str | None = None
    reconstruction_algorithm: str | None = None
    reference_frame_method: str | None = None
    frame_rate_hz: float | None = None


class EMGChannel(Entity):
    """Channel-level configuration for an EMG stream (doc Sec 7.6)."""

    emg_channel_id: str = Field(default_factory=lambda: new_id("emgch"))
    signal_id: str
    channel_name: str | None = None
    muscle_or_site: str | None = None
    electrode_placement: str | None = None
    reference_electrode: str | None = None
    amplifier_gain: float | None = None
    hardware_filter: str | None = None
    software_filter: str | None = None


class VentilatorSetting(Entity):
    """Ventilator mode/settings snapshot, scoped to a session (doc Sec 7.7).

    Settings that change during a session are handled by recording one
    ``VentilatorSetting`` row per change - ``measurement_time`` is when this
    particular snapshot was taken, not a single value for the whole session.

    The named fields cover common settings across manufacturers; ``extras``
    holds manufacturer/device-specific settings not covered by them (the
    ``Entity`` base forbids arbitrary extra fields, so this is the place for
    anything not already named here, rather than a schema change).
    """

    ventilator_record_id: str = Field(default_factory=lambda: new_id("ventset"))
    session_id: str
    device_id: str
    ventilator_mode: str | None = None
    peep_cmh2o: float | None = None
    fio2: float | None = None
    pressure_support_cmh2o: float | None = None
    respiratory_rate_set: float | None = None
    tidal_volume_set_ml: float | None = None
    measurement_time: datetime | None = None
    extras: dict[str, Any] = Field(default_factory=dict)


AsynchronyType = Literal[
    "none", "ineffective_effort", "double_trigger", "auto_trigger", "other"
]


class Breath(Entity):
    """One ventilator breath / respiratory cycle (doc Sec 7.8, optional)."""

    breath_id: str = Field(default_factory=lambda: new_id("breath"))
    session_id: str
    source_signal_id: str
    breath_start_time: datetime | None = None
    breath_end_time: datetime | None = None
    detection_method: Literal["algorithm", "manual"] | None = None
    asynchrony_type: AsynchronyType | None = None


# --- Recording context (doc Sec 7.9) -----------------------------------------

EventType = Literal[
    "position_change",
    "suctioning",
    "recruitment_maneuver",
    "ventilator_adjustment",
    "disconnection",
    "extubation",
    "other",
]
EventSource = Literal["manual_annotation", "device_log", "algorithm"]
Confidence = Literal["high", "medium", "low"]


class ClinicalEvent(Entity):
    """Time-stamped intervention or notable occurrence (doc Sec 7.9)."""

    event_id: str = Field(default_factory=lambda: new_id("event"))
    session_id: str
    event_type: EventType
    event_start_time: datetime | None = None
    event_end_time: datetime | None = None
    event_source: EventSource | None = None
    confidence: Confidence | None = None
    description: str | None = None


# --- Storage and provenance (doc Sec 7.10-7.11, 8.1) -------------------------

ProcessingStatus = Literal["pending", "running", "succeeded", "failed"]


class ProcessingRun(Entity):
    """One execution of a processing pipeline (doc Sec 7.10)."""

    processing_run_id: str = Field(default_factory=lambda: new_id("run"))
    pipeline_name: str
    pipeline_version: str | None = None
    code_commit_hash: str | None = None
    input_file_ids: list[str] = Field(default_factory=list)
    parameter_file_id: str | None = None
    run_time: datetime = Field(default_factory=_utc_now)
    operator_ref: str | None = None
    # Beyond the doc: forward-compat for the Task Runner / Error Handler
    # components in the roadmap image (see module docstring).
    status: ProcessingStatus = "succeeded"
    error: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


QualityFlag = Literal["valid", "suspect", "invalid", "missing"]


class DerivedFeature(Entity):
    """A quantitative result computed from one or more source streams (doc Sec 7.11)."""

    feature_id: str = Field(default_factory=lambda: new_id("feature"))
    source_signal_id: str | None = None
    processing_run_id: str
    feature_name: str
    time_window_start: datetime | None = None
    time_window_end: datetime | None = None
    value: float | None = None
    unit: str | None = None
    quality_flag: QualityFlag | None = None


FileFormat = Literal["hdf5", "edf", "csv", "json", "dicom", "parquet", "zarr", "other"]
FileRole = Literal["raw", "processed", "derived", "annotation", "parameter", "report"]


class DataFile(Entity):
    """A physical file holding raw/processed/derived data (doc Sec 8.1)."""

    file_id: str = Field(default_factory=lambda: new_id("file"))
    session_id: str | None = None
    signal_id: str | None = None
    file_path: str
    file_format: FileFormat | None = None
    file_role: FileRole | None = None
    checksum_sha256: str | None = None
    file_size_bytes: int | None = None
    created_at: datetime = Field(default_factory=_utc_now)


# --- Cross-cutting quality (doc Sec 11) --------------------------------------

TargetType = Literal["signal", "file", "session", "event", "feature"]
QualityLabel = Literal["valid", "suspect", "invalid", "missing", "artifact"]
ArtifactType = Literal[
    "motion",
    "electrode_detachment",
    "ventilator_disconnect",
    "noise",
    "saturation",
    "drift",
    "other",
]
AnnotationSource = Literal["manual", "automated", "device_log"]


class QualityAnnotation(Entity):
    """Polymorphic quality label (doc Sec 11)."""

    quality_annotation_id: str = Field(default_factory=lambda: new_id("qa"))
    target_type: TargetType
    target_id: str
    time_window_start: datetime | None = None
    time_window_end: datetime | None = None
    quality_label: QualityLabel
    artifact_type: ArtifactType | None = None
    annotation_source: AnnotationSource | None = None
    annotator_ref: str | None = None
