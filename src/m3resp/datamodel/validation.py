"""Checks for a ``DataModelStore`` (data model doc Section 10).

There are two kinds of checks, because a store is used in two different
situations:

- **Reference checks** always run. They confirm that every record points at
  other records that really exist (for example, that a signal stream names a
  recording session the store actually holds) and that no time window ends
  before it starts. A store that fails a reference check is inconsistent and
  should be treated as a problem to fix.

- **Completeness checks** only run when you ask for them
  (``require_complete=True``). They confirm that every record also carries the
  descriptive fields a finished dataset is expected to have - units, sampling
  rate, start time, file checksums, and so on. A store that is built while
  watching a live session often has not filled these in yet (an EIT or EMG
  recording usually knows only its time *since the start of the recording*,
  not the wall-clock time it started, and a file is not checksummed until it
  is written to disk). Those gaps are normal during processing, so they are
  reported only when you are about to save and freeze the dataset.
"""

from __future__ import annotations

from m3resp.datamodel.store import DataModelStore


def validate_store(
    store: DataModelStore, *, require_complete: bool = False
) -> list[str]:
    """Return a list of readable problem descriptions (empty if none found).

    By default only the reference checks run, so a store recorded from a live
    session passes as long as its records link up correctly. Pass
    ``require_complete=True`` before saving a finished dataset to also check
    that every record carries the descriptive fields the full data model
    expects.
    """

    problems: list[str] = []
    problems.extend(_check_references(store))
    problems.extend(_check_time_windows(store))
    if require_complete:
        problems.extend(_check_completeness(store))
    return problems


# -- reference checks (always run) -------------------------------------------


def _check_references(store: DataModelStore) -> list[str]:
    # Every SignalStream must belong to a recording session and a device that
    # the store actually holds; every derived feature must name a processing
    # run (and, when given, a source signal) that exists.
    problems = []
    for stream in store.signal_streams.values():
        if stream.session_id not in store.sessions:
            problems.append(
                f"SignalStream {stream.signal_id!r} has no matching RecordingSession"
            )
        if stream.device_id not in store.devices:
            problems.append(f"SignalStream {stream.signal_id!r} has no matching Device")
    for feature in store.derived_features.values():
        if feature.processing_run_id not in store.processing_runs:
            problems.append(
                f"DerivedFeature {feature.feature_id!r} has no matching ProcessingRun"
            )
        if (
            feature.source_signal_id is not None
            and feature.source_signal_id not in store.signal_streams
        ):
            problems.append(
                f"DerivedFeature {feature.feature_id!r} names a source signal that "
                "is not in the store"
            )
    return problems


def _check_time_windows(store: DataModelStore) -> list[str]:
    # A time window must not end before it starts.
    problems = []
    for feature in store.derived_features.values():
        start, end = feature.time_window_start, feature.time_window_end
        if start is not None and end is not None and end < start:
            problems.append(
                f"DerivedFeature {feature.feature_id!r} has a negative-duration "
                "time window"
            )
    for event in store.clinical_events.values():
        start, end = event.event_start_time, event.event_end_time
        if start is not None and end is not None and end < start:
            problems.append(
                f"ClinicalEvent {event.event_id!r} has a negative-duration time window"
            )
    return problems


# -- completeness checks (only when require_complete=True) --------------------


def _check_completeness(store: DataModelStore) -> list[str]:
    problems: list[str] = []
    problems.extend(_complete_signal_streams(store))
    problems.extend(_complete_data_files(store))
    problems.extend(_complete_processing_runs(store))
    problems.extend(_complete_clinical_events(store))
    problems.extend(_complete_derived_features(store))
    return problems


def _complete_signal_streams(store: DataModelStore) -> list[str]:
    # "Every SignalStream has a unit, sampling frequency, start time, and device."
    problems = []
    for stream in store.signal_streams.values():
        for field_name in ("unit", "sampling_frequency_hz", "start_time"):
            if getattr(stream, field_name) is None:
                problems.append(
                    f"SignalStream {stream.signal_id!r} is missing '{field_name}'"
                )
    return problems


def _complete_data_files(store: DataModelStore) -> list[str]:
    # "Every DataFile has a checksum and file format."
    # "Every processed or derived file references a ProcessingRun."
    problems = []
    for data_file in store.data_files.values():
        if data_file.checksum_sha256 is None:
            problems.append(f"DataFile {data_file.file_id!r} is missing a checksum")
        if data_file.file_format is None:
            problems.append(f"DataFile {data_file.file_id!r} is missing a file format")
        if data_file.file_role in ("processed", "derived") and not store.runs_for_file(
            data_file.file_id
        ):
            problems.append(
                f"DataFile {data_file.file_id!r} ({data_file.file_role}) has no "
                "ProcessingRun referencing it"
            )
    return problems


def _complete_processing_runs(store: DataModelStore) -> list[str]:
    # "Every ProcessingRun references its input files."
    problems = []
    for run in store.processing_runs.values():
        if not run.input_file_ids and run.status == "succeeded":
            problems.append(
                f"ProcessingRun {run.processing_run_id!r} references no input files"
            )
    return problems


def _complete_clinical_events(store: DataModelStore) -> list[str]:
    # "Every event has an event type and timestamp." (event_type is required by
    # the record itself already; check the timestamp.)
    problems = []
    for event in store.clinical_events.values():
        if event.event_start_time is None:
            problems.append(f"ClinicalEvent {event.event_id!r} is missing a timestamp")
    return problems


def _complete_derived_features(store: DataModelStore) -> list[str]:
    # "Every derived feature defines its source signal."
    problems = []
    for feature in store.derived_features.values():
        if feature.source_signal_id is None:
            problems.append(
                f"DerivedFeature {feature.feature_id!r} is missing its source signal"
            )
    return problems
