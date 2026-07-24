from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from m3resp.datamodel import (
    Case,
    DataFile,
    DataModelStore,
    DataModelStoreError,
    Device,
    ProcessingRun,
    RecordingSession,
    SignalStream,
    validate_store,
)
from m3resp.datamodel.entities import DerivedFeature


def test_entities_reject_unknown_coded_values():
    with pytest.raises(ValidationError):
        Device(device_type="eit", calibration_status="not_a_real_status")


def test_device_type_accepts_any_string():
    # device_type is an open vocabulary: a new loading function can use any
    # device type string without a schema change here.
    device = Device(device_type="not_a_real_device_type")
    assert device.device_type == "not_a_real_device_type"


def test_store_enforces_foreign_keys():
    store = DataModelStore()
    stream = SignalStream(
        session_id="missing-session",
        device_id="missing-device",
        signal_type="eit_waveform",
    )
    with pytest.raises(DataModelStoreError):
        store.add_signal_stream(stream)


def test_store_builds_case_session_stream_file_run_feature_chain():
    store = DataModelStore()

    case = store.add_case(Case(external_case_ref="subject-1"))
    session = store.add_session(RecordingSession(case_id=case.case_id))
    device = store.add_device(Device(device_type="eit"))
    stream = store.add_signal_stream(
        SignalStream(
            session_id=session.session_id,
            device_id=device.device_id,
            signal_type="eit_waveform",
            unit="a.u.",
            sampling_frequency_hz=50.0,
            start_time=datetime.now(UTC).timestamp(),
        )
    )
    data_file = store.add_data_file(
        DataFile(
            session_id=session.session_id,
            signal_id=stream.signal_id,
            file_path="/tmp/subject-1.h5",
            file_format="hdf5",
            file_role="raw",
            checksum_sha256="deadbeef",
        )
    )
    run = store.add_processing_run(
        ProcessingRun(pipeline_name="demo", input_file_ids=[data_file.file_id])
    )
    feature = store.add_derived_feature(
        DerivedFeature(
            source_signal_ids=[stream.signal_id],
            processing_run_id=run.processing_run_id,
            feature_name="tidal_impedance_variation",
            value=1.23,
        )
    )

    assert store.sessions_for_case(case.case_id) == [session]
    assert store.streams_for_session(session.session_id) == [stream]
    assert store.files_for_signal(stream.signal_id) == [data_file]
    assert store.runs_for_file(data_file.file_id) == [run]
    assert store.features_for_run(run.processing_run_id) == [feature]
    # A fully populated store passes both the reference checks and the stricter
    # completeness checks.
    assert validate_store(store) == []
    assert validate_store(store, require_complete=True) == []


def test_validate_store_flags_incomplete_signal_stream():
    store = DataModelStore()
    case = store.add_case(Case())
    session = store.add_session(RecordingSession(case_id=case.case_id))
    device = store.add_device(Device(device_type="emg"))
    store.add_signal_stream(
        SignalStream(
            session_id=session.session_id,
            device_id=device.device_id,
            signal_type="emg_raw",
        )
    )

    # References are all valid, so the default (reference-only) check passes;
    # the missing descriptive fields are reported only when completeness is
    # requested.
    assert validate_store(store) == []

    problems = validate_store(store, require_complete=True)

    assert any("unit" in problem for problem in problems)
    assert any("sampling_frequency_hz" in problem for problem in problems)
    assert any("start_time" in problem for problem in problems)
