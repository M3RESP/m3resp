"""Stage 2 wrapper tests: M3Session/pipeline activity -> DataModelStore."""

from __future__ import annotations

from typing import Any

import pytest

from m3resp.adapters import EITProcessingAdapter, ReSurfEMGAdapter
from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, ProcessingStep, QualityFlag, Signal
from m3resp.datamodel import (
    DataModelRecorder,
    DataModelStore,
    ProcessingRun,
    export_store,
    validate_store,
)
from m3resp.workflows import register_step, run_pipeline


@pytest.fixture(autouse=True)
def _temp_steps():
    from m3resp.workflows.registry import STEP_REGISTRY

    @register_step("t.constant", writes=("tidal_impedance_variation",))
    def _constant(*, value: float) -> dict[str, Any]:
        return {"tidal_impedance_variation": value}

    yield
    STEP_REGISTRY.pop("t.constant", None)


def test_recorder_mirrors_loads_and_provenance_into_store():
    session = M3Session(
        eit_adapter=EITProcessingAdapter(
            loader=lambda path, vendor=None, **kwargs: {"path": path, "vendor": vendor}
        ),
        emg_adapter=ReSurfEMGAdapter(loader=lambda path, **kwargs: {"path": path}),
    )
    store = DataModelStore()
    session.datamodel = DataModelRecorder(session, store)

    session.load_eit("subject.eit", vendor="sentec")
    session.load_emg("subject.edf")

    case = session.datamodel.case
    recording_session = session.datamodel.recording_session

    assert store.sessions_for_case(case.case_id) == [recording_session]
    streams = store.streams_for_session(recording_session.session_id)
    assert {s.signal_type for s in streams} == {"eit_waveform", "emg_raw"}

    eit_stream = next(s for s in streams if s.signal_type == "eit_waveform")
    files = store.files_for_signal(eit_stream.signal_id)
    assert [f.file_path for f in files] == ["subject.eit"]

    run_names = {run.pipeline_name for run in store.processing_runs.values()}
    assert run_names == {"load_eit", "load_emg"}


def test_pipeline_run_populates_derived_features_when_datamodel_attached():
    session = M3Session()
    store = DataModelStore()
    session.datamodel = DataModelRecorder(session, store)

    spec = {
        "name": "demo",
        "steps": [{"uses": "t.constant", "with": {"value": 1.23}}],
    }
    run_pipeline(spec, session=session)

    features = [
        f
        for f in store.derived_features.values()
        if f.feature_name == "tidal_impedance_variation"
    ]
    assert len(features) == 1
    assert features[0].value == 1.23

    run = store.processing_runs[features[0].processing_run_id]
    assert run.pipeline_name == "demo"


def test_record_signal_materializes_signal_stream_and_data_file():
    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)

    signal = Signal(
        values=[1.0, 2.0, 3.0],
        time=[0.0, 1.0, 2.0],
        modality="eit",
        channel="global_impedance",
        unit="a.u.",
        sample_frequency=1.0,
    )

    stream = recorder.record_signal(signal, file_path="subject.eit", file_role="raw")

    assert stream.signal_type == "eit_waveform"
    assert stream.unit == "a.u."
    assert stream.sampling_frequency_hz == 1.0
    files = store.files_for_signal(stream.signal_id)
    assert [f.file_path for f in files] == ["subject.eit"]


def _airway_pressure(channel: str) -> Signal:
    return Signal(
        values=[1.0, 2.0, 3.0],
        time=[0.0, 1.0, 2.0],
        modality="ventilator",
        category="airway_pressure",
        channel=channel,
        unit="cmH2O",
        sample_frequency=1.0,
    )


def test_two_instruments_recording_one_quantity_stay_separate():
    """A ventilator export and an EIT file's Medibus channels both carry an
    airway pressure. They are two instruments, not two channels of one, so
    they get their own device and their own stream rather than the second
    overwriting the first."""

    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)

    primary = recorder.record_signal(_airway_pressure("pressure"))
    second = recorder.record_signal(_airway_pressure("pressure__pod"))

    assert primary.signal_id != second.signal_id
    assert primary.device_id != second.device_id
    assert len(store.signal_streams) == 2
    assert len(store.devices) == 2


def test_a_result_naming_no_instrument_resolves_to_the_primary_recording():
    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)
    primary = recorder.record_signal(_airway_pressure("pressure"))
    recorder.record_signal(_airway_pressure("pressure__pod"))
    run = store.add_processing_run(ProcessingRun(pipeline_name="demo"))

    feature = recorder.record_parameter(
        ParameterResult(
            name="peak_pressure",
            value=22.0,
            modality="ventilator",
            category="airway_pressure",
            unit="cmH2O",
        ),
        processing_run_id=run.processing_run_id,
    )

    assert feature.source_signal_ids == [primary.signal_id]


def test_the_second_instrument_is_reachable_by_name():
    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)
    recorder.record_signal(_airway_pressure("pressure"))
    second = recorder.record_signal(_airway_pressure("pressure__pod"))

    resolved = recorder._lookup_signal_id("ventilator", "airway_pressure", "pod")
    assert resolved == second.signal_id


def test_reprocessing_one_instrument_replaces_its_own_stream():
    """Recording the same channel again (its filtered version, say) still
    replaces that instrument's entry, as it did before instruments were told
    apart - only a *different* instrument is kept separate."""

    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)
    recorder.record_signal(_airway_pressure("pressure"))
    reprocessed = recorder.record_signal(_airway_pressure("pressure"))
    run = store.add_processing_run(ProcessingRun(pipeline_name="demo"))

    feature = recorder.record_parameter(
        ParameterResult(
            name="peak_pressure",
            value=22.0,
            modality="ventilator",
            category="airway_pressure",
            unit="cmH2O",
        ),
        processing_run_id=run.processing_run_id,
    )

    assert feature.source_signal_ids == [reprocessed.signal_id]


def test_record_parameter_and_quality_flag_link_to_recorded_signal():
    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)
    stream = recorder.record_signal(
        Signal(values=[1.0], time=[0.0], modality="eit", unit="a.u.")
    )
    run = store.add_processing_run(ProcessingRun(pipeline_name="demo"))

    feature = recorder.record_parameter(
        ParameterResult(name="tiv", value=1.5, modality="eit", unit="a.u."),
        processing_run_id=run.processing_run_id,
    )
    annotation = recorder.record_quality_flag(
        QualityFlag(name="snr_check", passed=True, severity="info", modality="eit")
    )

    assert feature.source_signal_ids == [stream.signal_id]
    assert feature.value == 1.5
    assert annotation.target_type == "signal"
    assert annotation.target_id == stream.signal_id
    assert annotation.quality_label == "valid"


def test_record_quality_flag_falls_back_to_session_target_without_a_signal():
    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)

    annotation = recorder.record_quality_flag(
        QualityFlag(name="global_check", passed=False, severity="warning")
    )

    assert annotation.target_type == "session"
    assert annotation.target_id == recorder.recording_session.session_id
    assert annotation.quality_label == "suspect"


def test_record_processing_step_resolves_input_file_ids_from_recorded_signals():
    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)
    stream = recorder.record_signal(
        Signal(values=[1.0], time=[0.0], modality="eit"), file_path="subject.eit"
    )

    step = ProcessingStep(
        name="preprocess_eit", input_keys=["eit"], output_keys=["filtered_eit"]
    )
    run = recorder.record_processing_step(step)

    expected_file_id = store.files_for_signal(stream.signal_id)[0].file_id
    assert run.input_file_ids == [expected_file_id]


def test_pipeline_result_prefers_layer1_objects_over_bare_scalars():
    from m3resp.workflows.registry import STEP_REGISTRY

    @register_step("t.layer1", writes=("tiv", "quality"))
    def _layer1(**kwargs: Any) -> dict[str, Any]:
        return {
            "tiv": ParameterResult(name="tiv", value=2.5, modality="eit", unit="a.u."),
            "quality": QualityFlag(
                name="snr_check", passed=True, severity="info", modality="eit"
            ),
        }

    try:
        session = M3Session()
        store = DataModelStore()
        session.datamodel = DataModelRecorder(session, store)
        session.datamodel.record_signal(
            Signal(values=[1.0], time=[0.0], modality="eit")
        )

        run_pipeline({"name": "demo", "steps": [{"uses": "t.layer1"}]}, session=session)

        features = [
            f for f in store.derived_features.values() if f.feature_name == "tiv"
        ]
        assert len(features) == 1
        assert features[0].value == 2.5
        assert any(
            a.annotator_ref == "snr_check" for a in store.quality_annotations.values()
        )
    finally:
        STEP_REGISTRY.pop("t.layer1", None)


def test_export_store_writes_one_json_file_per_table(tmp_path):
    session = M3Session(
        eit_adapter=EITProcessingAdapter(
            loader=lambda path, vendor=None, **kwargs: {"path": path}
        )
    )
    store = DataModelStore()
    session.datamodel = DataModelRecorder(session, store)
    session.load_eit("subject.eit")

    written = export_store(store, tmp_path)

    assert written["sessions"].exists()
    assert written["signal_streams"].exists()
    # A store recorded from a live session links up correctly, so the default
    # reference checks pass...
    assert validate_store(store) == []
    # ...but it has not been given wall-clock start times or file checksums yet,
    # so the stricter completeness check reports those expected gaps.
    assert validate_store(store, require_complete=True)


def test_record_signal_skips_signal_with_unrecordable_modality():
    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)

    # The default modality is "unknown", which the data model has no stream
    # type for: the signal is skipped (returns None) instead of raising.
    stream = recorder.record_signal(Signal(values=[1.0, 2.0], time=[0.0, 1.0]))

    assert stream is None
    assert store.signal_streams == {}


def test_export_store_survives_array_valued_parameters(tmp_path):
    import json

    import numpy as np

    from m3resp.core.provenance import record

    session = M3Session()
    store = DataModelStore()
    session.datamodel = DataModelRecorder(session, store)

    # A ventilator array passed as a postprocessing argument would otherwise
    # land unserializable in ProcessingRun.parameters and break export.
    session.datamodel.record_provenance(
        record("postprocess_emg", "emg", ventilator=np.arange(5.0), peep=5.0)
    )

    written = export_store(store, tmp_path)  # must not raise

    rows = json.loads(written["processing_runs"].read_text())["rows"]
    params = next(
        r["parameters"] for r in rows if r["pipeline_name"] == "postprocess_emg"
    )
    assert params["peep"] == 5.0
    assert params["ventilator"].startswith("<array shape=(5,)")
