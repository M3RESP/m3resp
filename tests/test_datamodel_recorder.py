"""Stage 2 wrapper tests: M3Session/workflow activity -> DataModelStore."""

from __future__ import annotations

import os
from typing import Any

import pytest

from m3resp.adapters import EITProcessingAdapter, ReSurfEMGAdapter
from m3resp.core.session import M3Session
from m3resp.data import (
    BreathEvent,
    Event,
    EventData,
    IntervalData,
    ParameterResult,
    PixelMask,
    ProcessingStep,
    QualityFlag,
    Signal,
)
from m3resp.datamodel import (
    DataModelRecorder,
    DataModelStore,
    ProcessingRun,
    export_store,
    validate_store,
)
from m3resp.workflows import register_step, run_workflow


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

    run_names = {run.name for run in store.processing_runs.values()}
    assert run_names == {"load_eit", "load_emg"}


def test_workflow_run_populates_derived_features_when_datamodel_attached():
    session = M3Session()
    store = DataModelStore()
    session.datamodel = DataModelRecorder(session, store)

    spec = {
        "name": "demo",
        "steps": [{"uses": "t.constant", "with": {"value": 1.23}}],
    }
    run_workflow(spec, session=session)

    features = [
        f
        for f in store.derived_features.values()
        if f.feature_name == "tidal_impedance_variation"
    ]
    assert len(features) == 1
    assert features[0].value == 1.23

    run = store.processing_runs[features[0].processing_run_id]
    assert run.name == "demo"


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

    primary = recorder.record_signal(_airway_pressure("airway_pressure"))
    second = recorder.record_signal(_airway_pressure("airway_pressure__pod"))

    assert primary.signal_id != second.signal_id
    assert primary.device_id != second.device_id
    assert len(store.signal_streams) == 2
    assert len(store.devices) == 2


def test_a_result_naming_no_instrument_resolves_to_the_primary_recording():
    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)
    primary = recorder.record_signal(_airway_pressure("airway_pressure"))
    recorder.record_signal(_airway_pressure("airway_pressure__pod"))
    run = store.add_processing_run(ProcessingRun(name="demo"))

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
    """Streams are keyed by their full channel, so the second instrument is
    reached by the qualified channel it was recorded under."""

    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)
    recorder.record_signal(_airway_pressure("airway_pressure"))
    second = recorder.record_signal(_airway_pressure("airway_pressure__pod"))

    resolved = recorder._lookup_signal_id(
        "ventilator", "airway_pressure", "airway_pressure__pod"
    )
    assert resolved == second.signal_id


def test_reprocessing_one_instrument_replaces_its_own_stream():
    """Recording the same channel again (its filtered version, say) still
    replaces that instrument's entry, as it did before instruments were told
    apart - only a *different* instrument is kept separate."""

    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)
    recorder.record_signal(_airway_pressure("airway_pressure"))
    reprocessed = recorder.record_signal(_airway_pressure("airway_pressure"))
    run = store.add_processing_run(ProcessingRun(name="demo"))

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


def test_two_eit_impedance_streams_keep_separate_attribution():
    """The global and pixel impedance are both eit/impedance and are told
    apart only by their channel. A result naming its channel must reach its
    own stream, and one naming only the modality must stay with the first
    stream recorded rather than following the most recent one."""

    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)

    global_stream = recorder.record_signal(
        Signal(
            values=[1.0, 2.0],
            time=[0.0, 1.0],
            modality="eit",
            category="impedance",
            channel="global_impedance",
        )
    )
    pixel_stream = recorder.record_signal(
        Signal(
            values=[[[1.0]], [[2.0]]],
            time=[0.0, 1.0],
            modality="eit",
            category="impedance",
            channel="pixel_impedance",
        )
    )
    assert global_stream.signal_id != pixel_stream.signal_id

    run = store.add_processing_run(ProcessingRun(name="demo"))
    pixel_feature = recorder.record_parameter(
        ParameterResult(
            name="pixel_tiv",
            value=1.0,
            modality="eit",
            category="impedance",
            channel="pixel_impedance",
        ),
        processing_run_id=run.processing_run_id,
    )
    global_feature = recorder.record_parameter(
        ParameterResult(
            name="tiv",
            value=2.0,
            modality="eit",
            category="impedance",
            channel="global_impedance",
        ),
        processing_run_id=run.processing_run_id,
    )
    unchannelled_feature = recorder.record_parameter(
        ParameterResult(name="eeli", value=3.0, modality="eit"),
        processing_run_id=run.processing_run_id,
    )

    assert pixel_feature.source_signal_ids == [pixel_stream.signal_id]
    assert global_feature.source_signal_ids == [global_stream.signal_id]
    assert unchannelled_feature.source_signal_ids == [global_stream.signal_id]


def test_record_parameter_and_quality_flag_link_to_recorded_signal():
    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)
    stream = recorder.record_signal(
        Signal(values=[1.0], time=[0.0], modality="eit", unit="a.u.")
    )
    run = store.add_processing_run(ProcessingRun(name="demo"))

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


def test_record_parameter_converts_a_numpy_scalar_to_float():
    import numpy as np

    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)
    run = store.add_processing_run(ProcessingRun(name="demo"))

    feature = recorder.record_parameter(
        ParameterResult(name="sample_count", value=np.int64(12), modality="eit"),
        processing_run_id=run.processing_run_id,
    )

    assert feature.value == 12.0


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


def test_record_processing_step_converts_bare_numpy_scalar_parameters():
    import numpy as np

    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)

    run = recorder.record_processing_step(
        ProcessingStep(
            name="detect_breaths",
            parameters={
                "minimum_samples": np.int64(12),
                "threshold": np.float32(0.25),
                "nested": {"sample_index": np.int32(4)},
            },
        )
    )

    assert run.parameters == {
        "minimum_samples": 12,
        "threshold": pytest.approx(0.25),
        "nested": {"sample_index": 4},
    }


def test_workflow_result_records_bare_numpy_integer_outputs():
    import numpy as np

    from m3resp.workflows.registry import STEP_REGISTRY

    @register_step("t.numpy_integer", writes=("sample_count",))
    def _numpy_integer(**kwargs: Any) -> dict[str, Any]:
        return {"sample_count": np.int64(12)}

    try:
        session = M3Session()
        store = DataModelStore()
        session.datamodel = DataModelRecorder(session, store)

        result = run_workflow(
            {"name": "numpy-scalar", "steps": [{"uses": "t.numpy_integer"}]},
            session=session,
        )

        features = [
            feature
            for feature in store.derived_features.values()
            if feature.processing_run_id == result.processing_run_id
        ]
        assert [(feature.feature_name, feature.value) for feature in features] == [
            ("sample_count", 12.0)
        ]
    finally:
        STEP_REGISTRY.pop("t.numpy_integer", None)


def test_workflow_result_prefers_layer1_objects_over_bare_scalars():
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

        run_workflow({"name": "demo", "steps": [{"uses": "t.layer1"}]}, session=session)

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


def test_workflow_result_records_output_provenance_for_array_valued_results():
    """`record_workflow_result` stores an output-provenance
    mapping on the run for every native result, and an array-valued
    `ParameterResult` still gets a `DerivedFeature` (with a null value, since
    the array itself lives in the parameter artifact, not the store)."""

    import numpy as np

    from m3resp.workflows.registry import STEP_REGISTRY

    @register_step("t.array_result", writes=("mask_result",))
    def _array_result(**kwargs: Any) -> dict[str, Any]:
        return {
            "mask_result": ParameterResult(
                name="mask",
                value=np.array([1.0, float("nan")]),
                modality="eit",
                unit=None,
                method="eitprocessing.TIVLungspace",
                metadata={"operation": "eit.roi_tiv_lungspace", "shape": [2]},
            )
        }

    try:
        session = M3Session()
        store = DataModelStore()
        session.datamodel = DataModelRecorder(session, store)

        result = run_workflow(
            {"name": "demo", "steps": [{"uses": "t.array_result"}]}, session=session
        )

        run = store.processing_runs[result.processing_run_id]
        outputs = run.parameters["outputs"]
        assert outputs["mask_result"]["method"] == "eitprocessing.TIVLungspace"
        assert outputs["mask_result"]["is_scalar"] is False
        assert (
            outputs["mask_result"]["metadata"]["operation"] == "eit.roi_tiv_lungspace"
        )

        features = [
            f for f in store.derived_features.values() if f.feature_name == "mask"
        ]
        assert len(features) == 1
        assert features[0].value is None
        assert features[0].processing_run_id == result.processing_run_id
    finally:
        STEP_REGISTRY.pop("t.array_result", None)


def test_workflow_result_records_provenance_for_values_per_breath_and_masks():
    """`IntervalData` and `PixelMask` outputs are stored as `DerivedFeature`s,
    and the run records how they were made."""

    import numpy as np

    from m3resp.workflows.registry import STEP_REGISTRY

    @register_step("t.grouped_results", writes=("tiv_result", "mask_result"))
    def _grouped_results(**kwargs: Any) -> dict[str, Any]:
        return {
            "tiv_result": IntervalData(
                name="continuous_tivs",
                modality="eit",
                intervals=[BreathEvent("eit", 0.0, 1.0), BreathEvent("eit", 1.0, 2.0)],
                values=np.array([1.0, 2.0]),
                method="eitprocessing.TIV",
                metadata={"operation": "eit.continuous_tiv"},
            ),
            "mask_result": PixelMask(
                name="tiv_lungspace_mask",
                values=[[1.0, np.nan]],
                method="eitprocessing.TIVLungspace",
            ),
        }

    try:
        session = M3Session()
        store = DataModelStore()
        session.datamodel = DataModelRecorder(session, store)

        result = run_workflow(
            {"name": "demo", "steps": [{"uses": "t.grouped_results"}]},
            session=session,
        )

        outputs = store.processing_runs[result.processing_run_id].parameters["outputs"]
        assert outputs["tiv_result"]["type"] == "IntervalData"
        assert outputs["tiv_result"]["count"] == 2
        assert outputs["tiv_result"]["metadata"]["operation"] == "eit.continuous_tiv"
        assert outputs["mask_result"]["type"] == "PixelMask"
        assert outputs["mask_result"]["shape"] == [1, 2]
        assert outputs["mask_result"]["method"] == "eitprocessing.TIVLungspace"

        # Numbers per breath are stored one per breath, with the breath's
        # start and end as the time window; a mask is one feature without a
        # value (its grid lives in the export archive).
        tiv_features = [
            f
            for f in store.derived_features.values()
            if f.feature_name == "continuous_tivs"
        ]
        assert [f.value for f in tiv_features] == [1.0, 2.0]
        assert [(f.time_window_start, f.time_window_end) for f in tiv_features] == [
            (0.0, 1.0),
            (1.0, 2.0),
        ]
        [mask_feature] = [
            f
            for f in store.derived_features.values()
            if f.feature_name == "tiv_lungspace_mask"
        ]
        assert mask_feature.value is None
    finally:
        STEP_REGISTRY.pop("t.grouped_results", None)


def test_missing_values_per_breath_are_stored_without_a_value_and_export_as_json(
    tmp_path,
):
    """NaN has no place in a JSON file, so a missing value per breath is
    stored as "no value" and the exported store stays readable JSON."""

    import json
    import os

    import numpy as np

    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)
    run = store.add_processing_run(ProcessingRun(name="demo"))
    breaths = [BreathEvent("eit", 0.0, 1.0), BreathEvent("eit", 1.0, 2.0)]

    features = recorder.record_interval_data(
        IntervalData(
            name="continuous_tivs",
            modality="eit",
            intervals=breaths,
            values=np.array([1.5, np.nan]),
        ),
        processing_run_id=run.processing_run_id,
    )

    assert [f.value for f in features] == [1.5, None]
    export_store(store, tmp_path)
    with open(os.path.join(tmp_path, "derived_features.json"), encoding="utf-8") as f:
        text = f.read()
    json.loads(text, parse_constant=lambda name: pytest.fail(f"{name} in JSON"))


def test_values_that_are_not_numbers_are_stored_without_a_value():
    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)
    run = store.add_processing_run(ProcessingRun(name="demo"))

    features = recorder.record_interval_data(
        IntervalData(
            name="breath_quality",
            modality="eit",
            intervals=[BreathEvent("eit", 0.0, 1.0)],
            values=["good"],
        ),
        processing_run_id=run.processing_run_id,
    )

    assert [f.value for f in features] == [None]


def test_values_per_event_use_the_event_time_as_their_window():
    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)
    run = store.add_processing_run(ProcessingRun(name="demo"))

    [feature] = recorder.record_interval_data(
        EventData(
            name="ecg_peak_amplitude",
            modality="emg",
            events=[Event(name="ecg_peak", modality="emg", time=3.0)],
            values=[0.8],
        ),
        processing_run_id=run.processing_run_id,
    )

    assert (feature.time_window_start, feature.time_window_end) == (3.0, 3.0)
    assert feature.value == 0.8


def test_each_processing_run_says_what_kind_of_run_it_was():
    """A run is a whole workflow, one step, or one session method call; the
    `kind` field keeps these apart, so filtering by kind finds only workflows."""

    from m3resp.core.provenance import record

    session = M3Session()
    store = DataModelStore()
    recorder = DataModelRecorder(session, store)

    step_run = recorder.record_processing_step(ProcessingStep(name="eit.mdn_filter"))
    action_run = recorder.record_provenance(record("postprocess_emg", modality="emg"))
    session.datamodel = recorder

    from m3resp.workflows.registry import STEP_REGISTRY

    @register_step("t.kind_noop", writes=())
    def _noop(**kwargs: Any) -> dict[str, Any]:
        return {}

    try:
        workflow = run_workflow(
            {"name": "demo", "steps": [{"uses": "t.kind_noop"}]}, session=session
        )
    finally:
        STEP_REGISTRY.pop("t.kind_noop", None)
    workflow_run = store.processing_runs[workflow.processing_run_id]

    assert (step_run.kind, step_run.name) == ("step", "eit.mdn_filter")
    assert (action_run.kind, action_run.name) == ("session_action", "postprocess_emg")
    assert (workflow_run.kind, workflow_run.name) == ("workflow", "demo")


def test_record_parameter_file_links_data_file_onto_processing_run(tmp_path):
    """Phase 5.3/7: the array archive becomes a `DataFile` with role
    'parameter', listed in the run's `ProcessingRun.parameter_file_ids`."""

    session = M3Session()
    store = DataModelStore()
    session.datamodel = DataModelRecorder(session, store)
    session.parameter_results.add(
        ParameterResult(name="mask", value=[1.0, 2.0], modality="eit")
    )
    run = store.add_processing_run(ProcessingRun(name="demo"))

    output_dir = session.export_summary(
        tmp_path, processing_run_id=run.processing_run_id
    )

    stored_run = store.processing_runs[run.processing_run_id]
    [file_id] = stored_run.parameter_file_ids

    data_file = store.data_files[file_id]
    assert data_file.file_role == "parameter"
    assert data_file.file_format == "other"
    assert data_file.file_path == str(output_dir / "parameter_result_arrays.npz")
    assert data_file.checksum_sha256 is not None
    assert (
        data_file.file_size_bytes
        == (output_dir / "parameter_result_arrays.npz").stat().st_size
    )


def test_all_exported_array_files_are_linked_to_the_run(tmp_path):
    """A run that makes values per breath, masks and array parameters lists
    all three array files; exporting again does not list them twice."""

    import numpy as np

    session = M3Session()
    store = DataModelStore()
    session.datamodel = DataModelRecorder(session, store)
    session.parameter_results.add(
        ParameterResult(name="gate", value=[1.0, 2.0], modality="emg")
    )
    session.interval_data.add(
        IntervalData(
            name="pixel_tivs",
            modality="eit",
            intervals=[BreathEvent("eit", 0.0, 1.0)],
            values=[np.ones((2, 2))],
        )
    )
    session.pixel_masks.add(PixelMask(name="lung", values=[[1.0, np.nan]]))
    run = store.add_processing_run(ProcessingRun(name="demo"))

    session.export_summary(tmp_path, processing_run_id=run.processing_run_id)
    session.export_summary(tmp_path, processing_run_id=run.processing_run_id)

    linked = [
        os.path.basename(store.data_files[file_id].file_path)
        for file_id in store.processing_runs[run.processing_run_id].parameter_file_ids
    ]
    assert sorted(linked) == [
        "interval_data_arrays.npz",
        "parameter_result_arrays.npz",
        "pixel_masks.npz",
    ]
    assert validate_store(store) == []


def test_validate_store_reports_a_run_that_names_a_missing_file():
    store = DataModelStore()
    run = store.add_processing_run(ProcessingRun(name="demo"))
    run.parameter_file_ids.append("file_missing")

    problems = validate_store(store)

    assert any("file_missing" in problem for problem in problems)


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
        record(
            "postprocess_emg",
            "emg",
            ventilator=np.arange(5.0),
            peep=5.0,
            minimum_samples=np.int64(12),
            threshold=np.float32(0.25),
            nested={"sample_index": np.int32(4)},
        )
    )

    written = export_store(store, tmp_path)  # must not raise

    rows = json.loads(written["processing_runs"].read_text())["rows"]
    params = next(r["parameters"] for r in rows if r["name"] == "postprocess_emg")
    assert params["peep"] == 5.0
    assert params["ventilator"].startswith("<array shape=(5,)")
    assert params["minimum_samples"] == 12
    assert params["threshold"] == pytest.approx(0.25)
    assert params["nested"] == {"sample_index": 4}
