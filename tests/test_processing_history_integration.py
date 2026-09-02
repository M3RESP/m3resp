"""Tests for Phase 5 of the pipeline-structure plan: recording every
executed step onto ``M3Session.processing_history`` regardless of whether a
``DataModelRecorder`` is attached (5.1), recursively recording nested native
results (5.2), and linking input files onto the pipeline's ``ProcessingRun``
(5.3). See plan/stage2/3_pipeline_structure_implementation_plan.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, QualityFlag, Signal
from m3resp.datamodel.recorder import DataModelRecorder
from m3resp.workflows import PipelineExecutionError, register_step, run_pipeline
from m3resp.workflows.registry import STEP_REGISTRY


@pytest.fixture
def _history_steps():
    @register_step("history_test.ok", writes=("x",))
    def _ok(*, n: int = 1) -> dict[str, Any]:
        return {"x": n}

    @register_step("history_test.fail", writes=())
    def _fail() -> dict[str, Any]:
        raise ValueError("boom")

    yield
    STEP_REGISTRY.pop("history_test.ok", None)
    STEP_REGISTRY.pop("history_test.fail", None)


# --------------------------------------------------------------------------- #
# 5.1: every executed step recorded onto session.processing_history          #
# --------------------------------------------------------------------------- #


def test_successful_step_is_recorded_without_any_datamodel_attached(_history_steps):
    """Phase 5.1: "the recorder must not require export to be enabled for
    in-memory provenance" - here there isn't even a recorder attached."""

    session = M3Session()
    run_pipeline(
        {"name": "p", "steps": [{"uses": "history_test.ok", "with": {"n": 7}}]},
        session=session,
    )
    assert len(session.processing_history) == 1
    step = session.processing_history.steps[0]
    assert step.name == "history_test.ok"
    assert step.status == "succeeded"
    assert step.parameters == {"n": 7}
    assert step.output_keys == ["x"]


def test_failed_step_is_still_recorded_with_failed_status(_history_steps):
    session = M3Session()
    with pytest.raises(PipelineExecutionError):
        run_pipeline(
            {"name": "p", "steps": [{"uses": "history_test.fail"}]}, session=session
        )
    assert len(session.processing_history) == 1
    assert session.processing_history.steps[0].status == "failed"


def test_only_executed_steps_are_recorded_not_the_one_that_failed_after(
    _history_steps,
):
    session = M3Session()
    with pytest.raises(PipelineExecutionError):
        run_pipeline(
            {
                "name": "p",
                "steps": [
                    {"uses": "history_test.ok", "with": {"n": 1}},
                    {"uses": "history_test.fail"},
                ],
            },
            session=session,
        )
    assert [s.name for s in session.processing_history] == [
        "history_test.ok",
        "history_test.fail",
    ]
    assert [s.status for s in session.processing_history] == ["succeeded", "failed"]


def test_processing_history_does_not_create_a_second_processing_run(_history_steps):
    """Phase 5.1: "avoid duplicate processing runs" - the per-step
    ProcessingHistory log must not add extra ProcessingRun rows beyond the
    one record_pipeline_result already creates for the whole pipeline."""

    session = M3Session()
    session.datamodel = DataModelRecorder(session)
    run_pipeline(
        {
            "name": "p",
            "steps": [
                {"uses": "history_test.ok", "with": {"n": 1}},
                {"uses": "history_test.ok", "with": {"n": 2}, "out": {"x": "x2"}},
            ],
        },
        session=session,
    )
    assert len(session.processing_history) == 2  # one per step
    assert len(session.datamodel.store.processing_runs) == 1  # one per pipeline


# --------------------------------------------------------------------------- #
# 5.2: recursive native-result recording                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def _nested_result_step():
    @register_step("history_test.nested", writes=("bundle",))
    def _nested() -> dict[str, Any]:
        return {
            "bundle": {
                "features": [
                    ParameterResult(name="p1", value=1.0, modality="eit"),
                    ParameterResult(name="p2", value=2.0, modality="eit"),
                ],
                "flags": [
                    QualityFlag(
                        name="q1", passed=True, severity="info", modality="eit"
                    ),
                    QualityFlag(
                        name="q2", passed=False, severity="warning", modality="eit"
                    ),
                ],
            }
        }

    yield
    STEP_REGISTRY.pop("history_test.nested", None)


def test_dict_of_lists_of_native_results_is_recorded_recursively(_nested_result_step):
    session = M3Session()
    session.datamodel = DataModelRecorder(session)
    result = run_pipeline(
        {"name": "p", "steps": [{"uses": "history_test.nested"}]}, session=session
    )

    store = session.datamodel.store
    assert len(store.derived_features) == 2
    assert {f.value for f in store.derived_features.values()} == {1.0, 2.0}
    assert len(store.quality_annotations) == 2

    run = store.processing_runs[result.processing_run_id]
    nested_provenance = run.parameters["outputs"]["bundle"]
    assert len(nested_provenance["features"]) == 2
    # QualityFlag items have no JSON provenance entry of their own (by
    # design, unchanged from the pre-5.2 flat behavior) even though they are
    # still recorded as QualityAnnotations above.
    assert "flags" not in nested_provenance


def test_a_lone_quality_flag_still_records_without_a_provenance_entry(
    _nested_result_step,
):
    """Same non-list/dict path as before Phase 5.2 - regression guard."""

    session = M3Session()
    session.datamodel = DataModelRecorder(session)
    run_pipeline(
        {"name": "p", "steps": [{"uses": "history_test.nested"}]}, session=session
    )
    assert len(session.datamodel.store.quality_annotations) == 2


# --------------------------------------------------------------------------- #
# 5.3: input files linked onto the pipeline's ProcessingRun                  #
# --------------------------------------------------------------------------- #


@pytest.fixture
def _signal_producing_step():
    @register_step("history_test.load_like", writes=("raw_signal",))
    def _load_like() -> dict[str, Any]:
        return {
            "raw_signal": Signal(
                values=np.array([1.0, 2.0]),
                time=np.array([0.0, 1.0]),
                modality="eit",
                source="fake.bin",
            )
        }

    yield
    STEP_REGISTRY.pop("history_test.load_like", None)


def test_pipeline_result_links_produced_signal_files_to_the_run(
    _signal_producing_step, tmp_path
):
    session = M3Session()
    session.datamodel = DataModelRecorder(session)
    fake_path = tmp_path / "fake.bin"
    fake_path.write_text("data")

    result = session.datamodel.record_signal(
        Signal(values=np.array([1.0]), time=np.array([0.0]), modality="eit"),
        file_path=fake_path,
    )
    assert result is not None

    pipeline_result = run_pipeline(
        {"name": "p", "steps": [{"uses": "history_test.load_like"}]}, session=session
    )
    run = session.datamodel.store.processing_runs[pipeline_result.processing_run_id]
    expected_file_id = session.datamodel.store.files_for_signal(result.signal_id)[
        0
    ].file_id
    assert expected_file_id in run.input_file_ids


def test_pipeline_result_has_empty_input_file_ids_when_nothing_was_loaded(
    _history_steps,
):
    session = M3Session()
    session.datamodel = DataModelRecorder(session)
    result = run_pipeline(
        {"name": "p", "steps": [{"uses": "history_test.ok"}]}, session=session
    )
    run = session.datamodel.store.processing_runs[result.processing_run_id]
    assert run.input_file_ids == []
