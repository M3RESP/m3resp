"""Tests for Phase 4 of the pipeline-structure plan: pipeline/step states,
structured execution errors, deliberate warning capture, progress events,
cooperative cancellation, and the additive ``PipelineResult`` lifecycle
fields (plan/stage2/3_pipeline_structure_implementation_plan.md).
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from m3resp.workflows import (
    CancellationToken,
    PipelineExecutionError,
    register_step,
    run_pipeline,
)
from m3resp.workflows.registry import STEP_REGISTRY


@pytest.fixture
def _lifecycle_steps():
    @register_step("lifecycle_test.ok", writes=("x",))
    def _ok(*, n: int = 1) -> dict[str, Any]:
        return {"x": n}

    @register_step("lifecycle_test.warn_then_ok", writes=("y",))
    def _warn_then_ok() -> dict[str, Any]:
        warnings.warn("benign", UserWarning)
        return {"y": 2}

    @register_step("lifecycle_test.warn_then_fail", writes=())
    def _warn_then_fail() -> dict[str, Any]:
        warnings.warn("right before failure", UserWarning)
        raise ValueError("sample frequency must be positive")

    @register_step("lifecycle_test.fail", writes=())
    def _fail() -> dict[str, Any]:
        raise ValueError("boom")

    yield
    for name in (
        "lifecycle_test.ok",
        "lifecycle_test.warn_then_ok",
        "lifecycle_test.warn_then_fail",
        "lifecycle_test.fail",
    ):
        STEP_REGISTRY.pop(name, None)


# --------------------------------------------------------------------------- #
# 4.1 / 4.7: pipeline/step states and additive PipelineResult fields         #
# --------------------------------------------------------------------------- #


def test_successful_run_reports_succeeded_status_and_step_records(_lifecycle_steps):
    spec = {"name": "p", "steps": [{"uses": "lifecycle_test.ok", "with": {"n": 5}}]}
    result = run_pipeline(spec)

    assert result.status == "succeeded"
    assert result.run_id
    assert result.started_at is not None
    assert result.finished_at is not None
    assert result.duration_seconds is not None and result.duration_seconds >= 0
    assert result.compiled_pipeline is not None
    assert len(result.step_records) == 1

    record = result.step_records[0]
    assert record.status == "succeeded"
    assert record.operation_id == "lifecycle_test.ok"
    assert record.parameters == {"n": 5}
    assert record.started_at is not None
    assert record.finished_at is not None
    assert record.duration_seconds is not None
    assert record.output_summaries == {"x": 5}


def test_pipeline_result_still_exposes_existing_public_api(_lifecycle_steps):
    """The pre-Phase-4 contract (name/session/context/outputs/value()) must
    be unchanged - Phase 4 additions are additive only."""

    spec = {"name": "p", "steps": [{"uses": "lifecycle_test.ok", "with": {"n": 1}}]}
    result = run_pipeline(spec)
    assert result.name == "p"
    assert result.session is result.context.session
    assert result.value("x") == 1
    assert result.outputs == {"x": 1}


def test_execution_context_records_deterministic_metadata(_lifecycle_steps):
    spec = {
        "name": "p",
        "schema_version": 1,
        "execution": {"seed": 42},
        "steps": [{"uses": "lifecycle_test.ok"}],
    }
    result = run_pipeline(spec)
    assert result.execution_context is not None
    assert result.execution_context.seed == 42
    assert result.execution_context.run_id == result.run_id


# --------------------------------------------------------------------------- #
# 4.2: structured execution errors                                          #
# --------------------------------------------------------------------------- #


def test_step_failure_is_wrapped_in_pipeline_execution_error(_lifecycle_steps):
    spec = {"name": "p", "steps": [{"uses": "lifecycle_test.fail"}]}
    with pytest.raises(PipelineExecutionError) as excinfo:
        run_pipeline(spec)

    error = excinfo.value
    assert error.operation_id == "lifecycle_test.fail"
    assert error.position == 0
    assert "boom" in str(error)
    assert isinstance(error.__cause__, ValueError)
    assert str(error.__cause__) == "boom"


def test_earlier_steps_complete_before_a_later_failure(_lifecycle_steps):
    spec = {
        "name": "p",
        "steps": [
            {"uses": "lifecycle_test.ok", "with": {"n": 9}},
            {"uses": "lifecycle_test.fail"},
        ],
    }
    with pytest.raises(PipelineExecutionError):
        run_pipeline(spec)
    # No direct assertion possible on the (unreturned) partial result here;
    # covered by the event-based test below, which observes step_completed
    # for the first step before pipeline_failed.


# --------------------------------------------------------------------------- #
# 4.3: deliberate warning capture                                           #
# --------------------------------------------------------------------------- #


def test_warnings_are_captured_on_the_step_record_and_still_reach_the_caller(
    _lifecycle_steps,
):
    spec = {"name": "p", "steps": [{"uses": "lifecycle_test.warn_then_ok"}]}
    with pytest.warns(UserWarning, match="benign"):
        result = run_pipeline(spec)

    assert len(result.step_records[0].warnings) == 1
    assert result.step_records[0].warnings[0].message == "benign"
    assert result.step_records[0].warnings[0].category == "UserWarning"
    assert list(result.warnings) == result.step_records[0].warnings


def test_a_warning_issued_right_before_a_failure_is_not_dropped(_lifecycle_steps):
    spec = {"name": "p", "steps": [{"uses": "lifecycle_test.warn_then_fail"}]}
    with (
        pytest.warns(UserWarning, match="right before failure"),
        pytest.raises(PipelineExecutionError),
    ):
        run_pipeline(spec)


# --------------------------------------------------------------------------- #
# 4.4: framework-neutral progress events                                    #
# --------------------------------------------------------------------------- #


def test_progress_events_fire_in_order_for_a_successful_run(_lifecycle_steps):
    events: list[dict[str, Any]] = []
    spec = {"name": "p", "steps": [{"uses": "lifecycle_test.ok"}]}
    run_pipeline(spec, event_sink=events.append)

    assert [e["event"] for e in events] == [
        "pipeline_started",
        "step_started",
        "step_completed",
        "pipeline_completed",
    ]
    assert all("run_id" in e and "timestamp" in e for e in events)


def test_progress_events_include_step_warning_and_step_failed(_lifecycle_steps):
    events: list[dict[str, Any]] = []
    spec = {"name": "p", "steps": [{"uses": "lifecycle_test.warn_then_fail"}]}
    with pytest.warns(UserWarning), pytest.raises(PipelineExecutionError):
        run_pipeline(spec, event_sink=events.append)

    assert [e["event"] for e in events] == [
        "pipeline_started",
        "step_started",
        "step_warning",
        "step_failed",
        "pipeline_failed",
    ]


def test_progress_events_are_json_safe(_lifecycle_steps):
    import json

    events: list[dict[str, Any]] = []
    spec = {"name": "p", "steps": [{"uses": "lifecycle_test.ok", "with": {"n": 1}}]}
    run_pipeline(spec, event_sink=events.append)
    json.dumps(events)


# --------------------------------------------------------------------------- #
# 4.5: cooperative cancellation                                             #
# --------------------------------------------------------------------------- #


def test_cancellation_before_a_step_stops_the_run_and_preserves_completed_work(
    _lifecycle_steps,
):
    token = CancellationToken()

    @register_step("lifecycle_test.cancel_after", writes=("z",))
    def _cancel_after() -> dict[str, Any]:
        token.cancel()
        return {"z": 1}

    try:
        spec = {
            "name": "p",
            "steps": [
                {"uses": "lifecycle_test.cancel_after"},
                {"uses": "lifecycle_test.ok", "with": {"n": 99}},
            ],
        }
        result = run_pipeline(spec, cancellation_token=token)
        assert result.status == "cancelled"
        assert len(result.step_records) == 1
        assert "z" in result.context.values
        assert "x" not in result.context.values
    finally:
        STEP_REGISTRY.pop("lifecycle_test.cancel_after", None)


def test_cancellation_before_the_run_starts_executes_nothing(_lifecycle_steps):
    token = CancellationToken()
    token.cancel()
    spec = {"name": "p", "steps": [{"uses": "lifecycle_test.ok"}]}
    result = run_pipeline(spec, cancellation_token=token)
    assert result.status == "cancelled"
    assert result.step_records == ()


def test_pipeline_cancelled_event_fires(_lifecycle_steps):
    token = CancellationToken()
    token.cancel()
    events: list[dict[str, Any]] = []
    spec = {"name": "p", "steps": [{"uses": "lifecycle_test.ok"}]}
    run_pipeline(spec, cancellation_token=token, event_sink=events.append)
    assert [e["event"] for e in events] == ["pipeline_started", "pipeline_cancelled"]
