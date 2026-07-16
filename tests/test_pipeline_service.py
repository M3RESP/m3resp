"""Tests for Phase 7.3/7.4 of the pipeline-structure plan: the
framework-neutral ``PipelineService`` facade.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from m3resp.workflows.registry import STEP_REGISTRY, register_step
from m3resp.workflows.service import PipelineService


@pytest.fixture
def _service_step():
    @register_step("service_test.ok", writes=("x",))
    def _ok(*, n: int = 1) -> dict[str, Any]:
        return {"x": n}

    @register_step("service_test.fail", writes=())
    def _fail() -> dict[str, Any]:
        raise ValueError("boom")

    yield
    STEP_REGISTRY.pop("service_test.ok", None)
    STEP_REGISTRY.pop("service_test.fail", None)


def test_list_capabilities_returns_json_safe_descriptions():
    service = PipelineService()
    capabilities = service.list_capabilities(prefix="metric.")
    assert capabilities
    assert all(c["name"].startswith("metric.") for c in capabilities)
    json.dumps(capabilities)  # must not raise


def test_describe_capability_matches_registry():
    service = PipelineService()
    description = service.describe_capability("metric.interval_cv")
    assert description["name"] == "metric.interval_cv"
    assert description["capability"] == "available"


def test_validate_pipeline_never_raises_for_an_invalid_spec():
    service = PipelineService()
    report = service.validate_pipeline({"name": "p", "steps": [{"uses": "no.such"}]})
    assert report["is_valid"] is False
    assert any(d["code"] == "unknown_step" for d in report["structural"])


def test_compile_pipeline_returns_a_json_safe_plan(_service_step):
    service = PipelineService()
    compiled = service.compile_pipeline(
        {"name": "p", "steps": [{"uses": "service_test.ok", "with": {"n": 3}}]}
    )
    assert compiled["name"] == "p"
    assert len(compiled["steps"]) == 1
    json.dumps(compiled)


def test_run_pipeline_returns_a_json_safe_summary_not_a_pipeline_result(
    _service_step,
):
    service = PipelineService()
    summary = service.run_pipeline(
        {"name": "p", "steps": [{"uses": "service_test.ok", "with": {"n": 5}}]}
    )
    assert summary["status"] == "succeeded"
    assert summary["outputs"] == {"x": 5}
    assert "context" not in summary
    assert "session" not in summary
    json.dumps(summary)  # 7.4: no raw session/context/adapter objects


def test_run_pipeline_still_raises_pipeline_execution_error_on_failure(
    _service_step,
):
    from m3resp.workflows import PipelineExecutionError

    service = PipelineService()
    with pytest.raises(PipelineExecutionError):
        service.run_pipeline({"name": "p", "steps": [{"uses": "service_test.fail"}]})


def test_run_pipeline_forwards_event_sink(_service_step):
    events: list[dict[str, Any]] = []
    service = PipelineService()
    service.run_pipeline(
        {"name": "p", "steps": [{"uses": "service_test.ok"}]},
        event_sink=events.append,
    )
    assert [e["event"] for e in events] == [
        "pipeline_started",
        "step_started",
        "step_completed",
        "pipeline_completed",
    ]
