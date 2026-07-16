"""Tests for Phase 3.2/3.3/3.6 of the pipeline-structure plan: structured,
JSON-safe diagnostics collected in one pass (plan/stage2/
3_pipeline_structure_implementation_plan.md), and ``validate_spec()``'s
backward-compatible raising behavior on top of them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from m3resp.core.exceptions import PipelineSpecError, UnknownStepError
from m3resp.workflows.engine import collect_diagnostics, validate_spec
from m3resp.workflows.registry import STEP_REGISTRY, StepParameter, register_step
from m3resp.workflows.spec import load_spec


@pytest.fixture
def _diag_steps():
    @register_step(
        "diag_test.make",
        writes=("a",),
        parameters=(StepParameter(name="value", value_type="integer", required=True),),
    )
    def _make(*, value: int) -> dict[str, Any]:
        return {"a": value}

    @register_step(
        "diag_test.typed",
        reads={"x": "a"},
        writes=("result",),
        parameters=(
            StepParameter(
                name="mode",
                value_type="choice",
                choices=("fast", "slow"),
                default="fast",
            ),
            StepParameter(name="threshold", value_type="number", minimum=0, maximum=1),
        ),
    )
    def _typed(
        x: Any, *, mode: str = "fast", threshold: float | None = None
    ) -> dict[str, Any]:
        return {"result": x}

    yield
    STEP_REGISTRY.pop("diag_test.make", None)
    STEP_REGISTRY.pop("diag_test.typed", None)


def test_collect_diagnostics_returns_empty_for_a_valid_spec(_diag_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [{"uses": "diag_test.make", "with": {"value": 1}}],
        }
    )
    assert collect_diagnostics(spec) == []


def test_collect_diagnostics_reports_unknown_step():
    spec = load_spec({"name": "p", "steps": [{"uses": "no.such.step"}]})
    diagnostics = collect_diagnostics(spec)
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "unknown_step"
    assert diagnostics[0].severity == "error"


def test_collect_diagnostics_reports_unknown_input_binding(_diag_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "diag_test.make", "with": {"value": 1}},
                {
                    "uses": "diag_test.typed",
                    "in": {"x": "a", "typo_param": "a"},
                },
            ],
        }
    )
    diagnostics = collect_diagnostics(spec)
    assert any(d.code == "unknown_input_binding" for d in diagnostics)


def test_collect_diagnostics_reports_unknown_output_binding(_diag_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {
                    "uses": "diag_test.make",
                    "with": {"value": 1},
                    "out": {"not_a_real_output": "renamed"},
                },
            ],
        }
    )
    diagnostics = collect_diagnostics(spec)
    assert any(d.code == "unknown_output_binding" for d in diagnostics)


def test_collect_diagnostics_reports_duplicate_binding(_diag_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "diag_test.make", "with": {"value": 1}},
                {
                    "uses": "diag_test.typed",
                    "in": {"x": "a"},
                    "with": {"x": "@does_not_matter"},
                },
            ],
        }
    )
    diagnostics = collect_diagnostics(spec)
    assert any(d.code == "duplicate_binding" for d in diagnostics)


def test_collect_diagnostics_reports_missing_required_parameter(_diag_steps):
    spec = load_spec({"name": "p", "steps": [{"uses": "diag_test.make"}]})
    diagnostics = collect_diagnostics(spec)
    assert any(d.code == "missing_required_parameter" for d in diagnostics)


def test_collect_diagnostics_reports_parameter_type_mismatch(_diag_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [{"uses": "diag_test.make", "with": {"value": "not-an-int"}}],
        }
    )
    diagnostics = collect_diagnostics(spec)
    assert any(d.code == "parameter_type_mismatch" for d in diagnostics)


def test_collect_diagnostics_reports_parameter_choice_violation(_diag_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "diag_test.make", "with": {"value": 1}},
                {
                    "uses": "diag_test.typed",
                    "in": {"x": "a"},
                    "with": {"mode": "sideways"},
                },
            ],
        }
    )
    diagnostics = collect_diagnostics(spec)
    assert any(d.code == "parameter_choice_violation" for d in diagnostics)


def test_collect_diagnostics_reports_parameter_range_violation(_diag_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "diag_test.make", "with": {"value": 1}},
                {
                    "uses": "diag_test.typed",
                    "in": {"x": "a"},
                    "with": {"threshold": 5},
                },
            ],
        }
    )
    diagnostics = collect_diagnostics(spec)
    assert any(d.code == "parameter_range_violation" for d in diagnostics)


def test_collect_diagnostics_reports_every_independent_error_in_one_pass(_diag_steps):
    """Phase 3.6: report all independent validation errors in one pass,
    rather than forcing a user to fix one typo per run."""

    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "diag_test.make"},  # missing required 'value'
                {
                    "uses": "diag_test.typed",
                    "in": {"x": "a", "bogus": "a"},  # unknown input binding
                    "with": {"mode": "sideways"},  # choice violation
                },
            ],
        }
    )
    diagnostics = collect_diagnostics(spec)
    codes = {d.code for d in diagnostics}
    assert {
        "missing_required_parameter",
        "unknown_input_binding",
        "parameter_choice_violation",
    } <= codes
    assert len(diagnostics) >= 3


def test_diagnostic_as_dict_is_json_serializable(_diag_steps):
    spec = load_spec({"name": "p", "steps": [{"uses": "diag_test.make"}]})
    diagnostics = collect_diagnostics(spec)
    assert diagnostics
    json.dumps([d.as_dict() for d in diagnostics])


def test_validate_spec_raises_unknown_step_error_for_unknown_step():
    spec = load_spec({"name": "p", "steps": [{"uses": "no.such.step"}]})
    with pytest.raises(UnknownStepError, match="no.such.step"):
        validate_spec(spec)


def test_validate_spec_raises_pipeline_spec_error_with_first_message(_diag_steps):
    spec = load_spec({"name": "p", "steps": [{"uses": "diag_test.make"}]})
    with pytest.raises(PipelineSpecError, match="missing required parameter"):
        validate_spec(spec)
