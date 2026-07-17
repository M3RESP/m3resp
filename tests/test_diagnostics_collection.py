"""Tests for the pipeline-structure plan: structured,
JSON-safe diagnostics collected in one pass, and ``validate_spec()``'s
backward-compatible raising behavior on top of them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from m3resp.core.exceptions import PipelineSpecError, UnknownStepError
from m3resp.workflows.engine import collect_diagnostics, validate_spec
from m3resp.workflows.registry import (
    ANY_ARTIFACT_TYPE,
    STEP_REGISTRY,
    StepArtifact,
    StepParameter,
    register_step,
)
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
    """report all independent validation errors in one pass,
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


@pytest.fixture
def _artifact_type_steps():
    @register_step(
        "artifact_type_test.produce_a",
        writes=("out",),
        output_artifacts=(StepArtifact(name="out", artifact_type="type_a"),),
    )
    def _produce_a() -> dict[str, Any]:
        return {"out": 1}

    @register_step(
        "artifact_type_test.produce_b",
        writes=("out",),
        output_artifacts=(StepArtifact(name="out", artifact_type="type_b"),),
    )
    def _produce_b() -> dict[str, Any]:
        return {"out": 1}

    @register_step(
        "artifact_type_test.produce_any",
        writes=("out",),
        output_artifacts=(StepArtifact(name="out", artifact_type=ANY_ARTIFACT_TYPE),),
    )
    def _produce_any() -> dict[str, Any]:
        return {"out": 1}

    @register_step(
        "artifact_type_test.consume_a",
        reads={"value": "out"},
        input_artifacts=(StepArtifact(name="value", artifact_type="type_a"),),
    )
    def _consume_a(value: Any) -> None:
        return None

    @register_step(
        "artifact_type_test.consume_untyped",
        reads={"value": "out"},
    )
    def _consume_untyped(value: Any) -> None:
        return None

    yield
    for name in (
        "produce_a",
        "produce_b",
        "produce_any",
        "consume_a",
        "consume_untyped",
    ):
        STEP_REGISTRY.pop(f"artifact_type_test.{name}", None)


def test_matching_artifact_types_report_no_diagnostic(_artifact_type_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "artifact_type_test.produce_a"},
                {"uses": "artifact_type_test.consume_a"},
            ],
        }
    )
    assert collect_diagnostics(spec) == []


def test_mismatched_artifact_types_report_a_diagnostic(_artifact_type_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "artifact_type_test.produce_b"},
                {"uses": "artifact_type_test.consume_a"},
            ],
        }
    )
    diagnostics = collect_diagnostics(spec)
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "artifact_type_mismatch"
    assert diagnostics[0].severity == "error"
    assert "type_a" in diagnostics[0].message
    assert "type_b" in diagnostics[0].message


def test_any_artifact_type_producer_is_exempt_from_the_check(_artifact_type_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "artifact_type_test.produce_any"},
                {"uses": "artifact_type_test.consume_a"},
            ],
        }
    )
    assert collect_diagnostics(spec) == []


def test_undeclared_artifact_type_on_either_side_is_skipped_not_flagged(
    _artifact_type_steps,
):
    """Additive metadata: a step with no declared input/output artifact type
    is simply not checked, since most built-in steps are still being
    backfilled rather than fully typed."""

    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "artifact_type_test.produce_b"},
                {"uses": "artifact_type_test.consume_untyped"},
            ],
        }
    )
    assert collect_diagnostics(spec) == []


@pytest.fixture
def _mutex_steps():
    @register_step(
        "mutex_test.gate",
        parameters=(
            StepParameter(name="width_seconds", value_type="number", default=None),
            StepParameter(name="width_samples", value_type="integer", default=None),
            StepParameter(name="fill_method", value_type="integer", default=1),
        ),
        mutually_exclusive_parameters=(("width_seconds", "width_samples"),),
    )
    def _gate(
        *,
        width_seconds: float | None = None,
        width_samples: int | None = None,
        fill_method: int = 1,
    ) -> dict[str, Any]:
        return {}

    yield
    STEP_REGISTRY.pop("mutex_test.gate", None)


def test_setting_only_one_of_a_mutually_exclusive_group_is_fine(_mutex_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [{"uses": "mutex_test.gate", "with": {"width_seconds": 0.1}}],
        }
    )
    assert collect_diagnostics(spec) == []


def test_setting_neither_of_a_mutually_exclusive_group_is_fine(_mutex_steps):
    spec = load_spec({"name": "p", "steps": [{"uses": "mutex_test.gate"}]})
    assert collect_diagnostics(spec) == []


def test_setting_both_of_a_mutually_exclusive_group_is_a_diagnostic(_mutex_steps):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {
                    "uses": "mutex_test.gate",
                    "with": {"width_seconds": 0.1, "width_samples": 100},
                }
            ],
        }
    )
    diagnostics = collect_diagnostics(spec)
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "mutually_exclusive_parameters_conflict"
    assert diagnostics[0].severity == "error"
    assert "width_seconds" in diagnostics[0].message
    assert "width_samples" in diagnostics[0].message


@pytest.fixture
def _reserved_key_steps():
    @register_step("reserved_test.make", writes=("result",))
    def _make() -> dict[str, Any]:
        return {"result": 1}

    yield
    STEP_REGISTRY.pop("reserved_test.make", None)


@pytest.mark.parametrize(
    "reserved_key",
    [
        "session",
        "_spec_outputs",
        "_spec_experiment",
        "_resolved_output_dir",
        "_run_timestamp",
    ],
)
def test_renaming_an_output_onto_a_reserved_key_is_a_diagnostic(
    _reserved_key_steps, reserved_key
):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {
                    "uses": "reserved_test.make",
                    "out": {"result": reserved_key},
                }
            ],
        }
    )
    diagnostics = collect_diagnostics(spec)
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "reserved_context_key_overwrite"
    assert diagnostics[0].severity == "error"
    assert reserved_key in diagnostics[0].message


def test_default_output_binding_never_collides_with_a_reserved_key(_reserved_key_steps):
    """No built-in step's own natural output name can equal a reserved key
    (enforced at registration time in registry.py), so this only ever fires
    for an explicit 'out:' rename, never a step's default binding."""

    spec = load_spec({"name": "p", "steps": [{"uses": "reserved_test.make"}]})
    assert collect_diagnostics(spec) == []


def test_emg_ecg_gating_declares_its_real_mutually_exclusive_group():
    """Locks in the actual wiring on the built-in step this feature was
    built for - a spec setting both forms should fail at validate time, not
    only once the step function itself raises at execution time."""

    import m3resp.workflows.steps  # noqa: F401
    from m3resp.workflows.registry import get_step

    definition = get_step("emg.ecg_gating")
    assert ("gate_width_seconds", "gate_width_samples") in (
        definition.mutually_exclusive_parameters
    )
