"""Tests for Phase 2 of the pipeline-structure plan: the versioned, strict
spec schema (plan/stage2/3_pipeline_structure_implementation_plan.md).

Covers: schema_version-gated legacy vs. strict parsing, strict booleans,
unknown top-level/step keys, unsupported schema_version, and step id
generation/uniqueness (2.1, 2.2, 2.3). Recursive @ref resolution (2.4) and
path-parameter resolution from the spec root (2.5) are separate, later work.
"""

from __future__ import annotations

import pytest

from m3resp.core.exceptions import PipelineSpecError
from m3resp.workflows.spec import load_spec

_MINIMAL_STEPS = [{"uses": "t.make", "with": {"value": 1}}]


# --------------------------------------------------------------------------- #
# Legacy mode (no schema_version): preserved behavior, but warns             #
# --------------------------------------------------------------------------- #


def test_legacy_spec_has_no_schema_version_and_is_legacy():
    spec = load_spec({"name": "p", "steps": _MINIMAL_STEPS})
    assert spec.schema_version is None
    assert spec.is_legacy is True


def test_legacy_spec_still_coerces_non_bool_output_fields_with_warning():
    with pytest.warns(FutureWarning, match="timestamped"):
        spec = load_spec(
            {
                "name": "p",
                "steps": _MINIMAL_STEPS,
                "outputs": {"timestamped": "false"},
            }
        )
    assert spec.outputs.timestamped is True  # bool("false") is True


def test_legacy_spec_accepts_real_booleans_without_warning(recwarn):
    spec = load_spec(
        {"name": "p", "steps": _MINIMAL_STEPS, "outputs": {"timestamped": False}}
    )
    assert spec.outputs.timestamped is False
    assert not any(w.category is FutureWarning for w in recwarn.list)


def test_legacy_spec_ignores_unknown_top_level_keys():
    spec = load_spec({"name": "p", "steps": _MINIMAL_STEPS, "totally_unknown": 123})
    assert spec.name == "p"


# --------------------------------------------------------------------------- #
# Versioned mode: strict                                                     #
# --------------------------------------------------------------------------- #


def test_versioned_spec_is_not_legacy():
    spec = load_spec({"schema_version": 1, "name": "p", "steps": _MINIMAL_STEPS})
    assert spec.schema_version == 1
    assert spec.is_legacy is False


def test_versioned_spec_rejects_unknown_top_level_key():
    with pytest.raises(PipelineSpecError, match="bogus"):
        load_spec(
            {
                "schema_version": 1,
                "name": "p",
                "steps": _MINIMAL_STEPS,
                "bogus": 1,
            }
        )


def test_versioned_spec_rejects_unknown_step_key():
    with pytest.raises(PipelineSpecError, match="bogus"):
        load_spec(
            {
                "schema_version": 1,
                "name": "p",
                "steps": [{"uses": "t.make", "bogus": 1}],
            }
        )


def test_versioned_spec_rejects_non_boolean_output_field():
    with pytest.raises(PipelineSpecError, match="timestamped"):
        load_spec(
            {
                "schema_version": 1,
                "name": "p",
                "steps": _MINIMAL_STEPS,
                "outputs": {"timestamped": "false"},
            }
        )


def test_versioned_spec_rejects_unsupported_schema_version():
    with pytest.raises(PipelineSpecError, match="Unsupported schema_version"):
        load_spec({"schema_version": 2, "name": "p", "steps": _MINIMAL_STEPS})


def test_versioned_spec_rejects_non_fail_fast_error_policy():
    with pytest.raises(PipelineSpecError):
        load_spec(
            {
                "schema_version": 1,
                "name": "p",
                "steps": _MINIMAL_STEPS,
                "execution": {"error_policy": "retry"},
            }
        )


def test_versioned_spec_accepts_metadata_and_execution_seed():
    spec = load_spec(
        {
            "schema_version": 1,
            "name": "p",
            "steps": _MINIMAL_STEPS,
            "metadata": {"site": "hospital-a"},
            "execution": {"seed": 42},
        }
    )
    assert spec.metadata == {"site": "hospital-a"}
    assert spec.execution.seed == 42
    assert spec.execution.error_policy == "fail_fast"


def test_versioned_spec_requires_non_empty_steps():
    with pytest.raises(PipelineSpecError):
        load_spec({"schema_version": 1, "name": "p", "steps": []})


# --------------------------------------------------------------------------- #
# Step ids: generated, stable, unique (2.3) - applies in both modes          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("schema_version", [None, 1])
def test_generated_step_ids_are_stable_and_position_based(schema_version):
    raw = {"name": "p", "steps": [{"uses": "eit.load"}, {"uses": "eit.slice"}]}
    if schema_version is not None:
        raw["schema_version"] = schema_version
    spec = load_spec(raw)
    assert spec.steps[0].id == "step_000_eit_load"
    assert spec.steps[1].id == "step_001_eit_slice"


@pytest.mark.parametrize("schema_version", [None, 1])
def test_explicit_step_id_is_preserved(schema_version):
    raw = {"name": "p", "steps": [{"uses": "t.make", "id": "my-step"}]}
    if schema_version is not None:
        raw["schema_version"] = schema_version
    spec = load_spec(raw)
    assert spec.steps[0].id == "my-step"


@pytest.mark.parametrize("schema_version", [None, 1])
def test_duplicate_explicit_step_ids_are_rejected(schema_version):
    raw = {
        "name": "p",
        "steps": [
            {"uses": "t.make", "id": "dup"},
            {"uses": "t.make", "id": "dup"},
        ],
    }
    if schema_version is not None:
        raw["schema_version"] = schema_version
    with pytest.raises(PipelineSpecError, match="Duplicate step id"):
        load_spec(raw)


def test_reordering_a_spec_changes_generated_ids():
    forward = load_spec({"name": "p", "steps": [{"uses": "a.one"}, {"uses": "b.two"}]})
    backward = load_spec({"name": "p", "steps": [{"uses": "b.two"}, {"uses": "a.one"}]})
    assert [s.id for s in forward.steps] != [s.id for s in backward.steps]
