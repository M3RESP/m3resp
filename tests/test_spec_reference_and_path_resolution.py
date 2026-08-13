"""Tests for Phase 2.4 (recursive ``@ref`` resolution + ``@@`` escaping) and
2.5 (path-parameter resolution from the spec root) of the pipeline-structure
plan (plan/stage2/3_pipeline_structure_implementation_plan.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from m3resp.core.exceptions import PipelineSpecError
from m3resp.workflows.context import PipelineContext
from m3resp.workflows.engine import run_pipeline, validate_spec
from m3resp.workflows.registry import STEP_REGISTRY, StepParameter, register_step
from m3resp.workflows.spec import load_spec


def _ctx(**inputs: Any) -> PipelineContext:
    from m3resp.core.session import M3Session

    return PipelineContext(session=M3Session(), inputs=inputs)


# --------------------------------------------------------------------------- #
# 2.4: recursive @ref resolution and @@ escaping                             #
# --------------------------------------------------------------------------- #


def test_resolve_input_still_handles_whole_string_reference():
    ctx = _ctx(x=42)
    assert ctx.resolve_input("@x") == 42


def test_resolve_input_recurses_into_lists():
    ctx = _ctx(a=1, b=2)
    assert ctx.resolve_input(["@a", "literal", "@b"]) == [1, "literal", 2]


def test_resolve_input_recurses_into_nested_mappings():
    ctx = _ctx(a=1, b=2)
    resolved = ctx.resolve_input({"x": "@a", "nested": {"y": "@b", "z": [1, "@a"]}})
    assert resolved == {"x": 1, "nested": {"y": 2, "z": [1, 1]}}


def test_double_at_escapes_a_literal_at_string():
    ctx = _ctx()
    assert ctx.resolve_input("@@literal") == "@literal"


def test_escaped_literal_inside_a_list_is_not_resolved():
    ctx = _ctx(a=1)
    assert ctx.resolve_input(["@a", "@@a"]) == [1, "@a"]


def test_resolve_input_still_rejects_unknown_top_level_reference():
    ctx = _ctx()
    with pytest.raises(PipelineSpecError, match="unknown input"):
        ctx.resolve_input("@missing")


def test_resolve_input_rejects_unknown_reference_nested_in_a_mapping():
    ctx = _ctx()
    with pytest.raises(PipelineSpecError, match="unknown input"):
        ctx.resolve_input({"nested": ["@missing"]})


def test_non_string_and_non_container_values_pass_through_unchanged():
    ctx = _ctx()
    assert ctx.resolve_input(42) == 42
    assert ctx.resolve_input(None) is None
    assert ctx.resolve_input(True) is True


@pytest.fixture
def _nested_ref_step():
    @register_step(
        "ref_test.echo",
        writes=("value",),
    )
    def _echo(*, value: Any) -> dict[str, Any]:
        return {"value": value}

    yield
    STEP_REGISTRY.pop("ref_test.echo", None)


def test_validate_spec_rejects_unknown_reference_nested_in_with_block(
    _nested_ref_step,
):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "ref_test.echo", "with": {"value": ["@missing"]}},
            ],
        }
    )
    with pytest.raises(PipelineSpecError, match="unknown input '@missing'"):
        validate_spec(spec)


def test_run_pipeline_resolves_nested_references_end_to_end(_nested_ref_step):
    spec = {
        "name": "p",
        "inputs": {"a": 1, "b": 2},
        "steps": [
            {
                "uses": "ref_test.echo",
                "with": {"value": {"x": "@a", "y": ["@b", "@@a"]}},
            }
        ],
    }
    result = run_pipeline(spec)
    assert result.value("value") == {"x": 1, "y": [2, "@a"]}


# --------------------------------------------------------------------------- #
# 2.5: path-parameter resolution from the spec root                          #
# --------------------------------------------------------------------------- #


@pytest.fixture
def _path_echo_step():
    @register_step(
        "path_test.echo",
        writes=("resolved_path",),
        parameters=(
            StepParameter(
                name="p",
                value_type="path",
                required=True,
                path_kind="file",
                description="x",
            ),
        ),
    )
    def _echo(*, p: str) -> dict[str, Any]:
        return {"resolved_path": p}

    yield
    STEP_REGISTRY.pop("path_test.echo", None)


def test_relative_path_parameter_resolves_against_spec_root(_path_echo_step, tmp_path):
    raw = {
        "name": "p",
        "steps": [{"uses": "path_test.echo", "with": {"p": "sub/file.txt"}}],
    }
    spec = load_spec(raw, root=tmp_path)
    result = run_pipeline(spec)
    assert result.value("resolved_path") == str(
        (tmp_path / "sub" / "file.txt").resolve()
    )


def test_absolute_path_parameter_is_left_absolute(_path_echo_step, tmp_path):
    absolute = str(tmp_path / "abs.txt")
    raw = {
        "name": "p",
        "steps": [{"uses": "path_test.echo", "with": {"p": absolute}}],
    }
    spec = load_spec(raw, root=tmp_path / "unrelated" / "dir")
    result = run_pipeline(spec)
    assert result.value("resolved_path") == str(Path(absolute).resolve())


def test_at_referenced_relative_path_input_also_resolves_against_spec_root(
    _path_echo_step, tmp_path
):
    raw = {
        "name": "p",
        "inputs": {"my_file": "data/thing.bin"},
        "steps": [{"uses": "path_test.echo", "with": {"p": "@my_file"}}],
    }
    spec = load_spec(raw, root=tmp_path)
    result = run_pipeline(spec)
    assert result.value("resolved_path") == str(
        (tmp_path / "data" / "thing.bin").resolve()
    )


def test_path_resolution_defaults_to_cwd_for_dict_specs_without_root(
    _path_echo_step,
):
    raw = {
        "name": "p",
        "steps": [{"uses": "path_test.echo", "with": {"p": "relative.txt"}}],
    }
    spec = load_spec(raw)
    result = run_pipeline(spec)
    assert result.value("resolved_path") == str((Path.cwd() / "relative.txt").resolve())
