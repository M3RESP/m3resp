"""Tests for Phase 3.1/3.5 of the pipeline-structure plan: the compiled,
read-only execution plan (``compile_pipeline``/``CompiledPipeline``) and the
structural-vs-readiness validation report (``validate_pipeline``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from m3resp.core.exceptions import PipelineSpecError, UnknownStepError
from m3resp.workflows.compiler import compile_pipeline, validate_pipeline
from m3resp.workflows.registry import STEP_REGISTRY, StepParameter, register_step
from m3resp.workflows.spec import load_spec


@pytest.fixture
def _compiler_steps():
    @register_step(
        "compiler_test.make",
        writes=("a",),
        parameters=(StepParameter(name="value", value_type="integer", default=0),),
    )
    def _make(*, value: int = 0) -> dict[str, Any]:
        return {"a": value}

    @register_step(
        "compiler_test.echo_path",
        reads={"seed": "a"},
        writes=("resolved",),
        optional_packages=("not_a_real_package_xyz",),
        parameters=(
            StepParameter(name="p", value_type="path", path_kind="file", required=True),
        ),
    )
    def _echo_path(seed: Any, *, p: str) -> dict[str, Any]:
        return {"resolved": p}

    yield
    STEP_REGISTRY.pop("compiler_test.make", None)
    STEP_REGISTRY.pop("compiler_test.echo_path", None)


# --------------------------------------------------------------------------- #
# compile_pipeline / CompiledPipeline (3.1)                                   #
# --------------------------------------------------------------------------- #


def test_compile_pipeline_resolves_bindings_and_parameters(_compiler_steps, tmp_path):
    spec = load_spec(
        {
            "name": "p",
            "inputs": {"my_file": "data/thing.bin"},
            "steps": [
                {"uses": "compiler_test.make", "with": {"value": 7}},
                {
                    "uses": "compiler_test.echo_path",
                    "in": {"seed": "a"},
                    "with": {"p": "@my_file"},
                },
            ],
        },
        root=tmp_path,
    )
    compiled = compile_pipeline(spec)
    assert compiled.name == "p"
    assert len(compiled.steps) == 2

    first, second = compiled.steps
    assert first.operation_id == "compiler_test.make"
    assert first.output_bindings == {"a": "a"}
    assert first.parameters == {"value": 7}

    assert second.operation_id == "compiler_test.echo_path"
    assert second.input_bindings == {"seed": "a"}
    # @ref substituted AND resolved against the spec root (Phase 2.4 + 2.5).
    assert second.parameters["p"] == str((tmp_path / "data" / "thing.bin").resolve())


def test_compile_pipeline_fills_unset_optional_parameter_defaults(_compiler_steps):
    spec = load_spec({"name": "p", "steps": [{"uses": "compiler_test.make"}]})
    compiled = compile_pipeline(spec)
    assert compiled.steps[0].parameters == {"value": 0}


def test_compile_pipeline_carries_artifact_and_capability_metadata(
    _compiler_steps, tmp_path
):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "compiler_test.make"},
                {
                    "uses": "compiler_test.echo_path",
                    "in": {"seed": "a"},
                    "with": {"p": "x.bin"},
                },
            ],
        },
        root=tmp_path,
    )
    compiled = compile_pipeline(spec)
    step = compiled.steps[1]
    assert step.optional_packages == ("not_a_real_package_xyz",)


def test_compile_pipeline_raises_pipeline_spec_error_for_structural_problems(
    _compiler_steps,
):
    bad_spec = load_spec(
        {
            "name": "p",
            "steps": [{"uses": "compiler_test.echo_path", "in": {"seed": "missing"}}],
        }
    )
    with pytest.raises(PipelineSpecError):
        compile_pipeline(bad_spec)


def test_compile_pipeline_raises_unknown_step_error():
    spec = load_spec({"name": "p", "steps": [{"uses": "no.such.step"}]})
    with pytest.raises(UnknownStepError):
        compile_pipeline(spec)


def test_compiled_pipeline_as_dict_is_json_serializable(_compiler_steps, tmp_path):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "compiler_test.make", "with": {"value": 1}},
                {
                    "uses": "compiler_test.echo_path",
                    "in": {"seed": "a"},
                    "with": {"p": "x.bin"},
                },
            ],
        },
        root=tmp_path,
    )
    compiled = compile_pipeline(spec)
    json.dumps(compiled.as_dict())


def test_compile_pipeline_does_not_import_optional_packages(_compiler_steps, tmp_path):
    """Compilation must not import optional scientific packages (Phase 3.1) -
    'compiler_test.echo_path' declares a nonexistent optional package and
    compiles fine, since compiling never checks capability/imports it."""

    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "compiler_test.make"},
                {
                    "uses": "compiler_test.echo_path",
                    "in": {"seed": "a"},
                    "with": {"p": "x.bin"},
                },
            ],
        },
        root=tmp_path,
    )
    compile_pipeline(spec)  # must not raise despite the fake optional package


# --------------------------------------------------------------------------- #
# validate_pipeline / ValidationReport (3.5)                                  #
# --------------------------------------------------------------------------- #


def test_validate_pipeline_structural_only_by_default(_compiler_steps, tmp_path):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "compiler_test.make"},
                {
                    "uses": "compiler_test.echo_path",
                    "in": {"seed": "a"},
                    "with": {"p": "missing.bin"},
                },
            ],
        },
        root=tmp_path,
    )
    report = validate_pipeline(spec)
    assert report.is_valid
    assert report.structural == ()
    assert report.readiness == ()  # readiness not requested


def test_validate_pipeline_readiness_reports_missing_optional_dependency(
    _compiler_steps, tmp_path
):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "compiler_test.make"},
                {
                    "uses": "compiler_test.echo_path",
                    "in": {"seed": "a"},
                    "with": {"p": "x.bin"},
                },
            ],
        },
        root=tmp_path,
    )
    (tmp_path / "x.bin").write_text("data")
    report = validate_pipeline(spec, readiness=True)
    assert report.is_valid
    assert any(
        d.code == "capability_missing_optional_dependency" for d in report.readiness
    )


def test_validate_pipeline_readiness_reports_missing_file(_compiler_steps, tmp_path):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "compiler_test.make"},
                {
                    "uses": "compiler_test.echo_path",
                    "in": {"seed": "a"},
                    "with": {"p": "does_not_exist.bin"},
                },
            ],
        },
        root=tmp_path,
    )
    report = validate_pipeline(spec, readiness=True)
    missing_file = [d for d in report.readiness if d.code == "missing_file"]
    assert len(missing_file) == 1
    assert "does_not_exist.bin" in missing_file[0].message


def test_validate_pipeline_readiness_is_clean_when_file_exists_and_package_installed(
    tmp_path,
):
    @register_step(
        "compiler_test.clean",
        parameters=(
            StepParameter(name="p", value_type="path", path_kind="file", required=True),
        ),
    )
    def _clean(*, p: str) -> dict[str, Any]:
        return {}

    try:
        (tmp_path / "present.bin").write_text("data")
        spec = load_spec(
            {
                "name": "p",
                "steps": [
                    {"uses": "compiler_test.clean", "with": {"p": "present.bin"}}
                ],
            },
            root=tmp_path,
        )
        report = validate_pipeline(spec, readiness=True)
        assert report.is_valid
        assert report.readiness == ()
    finally:
        STEP_REGISTRY.pop("compiler_test.clean", None)


def test_validate_pipeline_reports_structural_errors_without_raising(_compiler_steps):
    spec = load_spec({"name": "p", "steps": [{"uses": "no.such.step"}]})
    report = validate_pipeline(spec)
    assert not report.is_valid
    assert any(d.code == "unknown_step" for d in report.structural)


def test_validation_report_as_dict_is_json_serializable(_compiler_steps, tmp_path):
    spec = load_spec(
        {
            "name": "p",
            "steps": [
                {"uses": "compiler_test.make"},
                {
                    "uses": "compiler_test.echo_path",
                    "in": {"seed": "a"},
                    "with": {"p": "missing.bin"},
                },
            ],
        },
        root=tmp_path,
    )
    report = validate_pipeline(spec, readiness=True)
    json.dumps(report.as_dict())
