"""Tests for the typed step metadata (StepParameter/StepArtifact) and
discovery API added to ``m3resp.workflows.registry``.

This metadata is additive and optional per step, backfilled module by module,
so a step declaring none of it must remain fully valid (most built-ins still do).
"""

from __future__ import annotations

from typing import Any

import pytest

import m3resp.workflows.steps  # noqa: F401 - ensure built-in steps are registered
from m3resp.core.exceptions import StepMetadataError
from m3resp.workflows.registry import (
    STEP_REGISTRY,
    StepArtifact,
    StepParameter,
    describe_step,
    describe_steps,
    register_step,
    step_capability_state,
)


@pytest.fixture
def _cleanup():
    created: list[str] = []
    yield created
    for name in created:
        STEP_REGISTRY.pop(name, None)


# --------------------------------------------------------------------------- #
# Backward compatibility: metadata-free registration still works              #
# --------------------------------------------------------------------------- #


def test_register_step_without_metadata_is_unaffected(_cleanup):
    @register_step("meta_test.plain", writes=("value",))
    def _plain(*, x: int) -> dict[str, Any]:
        return {"value": x}

    _cleanup.append("meta_test.plain")
    definition = STEP_REGISTRY["meta_test.plain"]
    assert definition.parameters == ()
    assert definition.input_artifacts == ()
    assert definition.output_artifacts == ()


# --------------------------------------------------------------------------- #
# Registration-time validation                                                #
# --------------------------------------------------------------------------- #


def test_rejects_malformed_step_name():
    with pytest.raises(StepMetadataError, match="lowercase snake_case"):

        @register_step("NoDot", writes=())
        def _bad() -> dict[str, Any]:
            return {}


def test_rejects_reserved_output_name():
    with pytest.raises(StepMetadataError, match="reserved output name"):

        @register_step("meta_test.reserved", writes=("session",))
        def _bad() -> dict[str, Any]:
            return {"session": None}


def test_rejects_underscore_output_name():
    with pytest.raises(StepMetadataError, match="reserved output name"):

        @register_step("meta_test.underscore", writes=("_private",))
        def _bad() -> dict[str, Any]:
            return {"_private": None}


def test_rejects_duplicate_parameter_names():
    with pytest.raises(StepMetadataError, match="duplicate parameter"):

        @register_step(
            "meta_test.dup_param",
            parameters=(
                StepParameter(name="x", value_type="integer"),
                StepParameter(name="x", value_type="string"),
            ),
        )
        def _bad(*, x: Any) -> dict[str, Any]:
            return {}


def test_rejects_parameter_colliding_with_read():
    with pytest.raises(
        StepMetadataError, match="collides with a declared context read"
    ):

        @register_step(
            "meta_test.collision",
            reads={"x": "some_key"},
            parameters=(StepParameter(name="x", value_type="integer"),),
        )
        def _bad(x: Any) -> dict[str, Any]:
            return {}


def test_rejects_unknown_value_type():
    with pytest.raises(StepMetadataError, match="unknown value_type"):

        @register_step(
            "meta_test.bad_type",
            parameters=(StepParameter(name="x", value_type="not_a_type"),),  # type: ignore[arg-type]
        )
        def _bad(*, x: Any) -> dict[str, Any]:
            return {}


def test_rejects_minimum_greater_than_maximum():
    with pytest.raises(StepMetadataError, match="greater than maximum"):

        @register_step(
            "meta_test.bad_range",
            parameters=(
                StepParameter(name="x", value_type="number", minimum=10, maximum=1),
            ),
        )
        def _bad(*, x: Any) -> dict[str, Any]:
            return {}


def test_rejects_mutually_exclusive_group_naming_an_undeclared_parameter():
    with pytest.raises(StepMetadataError, match="not a declared parameter"):

        @register_step(
            "meta_test.bad_mutex_unknown_name",
            parameters=(StepParameter(name="a", value_type="number"),),
            mutually_exclusive_parameters=(("a", "b"),),
        )
        def _bad(*, a: Any = None, b: Any = None) -> dict[str, Any]:
            return {}


def test_rejects_mutually_exclusive_group_with_fewer_than_two_names():
    with pytest.raises(StepMetadataError, match="fewer than two names"):

        @register_step(
            "meta_test.bad_mutex_singleton",
            parameters=(StepParameter(name="a", value_type="number"),),
            mutually_exclusive_parameters=(("a",),),
        )
        def _bad(*, a: Any = None) -> dict[str, Any]:
            return {}


def test_accepts_a_valid_mutually_exclusive_group():
    @register_step(
        "meta_test.good_mutex",
        parameters=(
            StepParameter(name="a", value_type="number", default=None),
            StepParameter(name="b", value_type="number", default=None),
        ),
        mutually_exclusive_parameters=(("a", "b"),),
    )
    def _good(*, a: Any = None, b: Any = None) -> dict[str, Any]:
        return {}

    STEP_REGISTRY.pop("meta_test.good_mutex", None)


def test_rejects_default_not_in_choices():
    with pytest.raises(StepMetadataError, match="not one of its declared choices"):

        @register_step(
            "meta_test.bad_choice",
            parameters=(
                StepParameter(
                    name="x", value_type="choice", default="c", choices=("a", "b")
                ),
            ),
        )
        def _bad(*, x: Any) -> dict[str, Any]:
            return {}


def test_rejects_duplicate_artifact_names():
    with pytest.raises(StepMetadataError, match="duplicate output_artifacts"):

        @register_step(
            "meta_test.dup_artifact",
            output_artifacts=(
                StepArtifact(name="a", artifact_type="scalar_metric"),
                StepArtifact(name="a", artifact_type="scalar_metric"),
            ),
        )
        def _bad() -> dict[str, Any]:
            return {"a": 1}


def test_rejects_parameter_not_in_function_signature():
    with pytest.raises(StepMetadataError, match="not a keyword argument"):

        @register_step(
            "meta_test.not_a_kwarg",
            parameters=(StepParameter(name="not_present", value_type="integer"),),
        )
        def _bad(*, x: Any) -> dict[str, Any]:
            return {}


def test_allows_parameter_when_function_accepts_var_keyword(_cleanup):
    @register_step(
        "meta_test.var_kwargs",
        parameters=(StepParameter(name="anything", value_type="integer"),),
    )
    def _ok(**kwargs: Any) -> dict[str, Any]:
        return {}

    _cleanup.append("meta_test.var_kwargs")
    assert "meta_test.var_kwargs" in STEP_REGISTRY


# --------------------------------------------------------------------------- #
# Discovery API                                                               #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# parameters_reviewed                                                         #
# --------------------------------------------------------------------------- #


def test_parameters_reviewed_defaults_false_with_no_parameters(_cleanup):
    @register_step("meta_test.unreviewed", writes=("value",))
    def _plain() -> dict[str, Any]:
        return {"value": 1}

    _cleanup.append("meta_test.unreviewed")
    assert STEP_REGISTRY["meta_test.unreviewed"].parameters_reviewed is False


def test_parameters_reviewed_auto_true_when_parameters_declared(_cleanup):
    @register_step(
        "meta_test.has_params",
        parameters=(StepParameter(name="x", value_type="integer"),),
    )
    def _plain(*, x: Any) -> dict[str, Any]:
        return {}

    _cleanup.append("meta_test.has_params")
    assert STEP_REGISTRY["meta_test.has_params"].parameters_reviewed is True


def test_parameters_reviewed_explicit_true_with_no_parameters(_cleanup):
    @register_step("meta_test.audited_empty", parameters_reviewed=True)
    def _plain() -> dict[str, Any]:
        return {}

    _cleanup.append("meta_test.audited_empty")
    definition = STEP_REGISTRY["meta_test.audited_empty"]
    assert definition.parameters == ()
    assert definition.parameters_reviewed is True


def test_describe_step_carries_parameters_reviewed():
    description = describe_step("metric.interval_cv")
    assert description.parameters_reviewed is True
    assert description.as_dict()["parameters_reviewed"] is True


def test_every_registered_step_has_reviewed_parameters():
    """Guards the Phase A audit (plan/06_gui_readiness_plan.md §1): an empty
    ``parameters`` tuple must mean "confirmed no tunable parameter", not
    "nobody has looked yet". Any new step must set ``parameters_reviewed``
    (directly, or implicitly by declaring real ``parameters``) at
    registration time."""

    unreviewed = sorted(
        name
        for name, definition in STEP_REGISTRY.items()
        if not definition.parameters_reviewed
    )
    assert unreviewed == []


def test_describe_step_matches_registered_metadata():
    description = describe_step("metric.interval_cv")
    assert description.name == "metric.interval_cv"
    assert description.category == "reducer"
    assert [a.name for a in description.input_artifacts] == ["intervals"]
    assert {a.name for a in description.output_artifacts} == {"cv", "mean", "std", "n"}
    assert description.capability == "available"


def test_describe_step_falls_back_description_to_summary(_cleanup):
    @register_step("meta_test.no_description", summary="A summary only.")
    def _plain() -> dict[str, Any]:
        return {}

    _cleanup.append("meta_test.no_description")
    assert describe_step("meta_test.no_description").description == "A summary only."


def test_describe_steps_covers_every_registered_step():
    descriptions = describe_steps()
    assert len(descriptions) == len(STEP_REGISTRY)
    assert {d.name for d in descriptions} == set(STEP_REGISTRY)


def test_describe_steps_filters_by_prefix():
    descriptions = describe_steps(prefix="metric.")
    assert descriptions
    assert all(d.name.startswith("metric.") for d in descriptions)


def test_describe_step_output_is_json_serializable():
    import json

    description = describe_step("session.sync_raw")
    json.dumps(description.as_dict())  # must not raise


# --------------------------------------------------------------------------- #
# Capability discovery                                                        #
# --------------------------------------------------------------------------- #


def test_capability_available_for_step_without_optional_packages():
    assert step_capability_state("metric.interval_cv") == "available"


def test_capability_missing_optional_dependency(_cleanup):
    @register_step(
        "meta_test.needs_package", optional_packages=("not_a_real_package_xyz",)
    )
    def _needs() -> dict[str, Any]:
        return {}

    _cleanup.append("meta_test.needs_package")
    assert (
        step_capability_state("meta_test.needs_package")
        == "missing_optional_dependency"
    )


def test_capability_deprecated_takes_priority(_cleanup):
    @register_step("meta_test.deprecated_step", deprecated_since="0.2.0")
    def _dep() -> dict[str, Any]:
        return {}

    _cleanup.append("meta_test.deprecated_step")
    assert step_capability_state("meta_test.deprecated_step") == "deprecated"
