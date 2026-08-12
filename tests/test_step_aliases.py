"""Retired step names keep resolving after the ventilator re-namespace.

Ten ventilator steps moved out of the `emg.*` namespace into `ventilator.*`.
Their former ids stay registered as silent aliases so existing pipeline specs
keep compiling unchanged, while discovery only ever offers the current names.
"""

from __future__ import annotations

import pytest

import m3resp.workflows.steps  # noqa: F401 - registers built-in steps
from m3resp.core.exceptions import UnknownStepError
from m3resp.workflows.registry import (
    STEP_ALIASES,
    STEP_REGISTRY,
    available_steps,
    describe_steps,
    get_step,
    register_step,
)

RENAMED = {
    "emg.load_ventilator": "ventilator.load",
    "emg.ventilator_channels": "ventilator.channels",
    "emg.detect_ventilator_breath": "ventilator.detect_breaths",
    "emg.normalize_ventilator_breaths": "ventilator.normalize_breaths",
    "emg.ventilator_respiratory_rate": "ventilator.respiratory_rate",
    "emg.find_occluded_breaths": "ventilator.find_occluded_breaths",
    "emg.pocc_intervals": "ventilator.pocc_intervals",
    "emg.pocc_time_product": "ventilator.pocc_time_product",
    "emg.pocc_quality": "ventilator.pocc_quality",
    "emg.detect_non_consecutive_manoeuvres": (
        "ventilator.detect_non_consecutive_manoeuvres"
    ),
}


#: A spec as an existing user would already have written it, before the
#: ventilator steps moved out of the `emg.*` namespace.
_LEGACY_SPEC = {
    "name": "legacy-ids",
    "inputs": {"vent_file": "recording.txt"},
    "steps": [
        {"uses": "emg.load_ventilator", "with": {"file": "@vent_file"}},
        {"uses": "emg.ventilator_channels"},
        {"uses": "emg.detect_ventilator_breath"},
    ],
}


class TestRetiredNamesResolve:
    @pytest.mark.parametrize(("old", "new"), sorted(RENAMED.items()))
    def test_the_old_name_returns_the_new_step(self, old, new):
        assert get_step(old) is get_step(new)

    @pytest.mark.parametrize(("old", "new"), sorted(RENAMED.items()))
    def test_the_resolved_step_reports_its_current_name(self, old, new):
        # A spec written against the old id compiles to the canonical
        # operation_id, so provenance never records the retired name.
        assert get_step(old).name == new
        assert get_step(old).operation_id == new

    def test_every_rename_is_registered_as_an_alias(self):
        assert {
            alias: canonical
            for alias, canonical in STEP_ALIASES.items()
            if canonical.startswith("ventilator.")
        } == RENAMED


class TestAliasesStayOutOfDiscovery:
    def test_retired_names_are_not_registered_steps(self):
        for old in RENAMED:
            assert old not in STEP_REGISTRY

    def test_retired_names_are_not_listed(self):
        listed = set(available_steps())
        assert not listed & set(RENAMED)
        assert set(RENAMED.values()) <= listed

    def test_retired_names_are_not_described(self):
        described = {description.name for description in describe_steps()}
        assert not described & set(RENAMED)

    def test_an_alias_does_not_leak_into_the_emg_prefix_listing(self):
        emg = {d.name for d in describe_steps(prefix="emg.")}
        assert not emg & set(RENAMED)


class TestUnknownStepsStillFail:
    def test_an_unregistered_name_raises(self):
        with pytest.raises(UnknownStepError, match="genuinely.nonexistent"):
            get_step("genuinely.nonexistent")

    def test_the_error_lists_canonical_names_not_aliases(self):
        with pytest.raises(UnknownStepError) as excinfo:
            get_step("genuinely.nonexistent")
        message = str(excinfo.value)
        assert "ventilator.pocc_quality" in message
        assert "emg.pocc_quality" not in message


class TestAliasRegistrationGuards:
    def test_aliasing_an_existing_step_is_rejected(self):
        with pytest.raises(ValueError, match="is itself a registered step"):

            @register_step("t.alias_clash", aliases=("emg.load",))
            def _clash() -> dict:
                return {}

    def test_aliasing_a_name_already_aliased_elsewhere_is_rejected(self):
        try:
            with pytest.raises(ValueError, match="already mapped"):

                @register_step("t.alias_reuse", aliases=("emg.pocc_quality",))
                def _reuse() -> dict:
                    return {}
        finally:
            STEP_REGISTRY.pop("t.alias_reuse", None)

    def test_a_rejected_registration_leaves_the_alias_map_untouched(self):
        assert STEP_ALIASES["emg.pocc_quality"] == "ventilator.pocc_quality"
        assert "t.alias_clash" not in STEP_REGISTRY


class TestExistingSpecsKeepRunning:
    def test_a_spec_using_retired_ids_still_compiles(self):
        from m3resp.workflows.compiler import compile_pipeline
        from m3resp.workflows.spec import load_spec

        compiled = compile_pipeline(load_spec(_LEGACY_SPEC))

        # It compiles, and to the *current* operation ids - so provenance and
        # the execution plan never carry a retired name forward.
        assert [step.operation_id for step in compiled.steps] == [
            "ventilator.load",
            "ventilator.channels",
            "ventilator.detect_breaths",
        ]

    def test_a_legacy_spec_produces_no_error_diagnostics(self):
        from m3resp.workflows.engine.diagnostics import collect_diagnostics
        from m3resp.workflows.spec import load_spec

        diagnostics = collect_diagnostics(load_spec(_LEGACY_SPEC))
        assert not [d for d in diagnostics if d.severity == "error"]
