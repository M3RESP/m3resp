"""Coverage for the modules backfilled with Phase 1 step metadata so far:
``metrics``, ``session``, ``sync``, ``export`` (see plan/stage2/
3_pipeline_structure_implementation_plan.md, "Concrete Next Actions" item 4).

These check that the declared metadata reflects the step's real behavior
(e.g. a ``choice`` parameter's ``choices`` match what the underlying function
actually accepts) rather than just that *some* metadata was attached.
"""

from __future__ import annotations

import inspect
import json

import m3resp.workflows.steps  # noqa: F401 - ensure built-in steps are registered
from m3resp.workflows.registry import describe_step, describe_steps

MIGRATED_STEPS = [
    "metric.interval_cv",
    "session.sync_raw",
    "sync.estimate_offset",
    "sync.apply_estimated_offset",
    "export.scalar_file",
    "export.json_file",
    "export.session_summary",
    "export.rotarc_result",
]


def test_migrated_steps_declare_parameters_or_artifacts():
    for name in MIGRATED_STEPS:
        description = describe_step(name)
        assert description.parameters or description.output_artifacts, name
        assert description.description, f"{name} has no description"
        json.dumps(description.as_dict())  # must not raise, for every step


def test_session_sync_raw_method_choices_match_supported_methods():
    description = describe_step("session.sync_raw")
    method = next(p for p in description.parameters if p.name == "method")
    assert method.choices == ("manual_offset",)
    assert method.default == "manual_offset"


def test_sync_estimate_offset_method_choices_match_estimator():
    from m3resp.synchronization.offset_estimation import _METHODS

    description = describe_step("sync.estimate_offset")
    method = next(p for p in description.parameters if p.name == "method")
    assert set(method.choices or ()) == set(_METHODS)


def test_export_session_summary_boolean_defaults_match_function():
    from m3resp.workflows.steps.export import session_summary

    description = describe_step("export.session_summary")
    defaults = {
        "summary_json": True,
        "event_csvs": True,
        "parameters_csv": True,
        "postprocessing": True,
        "structured_export": True,
    }
    for param_name, expected_default in defaults.items():
        param = next(p for p in description.parameters if p.name == param_name)
        assert param.value_type == "boolean"
        assert param.default is expected_default

    # Cross-check against the actual function signature, not just our metadata.
    sig_defaults = {
        name: p.default
        for name, p in inspect.signature(session_summary).parameters.items()
        if name in defaults
    }
    assert sig_defaults == defaults


def test_export_scalar_file_and_rotarc_result_share_precision_default():
    scalar_precision = next(
        p
        for p in describe_step("export.scalar_file").parameters
        if p.name == "precision"
    )
    rotarc_precision = next(
        p
        for p in describe_step("export.rotarc_result").parameters
        if p.name == "precision"
    )
    assert scalar_precision.default == rotarc_precision.default == 8


def test_describe_steps_prefix_filters_cover_migrated_modules():
    for prefix in ("metric.", "session.", "sync.", "export."):
        names = {d.name for d in describe_steps(prefix=prefix)}
        assert names, prefix


def test_all_60_plus_steps_still_describe_without_error():
    descriptions = describe_steps()
    assert len(descriptions) >= 60
    migrated_with_params = [
        d for d in descriptions if d.parameters or d.output_artifacts
    ]
    assert {d.name for d in migrated_with_params} >= set(MIGRATED_STEPS)
