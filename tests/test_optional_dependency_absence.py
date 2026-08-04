"""Capability discovery and structural validation with upstream packages absent.

Every other test in this repo runs with ``eitprocessing``/``resurfemg``
actually installed (they're dev dependencies), and confirms only that no
step module *eagerly imports* one at module scope
(``test_registry_coverage.py::test_no_step_module_eagerly_imports_an_optional_package``).
That is necessary but not sufficient: it has never been verified that
discovery/validation/dry-run actually keep working end-to-end when a
package is genuinely unimportable.

``importlib.util.find_spec(name)`` - what ``step_capability_state`` uses -
treats ``sys.modules[name] is None`` as the standard "known absent, do not
attempt to import" sentinel (see the CPython docs for
``importlib.util.find_spec``), so setting that is a faithful simulation of
"not installed" without needing a second virtualenv.
"""

from __future__ import annotations

import json
import sys

import pytest

import m3resp.workflows.steps  # noqa: F401 - ensure built-in steps are registered
from m3resp.__main__ import EXIT_SUCCESS, main
from m3resp.workflows.compiler import compile_pipeline, validate_pipeline
from m3resp.workflows.registry import (
    describe_step,
    describe_steps,
    step_capability_state,
)
from m3resp.workflows.spec import load_spec

pytestmark = pytest.mark.usefixtures("_absent_optional_packages")


@pytest.fixture
def _absent_optional_packages(monkeypatch):
    monkeypatch.setitem(sys.modules, "eitprocessing", None)
    monkeypatch.setitem(sys.modules, "resurfemg", None)


def _eit_emg_spec() -> dict:
    return {
        "name": "p",
        "inputs": {"eit_file": "does-not-need-to-exist.bin"},
        "steps": [
            {"uses": "eit.load", "with": {"file": "@eit_file", "vendor": "draeger"}}
        ],
    }


def test_step_capability_state_reports_missing_for_an_eit_step():
    assert step_capability_state("eit.load") == "missing_optional_dependency"


def test_step_capability_state_reports_missing_for_an_emg_step():
    assert step_capability_state("emg.preprocess") == "missing_optional_dependency"


def test_step_capability_state_is_unaffected_for_a_native_step():
    """Only steps that actually declare the (now-absent) optional package
    are affected - a purely native step (no ``optional_packages``) stays
    ``"available"``."""

    assert step_capability_state("metric.interval_cv") == "available"


def test_describe_steps_still_returns_every_step_with_no_import_error():
    descriptions = describe_steps()
    names = {d.name for d in descriptions}
    assert "eit.load" in names
    assert "emg.preprocess" in names
    # JSON-safe, exactly like `m3resp steps --json` would print.
    json.dumps([d.as_dict() for d in descriptions])


def test_describe_step_reports_capability_state_for_a_single_operation():
    description = describe_step("eit.load")
    assert description.capability == "missing_optional_dependency"


def test_compile_pipeline_succeeds_without_the_optional_package():
    spec = load_spec(_eit_emg_spec())
    compiled = compile_pipeline(spec)
    assert compiled.steps


def test_validate_readiness_reports_a_warning_not_a_hard_failure():
    """A missing optional dependency is a *readiness* warning, not a
    structural error - the spec is still valid, just not runnable here."""

    spec = load_spec(_eit_emg_spec())
    report = validate_pipeline(spec, readiness=True)
    assert report.is_valid
    codes = {d.code for d in report.readiness}
    assert "capability_missing_optional_dependency" in codes
    assert all(
        d.severity != "error"
        for d in report.readiness
        if d.code == "capability_missing_optional_dependency"
    )


def test_cli_steps_json_works_without_the_optional_packages(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["steps", "--json"])
    assert excinfo.value.code == EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out)
    names = {entry["name"] for entry in payload}
    assert "eit.load" in names


def test_cli_describe_reports_missing_dependency_capability(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["describe", "eit.load"])
    assert excinfo.value.code == EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out)
    assert payload["capability"] == "missing_optional_dependency"


def test_cli_validate_readiness_succeeds_with_a_warning_diagnostic(tmp_path, capsys):
    import yaml

    # An existing (if empty) fixture file, so the only readiness diagnostic
    # is the missing-optional-dependency warning this test is about, not
    # also an unrelated missing_file error from a dummy path.
    (tmp_path / "does-not-need-to-exist.bin").write_bytes(b"")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_eit_emg_spec()), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["validate", str(spec_path), "--readiness", "--json"])
    assert excinfo.value.code == EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out)
    codes = {d["code"] for d in payload["readiness"]}
    assert "capability_missing_optional_dependency" in codes


def test_cli_dry_run_succeeds_without_the_optional_packages(tmp_path, capsys):
    import yaml

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_eit_emg_spec()), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["run", str(spec_path), "--dry-run"])
    assert excinfo.value.code == EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out)
    assert payload["steps"][0]["operation_id"] == "eit.load"
