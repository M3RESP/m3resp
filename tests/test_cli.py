"""Tests for Phase 7.1/7.2 of the pipeline-structure plan: the CLI's
preserved commands, new inspection/validation commands, and stable exit
codes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from m3resp.__main__ import (
    EXIT_CANCELLED,
    EXIT_EXECUTION_FAILURE,
    EXIT_INVALID_SPEC,
    EXIT_READINESS_FAILURE,
    EXIT_SUCCESS,
    main,
)
from m3resp.workflows.registry import STEP_REGISTRY, register_step


def _run_cli(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    code = excinfo.value.code
    assert isinstance(code, int)
    return code


@pytest.fixture
def _cli_step():
    @register_step("cli_test.ok", writes=("x",))
    def _ok(*, n: int = 1) -> dict[str, Any]:
        return {"x": n}

    @register_step("cli_test.fail", writes=())
    def _fail() -> dict[str, Any]:
        raise ValueError("boom")

    yield
    STEP_REGISTRY.pop("cli_test.ok", None)
    STEP_REGISTRY.pop("cli_test.fail", None)


def _write_spec(tmp_path, raw: dict[str, Any]):
    import yaml

    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 7.1: existing commands preserved                                           #
# --------------------------------------------------------------------------- #


def test_steps_command_still_works(capsys):
    exit_code = _run_cli(["steps"])
    assert exit_code == EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "eit.load" in out


def test_run_command_still_works(_cli_step, tmp_path, capsys):
    spec_path = _write_spec(
        tmp_path, {"name": "p", "steps": [{"uses": "cli_test.ok", "with": {"n": 1}}]}
    )
    exit_code = _run_cli(["run", str(spec_path)])
    assert exit_code == EXIT_SUCCESS


def test_no_command_prints_help_and_exits_zero(capsys):
    exit_code = _run_cli([])
    assert exit_code == EXIT_SUCCESS
    assert "usage" in capsys.readouterr().out.lower()


# --------------------------------------------------------------------------- #
# 7.2: new inspection/validation commands and exit codes                     #
# --------------------------------------------------------------------------- #


def test_validate_valid_spec_exits_zero(_cli_step, tmp_path, capsys):
    spec_path = _write_spec(
        tmp_path, {"name": "p", "steps": [{"uses": "cli_test.ok", "with": {"n": 1}}]}
    )
    exit_code = _run_cli(["validate", str(spec_path)])
    assert exit_code == EXIT_SUCCESS
    assert "Valid." in capsys.readouterr().out


def test_validate_invalid_spec_exits_with_invalid_spec_code(tmp_path, capsys):
    spec_path = _write_spec(
        tmp_path, {"name": "p", "steps": [{"uses": "no.such.step"}]}
    )
    exit_code = _run_cli(["validate", str(spec_path)])
    assert exit_code == EXIT_INVALID_SPEC
    assert "unknown_step" in capsys.readouterr().out


def test_validate_json_output_is_parseable(_cli_step, tmp_path, capsys):
    spec_path = _write_spec(
        tmp_path, {"name": "p", "steps": [{"uses": "cli_test.ok", "with": {"n": 1}}]}
    )
    exit_code = _run_cli(["validate", str(spec_path), "--json"])
    assert exit_code == EXIT_SUCCESS
    report = json.loads(capsys.readouterr().out)
    assert report["is_valid"] is True


def test_validate_readiness_reports_missing_file(tmp_path, capsys):
    @register_step(
        "cli_test.needs_file",
        writes=(),
    )
    def _needs_file(*, file: str) -> dict[str, Any]:
        return {}

    from m3resp.workflows.registry import StepParameter

    STEP_REGISTRY.pop("cli_test.needs_file", None)
    register_step(
        "cli_test.needs_file",
        writes=(),
        parameters=(
            StepParameter(
                name="file", value_type="path", path_kind="file", required=True
            ),
        ),
    )(_needs_file)

    try:
        spec_path = _write_spec(
            tmp_path,
            {
                "name": "p",
                "steps": [
                    {"uses": "cli_test.needs_file", "with": {"file": "missing.bin"}}
                ],
            },
        )
        exit_code = _run_cli(["validate", str(spec_path), "--readiness"])
        assert exit_code == EXIT_READINESS_FAILURE
        assert "missing_file" in capsys.readouterr().out
    finally:
        STEP_REGISTRY.pop("cli_test.needs_file", None)


def test_run_dry_run_prints_compiled_plan_without_executing(
    _cli_step, tmp_path, capsys
):
    spec_path = _write_spec(
        tmp_path, {"name": "p", "steps": [{"uses": "cli_test.ok", "with": {"n": 1}}]}
    )
    exit_code = _run_cli(["run", str(spec_path), "--dry-run"])
    assert exit_code == EXIT_SUCCESS
    compiled = json.loads(capsys.readouterr().out)
    assert compiled["name"] == "p"
    assert len(compiled["steps"]) == 1


def test_run_execution_failure_exits_with_execution_failure_code(
    _cli_step, tmp_path, capsys
):
    spec_path = _write_spec(
        tmp_path, {"name": "p", "steps": [{"uses": "cli_test.fail"}]}
    )
    exit_code = _run_cli(["run", str(spec_path)])
    assert exit_code == EXIT_EXECUTION_FAILURE
    assert "boom" in capsys.readouterr().err


def test_run_missing_spec_file_exits_with_invalid_spec_code(capsys):
    exit_code = _run_cli(["run", "/does/not/exist.yaml"])
    assert exit_code == EXIT_INVALID_SPEC


def test_describe_known_operation_prints_json(capsys):
    exit_code = _run_cli(["describe", "metric.interval_cv"])
    assert exit_code == EXIT_SUCCESS
    description = json.loads(capsys.readouterr().out)
    assert description["name"] == "metric.interval_cv"


def test_describe_unknown_operation_exits_with_invalid_spec_code(capsys):
    exit_code = _run_cli(["describe", "no.such.step"])
    assert exit_code == EXIT_INVALID_SPEC


def test_steps_json_output_is_parseable(capsys):
    exit_code = _run_cli(["steps", "--json"])
    assert exit_code == EXIT_SUCCESS
    descriptions = json.loads(capsys.readouterr().out)
    assert any(d["name"] == "metric.interval_cv" for d in descriptions)


def test_steps_details_shows_parameters(capsys):
    exit_code = _run_cli(["steps", "--details"])
    assert exit_code == EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "session.sync_raw" in out
    assert "with: method (choice)" in out


def test_run_debug_flag_reraises_instead_of_printing_short_message(_cli_step, tmp_path):
    from m3resp.workflows import PipelineExecutionError

    spec_path = _write_spec(
        tmp_path, {"name": "p", "steps": [{"uses": "cli_test.fail"}]}
    )
    with pytest.raises(PipelineExecutionError):
        main(["run", str(spec_path), "--debug"])


def test_cancelled_run_exits_with_cancelled_code(_cli_step, tmp_path, capsys):
    """A step raising SIGINT at itself simulates a real Ctrl-C during a
    ``m3resp run`` - ``_cmd_run`` installs a handler that cancels its
    internal token, so the run stops cooperatively (Phase 4.5) and the CLI
    reports EXIT_CANCELLED (Phase 7.2), rather than a raw KeyboardInterrupt."""

    import signal

    @register_step("cli_test.self_interrupt", writes=())
    def _self_interrupt() -> dict[str, Any]:
        signal.raise_signal(signal.SIGINT)
        return {}

    try:
        spec_path = _write_spec(
            tmp_path,
            {
                "name": "p",
                "steps": [
                    {"uses": "cli_test.self_interrupt"},
                    {"uses": "cli_test.ok", "with": {"n": 1}},
                ],
            },
        )
        exit_code = _run_cli(["run", str(spec_path)])
        assert exit_code == EXIT_CANCELLED
        assert "cancelled" in capsys.readouterr().err.lower()
    finally:
        STEP_REGISTRY.pop("cli_test.self_interrupt", None)
