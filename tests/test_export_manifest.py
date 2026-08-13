"""Tests for Phase 6 of the pipeline-structure plan: explicit output modes
(6.1/6.2) and the atomic run manifest (6.3/6.4). See
plan/stage2/3_pipeline_structure_implementation_plan.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from m3resp.core.session import M3Session
from m3resp.workflows import (
    CancellationToken,
    PipelineExecutionError,
    load_spec,
    register_step,
    run_spec,
)
from m3resp.workflows.registry import STEP_REGISTRY


@pytest.fixture
def _manifest_steps():
    @register_step("manifest_test.ok", writes=("x",))
    def _ok(*, n: int = 1) -> dict[str, Any]:
        return {"x": n}

    @register_step("manifest_test.fail", writes=())
    def _fail() -> dict[str, Any]:
        raise ValueError("sample frequency must be positive")

    @register_step(
        "export.manifest_test_like",
        reads={"session": "session"},
        writes=(),
    )
    def _export_like(session: M3Session) -> dict[str, Any]:
        return {}

    yield
    for name in (
        "manifest_test.ok",
        "manifest_test.fail",
        "export.manifest_test_like",
    ):
        STEP_REGISTRY.pop(name, None)


def _write_spec(tmp_path: Path, raw: dict[str, Any]) -> Path:
    import yaml

    spec_path = tmp_path / "spec.pipeline.yaml"
    spec_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return spec_path


# --------------------------------------------------------------------------- #
# 6.2: explicit output mode                                                  #
# --------------------------------------------------------------------------- #


def test_versioned_spec_requires_mode_when_dir_is_set(tmp_path):
    with pytest.raises(Exception, match="outputs.mode must be set"):
        load_spec(
            {
                "schema_version": 1,
                "name": "p",
                "outputs": {"dir": str(tmp_path)},
                "steps": [{"uses": "manifest_test.ok"}],
            }
        )


def test_versioned_spec_rejects_mode_without_dir(tmp_path):
    with pytest.raises(Exception, match="requires outputs.dir"):
        load_spec(
            {
                "schema_version": 1,
                "name": "p",
                "outputs": {"mode": "automatic"},
                "steps": [{"uses": "manifest_test.ok"}],
            }
        )


def test_versioned_spec_allows_mode_none_with_dir_set(tmp_path):
    spec = load_spec(
        {
            "schema_version": 1,
            "name": "p",
            "outputs": {"dir": str(tmp_path), "mode": "none"},
            "steps": [{"uses": "manifest_test.ok"}],
        }
    )
    assert spec.outputs.mode == "none"


def test_legacy_spec_infers_automatic_mode_with_warning(_manifest_steps, tmp_path):
    spec_path = _write_spec(
        tmp_path,
        {
            "name": "p",
            "outputs": {"dir": str(tmp_path / "out"), "timestamped": False},
            "steps": [{"uses": "manifest_test.ok", "with": {"n": 1}}],
        },
    )
    with pytest.warns(FutureWarning, match="inferred 'automatic'"):
        run_spec(spec_path, session=M3Session())


def test_legacy_spec_infers_explicit_mode_when_export_step_present(
    _manifest_steps, tmp_path
):
    spec_path = _write_spec(
        tmp_path,
        {
            "name": "p",
            "outputs": {"dir": str(tmp_path / "out"), "timestamped": False},
            "steps": [
                {"uses": "manifest_test.ok", "with": {"n": 1}},
                {"uses": "export.manifest_test_like"},
            ],
        },
    )
    with pytest.warns(FutureWarning, match="inferred 'explicit'"):
        run_spec(spec_path, session=M3Session())


def test_legacy_spec_can_opt_in_to_explicit_mode_without_a_warning(
    _manifest_steps, tmp_path, recwarn
):
    spec_path = _write_spec(
        tmp_path,
        {
            "name": "p",
            "outputs": {
                "dir": str(tmp_path / "out"),
                "timestamped": False,
                "mode": "explicit",
            },
            "steps": [{"uses": "manifest_test.ok", "with": {"n": 1}}],
        },
    )
    run_spec(spec_path, session=M3Session())
    assert not any(w.category is FutureWarning for w in recwarn.list)
    # explicit mode: no automatic session_summary.json written
    assert not (tmp_path / "out" / "session_summary.json").exists()


def test_mode_none_writes_nothing_at_all(_manifest_steps, tmp_path):
    spec_path = _write_spec(
        tmp_path,
        {
            "name": "p",
            "outputs": {
                "dir": str(tmp_path / "out"),
                "timestamped": False,
                "mode": "none",
            },
            "steps": [{"uses": "manifest_test.ok", "with": {"n": 1}}],
        },
    )
    result = run_spec(spec_path, session=M3Session())
    assert result.manifest_path is None
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- #
# 6.3/6.4: the run manifest                                                  #
# --------------------------------------------------------------------------- #


def test_successful_run_writes_a_terminal_manifest(_manifest_steps, tmp_path):
    spec_path = _write_spec(
        tmp_path,
        {
            "name": "p",
            "outputs": {
                "dir": str(tmp_path / "out"),
                "timestamped": False,
                "mode": "automatic",
            },
            "steps": [{"uses": "manifest_test.ok", "with": {"n": 5}}],
        },
    )
    result = run_spec(spec_path, session=M3Session())

    assert result.manifest_path is not None
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["status"] == "succeeded"
    assert manifest["run_id"] == result.run_id
    assert len(manifest["step_records"]) == 1
    assert manifest["error"] is None
    assert manifest["pipeline_name"] == "p"


def test_failed_run_still_writes_an_honestly_failed_manifest(_manifest_steps, tmp_path):
    spec_path = _write_spec(
        tmp_path,
        {
            "name": "p",
            "outputs": {
                "dir": str(tmp_path / "out"),
                "timestamped": False,
                "mode": "automatic",
            },
            "steps": [{"uses": "manifest_test.fail"}],
        },
    )
    manifest_path = tmp_path / "out" / "run_manifest.json"

    with pytest.raises(PipelineExecutionError):
        run_spec(spec_path, session=M3Session())

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert "sample frequency must be positive" in manifest["error"]["message"]
    # never a success summary for a failed run (6.4)
    assert not (tmp_path / "out" / "session_summary.json").exists()


def test_cancelled_run_manifest_is_marked_cancelled_not_succeeded(
    _manifest_steps, tmp_path
):
    token = CancellationToken()

    @register_step("manifest_test.cancel_after", writes=())
    def _cancel_after() -> dict[str, Any]:
        token.cancel()
        return {}

    try:
        spec_path = _write_spec(
            tmp_path,
            {
                "name": "p",
                "outputs": {
                    "dir": str(tmp_path / "out"),
                    "timestamped": False,
                    "mode": "automatic",
                },
                "steps": [
                    {"uses": "manifest_test.cancel_after"},
                    {"uses": "manifest_test.ok", "with": {"n": 1}},
                ],
            },
        )
        result = run_spec(spec_path, session=M3Session(), cancellation_token=token)
        assert result.status == "cancelled"
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["status"] == "cancelled"
        # never a success summary for a cancelled run (6.4)
        assert not (tmp_path / "out" / "session_summary.json").exists()
    finally:
        STEP_REGISTRY.pop("manifest_test.cancel_after", None)


def test_manifest_write_is_atomic_no_leftover_temp_file(_manifest_steps, tmp_path):
    spec_path = _write_spec(
        tmp_path,
        {
            "name": "p",
            "outputs": {
                "dir": str(tmp_path / "out"),
                "timestamped": False,
                "mode": "automatic",
            },
            "steps": [{"uses": "manifest_test.ok", "with": {"n": 1}}],
        },
    )
    run_spec(spec_path, session=M3Session())
    out_dir = tmp_path / "out"
    leftovers = [p for p in out_dir.iterdir() if p.name.startswith(".run_manifest")]
    assert leftovers == []


def test_manifest_starts_as_running_then_becomes_terminal(_manifest_steps, tmp_path):
    """Can't observe the exact "running" instant deterministically without a
    lot of coupling, but the manifest must never regress to "running" once
    the run finished - the same path was overwritten in place."""

    spec_path = _write_spec(
        tmp_path,
        {
            "name": "p",
            "outputs": {
                "dir": str(tmp_path / "out"),
                "timestamped": False,
                "mode": "automatic",
            },
            "steps": [{"uses": "manifest_test.ok", "with": {"n": 1}}],
        },
    )
    result = run_spec(spec_path, session=M3Session())
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["status"] != "running"


def test_checksums_are_computed_only_when_configured(_manifest_steps, tmp_path):
    input_file = tmp_path / "input.bin"
    input_file.write_bytes(b"hello world")

    @register_step(
        "manifest_test.uses_path",
        parameters=(),
        writes=(),
    )
    def _uses_path(*, file: str) -> dict[str, Any]:
        return {}

    from m3resp.workflows.registry import StepParameter

    STEP_REGISTRY.pop("manifest_test.uses_path", None)
    register_step(
        "manifest_test.uses_path",
        writes=(),
        parameters=(StepParameter(name="file", value_type="path", required=True),),
    )(_uses_path)

    try:
        spec_path = _write_spec(
            tmp_path,
            {
                "name": "p",
                "outputs": {
                    "dir": str(tmp_path / "out"),
                    "mode": "automatic",
                    "checksums": True,
                },
                "steps": [
                    {
                        "uses": "manifest_test.uses_path",
                        "with": {"file": str(input_file)},
                    }
                ],
            },
        )
        result = run_spec(spec_path, session=M3Session())
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["checksums"]
        assert str(input_file) in manifest["checksums"]
    finally:
        STEP_REGISTRY.pop("manifest_test.uses_path", None)


def test_checksums_are_none_when_not_configured(_manifest_steps, tmp_path):
    spec_path = _write_spec(
        tmp_path,
        {
            "name": "p",
            "outputs": {
                "dir": str(tmp_path / "out"),
                "timestamped": False,
                "mode": "automatic",
            },
            "steps": [{"uses": "manifest_test.ok", "with": {"n": 1}}],
        },
    )
    result = run_spec(spec_path, session=M3Session())
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["checksums"] is None


def test_sensitive_looking_inputs_are_redacted_in_the_manifest(
    _manifest_steps, tmp_path
):
    spec_path = _write_spec(
        tmp_path,
        {
            "name": "p",
            "inputs": {"api_token": "super-secret-value", "n": 1},
            "outputs": {
                "dir": str(tmp_path / "out"),
                "timestamped": False,
                "mode": "automatic",
            },
            "steps": [{"uses": "manifest_test.ok", "with": {"n": "@n"}}],
        },
    )
    result = run_spec(spec_path, session=M3Session())
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["inputs"]["api_token"] == "***redacted***"
    assert manifest["inputs"]["n"] == 1
