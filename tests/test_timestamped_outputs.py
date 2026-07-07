"""General ``outputs.timestamped`` support: resolved once per run and shared
by every export path (automatic export, built-in export steps, and any
custom step that reads ``_resolved_output_dir`` from context).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from m3resp.core.session import M3Session
from m3resp.workflows import register_step, run_spec
from m3resp.workflows.utils import resolve_output_dir

_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


@pytest.fixture(autouse=True)
def _temp_steps():
    from m3resp.workflows.registry import STEP_REGISTRY

    @register_step("t.constant", writes=("value",))
    def _constant(**kwargs: Any) -> dict[str, Any]:
        return {"value": 1.0}

    @register_step(
        "t.echo_resolved_dir",
        reads={"output_dir": "_resolved_output_dir"},
        writes=("echoed_dir",),
    )
    def _echo(output_dir: Path | None, **kwargs: Any) -> dict[str, Any]:
        return {"echoed_dir": str(output_dir)}

    yield
    STEP_REGISTRY.pop("t.constant", None)
    STEP_REGISTRY.pop("t.echo_resolved_dir", None)


def _write_spec(path: Path, spec: dict[str, Any]) -> Path:
    spec_path = path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path


class TestResolveOutputDir:
    def test_returns_base_dir_unchanged_when_not_timestamped(self, tmp_path):
        result = resolve_output_dir(tmp_path, timestamped=False)
        assert result == Path(tmp_path)

    def test_appends_a_timestamp_subfolder_when_timestamped(self, tmp_path):
        result = resolve_output_dir(tmp_path, timestamped=True)
        assert result.parent == Path(tmp_path)
        assert _TIMESTAMP_RE.match(result.name)

    def test_uses_the_given_timestamp_instead_of_generating_one(self, tmp_path):
        result = resolve_output_dir(tmp_path, timestamped=True, timestamp="fixed-stamp")
        assert result == Path(tmp_path) / "fixed-stamp"


class TestAutomaticExportHonorsTimestamped:
    def test_summary_lands_in_a_timestamped_subfolder(self, tmp_path):
        spec = {
            "name": "auto-export-timestamped",
            "outputs": {"dir": str(tmp_path), "timestamped": True},
            "steps": [{"uses": "t.constant"}],
        }
        spec_path = _write_spec(tmp_path, spec)

        result = run_spec(spec_path, session=M3Session())

        subfolders = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(subfolders) == 1
        assert _TIMESTAMP_RE.match(subfolders[0].name)
        assert (subfolders[0] / "summary.json").exists()
        assert result.outputs["value"] == 1.0

    def test_no_subfolder_when_not_timestamped(self, tmp_path):
        spec = {
            "name": "auto-export-not-timestamped",
            "outputs": {"dir": str(tmp_path), "timestamped": False},
            "steps": [{"uses": "t.constant"}],
        }
        spec_path = _write_spec(tmp_path, spec)

        run_spec(spec_path, session=M3Session())

        assert (tmp_path / "summary.json").exists()
        assert not any(p.is_dir() for p in tmp_path.iterdir())


class TestRotarcResultHonorsTimestamped:
    def test_result_file_lands_in_a_timestamped_subfolder(self, tmp_path):
        spec = {
            "name": "rotarc-timestamped",
            "experiment": {
                "subject_id": "s01",
                "mode": "quiet",
                "run_identifier": "run-1",
                "selection": "selected",
            },
            "outputs": {
                "dir": str(tmp_path),
                "timestamped": True,
                "summary_json": False,
                "event_csvs": False,
            },
            "steps": [
                {"uses": "t.constant", "out": {"value": "cv"}},
                {"uses": "export.rotarc_result", "in": {"value": "cv"}},
            ],
        }
        spec_path = _write_spec(tmp_path, spec)

        result = run_spec(spec_path, session=M3Session())

        result_path = Path(result.value("result_path"))
        assert result_path.exists()
        # <tmp_path>/<timestamp>/subject_results/run-1/<file>.txt
        timestamp_dir = result_path.parents[2]
        assert timestamp_dir.parent == tmp_path
        assert _TIMESTAMP_RE.match(timestamp_dir.name)


class TestCustomStepsShareTheSameResolvedDirectory:
    def test_custom_step_and_rotarc_result_agree_on_one_timestamp(self, tmp_path):
        spec = {
            "name": "shared-timestamp",
            "experiment": {
                "subject_id": "s01",
                "mode": "quiet",
                "run_identifier": "run-1",
                "selection": "selected",
            },
            "outputs": {
                "dir": str(tmp_path),
                "timestamped": True,
                "summary_json": False,
                "event_csvs": False,
            },
            "steps": [
                {"uses": "t.constant", "out": {"value": "cv"}},
                {"uses": "t.echo_resolved_dir"},
                {"uses": "export.rotarc_result", "in": {"value": "cv"}},
            ],
        }
        spec_path = _write_spec(tmp_path, spec)

        result = run_spec(spec_path, session=M3Session())

        echoed_dir = Path(result.outputs["echoed_dir"])
        result_path = Path(result.value("result_path"))
        rotarc_timestamp_dir = result_path.parents[2]

        assert echoed_dir == rotarc_timestamp_dir
