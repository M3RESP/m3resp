"""Registered export steps (`m3resp.workflows.steps.export`), called directly."""

from __future__ import annotations

import json
import re

import pytest

from m3resp import M3Session
from m3resp.workflows.spec import SpecExperimentConfig, SpecOutputsConfig
from m3resp.workflows.steps import export


def test_scalar_file_writes_fixed_precision_and_creates_dirs(tmp_path):
    target = tmp_path / "nested" / "value.txt"

    result = export.scalar_file(0.123456789, path=str(target), precision=3)

    assert result == {"result_path": str(target)}
    assert target.read_text(encoding="utf-8") == "0.123"


def test_json_file_writes_sorted_json(tmp_path):
    target = tmp_path / "nested" / "summary.json"

    result = export.json_file({"b": 1, "a": 2}, path=str(target))

    assert result == {"json_path": str(target)}
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2, "b": 1}


def test_session_summary_creates_the_directory(tmp_path):
    target = tmp_path / "export"

    result = export.session_summary(
        M3Session(),
        output_dir=str(target),
        event_csvs=False,
        parameters_csv=False,
        postprocessing=False,
        structured_export=False,
    )

    assert result == {"output_dir": str(target)}
    assert (target / "summary.json").is_file()


class TestRotarcResult:
    def _call(self, tmp_path, **experiment):
        fields = {
            "subject_id": "S01",
            "mode": "pocc",
            "timepoint": "T1",
            "run_identifier": "run-1",
            **experiment,
        }
        return export.rotarc_result(
            0.25,
            _spec_outputs=SpecOutputsConfig(
                summary_json=False, event_csvs=False, parameters_csv=False
            ),
            _spec_experiment=SpecExperimentConfig(**fields),
            _resolved_output_dir=str(tmp_path),
            session=M3Session(),
            precision=4,
        )

    def test_writes_result_file_and_summary(self, tmp_path):
        result = self._call(tmp_path)

        output_dir = tmp_path / "subject_results" / "run-1"
        result_path = output_dir / "S01-pocc-T1-selected.txt"
        assert result == {"result_path": str(result_path)}
        assert result_path.read_text(encoding="utf-8") == "0.2500"
        summary = json.loads((output_dir / "rotarc_summary.json").read_text("utf-8"))
        assert summary["breath_duration_cv"] == 0.25
        assert summary["subject_id"] == "S01"

    def test_requires_an_output_directory(self, tmp_path):
        with pytest.raises(ValueError, match=r"requires 'outputs\.dir'"):
            export.rotarc_result(
                0.25,
                _spec_outputs=SpecOutputsConfig(),
                _spec_experiment=SpecExperimentConfig(
                    subject_id="S01", run_identifier="run-1"
                ),
                _resolved_output_dir=None,
                session=M3Session(),
            )

    @pytest.mark.parametrize(
        ("missing", "field"),
        [
            ({"subject_id": None}, "experiment.subject_id"),
            ({"run_identifier": ""}, "experiment.run_identifier"),
        ],
    )
    def test_requires_subject_and_run_identifier(self, tmp_path, missing, field):
        with pytest.raises(ValueError, match=re.escape(f"requires '{field}'")):
            self._call(tmp_path, **missing)
