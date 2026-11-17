"""Post-run summaries (`m3resp.workflows.summaries`) and the small workflow
helpers in `m3resp.workflows.utils` (output paths, filenames, JSON, logging).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from loguru import logger

from m3resp import M3Session
from m3resp.workflows import summaries, utils


@pytest.fixture
def log_lines():
    """Collect loguru messages (without color markup) for the test."""
    lines: list[str] = []
    handler_id = logger.add(lambda message: lines.append(message.record["message"]))
    yield lines
    logger.remove(handler_id)


def _eit_session(**overrides) -> M3Session:
    session = M3Session()
    session.processed["eit"] = {"filter_mode": "lowpass", **overrides}
    return session


def _emg_session(**overrides) -> M3Session:
    session = M3Session()
    session.processed["emg"] = {
        "channel": 1,
        "fs": 2048.0,
        "filter": {"high_pass_hz": 20.0},
        "raw_channel": np.zeros(10),
        "filtered": np.zeros(10),
        **overrides,
    }
    return session


class TestSummarizeEIT:
    def test_minimal_summary_has_only_filter_mode(self):
        assert summaries.summarize_eit(_eit_session()) == {"filter_mode": "lowpass"}

    def test_full_summary_converts_rates_and_counts_values(self):
        pixel_tiv = SimpleNamespace(values=[np.zeros((32, 32)), np.zeros((32, 32))])
        session = _eit_session(
            respiratory_rate_hz=0.25,
            heart_rate_hz=1.5,
            continuous_tiv=[1, 2, 3],
            eeli=[4, 5],
            pixel_tiv=pixel_tiv,
        )
        session.events["eit_breaths"] = ["b1", "b2", "b3", "b4"]

        summary = summaries.summarize_eit(session)

        assert summary == {
            "filter_mode": "lowpass",
            "respiratory_rate_bpm": 15.0,
            "heart_rate_bpm": 90.0,
            "n_eit_breaths": 4,
            "n_continuous_tiv_values": 3,
            "n_eeli_values": 2,
            "pixel_tiv_shape_per_breath": (32, 32),
        }

    @pytest.mark.parametrize(
        "pixel_tiv", [None, SimpleNamespace(values=[])], ids=["none", "empty"]
    )
    def test_missing_pixel_tiv_gives_no_shape(self, pixel_tiv):
        summary = summaries.summarize_eit(_eit_session(pixel_tiv=pixel_tiv))
        assert summary["pixel_tiv_shape_per_breath"] is None


class TestSummarizeEMG:
    def test_minimal_summary(self):
        summary = summaries.summarize_emg(_emg_session())

        assert summary == {
            "channel": 1,
            "fs": 2048.0,
            "filter": {"high_pass_hz": 20.0},
            "n_raw_samples": 10,
            "n_filtered_samples": 10,
            "n_ecg_cleaned_samples": None,
            "n_envelope_samples": None,
            "n_ventilator_breaths": 0,
        }

    def test_full_summary_with_events_and_postprocessing(self):
        session = _emg_session(ecg_cleaned=np.zeros(8), envelope=np.zeros(6))
        session.events["emg_breaths"] = [1, 2]
        session.events["ventilator_breaths"] = [1, 2, 3]
        session.parameters["emg_postprocessing"] = {
            "available": {"features": {"amplitude": True, "time_product": True}},
            "computed": {"features": {"amplitude": [1.0]}, "quality_assessment": {}},
            "skipped": {"quality_assessment.snr_pseudo": "no baseline"},
        }

        summary = summaries.summarize_emg(session)

        assert summary["n_ecg_cleaned_samples"] == 8
        assert summary["n_envelope_samples"] == 6
        assert summary["n_emg_breaths"] == 2
        assert summary["n_ventilator_breaths"] == 3
        assert summary["postprocessing_available"] == {"features": 2}
        # Categories with no results are left out.
        assert summary["postprocessing_computed"] == {"features": ["amplitude"]}
        assert summary["postprocessing_skipped"] == {
            "quality_assessment.snr_pseudo": "no baseline"
        }


class TestSummarizeMultimodal:
    def test_nothing_included_gives_empty_summary(self):
        assert (
            summaries.summarize_multimodal(
                M3Session(), include_eit=False, include_emg=False
            )
            == {}
        )

    def test_combines_eit_summary_with_selected_emg_keys(self):
        session = _emg_session()
        session.processed["eit"] = {"filter_mode": "none"}
        session.events["emg_breaths"] = [1]
        session.parameters["emg_postprocessing"] = {
            "available": {"features": {"amplitude": True}},
            "computed": {},
            "skipped": {},
        }

        summary = summaries.summarize_multimodal(
            session, include_eit=True, include_emg=True
        )

        assert summary == {
            "filter_mode": "none",
            "n_emg_breaths": 1,
            "n_ventilator_breaths": 0,
            "emg_postprocessing_available": {"features": 1},
            "emg_postprocessing_computed": {},
            "emg_postprocessing_skipped": {},
        }


class TestOutputPaths:
    def test_untimestamped_dir_is_the_base(self, tmp_path):
        assert utils.resolve_output_dir(tmp_path, timestamped=False) == tmp_path

    def test_explicit_timestamp_is_appended(self, tmp_path):
        resolved = utils.resolve_output_dir(
            str(tmp_path), timestamped=True, timestamp="20260101_120000"
        )
        assert resolved == Path(tmp_path) / "20260101_120000"

    def test_default_timestamp_is_used_when_none_given(self, tmp_path):
        resolved = utils.resolve_output_dir(tmp_path, timestamped=True)
        assert resolved.parent == tmp_path
        assert re.fullmatch(r"\d{8}_\d{6}", resolved.name)

    def test_default_run_timestamp_format(self):
        assert re.fullmatch(r"\d{8}_\d{6}", utils.default_run_timestamp())

    @pytest.mark.parametrize(
        ("timepoint", "expected"),
        [
            ("T1", "S01-pocc-T1-selected.txt"),
            (None, "S01-pocc-selected.txt"),
            ("", "S01-pocc-selected.txt"),
        ],
    )
    def test_subject_result_filename(self, timepoint, expected):
        assert (
            utils.subject_result_filename("S01", "pocc", timepoint, "selected")
            == expected
        )

    def test_write_json_is_sorted_and_indented(self, tmp_path):
        path = tmp_path / "out.json"
        utils.write_json(path, {"b": 1, "a": {"d": 2, "c": 3}})

        text = path.read_text(encoding="utf-8")
        assert json.loads(text) == {"a": {"c": 3, "d": 2}, "b": 1}
        assert text.index('"a"') < text.index('"b"')
        assert '\n  "a"' in text


class TestLogWorkflowSummary:
    def test_logs_title_output_dir_and_plain_keys(self, log_lines, tmp_path):
        utils.log_workflow_summary("EMG run", tmp_path, {"n_emg_breaths": 12})

        joined = "\n".join(log_lines)
        assert "EMG run" in joined
        assert str(tmp_path) in joined
        assert "12" in joined

    def test_postprocessing_entries_are_expanded(self, log_lines, tmp_path):
        utils.log_workflow_summary(
            "EMG run",
            tmp_path,
            {
                "emg_postprocessing_available": {"quality_assessment": 3},
                "emg_postprocessing_computed": {
                    "features": ["amplitude", "time_product"],
                    "baseline": [],
                },
                "emg_postprocessing_skipped": {"event_detection": "no ventilator"},
            },
        )

        joined = "\n".join(log_lines)
        assert "quality assessment (3)" in joined
        assert "amplitude, time_product" in joined
        assert "none" in joined  # the empty baseline list
        assert "no ventilator" in joined

    def test_empty_postprocessing_entries_log_none(self, log_lines, tmp_path):
        utils.log_workflow_summary(
            "run",
            tmp_path,
            {
                "postprocessing_available": {},
                "postprocessing_computed": {},
                "postprocessing_skipped": {},
            },
        )

        assert log_lines.count("  none") == 2
        assert any(line.endswith("none") for line in log_lines)

    def test_other_mappings_are_logged_as_plain_values(self, log_lines, tmp_path):
        utils.log_workflow_summary("run", tmp_path, {"filter": {"high_pass_hz": 20}})

        assert any("high_pass_hz" in line for line in log_lines)
