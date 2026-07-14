"""Milestone 2.6 - structured export (plan_stage2.md Sec 22).

Acceptance criterion under test: a complete M3Session can be exported and
re-inspected outside Python.
"""

from __future__ import annotations

import csv
import json

from m3resp import M3Session
from m3resp.core.events import BreathEvent
from m3resp.data import ParameterResult, QualityFlag, Signal
from m3resp.data.linked_breath import LinkedBreath


def _populated_session() -> M3Session:
    session = M3Session()
    session.signals.add(
        Signal(values=[1.0, 2.0], time=[0.0, 1.0], modality="eit", unit="a.u.")
    )
    session.parameter_results.add(
        ParameterResult(name="tiv", value=1.5, modality="eit", unit="a.u.")
    )
    session.quality.add(
        QualityFlag(name="snr_check", passed=True, severity="info", modality="eit")
    )
    eit_breath = BreathEvent(
        modality="eit", start_time=1.0, end_time=2.0, peak_time=1.5
    )
    emg_breath = BreathEvent(
        modality="emg", start_time=1.1, end_time=2.1, peak_time=1.6
    )
    session.linked_breaths.append(
        LinkedBreath(breaths={"eit": eit_breath, "emg": emg_breath}, confidence=0.9)
    )
    return session


def test_structured_export_writes_all_new_files(tmp_path):
    session = _populated_session()

    output_dir = session.export_summary(tmp_path)

    for filename in (
        "summary.json",
        "session_metadata.json",
        "processing_history.json",
        "signals_manifest.csv",
        "parameter_results.csv",
        "quality_flags.csv",
        "linked_breaths.csv",
    ):
        assert (output_dir / filename).exists(), f"missing {filename}"


def test_signals_manifest_csv_has_one_row_per_signal(tmp_path):
    session = _populated_session()

    output_dir = session.export_summary(tmp_path)

    with (output_dir / "signals_manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["modality"] == "eit"
    assert rows[0]["unit"] == "a.u."


def test_linked_breaths_csv_flattens_both_modalities(tmp_path):
    session = _populated_session()

    output_dir = session.export_summary(tmp_path)

    with (output_dir / "linked_breaths.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["modalities"] == "eit+emg"
    assert float(row["eit_start_time"]) == 1.0
    assert float(row["emg_start_time"]) == 1.1
    assert float(row["confidence"]) == 0.9


def test_processing_history_json_contains_provenance(tmp_path):
    session = M3Session()
    session._record("demo_action", parameters={"x": 1})

    output_dir = session.export_summary(tmp_path)

    payload = json.loads((output_dir / "processing_history.json").read_text())

    assert payload["provenance"][0]["action"] == "demo_action"


def test_structured_export_false_skips_new_files(tmp_path):
    from m3resp.export.session_export import export_session_summary

    session = _populated_session()

    output_dir = export_session_summary(session, tmp_path, structured_export=False)

    assert (output_dir / "summary.json").exists()
    assert not (output_dir / "signals_manifest.csv").exists()
    assert not (output_dir / "session_metadata.json").exists()


def test_empty_collections_do_not_produce_empty_csv_files(tmp_path):
    session = M3Session()

    output_dir = session.export_summary(tmp_path)

    assert (output_dir / "session_metadata.json").exists()
    assert (output_dir / "processing_history.json").exists()
    assert not (output_dir / "signals_manifest.csv").exists()
    assert not (output_dir / "parameter_results.csv").exists()
    assert not (output_dir / "quality_flags.csv").exists()
    assert not (output_dir / "linked_breaths.csv").exists()
