"""Milestone 2.6 - structured export (plan_stage2.md Sec 22).

Acceptance criterion under test: a complete M3Session can be exported and
re-inspected outside Python.
"""

from __future__ import annotations

import ast
import csv
import json

import numpy as np

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


# -- Phase 5.3 / 7 (plan/stage2/1_eit_gap_migration_implementation_plan.md):
# array-valued ParameterResult export -----------------------------------------


def _read_parameter_results_csv(output_dir) -> list[dict]:
    with (output_dir / "parameter_results.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_scalar_only_run_produces_the_same_csv_content_as_before(tmp_path):
    session = M3Session()
    session.parameter_results.add(
        ParameterResult(name="tiv", value=1.5, modality="eit", unit="a.u.")
    )

    output_dir = session.export_summary(tmp_path)

    assert not (output_dir / "parameter_result_arrays.npz").exists()
    rows = _read_parameter_results_csv(output_dir)
    assert rows == [
        {
            "name": "tiv",
            "value": "1.5",
            "modality": "eit",
            "unit": "a.u.",
            "breath_id": "",
            "breath_ids": "",
            "start_time": "",
            "end_time": "",
            "region": "",
            "channel": "",
            "method": "",
            "metadata": "{}",
        }
    ]


def test_array_result_round_trips_shape_dtype_nan_pattern_and_values(tmp_path):
    session = M3Session()
    value = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]], dtype=np.float32)
    session.parameter_results.add(
        ParameterResult(
            name="pixel_tivs",
            value=value,
            modality="eit",
            unit="a.u.",
            method="eitprocessing.TIV",
            metadata={"axes": ["row", "column"], "operation": "eit.pixel_tiv"},
        )
    )

    output_dir = session.export_summary(tmp_path)

    rows = _read_parameter_results_csv(output_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["value"] == ""
    assert row["value_file"] == "parameter_result_arrays.npz"
    assert row["array_key"] == "pixel_tivs_0"
    assert row["shape"] == "[2, 3]"
    assert row["dtype"] == str(value.dtype)
    assert row["method"] == "eitprocessing.TIV"

    metadata = ast.literal_eval(row["metadata"])
    assert metadata["axes"] == ["row", "column"]
    assert metadata["operation"] == "eit.pixel_tiv"

    with np.load(output_dir / "parameter_result_arrays.npz") as archive:
        reloaded = archive["pixel_tivs_0"]
        assert reloaded.shape == value.shape
        assert reloaded.dtype == value.dtype
        np.testing.assert_array_equal(reloaded, value)  # NaN-aware


def test_non_scalar_metadata_time_moves_into_the_archive_with_a_reference(tmp_path):
    session = M3Session()
    value = np.array([1.0, 2.0, np.nan])
    time = np.array([0.0, 1.0, 2.0])
    session.parameter_results.add(
        ParameterResult(
            name="continuous_eelis",
            value=value,
            modality="eit",
            unit="a.u.",
            metadata={"time": time.tolist()},
        )
    )

    output_dir = session.export_summary(tmp_path)

    rows = _read_parameter_results_csv(output_dir)
    row = rows[0]
    assert row["time_array_key"] == "continuous_eelis_0_time"
    metadata = ast.literal_eval(row["metadata"])
    assert metadata["time"] == "npz:continuous_eelis_0_time"

    with np.load(output_dir / "parameter_result_arrays.npz") as archive:
        np.testing.assert_array_equal(archive["continuous_eelis_0_time"], time)


def test_repeated_result_names_get_distinct_stable_archive_keys(tmp_path):
    session = M3Session()
    for _ in range(2):
        session.parameter_results.add(
            ParameterResult(
                name="tiv_lungspace_mask",
                value=np.array([[1.0, np.nan]]),
                modality="eit",
            )
        )

    output_dir = session.export_summary(tmp_path)

    rows = _read_parameter_results_csv(output_dir)
    array_keys = [row["array_key"] for row in rows]
    assert array_keys == ["tiv_lungspace_mask_0", "tiv_lungspace_mask_1"]
    with np.load(output_dir / "parameter_result_arrays.npz") as archive:
        assert {"tiv_lungspace_mask_0", "tiv_lungspace_mask_1"} <= set(archive.keys())


def test_manual_export_without_a_processing_run_still_writes_the_archive_but_does_not_link_it(
    tmp_path,
):
    from m3resp.datamodel.recorder import DataModelRecorder

    session = M3Session()
    session.datamodel = DataModelRecorder(session)
    session.parameter_results.add(
        ParameterResult(name="mask", value=np.array([1.0, np.nan]), modality="eit")
    )

    output_dir = session.export_summary(tmp_path)  # no processing_run_id

    assert (output_dir / "parameter_result_arrays.npz").exists()
    assert session.datamodel.store.data_files == {}
