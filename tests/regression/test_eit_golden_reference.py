"""Compare the EIT results of m3resp and of eitprocessing with the golden files.

The golden files in ``tests/regression/golden/eit/`` hold the results of
eitprocessing 1.8.7 on three test files (Draeger, Sentec, Timpel). They are
made by ``scripts/make_eit_golden_files.py``. This file checks three things:

1. The input files are the files the golden files were made from (SHA-256).
2. eitprocessing itself, run again with the same settings, still gives the
   golden results. This needs eitprocessing 1.8.7.
3. The m3resp EIT steps give the golden results. The checks read only the
   m3resp result types, so the same test checks the m3resp code before and
   after the eitprocessing code is copied into m3resp.

Comparison rule (also in ``manifest.json``): shapes, counts, indices and
masks exact; other floating-point values within 1e-9 times the largest
absolute value of the same array; NaN must stay NaN.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from typing import Any

import numpy as np
import pytest

from m3resp.core.session import M3Session
from m3resp.workflows.engine import run_workflow

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "make_eit_golden_files.py")

RELATIVE_TOLERANCE = 1e-9


def _load_manifest() -> dict[str, Any]:
    with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as handle:
        return json.load(handle)


MANIFEST = _load_manifest()
DATASETS = sorted(MANIFEST["inputs"])


def _golden(dataset: str, name: str) -> dict[str, np.ndarray]:
    """Return all arrays of one golden ``.npz`` file."""

    with np.load(os.path.join(GOLDEN_DIR, dataset, f"{name}.npz")) as data:
        return {key: data[key] for key in data.files}


def _load_script() -> Any:
    """Import ``scripts/make_eit_golden_files.py`` as a module."""

    spec = importlib.util.spec_from_file_location("make_eit_golden_files", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_matches(actual: Any, expected: np.ndarray, what: str) -> None:
    """Check one result against its golden array with the comparison rule.

    Args:
        actual (Any): The result to check; anything NumPy turns into an array.
        expected (np.ndarray): The golden array.
        what (str): Name of the result, shown when the check fails.

    Raises:
        AssertionError: If the shapes differ, or the values differ by more
            than the comparison rule allows.
    """

    actual = np.asarray(actual)
    assert actual.shape == expected.shape, (
        f"{what}: shape {actual.shape}, golden {expected.shape}"
    )
    if expected.dtype.kind != "f":
        np.testing.assert_array_equal(actual, expected, err_msg=what)
        return
    actual = actual.astype(float)
    np.testing.assert_array_equal(
        np.isnan(actual), np.isnan(expected), err_msg=f"{what}: NaN"
    )
    finite = expected[np.isfinite(expected)]
    scale = float(np.max(np.abs(finite))) if finite.size else 0.0
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0,
        atol=RELATIVE_TOLERANCE * scale,
        equal_nan=True,
        err_msg=what,
    )


def assert_mask_matches(actual: Any, expected: np.ndarray, what: str) -> None:
    """Check a lung-space mask exactly (1 inside, NaN outside).

    Args:
        actual (Any): The mask to check.
        expected (np.ndarray): The golden mask.
        what (str): Name of the mask, shown when the check fails.

    Raises:
        AssertionError: If the shapes or any pixel differ.
    """

    actual = np.asarray(actual, dtype=float)
    assert actual.shape == expected.shape, what
    np.testing.assert_array_equal(actual, expected, err_msg=what)


# 1. Input files


@pytest.mark.parametrize("dataset", DATASETS)
def test_input_file_is_the_file_of_the_golden_files(dataset: str) -> None:
    entry = MANIFEST["inputs"][dataset]
    digest = hashlib.sha256()
    with open(os.path.join(REPO_ROOT, entry["file"]), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    assert digest.hexdigest() == entry["sha256"], (
        f"{entry['file']} changed; make the golden files again"
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_every_golden_file_listed_in_the_manifest_exists(dataset: str) -> None:
    for name, keys in MANIFEST["inputs"][dataset]["golden_files"].items():
        stored = _golden(dataset, name.removesuffix(".npz"))
        assert sorted(stored) == keys, f"{dataset}/{name}"


# 2. eitprocessing against the golden files


@pytest.fixture(scope="module")
def eitprocessing_results() -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Run the golden-file chain of eitprocessing again, for every test file."""

    pytest.importorskip("eitprocessing")
    reference = MANIFEST["eitprocessing"]["version"]
    installed = importlib.metadata.version("eitprocessing")
    if installed != reference:
        pytest.skip(
            f"eitprocessing {installed} installed; golden files are from {reference}"
        )
    script = _load_script()
    results = {}
    for dataset in DATASETS:
        entry = MANIFEST["inputs"][dataset]
        path = os.path.join(REPO_ROOT, entry["file"])
        results[dataset], _ = script.run_chain(path, entry["vendor"])
    return results


@pytest.mark.parametrize("dataset", DATASETS)
def test_eitprocessing_still_gives_the_golden_results(
    dataset: str, eitprocessing_results: dict
) -> None:
    for name, arrays in eitprocessing_results[dataset].items():
        golden = _golden(dataset, name)
        assert sorted(arrays) == sorted(golden), f"{dataset}/{name}: names differ"
        for key, expected in golden.items():
            assert_matches(arrays[key], expected, f"{dataset}/{name}/{key}")


# 3. m3resp against the golden files


def _spec(path: str, vendor: str) -> dict[str, Any]:
    """The m3resp workflow with the settings of the golden files."""

    settings = MANIFEST["settings"]
    return {
        "name": "eit-golden-reference",
        "steps": [
            {"uses": "eit.load", "with": {"file_path": path, "vendor": vendor}},
            {
                "uses": "eit.detect_rates",
                "in": {"signal": "raw_eit"},
                "with": {
                    "subject_type": settings["rate_detection"]["subject_type"],
                    "capture": True,
                },
            },
            {
                "uses": "eit.mdn_filter",
                "in": {"signal": "raw_eit"},
                "with": {"label": "mdn_filtered"},
            },
            {"uses": "eit.global_impedance", "in": {"signal": "filtered_eit"}},
            {
                "uses": "eit.detect_breaths",
                "in": {"signal": "global_impedance"},
                "with": {
                    "min_duration_s": settings["breath_detection"]["minimum_duration_s"]
                },
            },
            {"uses": "eit.continuous_tiv"},
            {"uses": "eit.eeli"},
            {"uses": "eit.pixel_tiv"},
            {"uses": "eit.pixel_breaths"},
            {
                "uses": "eit.roi_tiv_lungspace",
                "with": {"threshold": settings["tiv_lungspace"]["threshold"]},
            },
            {
                "uses": "eit.roi_amplitude_lungspace",
                "with": {"threshold": settings["amplitude_lungspace"]["threshold"]},
            },
            {
                "uses": "eit.roi_watershed",
                "with": {
                    "threshold_fraction": settings["watershed_lungspace"][
                        "threshold_fraction"
                    ]
                },
            },
            {
                "uses": "eit.roi_filter_by_size",
                "in": {"mask": "watershed_lungspace_mask"},
                "with": {
                    "min_region_size": settings["filter_by_size"]["min_region_size"]
                },
            },
        ],
    }


@pytest.fixture(scope="module")
def m3resp_results() -> dict[str, Any]:
    """Run the m3resp EIT steps once for every test file."""

    pytest.importorskip("eitprocessing")
    results = {}
    for dataset in DATASETS:
        entry = MANIFEST["inputs"][dataset]
        path = os.path.join(REPO_ROOT, entry["file"])
        results[dataset] = run_workflow(
            _spec(path, entry["vendor"]), session=M3Session()
        )
    return results


@pytest.mark.parametrize("dataset", DATASETS)
def test_m3resp_reader_matches_golden(dataset: str, m3resp_results: dict) -> None:
    result = m3resp_results[dataset]
    golden = _golden(dataset, "reader")
    pixels = result.value("raw_pixel_impedance_signal")
    assert_matches(pixels.time, golden["time"], "time")
    assert_matches(pixels.values, golden["pixel_impedance"], "pixel_impedance")
    assert_matches(
        pixels.sample_frequency, golden["sample_frequency"], "sample_frequency"
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_m3resp_ventilator_waveforms_from_the_eit_file_match_golden(
    dataset: str,
) -> None:
    """The airway pressure, flow and volume stored in the EIT file.

    m3resp reads them with ``session.load_ventilator(..., source="eit")``.
    A Sentec file holds no waveforms.
    """

    pytest.importorskip("eitprocessing")
    entry = MANIFEST["inputs"][dataset]
    path = os.path.join(REPO_ROOT, entry["file"])
    golden = _golden(dataset, "reader")
    session = M3Session()
    if entry["vendor"] == "sentec":
        from m3resp.core.exceptions import UnsupportedWorkflowError

        assert not any(key.startswith("continuous/") for key in golden)
        with pytest.raises(UnsupportedWorkflowError, match="no ventilator channel"):
            session.load_ventilator(path, source="eit", vendor="sentec")
        return

    payload = session.load_ventilator(path, source="eit", vendor=entry["vendor"])

    metadata = payload["metadata"]
    for row, label, unit in zip(
        payload["array"], metadata["labels"], metadata["units"], strict=True
    ):
        assert unit == str(golden[f"continuous/{label}/unit"]), label
        assert_matches(row, golden[f"continuous/{label}/values"], f"{label}/values")
        assert_matches(
            metadata["time"], golden[f"continuous/{label}/time"], f"{label}/time"
        )
    assert len(metadata["labels"]) == 3


@pytest.mark.parametrize("dataset", DATASETS)
def test_m3resp_global_impedance_matches_golden(
    dataset: str, m3resp_results: dict
) -> None:
    result = m3resp_results[dataset]
    golden = _golden(dataset, "global_impedance")
    raw = result.value("raw_global_impedance_signal")
    assert_matches(raw.time, golden["raw/time"], "raw/time")
    assert_matches(raw.values, golden["raw/values"], "raw/values")
    filtered = result.value("global_impedance")
    assert_matches(filtered.time, golden["filtered/time"], "filtered/time")
    assert_matches(filtered.values, golden["filtered/values"], "filtered/values")


@pytest.mark.parametrize("dataset", DATASETS)
def test_m3resp_rates_match_golden(dataset: str, m3resp_results: dict) -> None:
    result = m3resp_results[dataset]
    golden = _golden(dataset, "rates")
    assert_matches(
        result.value("respiratory_rate_hz"), golden["respiratory_rate_hz"], "RR"
    )
    assert_matches(result.value("heart_rate_hz"), golden["heart_rate_hz"], "HR")
    captures = result.value("rate_captures")
    for key, expected in golden.items():
        if key.startswith("captures/"):
            assert_matches(captures[key.removeprefix("captures/")], expected, key)


@pytest.mark.parametrize("dataset", DATASETS)
def test_m3resp_mdn_filter_matches_golden(dataset: str, m3resp_results: dict) -> None:
    result = m3resp_results[dataset]
    golden = _golden(dataset, "mdn_filter")
    signal = result.value("filtered_eit_signal")
    assert_matches(signal.values, golden["pixel_impedance"], "pixel_impedance")
    captures = result.value("filter_captures")
    for key, expected in golden.items():
        if key.startswith("captures/"):
            assert_matches(captures[key.removeprefix("captures/")], expected, key)


@pytest.mark.parametrize("dataset", DATASETS)
def test_m3resp_breaths_match_golden(dataset: str, m3resp_results: dict) -> None:
    result = m3resp_results[dataset]
    golden = _golden(dataset, "breaths")["breaths"]
    breaths = result.value("continuous_tiv_result").intervals
    actual = [(b.start_time, b.extremum_time, b.end_time) for b in breaths]
    assert_matches(np.asarray(actual, dtype=float).reshape(-1, 3), golden, "breaths")


@pytest.mark.parametrize("dataset", DATASETS)
def test_m3resp_tiv_and_eeli_match_golden(dataset: str, m3resp_results: dict) -> None:
    result = m3resp_results[dataset]
    golden = _golden(dataset, "tiv_eeli")
    tiv = result.value("continuous_tiv_result")
    assert_matches(tiv.values, golden["tiv/values"], "tiv/values")
    # TIV is placed at the middle of each breath, EELI at its end.
    assert_matches(
        [b.extremum_time for b in tiv.intervals], golden["tiv/time"], "tiv/time"
    )
    eeli = result.value("eeli_result")
    assert_matches(eeli.values, golden["eeli/values"], "eeli/values")
    assert_matches(
        [b.end_time for b in eeli.intervals], golden["eeli/time"], "eeli/time"
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_m3resp_pixel_results_match_golden(dataset: str, m3resp_results: dict) -> None:
    result = m3resp_results[dataset]
    golden = _golden(dataset, "pixel")
    pixel_tiv = result.value("pixel_tiv_result")
    values = np.stack([pixel_map.values for pixel_map in pixel_tiv.values])
    assert_matches(values, golden["tiv/values"], "tiv/values")
    # With continuous timing every pixel of a breath has the same time.
    assert_matches(
        [b.extremum_time for b in pixel_tiv.intervals],
        golden["tiv/time"][:, 0, 0],
        "tiv/time",
    )
    timing = result.value("pixel_breath_timing_result")
    assert_matches(
        np.stack(timing.values), golden["breaths/landmarks"], "breaths/landmarks"
    )
    start_end = [(b.start_time, b.end_time) for b in timing.intervals]
    assert_matches(
        np.asarray(start_end, dtype=float).reshape(-1, 2),
        golden["breaths/start_end"],
        "breaths/start_end",
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_m3resp_lung_space_masks_match_golden(
    dataset: str, m3resp_results: dict
) -> None:
    result = m3resp_results[dataset]
    golden = _golden(dataset, "roi")
    for name, value in (
        ("tiv_lungspace", "tiv_lungspace_result"),
        ("amplitude_lungspace", "amplitude_lungspace_result"),
        ("watershed_lungspace", "watershed_lungspace_result"),
        ("filter_by_size", "size_filtered_roi_result"),
    ):
        assert_mask_matches(result.value(value).values, golden[f"{name}/mask"], name)


# 4. m3resp through session.preprocess_eit() against the golden files


@pytest.fixture(scope="module")
def preprocess_results() -> dict[str, Any]:
    """Run ``session.preprocess_eit()`` once for every test file."""

    pytest.importorskip("eitprocessing")
    settings = MANIFEST["settings"]
    results = {}
    for dataset in DATASETS:
        entry = MANIFEST["inputs"][dataset]
        session = M3Session()
        session.load_eit(os.path.join(REPO_ROOT, entry["file"]), entry["vendor"])
        results[dataset] = session.preprocess_eit(
            subject_type=settings["rate_detection"]["subject_type"],
            breath_min_duration_seconds=settings["breath_detection"][
                "minimum_duration_s"
            ],
        )
    return results


@pytest.mark.parametrize("dataset", DATASETS)
def test_preprocess_rates_match_golden(dataset: str, preprocess_results: dict) -> None:
    result = preprocess_results[dataset]
    golden = _golden(dataset, "rates")
    assert_matches(result["respiratory_rate_hz"], golden["respiratory_rate_hz"], "RR")
    assert_matches(result["heart_rate_hz"], golden["heart_rate_hz"], "HR")
    for key, expected in golden.items():
        if key.startswith("captures/"):
            assert_matches(
                result["rate_captures"][key.removeprefix("captures/")], expected, key
            )


@pytest.mark.parametrize("dataset", DATASETS)
def test_preprocess_mdn_filter_matches_golden(
    dataset: str, preprocess_results: dict
) -> None:
    result = preprocess_results[dataset]
    golden = _golden(dataset, "mdn_filter")
    assert result["filter_mode"] == "mdn"
    assert_matches(
        result["filtered_eit"].pixel_impedance, golden["pixel_impedance"], "pixels"
    )
    for key, expected in golden.items():
        if key.startswith("captures/"):
            assert_matches(
                result["filter_captures"][key.removeprefix("captures/")], expected, key
            )
    stored = result["sequence"].eit_data["mdn_filtered"]
    assert stored is result["filtered_eit"]
    assert stored.name == "MDN-filtered EIT data"


@pytest.mark.parametrize("dataset", DATASETS)
def test_preprocess_breaths_match_golden(
    dataset: str, preprocess_results: dict
) -> None:
    result = preprocess_results[dataset]
    golden = _golden(dataset, "breaths")["breaths"]
    breaths = result["breath_intervals"].intervals
    actual = [(b.start_time, b.extremum_time, b.end_time) for b in breaths]
    assert_matches(np.asarray(actual, dtype=float).reshape(-1, 3), golden, "breaths")
    stored = result["sequence"].interval_data["eit_breaths"].values
    upstream = [(b.start_time, b.middle_time, b.end_time) for b in stored]
    assert_matches(np.asarray(upstream, dtype=float).reshape(-1, 3), golden, "Sequence")


@pytest.mark.parametrize("dataset", DATASETS)
def test_preprocess_tiv_and_eeli_match_golden(
    dataset: str, preprocess_results: dict
) -> None:
    result = preprocess_results[dataset]
    golden = _golden(dataset, "tiv_eeli")
    tiv, eeli = result["continuous_tiv"], result["eeli"]
    breaths = result["breath_intervals"].intervals
    assert_matches(tiv.values, golden["tiv/values"], "tiv/values")
    assert_matches(eeli.values, golden["eeli/values"], "eeli/values")
    # TIV and EELI hold the breaths of `breath_intervals`.
    assert all(a is b for a, b in zip(tiv.intervals, breaths, strict=True))
    assert all(a is b for a, b in zip(eeli.intervals, breaths, strict=True))
    # The Sequence holds no TIV or EELI.
    assert "continuous_tivs" not in result["sequence"].sparse_data
    assert "continuous_eelis" not in result["sequence"].sparse_data
