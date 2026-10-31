"""``m3resp.eit.processing.global_impedance`` gives the global impedance of eitprocessing."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from m3resp.core.session import M3Session
from m3resp.data import Signal
from m3resp.eit.io import read_eit_file
from m3resp.eit.processing import global_impedance
from m3resp.workflows import run_workflow

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")
with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as _handle:
    MANIFEST = json.load(_handle)
DATASETS = sorted(MANIFEST["inputs"])
#: Largest difference, as a part of the largest absolute golden value.
RELATIVE_TOLERANCE = 1e-9


def _golden(dataset: str) -> dict[str, np.ndarray]:
    with np.load(os.path.join(GOLDEN_DIR, dataset, "global_impedance.npz")) as data:
        return {key: data[key] for key in data.files}


def _assert_matches(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=RELATIVE_TOLERANCE * scale
    )


def _input(dataset: str) -> tuple[str, str]:
    entry = MANIFEST["inputs"][dataset]
    return os.path.join(REPO_ROOT, entry["file"]), entry["vendor"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_raw_global_impedance_matches_golden(dataset):
    pixels = read_eit_file(*_input(dataset)).pixel_impedance

    result = global_impedance(pixels)

    golden = _golden(dataset)
    np.testing.assert_array_equal(result.time, golden["raw/time"])
    _assert_matches(result.values, golden["raw/values"])


@pytest.mark.parametrize("dataset", DATASETS)
def test_filtered_global_impedance_equals_eitprocessing(dataset):
    pytest.importorskip("eitprocessing")
    from m3resp.adapters.eitprocessing_adapter import eit_data_to_signal

    path, vendor = _input(dataset)
    spec = {
        "name": "global-impedance",
        "steps": [
            {"uses": "eit.load", "with": {"file_path": path, "vendor": vendor}},
            {
                "uses": "eit.detect_rates",
                "in": {"signal": "raw_eit"},
                "with": {
                    "subject_type": MANIFEST["settings"]["rate_detection"][
                        "subject_type"
                    ]
                },
            },
            {
                "uses": "eit.mdn_filter",
                "in": {"signal": "raw_eit"},
                "with": {"label": "mdn_filtered"},
            },
        ],
    }
    sequence = run_workflow(spec, session=M3Session()).value("eit_sequence")
    filtered = sequence.eit_data["mdn_filtered"]
    upstream = filtered.get_summed_impedance()

    result = global_impedance(
        eit_data_to_signal(filtered, processing_state="intermediate", name="mdn")
    )

    np.testing.assert_array_equal(result.values, upstream.values)
    np.testing.assert_array_equal(result.time, upstream.time)
    assert result.sample_frequency == upstream.sample_frequency
    golden = _golden(dataset)
    np.testing.assert_array_equal(result.time, golden["filtered/time"])
    _assert_matches(result.values, golden["filtered/values"])


def test_result_fields():
    frames = np.arange(12.0).reshape(3, 2, 2)
    frames[1] = np.nan
    frames[2, 0, 0] = np.nan
    pixels = Signal(
        values=frames,
        time=[0.0, 0.05, 0.1],
        sample_frequency=20.0,
        modality="eit",
        source="sentec",
        processing_state="intermediate",
    )

    result = global_impedance(pixels, name="summed")

    np.testing.assert_array_equal(result.values, [6.0, 0.0, 9.0 + 10.0 + 11.0])
    assert result.time is not pixels.time
    np.testing.assert_array_equal(result.time, pixels.time)
    assert (result.name, result.unit, result.sample_frequency) == (
        "summed",
        "a.u.",
        20.0,
    )
    assert (result.modality, result.category, result.channel) == (
        "eit",
        "impedance",
        "global_impedance",
    )
    assert (result.source, result.processing_state) == ("sentec", "intermediate")
    assert result.method == "m3resp.eit.processing.global_impedance"


def test_needs_images():
    with pytest.raises(ValueError, match="shape \\(frames, rows, columns\\)"):
        global_impedance(Signal(values=np.zeros(3), time=np.arange(3.0), name="flat"))
