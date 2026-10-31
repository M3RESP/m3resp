"""A ``Signal`` made from an eitprocessing ``EITData`` or ``ContinuousData`` gives the same object back."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from m3resp.adapters.eitprocessing_adapter import (
    EITPROCESSING_FIELDS,
    continuous_data_to_signal,
    eit_data_to_signal,
    signal_to_continuous_data,
    signal_to_eit_data,
)
from m3resp.core.session import M3Session
from m3resp.data import Signal
from m3resp.eit.io import read_eit_file
from m3resp.workflows import run_workflow

pytest.importorskip("eitprocessing")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(
    os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit", "manifest.json"),
    encoding="utf-8",
) as _handle:
    MANIFEST = json.load(_handle)
DATASETS = sorted(MANIFEST["inputs"])


def _chain(dataset: str) -> dict:
    """The EIT steps of the golden files up to the global impedance."""

    entry = MANIFEST["inputs"][dataset]
    spec = {
        "name": "eit-signal-conversion",
        "steps": [
            {
                "uses": "eit.load",
                "with": {
                    "file_path": os.path.join(REPO_ROOT, entry["file"]),
                    "vendor": entry["vendor"],
                },
            },
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
            {"uses": "eit.global_impedance", "in": {"signal": "filtered_eit"}},
        ],
    }
    result = run_workflow(spec, session=M3Session())
    sequence = result.value("eit_sequence")
    filtered = sequence.eit_data["mdn_filtered"]
    return {
        "result": result,
        "eit_data": [sequence.eit_data["raw"], filtered],
        "continuous_data": [
            filtered.get_summed_impedance(),
            *sequence.continuous_data.values(),
        ],
    }


@pytest.fixture(scope="module")
def chains() -> dict[str, dict]:
    return {dataset: _chain(dataset) for dataset in DATASETS}


def _assert_same_eit_data(back, original) -> None:
    assert back == original
    for field in ("label", "name", "description", "path", "vendor", "nframes"):
        assert getattr(back, field) == getattr(original, field), field
    assert back.sample_frequency == original.sample_frequency
    assert type(back.sample_frequency) is type(original.sample_frequency)


def _assert_same_continuous_data(back, original) -> None:
    assert back == original
    for field in ("label", "name", "description", "unit", "category", "parameters"):
        assert getattr(back, field) == getattr(original, field), field
    assert back.locked == original.locked
    assert back.sample_frequency == original.sample_frequency
    assert type(back.sample_frequency) is type(original.sample_frequency)


@pytest.mark.parametrize("dataset", DATASETS)
def test_eit_data_round_trip(chains, dataset):
    for original in chains[dataset]["eit_data"]:
        signal = eit_data_to_signal(
            original, processing_state="intermediate", name=original.label
        )

        back = signal_to_eit_data(signal)

        _assert_same_eit_data(back, original)
        assert back.pixel_impedance is original.pixel_impedance


@pytest.mark.parametrize("dataset", DATASETS)
def test_continuous_data_round_trip(chains, dataset):
    originals = chains[dataset]["continuous_data"]
    assert len(originals) >= 2

    for original in originals:
        signal = continuous_data_to_signal(
            original,
            modality="eit",
            channel="global_impedance",
            processing_state="intermediate",
            name=original.label,
        )

        back = signal_to_continuous_data(signal)

        _assert_same_continuous_data(back, original)


@pytest.mark.parametrize("dataset", DATASETS)
def test_step_outputs_give_the_eitprocessing_objects_of_the_sequence(chains, dataset):
    result = chains[dataset]["result"]
    sequence = result.value("eit_sequence")

    raw = signal_to_eit_data(result.value("raw_eit"))
    filtered = signal_to_eit_data(result.value("filtered_eit"))
    summed = signal_to_continuous_data(result.value("global_impedance"))

    _assert_same_eit_data(raw, sequence.eit_data["raw"])
    _assert_same_eit_data(filtered, sequence.eit_data["mdn_filtered"])
    _assert_same_continuous_data(
        summed, sequence.eit_data["mdn_filtered"].get_summed_impedance()
    )


def test_unlocked_data_stays_unlocked_and_signal_arrays_stay_writable():
    raw = read_eit_file(
        os.path.join(REPO_ROOT, "tests", "data", "draeger_synthetic_draeger_20Hz.bin"),
        "draeger",
    )
    pixels = signal_to_eit_data(raw.pixel_impedance)
    summed = pixels.get_summed_impedance(return_label="global")
    summed.unlock()
    signal = continuous_data_to_signal(
        summed, modality="eit", channel="global_impedance", processing_state="raw"
    )

    back = signal_to_continuous_data(signal)

    assert not back.locked
    native = raw.signals["global_impedance_(raw)"]
    assert signal_to_continuous_data(native).locked
    assert native.values.flags["WRITEABLE"]
    assert native.time.flags["WRITEABLE"]


def test_m3resp_signal_without_eitprocessing_fields():
    path = os.path.join(REPO_ROOT, "tests", "data", "sentec_healthy_volunteer_1b.zri")
    pixels = read_eit_file(path, "sentec").pixel_impedance
    assert EITPROCESSING_FIELDS not in pixels.metadata

    eit_data = signal_to_eit_data(pixels)
    back = eit_data_to_signal(eit_data)

    assert eit_data.vendor.value == "sentec"
    assert eit_data.label == eit_data.name == pixels.name
    assert eit_data.sample_frequency == pixels.sample_frequency
    np.testing.assert_array_equal(back.values, pixels.values)
    np.testing.assert_array_equal(back.time, pixels.time)
    assert back.metadata[EITPROCESSING_FIELDS]["vendor"] == "sentec"


def test_global_impedance_signal_gives_its_category_and_name():
    signal = Signal(
        values=np.array([1.0, 2.0, 3.0]),
        time=np.array([0.0, 0.05, 0.1]),
        sample_frequency=20.0,
        unit="AU",
        name="global_impedance",
        modality="eit",
        category="impedance",
        channel="global_impedance",
    )

    data = signal_to_continuous_data(signal)

    assert (data.label, data.name, data.unit, data.category) == (
        "global_impedance",
        "global_impedance",
        "a.u.",
        "impedance",
    )
    assert data.parameters == {}
    assert data.derived_from == []


def test_signal_to_eit_data_needs_images():
    signal = Signal(
        values=np.zeros(3), time=np.arange(3.0), source="draeger", name="flat"
    )

    with pytest.raises(ValueError, match="shape \\(frames, rows, columns\\)"):
        signal_to_eit_data(signal)


def test_signal_to_eit_data_needs_a_device_maker():
    signal = Signal(
        values=np.zeros((3, 2, 2)),
        time=np.arange(3.0),
        sample_frequency=1.0,
        source="m3resp",
        name="images",
    )

    with pytest.raises(ValueError, match="device maker of 'images' is 'm3resp'"):
        signal_to_eit_data(signal)


def test_a_changed_signal_needs_a_name():
    raw = read_eit_file(
        os.path.join(REPO_ROOT, "tests", "data", "draeger_synthetic_draeger_20Hz.bin"),
        "draeger",
    )
    eit_data = signal_to_eit_data(raw.pixel_impedance)

    with pytest.raises(ValueError, match="give `name`"):
        eit_data_to_signal(eit_data, processing_state="intermediate")
