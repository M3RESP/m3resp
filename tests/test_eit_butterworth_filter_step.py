"""The step ``eit.butterworth_filter``, run by m3resp code.

The step must give the numbers of eitprocessing 1.8.7 ``ButterworthFilter``
on the EIT images of the 3 test files. A pixel without any value stays
without value; eitprocessing refuses NaN, so the reference filters those
pixels as 0 and sets them back to NaN, as the step does.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import m3resp.workflows.steps  # noqa: F401  (registers the steps)
from m3resp import M3Session
from m3resp.data import Signal
from m3resp.eit.io import read_eit_file
from m3resp.workflows.engine.spec_runner import _infer_filter_mode
from m3resp.workflows.registry import get_step

pytest.importorskip("eitprocessing")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")
with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as _handle:
    MANIFEST = json.load(_handle)
DATASETS = sorted(MANIFEST["inputs"])

#: Cutoffs in Hz of each mode, as the step takes them (lowpass_hz, highpass_hz).
CUTOFFS = {"lowpass_hz": 1.0, "highpass_hz": 0.05}


@pytest.fixture(scope="module")
def images() -> dict[str, Signal]:
    """The EIT images of the 3 test files, read by m3resp."""

    return {
        dataset: read_eit_file(
            os.path.join(REPO_ROOT, MANIFEST["inputs"][dataset]["file"]),
            MANIFEST["inputs"][dataset]["vendor"],
        ).pixel_impedance
        for dataset in DATASETS
    }


def _sequence() -> SimpleNamespace:
    """An eitprocessing image collection, as the ``Sequence`` of a session."""

    from eitprocessing.datahandling.datacollection import DataCollection
    from eitprocessing.datahandling.eitdata import EITData

    return SimpleNamespace(eit_data=DataCollection(EITData))


def _run_step(signal: Signal, **settings) -> tuple[dict, SimpleNamespace]:
    sequence = _sequence()
    output = get_step("eit.butterworth_filter").func(
        signal, eit_sequence=sequence, session=M3Session(), **settings
    )
    return output, sequence


def _upstream(signal: Signal, mode: str, order: int) -> tuple[np.ndarray, dict]:
    """eitprocessing ``ButterworthFilter.apply`` on the same images."""

    from eitprocessing.filters.butterworth_filters import ButterworthFilter

    cutoff = {"lowpass": 1.0, "highpass": 0.05}.get(mode, (0.05, 1.0))
    values = np.asarray(signal.values, dtype=float)
    unmeasured = np.isnan(values).all(axis=0)
    placeholder = values.copy()
    placeholder[:, unmeasured] = 0.0
    captures: dict = {}
    filtered = ButterworthFilter(
        filter_type=mode,
        cutoff_frequency=cutoff,
        order=order,
        sample_frequency=signal.sample_frequency,
    ).apply(placeholder, axis=0, captures=captures)
    filtered[:, unmeasured] = np.nan
    return filtered, captures


@pytest.mark.parametrize("order", [1, 4, 10])
@pytest.mark.parametrize("mode", ["lowpass", "highpass", "bandpass", "bandstop"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_the_step_gives_the_images_and_captures_of_eitprocessing(
    dataset: str, mode: str, order: int, images: dict
) -> None:
    signal = images[dataset]
    expected, expected_captures = _upstream(signal, mode, order)

    output, sequence = _run_step(
        signal, mode=mode, order=order, label="filtered", **CUTOFFS
    )

    filtered = output["filtered_eit"]
    np.testing.assert_array_equal(filtered.values, expected)
    captures = output["filter_captures"]
    assert set(captures) == set(expected_captures)
    for key in set(captures) - {"unfiltered_data", "filtered_data"}:
        assert captures[key] == expected_captures[key], key
    assert captures["unfiltered_data"] is signal
    assert captures["filtered_data"] is filtered
    np.testing.assert_array_equal(
        sequence.eit_data["filtered"].pixel_impedance, expected
    )


def test_the_result_describes_the_filter(images: dict) -> None:
    signal = images["sentec_healthy_volunteer_1b"]

    output, sequence = _run_step(signal, mode="bandpass", label="band", **CUTOFFS)

    filtered = output["filtered_eit"]
    assert filtered is output["filtered_eit_signal"]
    assert filtered.name == "Bandpass-filtered EIT data"
    assert filtered.source == "sentec"
    assert filtered.processing_state == "intermediate"
    assert filtered.method == "m3resp.processing.filters.butterworth_filter"
    assert filtered.metadata["source_package"] == "m3resp"
    assert filtered.metadata["parameters"]["mode"] == "bandpass"
    assert _infer_filter_mode(filtered) == "bandpass"
    np.testing.assert_array_equal(filtered.time, signal.time)
    assert filtered.time is not signal.time
    stored = sequence.eit_data["band"]
    assert stored.label == "band"
    assert stored.name == "Bandpass-filtered EIT data"
    assert stored.description == "EIT data filtered with a bandpass Butterworth filter."


def test_pixels_without_any_value_stay_without_value(images: dict) -> None:
    signal = images["timpel_healthy_volunteer_1_60-120s"]
    unmeasured = np.isnan(signal.values).all(axis=0)
    assert unmeasured.sum() == 501

    output, _ = _run_step(signal)

    filtered = output["filtered_eit"].values
    assert np.isnan(filtered[:, unmeasured]).all()
    assert np.isfinite(filtered[:, ~unmeasured]).all()


def test_a_pixel_missing_at_some_samples_is_refused(images: dict) -> None:
    signal = images["draeger_synthetic_20Hz"]
    values = np.array(signal.values, dtype=float)
    values[10:12, 5, 5] = np.nan
    gap = Signal(
        values=values,
        time=signal.time,
        sample_frequency=signal.sample_frequency,
        name="images with a gap",
        modality="eit",
        source="draeger",
    )

    with pytest.raises(ValueError, match="missing samples for part"):
        _run_step(gap)


@pytest.mark.parametrize("order", [0, 11])
def test_an_order_outside_1_to_10_is_refused_by_the_step_and_preprocess(
    order: int, images: dict
) -> None:
    signal = images["draeger_synthetic_20Hz"]
    with pytest.raises(ValueError, match="from 1 to 10"):
        _run_step(signal, order=order)

    session = M3Session()
    entry = MANIFEST["inputs"]["draeger_synthetic_20Hz"]
    session.load_eit(os.path.join(REPO_ROOT, entry["file"]), entry["vendor"])
    with pytest.raises(ValueError, match="from 1 to 10"):
        session.preprocess_eit(filter_mode="lowpass", filter_order=order)


def test_the_order_parameter_has_the_limits_1_and_10() -> None:
    order = next(
        p for p in get_step("eit.butterworth_filter").parameters if p.name == "order"
    )

    assert (order.minimum, order.maximum) == (1, 10)


@pytest.mark.parametrize("mode", ["lowpass", "bandpass"])
def test_preprocess_gives_the_eitprocessing_images_and_signal_captures(
    mode: str,
) -> None:
    import copy

    from eitprocessing.filters.butterworth_filters import ButterworthFilter

    session = M3Session()
    entry = MANIFEST["inputs"]["draeger_synthetic_20Hz"]
    session.load_eit(os.path.join(REPO_ROOT, entry["file"]), entry["vendor"])
    sequence = copy.deepcopy(session.eit.data)
    raw = sequence.eit_data["raw"]
    expected = copy.deepcopy(raw)
    expected.label = f"{mode}_filtered"
    expected.name = f"{mode.title()}-filtered EIT data"
    expected.description = f"EIT data filtered with a {mode} Butterworth filter."
    expected.pixel_impedance = ButterworthFilter(
        filter_type=mode,
        cutoff_frequency=1.0 if mode == "lowpass" else (0.05, 1.0),
        order=4,
        sample_frequency=raw.sample_frequency,
    ).apply(raw.pixel_impedance, axis=0)

    result = session.eit_adapter.preprocess(
        sequence,
        filter_mode=mode,
        compute_continuous_tiv=False,
        compute_eeli=False,
        compute_pixel_tiv=False,
    )

    filtered = result["filtered_eit"]
    assert filtered == expected
    for field in ("label", "name", "description", "path", "vendor"):
        assert getattr(filtered, field) == getattr(expected, field), field
    assert sequence.eit_data[f"{mode}_filtered"] is filtered
    assert result["filter_mode"] == mode
    captures = result["filter_captures"]
    assert isinstance(captures["unfiltered_data"], Signal)
    assert isinstance(captures["filtered_data"], Signal)
    np.testing.assert_array_equal(
        captures["filtered_data"].values, expected.pixel_impedance
    )
    np.testing.assert_array_equal(
        captures["unfiltered_data"].values, raw.pixel_impedance
    )
