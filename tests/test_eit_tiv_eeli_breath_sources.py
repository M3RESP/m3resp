"""TIV and EELI steps on breaths of any modality (#113).

``eit.continuous_tiv`` and ``eit.eeli`` take breaths through the input
``breaths``. EIT breaths give the values of the breath detection. Breaths of
another modality are moved to the EIT clock with the start times of the
session, and each time is read at the nearest sample. The expected values
here are hand calculations.
"""

from __future__ import annotations

import dataclasses
import os
import warnings

import numpy as np
import pytest

import m3resp.workflows.steps  # noqa: F401 - registers the steps
from m3resp.adapters.eitprocessing_adapter import as_signal, breaths_to_interval_data
from m3resp.core.exceptions import UnsynchronizedDataWarning
from m3resp.core.session import M3Session
from m3resp.data import BreathEvent, IntervalData, Signal
from m3resp.eit.io import read_eit_file
from m3resp.eit.processing import (
    BreathDetectionSettings,
    end_expiratory_lung_impedance,
    find_breaths,
    global_impedance,
)
from m3resp.synchronization.start_times import on_clock_of, shared_clock_shift
from m3resp.synchronization.sync_methods import clock_key
from m3resp.workflows import run_workflow
from m3resp.workflows.compiler import validate_workflow
from m3resp.workflows.spec import StepSpec, load_spec
from m3resp.workflows.steps.eit import continuous_tiv, eeli

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "tests", "data")
DRAEGER_FILE = os.path.join(DATA_DIR, "m3resp_multimodal_1_eit_draeger.bin")
TIMPEL_FILE = os.path.join(DATA_DIR, "timpel_healthy_volunteer_1_60-120s.txt")
MULTIMODAL_SPEC = os.path.join(
    REPO_ROOT, "examples", "multimodal_example", "multimodal.workflow.yaml"
)


def _nearest(signal: Signal, time: float) -> float:
    """The value of ``signal`` at the sample nearest to ``time``."""

    return float(np.asarray(signal.values)[np.argmin(np.abs(signal.time - time))])


def _eit_session() -> M3Session:
    session = M3Session()
    session.load_eit(DRAEGER_FILE, vendor="draeger")
    return session


def _synchronize(session: M3Session, start_times: dict[str, float]) -> None:
    """Set start times and mark each recording as synchronized."""

    session.start_times.update(start_times)
    for modality in start_times:
        session.sync_methods[clock_key(session, modality)] = "manual_offset"


# EIT breaths


def test_eit_breaths_through_the_input_give_the_values_of_the_detector() -> None:
    session = _eit_session()
    waveform = as_signal(session.eit.global_impedance)
    settings = BreathDetectionSettings()
    breaths = breaths_to_interval_data(find_breaths(waveform, settings))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        found_tiv = continuous_tiv(waveform, session=session, breath_detector=settings)
        given_tiv = continuous_tiv(waveform, session=session, breaths=breaths)
        found_eeli = eeli(waveform, session=session, breath_detector=settings)
        given_eeli = eeli(waveform, session=session, breaths=breaths)

    found_tiv = found_tiv["continuous_tiv_result"]
    given_tiv = given_tiv["continuous_tiv_result"]
    found_eeli, given_eeli = found_eeli["eeli_result"], given_eeli["eeli_result"]
    np.testing.assert_array_equal(given_tiv.values, found_tiv.values)
    np.testing.assert_array_equal(given_eeli.values, found_eeli.values)
    assert all(
        a is b for a, b in zip(given_tiv.intervals, breaths.intervals, strict=True)
    )
    assert given_tiv.metadata["parameters"]["breaths"] == "input"
    assert found_tiv.metadata["parameters"]["breaths"] == "breath_detector"


def test_timpel_device_breaths_give_the_impedance_at_their_samples() -> None:
    data = read_eit_file(TIMPEL_FILE, "timpel")
    waveform = global_impedance(data.pixel_impedance)
    device_breaths = data.intervals["breaths_(timpel)"]
    assert len(device_breaths) > 1

    function_result = end_expiratory_lung_impedance(waveform, device_breaths.intervals)
    step_result = eeli(waveform, session=M3Session(), breaths=device_breaths)[
        "eeli_result"
    ]

    expected = [_nearest(waveform, b.end_time) for b in device_breaths.intervals]
    np.testing.assert_array_equal(function_result.values, expected)
    np.testing.assert_array_equal(step_result.values, expected)
    assert step_result.metadata["breath_modality"] == "eit"


def test_given_breaths_are_refused_when_they_are_not_breath_events() -> None:
    session = _eit_session()
    with pytest.raises(TypeError, match="needs BreathEvent objects"):
        eeli(
            as_signal(session.eit.global_impedance),
            session=session,
            breaths=[(0.0, 1.0)],
        )


def test_eitprocessing_breaths_through_the_input_give_the_values_of_the_detector() -> (
    None
):
    pytest.importorskip("eitprocessing")
    from eitprocessing.features.breath_detection import BreathDetection

    from m3resp.adapters.eitprocessing_adapter import (
        EITProcessingAdapter,
        as_continuous_data,
    )

    session = _eit_session()
    waveform = as_signal(session.eit.global_impedance)
    native_breaths = BreathDetection().find_breaths(as_continuous_data(waveform))
    expected = eeli(
        waveform, session=session, breath_detector=BreathDetectionSettings()
    )

    step_result = eeli(waveform, session=session, breaths=native_breaths)
    adapter_result = EITProcessingAdapter().compute_eeli(
        waveform, breaths=native_breaths
    )

    expected_values = expected["eeli_result"].values
    np.testing.assert_array_equal(step_result["eeli_result"].values, expected_values)
    np.testing.assert_array_equal(adapter_result.values, expected_values)


def test_a_step_with_no_breaths_and_no_detector_is_refused() -> None:
    session = _eit_session()
    with pytest.raises(ValueError, match="needs the breaths"):
        eeli(as_signal(session.eit.global_impedance), session=session)


# Breaths of another modality


def _other_breaths() -> list[BreathEvent]:
    return [
        BreathEvent("ventilator", 1.0, 3.0, extremum_time=2.0),
        BreathEvent("ventilator", 3.0, 5.2, extremum_time=4.1),
        BreathEvent("emg", 6.03, 8.51, extremum_time=7.0),
    ]


def test_breaths_of_another_modality_are_moved_to_the_eit_clock() -> None:
    session = _eit_session()
    _synchronize(session, {"eit": 0.0, "ventilator": 3.0, "emg": -2.0})
    waveform = as_signal(session.eit.global_impedance)
    first_time = float(waveform.time[0])  # the time of day of the first frame
    breaths = _other_breaths()
    times = [(b.start_time, b.extremum_time, b.end_time) for b in breaths]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tiv = continuous_tiv(waveform, session=session, breaths=breaths)[
            "continuous_tiv_result"
        ]
        result = eeli(waveform, session=session, breaths=breaths)["eeli_result"]

    # On the EIT clock: own time + start time of the modality + first EIT time.
    shift = {"ventilator": 3.0 + first_time, "emg": -2.0 + first_time}
    expected_eeli = [
        _nearest(waveform, b.end_time + shift[b.modality]) for b in breaths
    ]
    expected_tiv = [
        _nearest(waveform, b.extremum_time + shift[b.modality])
        - _nearest(waveform, b.start_time + shift[b.modality])
        for b in breaths
    ]
    np.testing.assert_array_equal(result.values, expected_eeli)
    np.testing.assert_array_equal(tiv.values, expected_tiv)
    # The result keeps the given breaths, on their own clock.
    assert all(a is b for a, b in zip(result.intervals, breaths, strict=True))
    assert [(b.start_time, b.extremum_time, b.end_time) for b in breaths] == times
    assert result.metadata["breath_modality"] == ["emg", "ventilator"]
    assert result.modality == "eit"


def test_unsynchronized_breaths_of_another_modality_give_a_warning() -> None:
    session = _eit_session()
    waveform = as_signal(session.eit.global_impedance)
    first_time = float(waveform.time[0])
    breaths = _other_breaths()[:2]

    with pytest.warns(UnsynchronizedDataWarning, match="eit.eeli compares"):
        result = eeli(waveform, session=session, breaths=breaths)["eeli_result"]

    # With no start times, both recordings start at 0 s on the shared clock.
    expected = [_nearest(waveform, b.end_time + first_time) for b in breaths]
    np.testing.assert_array_equal(result.values, expected)


def test_breaths_of_another_modality_outside_the_eit_data_give_nan() -> None:
    session = _eit_session()
    _synchronize(session, {"eit": 0.0, "ventilator": -1000.0})
    breaths = _other_breaths()[:2]

    with pytest.warns(UserWarning, match="2 of 2 breaths lie partly outside"):
        result = eeli(
            as_signal(session.eit.global_impedance), session=session, breaths=breaths
        )

    assert np.isnan(result["eeli_result"].values).all()


def test_on_clock_of_moves_each_modality_by_its_start_time() -> None:
    session = _eit_session()
    _synchronize(session, {"eit": 0.0, "ventilator": 3.0})
    first_time = float(as_signal(session.eit.global_impedance).time[0])
    # "vent" is another spelling of the ventilator modality.
    ventilator = BreathEvent("vent", 1.0, 3.0, extremum_time=2.0)
    eit = BreathEvent("eit", first_time + 1.0, first_time + 2.0)

    moved_ventilator, moved_eit = on_clock_of(session, [ventilator, eit], "eit")

    shift = 3.0 + first_time
    assert moved_ventilator.start_time == pytest.approx(1.0 + shift)
    assert moved_ventilator.extremum_time == pytest.approx(2.0 + shift)
    assert moved_ventilator.end_time == pytest.approx(3.0 + shift)
    assert moved_ventilator.modality == "vent"
    assert (moved_eit.start_time, moved_eit.end_time) == (eit.start_time, eit.end_time)


def test_the_eeli_figure_shows_breaths_of_another_modality_on_the_eit_clock(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    from m3resp.visualization.eit_figures import save_eit_figures

    session = _eit_session()
    _synchronize(session, {"eit": 0.0, "ventilator": 3.0})
    waveform = as_signal(session.eit.global_impedance)
    breaths = [
        *_other_breaths()[:2],
        BreathEvent("ventilator", 1.0e5, 1.0e5 + 2.0, extremum_time=1.0e5 + 1.0),
    ]
    with pytest.warns(UserWarning, match="1 of 3 breaths lie partly outside"):
        result = eeli(waveform, session=session, breaths=breaths)["eeli_result"]

    lines_of_saved_figures = []
    savefig = Figure.savefig

    def keep_lines(figure, *args, **kwargs):
        lines = figure.axes[0].lines
        lines_of_saved_figures.append({line.get_label(): line for line in lines})
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", keep_lines)
    save_eit_figures(
        {"global_impedance": waveform, "eeli_result": result}, tmp_path, session
    )

    [lines] = lines_of_saved_figures
    # The x axis is in seconds from the first EIT sample: breath end + 3 s.
    np.testing.assert_allclose(lines["EELIs"].get_xdata()[:2], [3.0 + 3.0, 5.2 + 3.0])
    assert np.isnan(result.values[2])
    # The mean and the median leave out the breath outside the EIT data.
    mean = lines["EELI mean"].get_ydata()[0]
    median = lines["EELI median"].get_ydata()[0]
    assert mean == pytest.approx(np.mean(result.values[:2]))
    assert median == pytest.approx(np.median(result.values[:2]))


# Workflows


def _multimodal_spec_with(*steps: StepSpec):
    spec = load_spec(MULTIMODAL_SPEC)
    return dataclasses.replace(spec, steps=(*spec.steps, *steps))


def test_the_workflow_type_check_accepts_breaths_and_refuses_other_data() -> None:
    accepted = _multimodal_spec_with(
        StepSpec(
            uses="eit.eeli",
            id="eeli_on_ventilator_breaths",
            inputs={
                "signal": "global_impedance",
                "breaths": "ventilator_breath_events",
            },
            outputs={"eeli_result": "ventilator_eeli"},
        ),
        StepSpec(
            uses="eit.continuous_tiv",
            id="tiv_on_eit_breaths",
            inputs={"signal": "global_impedance", "breaths": "breath_intervals"},
            outputs={"continuous_tiv_result": "eit_breath_tiv"},
        ),
    )
    refused = _multimodal_spec_with(
        StepSpec(
            uses="eit.eeli",
            id="eeli_on_a_waveform",
            inputs={"signal": "global_impedance", "breaths": "global_impedance"},
            outputs={"eeli_result": "wrong_eeli"},
        )
    )

    assert validate_workflow(accepted, readiness=False).structural == ()
    [diagnostic] = validate_workflow(refused, readiness=False).structural
    assert diagnostic.code == "artifact_type_mismatch"
    assert "'breaths'" in diagnostic.message


def test_the_workflow_check_refuses_a_step_with_no_breaths_and_no_detector() -> None:
    spec = _multimodal_spec_with(
        StepSpec(
            uses="eit.eeli",
            id="eeli_with_no_breaths",
            inputs={"signal": "global_impedance", "breath_detector": "no_detector"},
            outputs={"eeli_result": "no_eeli"},
        )
    )

    [diagnostic] = validate_workflow(spec, readiness=False).structural

    assert diagnostic.code == "missing_one_of_inputs"
    assert diagnostic.step_id == "eeli_with_no_breaths"


def test_the_workflow_computes_eeli_and_tiv_on_ventilator_and_emg_breaths() -> None:
    pytest.importorskip("eitprocessing")
    pytest.importorskip("resurfemg")
    spec = _multimodal_spec_with(
        StepSpec(
            uses="eit.eeli",
            id="eeli_on_ventilator_breaths",
            inputs={
                "signal": "global_impedance",
                "breaths": "ventilator_breath_events",
            },
            outputs={"eeli_result": "ventilator_eeli"},
        ),
        StepSpec(
            uses="eit.continuous_tiv",
            id="tiv_on_emg_breaths",
            inputs={"signal": "global_impedance", "breaths": "emg_breath_events"},
            outputs={"continuous_tiv_result": "emg_tiv"},
        ),
    )

    result = run_workflow(spec, session=M3Session())

    assert result.status == "succeeded"
    session = result.session
    waveform = result.value("global_impedance")
    ventilator_breaths = session.events["ventilator_breaths"]
    emg_breaths = session.events["emg_breaths"]
    ventilator_eeli = result.value("ventilator_eeli")
    emg_tiv = result.value("emg_tiv")
    assert isinstance(ventilator_eeli, IntervalData) and len(ventilator_eeli) > 0
    assert all(
        a is b
        for a, b in zip(ventilator_eeli.intervals, ventilator_breaths, strict=True)
    )
    assert all(a is b for a, b in zip(emg_tiv.intervals, emg_breaths, strict=True))

    eit_shift = shared_clock_shift(session, "eit")

    def on_eit_clock(time: float, modality: str) -> float:
        return time + shared_clock_shift(session, modality) - eit_shift

    time = np.asarray(waveform.time)
    half_sample = 0.5 / waveform.sample_frequency

    def inside(*times: float) -> bool:
        return all(time[0] - half_sample <= t <= time[-1] + half_sample for t in times)

    expected_eeli = []
    for b in ventilator_breaths:
        start, end = (on_eit_clock(t, "ventilator") for t in (b.start_time, b.end_time))
        expected_eeli.append(_nearest(waveform, end) if inside(start, end) else np.nan)
    np.testing.assert_array_equal(ventilator_eeli.values, expected_eeli)
    expected_tiv = []
    for b in emg_breaths:
        start, extremum, end = (
            on_eit_clock(t, "emg") for t in (b.start_time, b.extremum_time, b.end_time)
        )
        expected_tiv.append(
            _nearest(waveform, extremum) - _nearest(waveform, start)
            if inside(start, extremum, end)
            else np.nan
        )
    np.testing.assert_array_equal(emg_tiv.values, expected_tiv)
