"""The pixel EIT steps in a workflow.

``eit.pixel_tiv`` and ``eit.pixel_eeli`` take the breaths of each pixel from
the optional input ``pixel_breaths``, by default the output ``pixel_breaths``
of an earlier ``eit.pixel_breaths`` step. The workflow type check accepts the
``PixelBreaths`` of ``eit.pixel_breaths`` there and refuses other data. With
the 'pixel' timing, the values are the same with and without that input, and
the same as eitprocessing 1.8.7 ``TIV.compute_pixel_parameter``. With the
'continuous' timing, the steps take breaths of every modality, as
``eit.continuous_tiv`` and ``eit.eeli``.
"""

from __future__ import annotations

import dataclasses
import os
import warnings

import numpy as np
import pytest

import m3resp.workflows.steps  # noqa: F401 - registers the steps
from m3resp.adapters.eitprocessing_adapter import (
    as_continuous_data,
    as_eit_data,
    as_signal,
)
from m3resp.core.session import M3Session
from m3resp.data import PixelBreaths
from m3resp.eit.processing import (
    pixel_end_expiratory_lung_impedance,
    pixel_tidal_impedance_variation,
)
from m3resp.synchronization.start_times import on_clock_of
from m3resp.workflows import WorkflowExecutionError, run_workflow
from m3resp.workflows.compiler import compile_workflow, validate_workflow
from m3resp.workflows.spec import StepSpec, load_spec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EIT_FULL_SPEC = os.path.join(
    REPO_ROOT, "examples", "eit_full_preprocessing", "eit-full.workflow.yaml"
)
MULTIMODAL_SPEC = os.path.join(
    REPO_ROOT, "examples", "multimodal_example", "multimodal.workflow.yaml"
)

#: The minimum breath duration of ``eit.detect_breaths`` in the eit-full
#: example.
EIT_FULL_MINIMUM_DURATION = 0.6666666667


def _spec_with(path: str, *steps: StepSpec, pixel_breaths_key: str | None = None):
    """The spec with extra steps. ``pixel_breaths_key`` renames the output
    ``pixel_breaths`` of ``eit.pixel_breaths``."""

    spec = load_spec(path)
    own_steps = spec.steps
    if pixel_breaths_key is not None:
        own_steps = tuple(
            dataclasses.replace(step, outputs={"pixel_breaths": pixel_breaths_key})
            if step.uses == "eit.pixel_breaths"
            else step
            for step in own_steps
        )
    return dataclasses.replace(spec, steps=(*own_steps, *steps))


def _maps(result) -> np.ndarray:
    return np.stack([np.asarray(pixel_map, dtype=float) for pixel_map in result.values])


def test_the_pixel_breaths_input_reads_the_output_of_eit_pixel_breaths() -> None:
    compiled = compile_workflow(load_spec(EIT_FULL_SPEC))

    [step] = [s for s in compiled.steps if s.operation_id == "eit.pixel_tiv"]
    assert step.input_bindings["pixel_breaths"] == "pixel_breaths"
    assert step.input_bindings["breath_detector"] == "breath_detector"


def test_the_pixel_timing_uses_the_pixel_breaths_of_an_earlier_step() -> None:
    """The phase shift mode of 'eit.pixel_breaths' reaches 'eit.pixel_tiv'
    without an 'in:' binding."""

    spec = load_spec(EIT_FULL_SPEC)
    steps = tuple(
        dataclasses.replace(step, params={"phase_correction_mode": "phase shift"})
        if step.uses == "eit.pixel_breaths"
        else step
        for step in spec.steps
    )
    spec = dataclasses.replace(
        spec,
        steps=(
            *steps,
            StepSpec(
                uses="eit.pixel_tiv",
                id="tiv_on_phase_shift",
                params={"tiv_timing": "pixel"},
                outputs={"pixel_tiv_result": "phase_shift_tiv"},
            ),
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run_workflow(spec, session=M3Session())

    assert result.status == "succeeded"
    found = result.value("pixel_breaths")
    assert found.metadata["phase_correction_mode"] == "phase shift"
    tiv = result.value("phase_shift_tiv")
    assert tiv.metadata["parameters"]["breaths"] == "pixel_breaths"
    np.testing.assert_array_equal(
        _maps(tiv),
        _maps(pixel_tidal_impedance_variation(result.value("filtered_eit"), found)),
    )


def test_the_type_check_accepts_pixel_breaths_and_refuses_other_data() -> None:
    accepted = _spec_with(
        EIT_FULL_SPEC,
        StepSpec(
            uses="eit.pixel_tiv",
            id="tiv_on_pixel_breaths",
            inputs={"pixel_breaths": "pixel_breaths"},
            params={"tiv_timing": "pixel"},
            outputs={"pixel_tiv_result": "pixel_timing_tiv"},
        ),
    )
    refused = _spec_with(
        EIT_FULL_SPEC,
        StepSpec(
            uses="eit.pixel_eeli",
            id="eeli_on_time_grids",
            inputs={"pixel_breaths": "pixel_breath_timing_result"},
            params={"eeli_timing": "pixel"},
        ),
    )

    assert validate_workflow(accepted, readiness=False).structural == ()
    [diagnostic] = validate_workflow(refused, readiness=False).structural
    assert diagnostic.code == "artifact_type_mismatch"
    assert "'pixel_breaths'" in diagnostic.message


def test_pixel_timing_in_a_workflow_gives_the_values_of_eitprocessing() -> None:
    pytest.importorskip("eitprocessing")
    from eitprocessing.features.breath_detection import BreathDetection
    from eitprocessing.parameters.tidal_impedance_variation import TIV

    spec = _spec_with(
        EIT_FULL_SPEC,
        StepSpec(
            uses="eit.pixel_tiv",
            id="tiv_on_pixel_breaths",
            inputs={"pixel_breaths": "own_pixel_breaths"},
            params={"tiv_timing": "pixel", "tiv_method": "expiratory"},
            outputs={"pixel_tiv_result": "tiv_connected"},
        ),
        StepSpec(
            uses="eit.pixel_tiv",
            id="tiv_with_own_pixel_breaths",
            params={"tiv_timing": "pixel", "tiv_method": "expiratory"},
            outputs={"pixel_tiv_result": "tiv_alone"},
        ),
        StepSpec(
            uses="eit.pixel_eeli",
            id="eeli_on_pixel_breaths",
            inputs={"pixel_breaths": "own_pixel_breaths"},
            params={"eeli_timing": "pixel"},
            outputs={"pixel_eeli_result": "eeli_connected"},
        ),
        StepSpec(
            uses="eit.pixel_eeli",
            id="eeli_with_own_pixel_breaths",
            params={"eeli_timing": "pixel"},
            outputs={"pixel_eeli_result": "eeli_alone"},
        ),
        pixel_breaths_key="own_pixel_breaths",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run_workflow(spec, session=M3Session())

    assert result.status == "succeeded"
    pixels = result.value("filtered_eit")
    found = result.value("own_pixel_breaths")
    assert isinstance(found, PixelBreaths)
    connected, alone = result.value("tiv_connected"), result.value("tiv_alone")
    np.testing.assert_array_equal(_maps(connected), _maps(alone))
    np.testing.assert_array_equal(
        _maps(connected),
        _maps(pixel_tidal_impedance_variation(pixels, found, method="expiratory")),
    )
    np.testing.assert_array_equal(
        _maps(result.value("eeli_connected")), _maps(result.value("eeli_alone"))
    )
    np.testing.assert_array_equal(
        _maps(result.value("eeli_connected")),
        _maps(pixel_end_expiratory_lung_impedance(pixels, found)),
    )
    assert connected.metadata["parameters"]["breaths"] == "pixel_breaths"
    assert alone.metadata["parameters"]["breaths"] == "breath_detector"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        upstream = TIV(
            breath_detection=BreathDetection(minimum_duration=EIT_FULL_MINIMUM_DURATION)
        ).compute_pixel_parameter(
            as_eit_data(pixels),
            as_continuous_data(result.value("global_impedance")),
            None,
            tiv_method="expiratory",
            tiv_timing="pixel",
            store=False,
        )
    upstream_maps = np.stack(
        [np.asarray(grid, dtype=float) for grid in upstream.values]
    )
    np.testing.assert_array_equal(_maps(alone), upstream_maps)


def test_continuous_timing_takes_breaths_of_another_modality() -> None:
    pytest.importorskip("eitprocessing")
    pytest.importorskip("resurfemg")
    spec = _spec_with(
        MULTIMODAL_SPEC,
        StepSpec(
            uses="eit.pixel_eeli",
            id="pixel_eeli_on_ventilator_breaths",
            inputs={
                "eit_data": "filtered_eit",
                "signal": "global_impedance",
                "breaths": "ventilator_breath_events",
            },
            outputs={"pixel_eeli_result": "ventilator_pixel_eeli"},
        ),
    )

    # The last ventilator breath ends after the EIT data.
    with pytest.warns(UserWarning, match="1 of 13 breaths lie partly outside"):
        result = run_workflow(spec, session=M3Session())

    assert result.status == "succeeded"
    session = result.session
    ventilator_breaths = session.events["ventilator_breaths"]
    pixel_eeli = result.value("ventilator_pixel_eeli")
    assert all(
        a is b for a, b in zip(pixel_eeli.intervals, ventilator_breaths, strict=True)
    )
    with pytest.warns(UserWarning, match="1 of 13 breaths lie partly outside"):
        expected = pixel_end_expiratory_lung_impedance(
            as_signal(result.value("filtered_eit")),
            on_clock_of(session, ventilator_breaths, "eit"),
        )
    np.testing.assert_array_equal(_maps(pixel_eeli), _maps(expected))
    assert np.isfinite(_maps(pixel_eeli)).any()


def test_pixel_timing_refuses_breaths_of_another_modality() -> None:
    pytest.importorskip("eitprocessing")
    pytest.importorskip("resurfemg")
    spec = _spec_with(
        MULTIMODAL_SPEC,
        StepSpec(
            uses="eit.pixel_tiv",
            id="pixel_tiv_on_ventilator_breaths",
            inputs={"breaths": "ventilator_breath_events"},
            params={"tiv_timing": "pixel"},
            outputs={"pixel_tiv_result": "ventilator_pixel_tiv"},
        ),
    )

    with pytest.raises(
        WorkflowExecutionError, match="EIT breaths only; got breaths of ventilator"
    ):
        run_workflow(spec, session=M3Session())


def _small_pixel_tiv():
    """Pixel TIV of 2 x 2 images with one breath every 4 s."""

    from m3resp.data import BreathEvent, Signal

    time = np.arange(400) / 20.0
    breathing = np.sin(2 * np.pi * 0.25 * time)
    pixels = Signal(
        values=10.0 + breathing[:, None, None] * np.array([[1.0, 2.0], [3.0, 4.0]]),
        time=time,
        sample_frequency=20.0,
        name="images",
        modality="eit",
    )
    breaths = [
        BreathEvent("eit", start, start + 4.0, extremum_time=start + 2.0)
        for start in (3.0, 7.0, 11.0)
    ]
    return pixel_tidal_impedance_variation(pixels, breaths)


def test_the_pixel_tiv_figure_shows_the_mean_map_of_pixel_tiv_result(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("eitprocessing")
    from matplotlib.figure import Figure

    from m3resp.visualization.eit_figures import save_eit_figures

    pixel_tiv = _small_pixel_tiv()
    shown = []
    savefig = Figure.savefig

    def keep_image(figure, *args, **kwargs):
        shown.append(np.asarray(figure.axes[0].images[0].get_array()))
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", keep_image)
    paths = save_eit_figures({"pixel_tiv_result": pixel_tiv}, tmp_path)

    assert [path.name for path in paths] == ["pixel_tiv.png"]
    [image] = shown
    np.testing.assert_allclose(image, np.nanmean(_maps(pixel_tiv), axis=0))


def test_the_eit_summary_gives_the_shape_of_the_pixel_tiv_maps() -> None:
    from m3resp.workflows.summaries import summarize_eit

    session = M3Session()
    session.processed["eit"] = {"filter_mode": "none", "pixel_tiv": _small_pixel_tiv()}

    assert summarize_eit(session)["pixel_tiv_shape_per_breath"] == (2, 2)
