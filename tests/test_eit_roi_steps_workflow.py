"""The lung-space steps in a workflow.

``eit.roi_tiv_lungspace``, ``eit.roi_amplitude_lungspace`` and
``eit.roi_watershed`` take the global breaths from the optional input
``breaths``; without it, they find the breaths on ``timing_data`` with the
default settings. ``eit.roi_filter_by_size`` takes a mask of type
``pixel_mask``, and the workflow type check refuses other data there.
"""

from __future__ import annotations

import dataclasses
import os
import warnings

import numpy as np
import pytest

import m3resp.workflows.steps  # noqa: F401 - registers the steps
from m3resp.core.session import M3Session
from m3resp.eit.roi import tiv_lungspace
from m3resp.synchronization.start_times import on_clock_of
from m3resp.workflows import WorkflowExecutionError, run_workflow
from m3resp.workflows.compiler import validate_workflow
from m3resp.workflows.spec import StepSpec, load_spec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EIT_FULL_SPEC = os.path.join(
    REPO_ROOT, "examples", "eit_full_preprocessing", "eit-full.workflow.yaml"
)
MULTIMODAL_SPEC = os.path.join(
    REPO_ROOT, "examples", "multimodal_example", "multimodal.workflow.yaml"
)

#: Each lung-space step with its captures output and its mask output.
_LUNGSPACE_STEPS = (
    ("eit.roi_tiv_lungspace", "tiv_lungspace_captures", "tiv_lungspace_result"),
    (
        "eit.roi_amplitude_lungspace",
        "amplitude_lungspace_captures",
        "amplitude_lungspace_result",
    ),
    ("eit.roi_watershed", "watershed_captures", "watershed_lungspace_result"),
)


def _spec_with(path: str, *steps: StepSpec):
    spec = load_spec(path)
    return dataclasses.replace(spec, steps=(*spec.steps, *steps))


def _run(spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return run_workflow(spec, session=M3Session())


def test_the_type_check_accepts_a_pixel_mask_and_refuses_other_data() -> None:
    accepted = _spec_with(
        EIT_FULL_SPEC,
        StepSpec(
            uses="eit.roi_filter_by_size",
            id="filter_tiv_mask",
            inputs={"mask": "tiv_lungspace_result"},
            outputs={"size_filtered_roi_result": "filtered_tiv_mask"},
        ),
    )
    refused = _spec_with(
        EIT_FULL_SPEC,
        StepSpec(
            uses="eit.roi_filter_by_size",
            id="filter_pixel_tiv",
            inputs={"mask": "pixel_tiv_result"},
            outputs={"size_filtered_roi_result": "filtered_pixel_tiv"},
        ),
    )

    assert validate_workflow(accepted, readiness=False).structural == ()
    [diagnostic] = validate_workflow(refused, readiness=False).structural
    assert diagnostic.code == "artifact_type_mismatch"
    assert "'mask'" in diagnostic.message


def test_the_eit_breaths_of_a_workflow_give_the_same_masks() -> None:
    """In the EIT full example, 'eit.detect_breaths' uses the default
    settings, so its breaths give the masks of the steps without 'breaths'."""

    spec = _spec_with(
        EIT_FULL_SPEC,
        *(
            StepSpec(
                uses=operation,
                id=f"{operation.split('.')[1]}_on_breaths",
                inputs={"breaths": "breath_intervals"},
                outputs={
                    captures: f"{captures}_on_breaths",
                    output: f"{output}_on_breaths",
                },
            )
            for operation, captures, output in _LUNGSPACE_STEPS
        ),
    )

    result = _run(spec)

    assert result.status == "succeeded"
    for _, _, output in _LUNGSPACE_STEPS:
        found = result.value(output)
        given = result.value(f"{output}_on_breaths")
        np.testing.assert_array_equal(given.values, found.values)
        assert found.metadata["parameters"]["breaths"] == "timing_data"
        assert given.metadata["parameters"]["breaths"] == "input"


def test_the_tiv_mask_takes_breaths_of_another_modality() -> None:
    pytest.importorskip("eitprocessing")
    pytest.importorskip("resurfemg")
    spec = _spec_with(
        MULTIMODAL_SPEC,
        StepSpec(
            uses="eit.roi_tiv_lungspace",
            id="tiv_mask_on_ventilator_breaths",
            inputs={
                "eit_data": "filtered_eit",
                "timing_data": "global_impedance",
                "breaths": "ventilator_breath_events",
            },
        ),
    )

    # The last ventilator breath ends after the EIT data.
    with pytest.warns(UserWarning, match="1 of 13 breaths lie partly outside"):
        result = run_workflow(spec, session=M3Session())

    assert result.status == "succeeded"
    session = result.session
    mask = result.value("tiv_lungspace_result")
    with pytest.warns(UserWarning, match="1 of 13 breaths lie partly outside"):
        expected = tiv_lungspace(
            result.value("filtered_eit"),
            on_clock_of(session, session.events["ventilator_breaths"], "eit"),
        )
    np.testing.assert_array_equal(mask.values, expected.values)
    assert mask.metadata["parameters"]["breaths"] == "input"


@pytest.mark.parametrize(
    "operation", ["eit.roi_amplitude_lungspace", "eit.roi_watershed"]
)
def test_amplitude_and_watershed_refuse_breaths_of_another_modality(
    operation: str,
) -> None:
    pytest.importorskip("eitprocessing")
    pytest.importorskip("resurfemg")
    spec = _spec_with(
        MULTIMODAL_SPEC,
        StepSpec(
            uses=operation,
            id="mask_on_ventilator_breaths",
            inputs={
                "eit_data": "filtered_eit",
                "timing_data": "global_impedance",
                "breaths": "ventilator_breath_events",
            },
        ),
    )

    with pytest.raises(
        WorkflowExecutionError, match="EIT breaths only; got breaths of ventilator"
    ):
        _run(spec)
