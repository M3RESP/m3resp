"""Tests for ``register_step(..., record=True)``: one provenance record per call."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import pytest

from m3resp import M3Session
from m3resp.core.exceptions import StepMetadataError
from m3resp.workflows import run_workflow
from m3resp.workflows.registry import STEP_REGISTRY, register_step


@pytest.fixture
def registered():
    """Names of test steps, removed from the registry after the test."""

    names: list[str] = []
    yield names
    for name in names:
        STEP_REGISTRY.pop(name, None)


def _register_scaling_step(registered: list[str], **options: Any):
    @register_step(
        "record_test.scale",
        reads={"session": "session", "values": "values"},
        writes=("scaled",),
        modality="emg",
        record=True,
        **options,
    )
    def scale(session: M3Session, values: list[float], factor: float = 2.0):
        return {
            "scaled": [factor * value for value in values],
            "_provenance": {"parameters": {"factor": factor}},
        }

    registered.append("record_test.scale")
    return scale


def test_a_direct_call_adds_one_record_with_the_step_details(registered):
    scale = _register_scaling_step(registered)
    session = M3Session()

    outputs = scale(session=session, values=[1.0, 2.0], factor=3.0)

    assert outputs == {"scaled": [3.0, 6.0]}
    assert len(session.provenance) == 1
    entry = session.provenance[0]
    assert entry.action == "record_test.scale"
    assert entry.modality == "emg"
    assert entry.parameters == {
        "step": "record_test.scale",
        "reads": ["session", "values"],
        "writes": ["scaled"],
        "upstream_version": None,
        "parameters": {"factor": 3.0},
    }


def test_a_session_passed_by_position_is_found(registered):
    scale = _register_scaling_step(registered)
    session = M3Session()

    scale(session, [1.0])

    assert [entry.action for entry in session.provenance] == ["record_test.scale"]


def test_a_workflow_run_records_once_and_keeps_the_details_out_of_its_outputs(
    registered,
):
    _register_scaling_step(registered)
    session = M3Session()

    result = run_workflow(
        {"name": "scale", "steps": [{"uses": "record_test.scale"}]},
        session=session,
        extra_context={"values": [1.0, 2.0]},
    )

    assert result.outputs["scaled"] == [2.0, 4.0]
    assert "_provenance" not in result.outputs
    step_records = [e for e in session.provenance if e.action == "record_test.scale"]
    assert len(step_records) == 1
    assert step_records[0].parameters["parameters"] == {"factor": 2.0}


def test_upstream_version_is_the_first_optional_package_version(registered):
    scale = _register_scaling_step(registered, optional_packages=("numpy", "scipy"))
    session = M3Session()

    scale(session=session, values=[1.0])

    assert session.provenance[0].parameters["upstream_version"] == version("numpy")


def test_upstream_version_is_none_for_a_package_that_is_not_installed(registered):
    scale = _register_scaling_step(
        registered, optional_packages=("m3resp_test_package_absent",)
    )
    session = M3Session()

    scale(session=session, values=[1.0])

    assert session.provenance[0].parameters["upstream_version"] is None


def test_record_needs_the_step_to_read_session(registered):
    with pytest.raises(StepMetadataError, match="must read 'session'"):

        @register_step(
            "record_test.no_session", reads={"values": "values"}, record=True
        )
        def _no_session(values: list[float]) -> dict[str, Any]:
            return {}

    assert "record_test.no_session" not in STEP_REGISTRY


def test_a_call_without_a_session_raises(registered):
    scale = _register_scaling_step(registered)

    with pytest.raises(TypeError, match="needs a session"):
        scale(session=None, values=[1.0])


def test_the_registry_holds_the_recording_function(registered):
    scale = _register_scaling_step(registered)

    definition = STEP_REGISTRY["record_test.scale"]

    assert definition.func is scale
    assert definition.record is True
